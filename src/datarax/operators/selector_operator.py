"""SelectorOperator - Random selection from multiple operators.

This operator wraps multiple OperatorModules and randomly selects ONE to apply
per batch element.

Key Features:

- Wraps multiple OperatorModules with random selection
- Configurable weights for weighted random selection (defaults to uniform)
- Uses jax.lax.switch for JIT-compatible dynamic selection
- Always stochastic (always makes a random choice)
- Full JAX compatibility (JIT, vmap)

Examples:
    Basic usage:

    ```python
    config = SelectorOperatorConfig(
        operators=[op1, op2, op3],
        weights=[0.5, 0.3, 0.2]
    )
    op = SelectorOperator(config, rngs=rngs)
    ```
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import PyTree

from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SelectorOperatorConfig(OperatorConfig):
    """Configuration for SelectorOperator.

    Extends OperatorConfig with operators list and optional weights.

    Attributes:
        operators: List of operators to select from (minimum 1)
        weights: Optional weights for random selection (defaults to uniform)
                 Will be normalized to sum to 1.0

    Note:

        - stochastic is always True (always makes random choice)
        - stream_name defaults to "augment" for random selection
    """

    operators: list[OperatorModule] = field(kw_only=True)
    weights: list[float] | None = field(default=None, kw_only=True)
    normalized_weights: jax.Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration and normalize weights."""
        # Validate at least one operator
        if not self.operators:
            raise ValueError("Must provide at least one operator")

        n_operators = len(self.operators)

        # Validate or create weights
        if self.weights is None:
            # Uniform weights by default
            weights = [1.0 / n_operators] * n_operators
        else:
            if len(self.weights) != n_operators:
                raise ValueError(
                    f"Number of weights ({len(self.weights)}) must match "
                    f"number of operators ({n_operators})"
                )
            weights = self.weights

        # Normalize weights to sum to 1.0
        total = sum(weights)
        normalized = [w / total for w in weights]
        object.__setattr__(self, "normalized_weights", jnp.array(normalized))

        # SelectorOperator is ALWAYS stochastic (always makes random choice)
        object.__setattr__(self, "stochastic", True)

        # Set stream_name for random selection
        if self.stream_name is None:
            object.__setattr__(self, "stream_name", "augment")

        super().__post_init__()


class SelectorOperator(OperatorModule):
    """Wrapper operator that randomly selects ONE operator to apply.

    Wraps multiple OperatorModules and uses weighted random selection to
    choose which one to apply per batch element.

    Uses jax.lax.switch for JIT-compatible operator selection with the
    unified operator interface.

    Examples:
        ```python
        op1 = BrightnessOperator(brightness_config, rngs=nnx.Rngs(0))  # Different transforms
        op2 = NoiseOperator(noise_config, rngs=nnx.Rngs(0))
        op3 = RotationOperator(rotation_config, rngs=nnx.Rngs(0))
        selector_config = SelectorOperatorConfig(  # 50% brightness, 30% noise, 20% rotation
            operators=[op1, op2, op3],
            weights=[0.5, 0.3, 0.2]
        )
        selector = SelectorOperator(selector_config, rngs=nnx.Rngs(0))
        result_batch = selector(batch)  # Each element gets one randomly selected operator
        ```
    """

    def __init__(
        self,
        config: SelectorOperatorConfig,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        """Initialize selector operator.

        Args:
            config: SelectorOperatorConfig with operators list and optional weights
            rngs: Random number generators (required for random selection)
        """
        super().__init__(config, rngs=rngs)

        # Type narrowing for pyright
        self.config: SelectorOperatorConfig = config

        # Store operators in NNX List for proper state management
        self.operators = nnx.List(config.operators)
        self.weights = nnx.static(tuple(float(w) for w in config.normalized_weights.tolist()))

    def get_output_structure(
        self,
        sample_data: PyTree,
        sample_state: PyTree,
    ) -> tuple[PyTree, PyTree]:
        """Declare output structure using first operator.

        SelectorOperator's apply() requires random_params which isn't available
        during jax.eval_shape tracing. Since all child operators should produce
        compatible output structures, we use the first operator's structure.

        Args:
            sample_data: Single element data (not batched)
            sample_state: Single element state (not batched)

        Returns:
            Tuple of (output_data_structure, output_state_structure) with 0 leaves.
        """
        # Use first operator to determine output structure
        # All operators should have compatible output structures
        first_op = self.operators[0]
        return first_op.get_output_structure(sample_data, sample_state)

    def generate_random_params(
        self,
        element_keys: jax.Array,
        data_shapes: PyTree,
    ) -> dict[str, Any]:
        """Generate per-record operator selection indices and child params.

        Each record independently selects an operator from its own key
        (``fold_in``), and each child receives its own per-record key set, so
        both the selection and each child's randomness are reproducible per
        record regardless of batch composition, shuffle, host count, or resume.

        Args:
            element_keys: ``(batch_size,)`` per-record PRNG keys.
            data_shapes: PyTree with same structure as batch.data, containing shapes.

        Returns:
            Dict with:

                - "selected_indices": Array of operator indices per record
                - "child_params": Dict mapping operator index to its random params
        """
        n_children = len(self.operators)
        weights = jnp.asarray(self.weights)

        # Per-record operator selection (fold index 0 reserved for selection).
        select_keys = jax.vmap(lambda key: jax.random.fold_in(key, 0))(element_keys)
        selected_indices = jax.vmap(
            lambda key: jax.random.choice(key, n_children, shape=(), p=weights)
        )(select_keys)

        # Per-child per-record keys (fold index i+1 for child i), delegated to
        # each child. All children's params are generated because selection is
        # resolved per record at apply time.
        child_params: dict[str, Any] = {}
        for i, operator in enumerate(self.operators):
            if hasattr(operator, "generate_random_params"):
                child_keys = jax.vmap(lambda key, offset=i + 1: jax.random.fold_in(key, offset))(
                    element_keys
                )
                child_params[f"operator_{i}"] = operator.generate_random_params(
                    child_keys, data_shapes
                )
            else:
                child_params[f"operator_{i}"] = None

        return {
            "selected_indices": selected_indices,
            "child_params": child_params,
        }

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply the randomly selected operator to the data.

        Uses jax.lax.switch for JIT-compatible operator selection based on
        the pre-generated random index.

        Args:
            data: Element data PyTree (no batch dimension)
            state: Element state PyTree
            metadata: Element metadata
            random_params: Dict with "selected_indices" (int) and "child_params"
            stats: Optional statistics

        Returns:
            Tuple of (transformed_data, state, metadata) from selected operator
        """
        # Get selection index for this element
        selected_idx = random_params["selected_indices"]
        child_params = random_params.get("child_params", {})

        # Create branch functions for each operator
        # Each branch applies its operator with its specific random params
        def make_branch_fn(i: int, operator: OperatorModule) -> Callable:
            def branch_fn(operands: Any) -> tuple[Any, Any, Any]:
                d, s, m, cp_dict, st = operands
                # Get this operator's random params
                op_params = cp_dict.get(f"operator_{i}", None)
                return operator.apply(d, s, m, op_params, st)

            return branch_fn

        branches = [make_branch_fn(i, op) for i, op in enumerate(self.operators)]

        # Use jax.lax.switch for JIT-compatible selection
        # This compiles efficiently and only executes the selected branch
        result_data, result_state, result_metadata = jax.lax.switch(
            selected_idx,
            branches,
            (data, state, metadata, child_params, stats),
        )

        return result_data, result_state, result_metadata
