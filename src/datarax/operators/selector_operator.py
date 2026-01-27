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

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import PyTree

from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule


@dataclass
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

    def __post_init__(self):
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
    ):
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
        self.weights = config.normalized_weights

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
        rng: jax.Array,
        data_shapes: PyTree,
    ) -> dict[str, Any]:
        """Generate random operator selection indices for each batch element.

        Creates integer indices determining which operator to apply per element,
        plus delegates to all child operators for their random params.

        Args:
            rng: JAX random key
            data_shapes: PyTree with same structure as batch.data, containing shapes
                        Examples: {"image": (batch_size, H, W, C)}

        Returns:
            Dict with:

                - "selected_indices": Array of operator indices per batch element
                - "child_params": Dict mapping operator index to its random params
        """
        # Split RNG: one for selection, one for children
        rng_select, rng_children = jax.random.split(rng)

        # Extract batch size from shape tuples (same pattern as MapOperator)
        batch_sizes = jax.tree.map(
            lambda shape: shape[0], data_shapes, is_leaf=lambda x: isinstance(x, tuple)
        )
        batch_size_leaves = jax.tree.leaves(batch_sizes)
        batch_size = batch_size_leaves[0] if batch_size_leaves else 1

        # Generate operator selection indices for each batch element
        # jax.random.choice samples from [0, n_operators) with given weights
        selected_indices = jax.random.choice(
            rng_select,
            len(self.operators),
            shape=(batch_size,),
            p=self.weights,
        )

        # Generate random params for ALL child operators
        # (we need them all because selection happens per-element at apply time)
        child_params = {}
        n_children = len(self.operators)
        child_rngs = jax.random.split(rng_children, n_children)

        for i, operator in enumerate(self.operators):
            if hasattr(operator, "generate_random_params"):
                child_params[f"operator_{i}"] = operator.generate_random_params(
                    child_rngs[i], data_shapes
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
        def make_branch_fn(i: int, operator: OperatorModule):
            def branch_fn(operands):
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
