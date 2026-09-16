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
from datarax.core.operator import OperatorModule, require_key


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SelectorOperatorConfig(OperatorConfig):
    """Configuration for SelectorOperator.

    Extends OperatorConfig with operators list and optional weights.

    Attributes:
        operators: List of operators to select from (minimum 1)
        weights: Optional weights for random selection (defaults to uniform)
                 Will be normalized to sum to 1.0
        normalized_weights: The weights normalized to sum to 1.0, as a tuple of floats.
                 Derived in __post_init__; a config is graphdef metadata, which jit
                 dispatch compares, so it holds no array.

    Note:
        - stochastic is always True (always makes random choice)
        - stream_name defaults to "augment" for random selection
    """

    operators: list[OperatorModule] = field(kw_only=True)
    weights: list[float] | None = field(default=None, kw_only=True)
    normalized_weights: tuple[float, ...] = field(init=False, repr=False)

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
        object.__setattr__(self, "normalized_weights", tuple(normalized))

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
        self.weights = nnx.static(config.normalized_weights)

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply one child operator, chosen for this record from its own key.

        The choice and every child's randomness are folded out of this record's key — index
        0 for the selection, ``i + 1`` for child ``i`` — so which operator a record gets, and
        what that operator draws, depend on the record alone. ``jax.lax.switch`` keeps the
        choice traceable and executes only the branch taken.

        Args:
            data: Element data PyTree (no batch dimension)
            state: Element state PyTree
            metadata: Element metadata
            key: This record's PRNG key
            stats: Optional statistics

        Returns:
            Tuple of (transformed_data, state, metadata) from selected operator
        """
        record_key = require_key(key, self)
        weights = jnp.asarray(self.weights)

        # Which child this record gets (fold index 0 is reserved for the selection).
        selected_idx = jax.random.choice(
            jax.random.fold_in(record_key, 0), len(self.operators), shape=(), p=weights
        )

        # Create branch functions for each operator
        # Each branch applies its operator with its own key, folded from the record's
        def make_branch_fn(i: int, operator: OperatorModule) -> Callable:
            def branch_fn(operands: Any) -> tuple[Any, Any, Any]:
                d, s, m, k, st = operands
                return operator.apply(d, s, m, jax.random.fold_in(k, i + 1), st)

            return branch_fn

        branches = [make_branch_fn(i, op) for i, op in enumerate(self.operators)]

        # Use jax.lax.switch for JIT-compatible selection
        # This compiles efficiently and only executes the selected branch
        result_data, result_state, result_metadata = jax.lax.switch(
            selected_idx,
            branches,
            (data, state, metadata, record_key, stats),
        )

        return result_data, result_state, result_metadata
