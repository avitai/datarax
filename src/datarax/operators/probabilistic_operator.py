"""ProbabilisticOperator - Wrapper for probability-based operator application.

This operator wraps any OperatorModule and applies it with a configured probability.

Key Features:
- Wraps any OperatorModule with probabilistic application
- Configurable probability (0.0 to 1.0)
- Stochastic mode when 0 < p < 1
- Deterministic mode when p = 0.0 or p = 1.0
- Full JAX compatibility with JIT compilation
- Minimal overhead wrapper pattern

Examples:
    Basic usage:

    ```python
    config = ProbabilisticOperatorConfig(operator=child_op, probability=0.5)
    op = ProbabilisticOperator(config, rngs=rngs)
    ```
"""

from dataclasses import dataclass, field
from typing import Any

import jax
from flax import nnx
from jaxtyping import PyTree

from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule


@dataclass
class ProbabilisticOperatorConfig(OperatorConfig):
    """Configuration for ProbabilisticOperator.

    Extends OperatorConfig with probability parameter and child operator.

    Attributes:
        operator: Child operator to wrap with probabilistic application
        probability: Probability of applying the operator (0.0 to 1.0)
                    - 0.0: never apply (deterministic)
                    - 1.0: always apply (deterministic)
                    - 0 < p < 1: probabilistic (stochastic)

    Note:
        - stochastic is automatically set based on probability
        - stream_name is inherited from child operator if stochastic
    """

    operator: OperatorModule = field(kw_only=True)
    probability: float = field(default=0.5, kw_only=True)

    def __post_init__(self):
        """Validate configuration and infer stochastic mode."""
        # Validate probability range
        if not isinstance(self.probability, int | float):
            raise TypeError(f"probability must be a number, got {type(self.probability)}")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability must be in [0.0, 1.0], got {self.probability}")

        # Infer stochastic mode from probability
        # ProbabilisticOperator is stochastic when it makes a random decision (0 < p < 1)
        # It's deterministic when decision is fixed (p=0 always skip, p=1 always apply)
        is_stochastic = 0.0 < self.probability < 1.0
        object.__setattr__(self, "stochastic", is_stochastic)

        # Set stream_name BEFORE calling super().__post_init__() for validation
        if is_stochastic:
            # Stochastic mode needs stream_name for random decision
            # Use provided stream_name, or inherit from child, or default to "augment"
            if self.stream_name is None:
                if hasattr(self.operator, "stream_name") and self.operator.stream_name is not None:
                    object.__setattr__(self, "stream_name", self.operator.stream_name)
                else:
                    object.__setattr__(self, "stream_name", "augment")
        else:
            # Deterministic mode (p=0.0 or p=1.0) doesn't need RNG for decision
            object.__setattr__(self, "stream_name", None)

        super().__post_init__()


class ProbabilisticOperator(OperatorModule):
    """Wrapper operator that applies child operator with configured probability.

    Wraps any OperatorModule and applies it probabilistically:
        - p=0.0: never apply (passthrough)
        - p=1.0: always apply (equivalent to child operator)
        - 0<p<1: apply with probability p (stochastic)

    Uses jax.lax.cond for JIT-compatible conditional execution.

    Examples:
        Probabilistic application:

        ```python
        # Wrap any operator with 50% application probability
        child_config = BrightnessOperatorConfig(field_key="image", factor_range=(0.8, 1.2))
        child_op = BrightnessOperator(child_config, rngs=nnx.Rngs(0))

        prob_config = ProbabilisticOperatorConfig(
            operator=child_op,
            probability=0.5
        )
        prob_op = ProbabilisticOperator(prob_config, rngs=nnx.Rngs(0))

        # Apply to batch - each element has 50% chance of brightness adjustment
        result_batch = prob_op(batch)
        ```
    """

    def __init__(
        self,
        config: ProbabilisticOperatorConfig,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        """Initialize probabilistic operator.

        Args:
            config: ProbabilisticOperatorConfig with child operator and probability
            rngs: Random number generators (required if stochastic=True)
        """
        super().__init__(config, rngs=rngs)

        # Type narrowing for pyright
        self.config: ProbabilisticOperatorConfig = config

        # Store child operator
        self.operator = config.operator
        self.probability = config.probability

    def generate_random_params(
        self,
        rng: jax.Array,
        data_shapes: PyTree,
    ) -> dict[str, Any] | PyTree:
        """Generate random application decisions for each batch element.

        Creates boolean mask determining which elements get the operator applied.

        Args:
            rng: JAX random key
            data_shapes: PyTree with same structure as batch.data, containing shapes
                        Examples: {"image": (batch_size, H, W, C)}

        Returns:
            - If probabilistic (0 < p < 1): Dict with "apply_mask" and "child_params"
            - If deterministic (p=0 or p=1): Child's random params (or None if child deterministic)

        Note:
            The base class ALWAYS calls generate_random_params, so we must always
            delegate to child to get its params (even for deterministic ProbabilisticOperator).
        """
        # Always generate child's random params (child might be stochastic)
        # Split RNG for our use and child's use
        rng_mask, rng_child = jax.random.split(rng)

        child_params = None
        if hasattr(self.operator, "generate_random_params"):
            child_params = self.operator.generate_random_params(rng_child, data_shapes)

        # Deterministic cases (p=0 or p=1): just return child's params
        if self.probability == 0.0 or self.probability == 1.0:
            return child_params

        # Stochastic case (0 < p < 1): generate mask + child params
        # Extract batch size from shape tuples (same pattern as MapOperator)
        # Use is_leaf to treat tuples as atomic values
        batch_sizes = jax.tree.map(
            lambda shape: shape[0], data_shapes, is_leaf=lambda x: isinstance(x, tuple)
        )
        batch_size_leaves = jax.tree.leaves(batch_sizes)
        batch_size = batch_size_leaves[0] if batch_size_leaves else 1

        # Generate boolean mask based on probability
        # Each element independently sampled
        random_values = jax.random.uniform(rng_mask, shape=(batch_size,))
        apply_mask = random_values < self.probability

        return {
            "apply_mask": apply_mask,
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
        """Apply child operator conditionally based on probability.

        Uses jax.lax.cond for JIT-compatible conditional execution.

        Args:
            data: Element data PyTree (no batch dimension)
            state: Element state PyTree
            metadata: Element metadata
            random_params: Dict with "apply_mask" (bool) and "child_params"
                          For deterministic modes, can be None
            stats: Optional statistics

        Returns:
            Tuple of (transformed_data, state, metadata)
            - If applied: child operator's output
            - If not applied: input data/state/metadata unchanged
        """
        # Deterministic cases - direct path without conditionals
        if self.probability == 0.0:
            # Never apply - passthrough
            return data, state, metadata

        if self.probability == 1.0:
            # Always apply - direct child call
            # random_params IS the child_params directly
            return self.operator.apply(data, state, metadata, random_params, stats)

        # Stochastic case (0 < p < 1) - use jax.lax.cond
        # random_params is dict with "apply_mask" and "child_params"
        should_apply = random_params["apply_mask"] if isinstance(random_params, dict) else True
        child_params = (
            random_params.get("child_params", None) if isinstance(random_params, dict) else None
        )

        # Define branch functions for jax.lax.cond
        def apply_fn(operands):
            """Branch: apply child operator."""
            d, s, m, cp, st = operands
            return self.operator.apply(d, s, m, cp, st)

        def passthrough_fn(operands):
            """Branch: return input unchanged."""
            d, s, m, _, _ = operands
            return d, s, m

        # Use jax.lax.cond for JIT-compatible branching
        return jax.lax.cond(
            should_apply,
            apply_fn,
            passthrough_fn,
            (data, state, metadata, child_params, stats),
        )
