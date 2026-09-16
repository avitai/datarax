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

import logging
from dataclasses import dataclass, field
from typing import Any

import jax
from flax import nnx
from jaxtyping import PyTree

from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule, require_key


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
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

    def __post_init__(self) -> None:
        """Validate configuration and infer stochastic mode."""
        # Validate probability range
        if not isinstance(self.probability, int | float):
            raise TypeError(f"probability must be a number, got {type(self.probability)}")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability must be in [0.0, 1.0], got {self.probability}")

        # A wrapper needs a key when it makes a random decision (0 < p < 1) AND when it has a
        # stochastic child to hand one to. At p == 1 the decision is fixed but the child still
        # draws, so a wrapper that is deterministic on its own account would receive no key and
        # silence the operator it wraps. At p == 0 the child is never reached, so nothing draws.
        decides_randomly = 0.0 < self.probability < 1.0
        child_is_stochastic = bool(getattr(self.operator.config, "stochastic", False))
        is_stochastic = decides_randomly or (child_is_stochastic and self.probability > 0.0)
        object.__setattr__(self, "stochastic", is_stochastic)

        # Set stream_name BEFORE calling super().__post_init__() for validation
        if is_stochastic:
            # Stochastic mode needs stream_name for the decision, the child's draw, or both.
            # Use provided stream_name, or inherit from child, or default to "augment"
            if self.stream_name is None:
                if hasattr(self.operator, "stream_name") and self.operator.stream_name is not None:
                    object.__setattr__(self, "stream_name", self.operator.stream_name)
                else:
                    object.__setattr__(self, "stream_name", "augment")
        else:
            # Nothing downstream draws: p == 0, or a fixed decision over a deterministic child.
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
        child_config = BrightnessOperatorConfig(field_key="image", brightness_delta=0.2)
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
    ) -> None:
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

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply the child operator with probability ``p``, decided for this record.

        The decision and the child's randomness come from two keys folded out of this
        record's key, so whether a record is augmented — and how — depends on the record
        alone, not on the batch it arrives in. Uses ``jax.lax.cond`` so the choice survives
        tracing.

        At ``p == 1`` the child is still handed its own key. Passing the wrapper's fourth
        argument straight through, as this did before, gave a deterministic wrapper's child
        ``None`` and silently turned a stochastic child into a fixed one.

        Args:
            data: Element data PyTree (no batch dimension)
            state: Element state PyTree
            metadata: Element metadata
            key: This record's PRNG key, or ``None`` for a deterministic wrapper
            stats: Optional statistics

        Returns:
            Tuple of (transformed_data, state, metadata)
            - If applied: child operator's output
            - If not applied: input data/state/metadata unchanged
        """
        # Never applied: the child is not reached, so no key is needed.
        if self.probability == 0.0:
            return data, state, metadata

        # The child's key is independent of the decision key drawn below.
        child_key = None if key is None else jax.random.fold_in(key, 1)

        if self.probability == 1.0:
            return self.operator.apply(data, state, metadata, child_key, stats)

        # Stochastic case (0 < p < 1): decide per record, from its own key.
        should_apply = (
            jax.random.uniform(jax.random.fold_in(require_key(key, self), 0), shape=())
            < self.probability
        )

        # Define branch functions for jax.lax.cond
        def apply_fn(operands: Any) -> tuple[Any, Any, Any]:
            """Branch: apply child operator."""
            d, s, m, cp, st = operands
            return self.operator.apply(d, s, m, cp, st)

        def passthrough_fn(operands: Any) -> tuple[Any, Any, Any]:
            """Branch: return input unchanged."""
            d, s, m, _, _ = operands
            return d, s, m

        # Use jax.lax.cond for JIT-compatible branching
        return jax.lax.cond(
            should_apply,
            apply_fn,
            passthrough_fn,
            (data, state, metadata, child_key, stats),
        )
