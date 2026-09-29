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
    config = ProbabilisticOperatorConfig(probability=0.5)
    op = ProbabilisticOperator(config, operator=child_op, rngs=rngs)
    ```
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import jax
from flax import nnx

from datarax.core.config import OperatorConfig
from datarax.core.element_batch import Batch, Element
from datarax.core.operator import (
    apply_where,
    child_statistics,
    OperatorModule,
    require_key,
    require_record_form,
    statistics_for_child,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbabilisticOperatorConfig(OperatorConfig):
    """Configuration for ProbabilisticOperator.

    Extends OperatorConfig with the application probability. The child operator is a
    constructor argument of ``ProbabilisticOperator``, never configuration: a configuration is
    static metadata a transform compares, and a module in it would compare by identity.

    Attributes:
        probability: Probability of applying the operator (0.0 to 1.0)
                    - 0.0: never apply (deterministic)
                    - 1.0: always apply (deterministic)
                    - 0 < p < 1: probabilistic (stochastic)

    Note:
        The wrapper draws a key only for its own decision (0 < p < 1); the child keys each record
        from its own base key, so it draws the same values here as at top level.
    """

    probability: float = field(default=0.5, kw_only=True)

    def __post_init__(self) -> None:
        """Validate the probability; only a random decision (0 < p < 1) draws a key."""
        if not isinstance(self.probability, int | float):
            raise TypeError(f"probability must be a number, got {type(self.probability)}")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability must be in [0.0, 1.0], got {self.probability}")
        decides_randomly = 0.0 < self.probability < 1.0
        object.__setattr__(self, "stochastic", decides_randomly)
        object.__setattr__(
            self, "stream_name", (self.stream_name or "augment") if decides_randomly else None
        )
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

        prob_op = ProbabilisticOperator(
            ProbabilisticOperatorConfig(probability=0.5), operator=child_op, rngs=nnx.Rngs(0)
        )

        # Apply to batch - each element has 50% chance of brightness adjustment
        result_batch = prob_op(batch)
        ```
    """

    def __init__(
        self,
        config: ProbabilisticOperatorConfig,
        operator: OperatorModule,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        """Initialize probabilistic operator.

        Args:
            config: ProbabilisticOperatorConfig with the application probability
            operator: The child operator, held as a graph child
            rngs: Random number generators (required if the wrapper is stochastic)
        """
        require_record_form(operator, self)
        super().__init__(config, rngs=rngs)

        # Type narrowing for pyright
        self.config: ProbabilisticOperatorConfig = config

        self.operator = operator
        self.probability = config.probability

    def fits_statistics_per_batch(self) -> bool:
        """Whether the child computes its statistics from each batch."""
        return self.operator.fits_statistics_per_batch()

    def compute_statistics(self, batch: Batch) -> dict[str, Any] | None:
        """Return the child's statistics for this batch, computed on this wrapper's input.

        They are computed whatever the probability: whether a given record reaches the child is
        decided per record inside the vectorized call, long after the batch is gone.

        Args:
            batch: The batch about to be applied.

        Returns:
            The child's statistics, or ``None`` when it has none.
        """
        return child_statistics([self.operator], batch)

    def apply(
        self, element: Element, key: jax.Array | None = None, stats: dict[str, Any] | None = None
    ) -> Element:
        """Apply the child with probability ``p``, decided for this record from its own key.

        The child draws from its own key for the record (``apply_record``), so whether a record
        is augmented depends on this wrapper's key and how it is augmented on the child's.
        ``jax.lax.cond`` keeps the choice traceable (``apply_where``).

        Args:
            element: The record.
            key: This record's key when 0 < p < 1, else ``None``.
            stats: The child's statistics, as ``compute_statistics`` built them.

        Returns:
            The record, transformed by the child or unchanged.
        """
        if self.probability == 0.0:
            return element
        child_stats = statistics_for_child(stats, 0)
        if self.probability == 1.0:
            return self.operator.apply_record(element, child_stats)

        should_apply = jax.random.uniform(require_key(key, self), shape=()) < self.probability
        return apply_where(self.operator, should_apply, element, child_stats)
