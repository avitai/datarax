"""SelectorOperator - Random selection from multiple operators.

This operator wraps multiple OperatorModules and randomly selects ONE to apply
per record, each child drawing from its own key.

Key Features:

- Wraps multiple OperatorModules with random selection
- Configurable weights for weighted random selection (defaults to uniform)
- Uses jax.lax.switch for JIT-compatible dynamic selection
- Always stochastic (always makes a random choice)
- Full JAX compatibility (JIT, vmap)

Examples:
    Basic usage:

    ```python
    config = SelectorOperatorConfig(weights=[0.5, 0.3, 0.2])
    op = SelectorOperator(config, operators=[op1, op2, op3], rngs=rngs)
    ```
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from datarax.core.config import OperatorConfig
from datarax.core.element_batch import Batch, Element
from datarax.core.operator import (
    apply_selected,
    child_statistics,
    OperatorModule,
    require_distinct,
    require_key,
    require_record_form,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SelectorOperatorConfig(OperatorConfig):
    """Configuration for SelectorOperator.

    Extends OperatorConfig with optional selection weights. The operators to select from are a
    constructor argument of ``SelectorOperator``, never configuration: a configuration is static
    metadata a transform compares, and a module in it would compare by identity.

    Attributes:
        weights: Optional weights for random selection (defaults to uniform), one per operator;
            normalized to sum to 1.0 when the selector is built. Stored as a tuple, so the
            configuration stays hashable.

    Note:
        - stochastic is always True (always makes random choice)
        - stream_name defaults to "augment" for random selection
    """

    weights: Sequence[float] | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        """Validate the weights and set the random mode."""
        if self.weights is not None:
            if any(weight < 0 for weight in self.weights) or sum(self.weights) <= 0:
                raise ValueError(
                    f"weights must be non-negative with a positive sum, got {self.weights}"
                )
            object.__setattr__(self, "weights", tuple(float(weight) for weight in self.weights))

        # SelectorOperator is ALWAYS stochastic (always makes random choice)
        object.__setattr__(self, "stochastic", True)
        if self.stream_name is None:
            object.__setattr__(self, "stream_name", "augment")

        super().__post_init__()

    def normalized_weights(self, n_operators: int) -> tuple[float, ...]:
        """Return the selection weights for ``n_operators`` operators, summing to 1.0.

        Args:
            n_operators: How many operators the selector chooses among.

        Returns:
            One weight per operator.

        Raises:
            ValueError: If there are no operators, or the weights name a different number.
        """
        if n_operators < 1:
            raise ValueError("Must provide at least one operator")
        if self.weights is None:
            return (1.0 / n_operators,) * n_operators
        if len(self.weights) != n_operators:
            raise ValueError(
                f"Number of weights ({len(self.weights)}) must match "
                f"number of operators ({n_operators})"
            )
        total = sum(self.weights)
        return tuple(weight / total for weight in self.weights)


class SelectorOperator(OperatorModule):
    """Wrapper operator that randomly selects ONE operator to apply.

    Wraps multiple OperatorModules and uses weighted random selection to
    choose which one to apply per record, each child drawing from its own key.

    Uses jax.lax.switch for JIT-compatible operator selection with the
    unified operator interface.

    Examples:
        ```python
        op1 = BrightnessOperator(brightness_config, rngs=nnx.Rngs(0))  # Different transforms
        op2 = NoiseOperator(noise_config, rngs=nnx.Rngs(0))
        op3 = RotationOperator(rotation_config, rngs=nnx.Rngs(0))
        selector = SelectorOperator(  # 50% brightness, 30% noise, 20% rotation
            SelectorOperatorConfig(weights=(0.5, 0.3, 0.2)),
            operators=[op1, op2, op3],
            rngs=nnx.Rngs(0),
        )
        result_batch = selector(batch)  # Each element gets one randomly selected operator
        ```
    """

    def __init__(
        self,
        config: SelectorOperatorConfig,
        operators: Sequence[OperatorModule],
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        """Initialize selector operator.

        Args:
            config: SelectorOperatorConfig with optional selection weights
            operators: The operators to select among, held as graph children
            rngs: Random number generators (required for random selection)
        """
        weights = config.normalized_weights(len(operators))
        require_distinct(operators)
        for operator in operators:
            require_record_form(operator, self)
        super().__init__(config, rngs=rngs)

        # Type narrowing for pyright
        self.config: SelectorOperatorConfig = config

        self.operators = nnx.List(operators)
        self.weights = nnx.static(weights)

    def fits_statistics_per_batch(self) -> bool:
        """Whether a child computes its statistics from each batch."""
        return any(operator.fits_statistics_per_batch() for operator in self.operators)

    def compute_statistics(self, batch: Batch) -> dict[str, Any] | None:
        """Return one entry per child, each computed on this selector's input.

        Every child's statistics are computed, not only the selected one's: which child a record
        gets is decided per record inside the vectorized call, long after the batch is gone.

        Args:
            batch: The batch about to be applied.

        Returns:
            The children's statistics, or ``None`` when no child has any.
        """
        return child_statistics(list(self.operators), batch)

    def apply(
        self, element: Element, key: jax.Array | None = None, stats: dict[str, Any] | None = None
    ) -> Element:
        """Apply one child, chosen for this record from the selector's key for it.

        The choice is drawn from this record's key; the chosen child draws from its own key for
        the record (``apply_record``), as it would at top level. ``jax.lax.switch`` keeps the
        choice traceable and runs only the branch taken (``apply_selected``).

        Args:
            element: The record.
            key: This record's key for the selection.
            stats: The children's statistics, as ``compute_statistics`` built them.

        Returns:
            The record, transformed by the chosen child.
        """
        selected = jax.random.choice(
            require_key(key, self), len(self.operators), shape=(), p=jnp.asarray(self.weights)
        )

        return apply_selected(list(self.operators), selected, element, stats)
