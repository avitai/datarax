"""Ensemble composition strategies."""

import logging
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp

from datarax.core.element_batch import Element
from datarax.core.operator import OperatorModule
from datarax.operators.strategies.base import CompositionStrategyImpl


logger = logging.getLogger(__name__)


class EnsembleStrategy(CompositionStrategyImpl):
    """Applies operators to the same record and reduces their data (mean, sum, max, min)."""

    def __init__(self, mode: str) -> None:
        """Initialize ensemble strategy.

        Args:
            mode: Reduction mode ("mean", "sum", "max", "min")
        """
        self.mode = mode

    def apply(
        self,
        operators: Sequence[OperatorModule],
        element: Element,
        stats: dict[str, Any] | None,
    ) -> Element:
        """Apply the operators to the same record and reduce their data element-wise.

        Args:
            operators: Operators to apply to the record.
            element: The record.
            stats: One entry per child, or ``None``.

        Returns:
            The record with the reduced data and the last operator's state.

        Raises:
            ValueError: If ``self.mode`` is not one of mean/sum/max/min.
        """
        outputs = self._apply_each(operators, element, stats)
        datas = [output.data for output in outputs]
        if self.mode == "mean":
            reduced = jax.tree.map(lambda *args: jnp.mean(jnp.stack(args), axis=0), *datas)
        elif self.mode == "sum":
            reduced = jax.tree.map(lambda *args: sum(args), *datas)
        elif self.mode == "max":
            reduced = jax.tree.map(lambda *args: jnp.max(jnp.stack(args), axis=0), *datas)
        elif self.mode == "min":
            reduced = jax.tree.map(lambda *args: jnp.min(jnp.stack(args), axis=0), *datas)
        else:
            raise ValueError(f"Unknown ensemble mode: {self.mode}")
        return element.replace(data=reduced, state=outputs[-1].state)
