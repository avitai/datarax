"""Base class for composition strategies.

A strategy combines a composite's children for one record: it receives the children, the record
and the statistics the composite computed for them, and returns the record. Each child is applied
through ``apply_record``, so it keys the record from its own base key and follows its own mode.
"""

import abc
import logging
from collections.abc import Sequence
from typing import Any

from datarax.core.element_batch import Element
from datarax.core.operator import OperatorModule, statistics_for_child


logger = logging.getLogger(__name__)


class CompositionStrategyImpl(abc.ABC):
    """Abstract base class for composition strategies."""

    def describe(self) -> dict[str, Any]:
        """Return a serializable description of this strategy."""
        return {"strategy": type(self).__name__}

    @staticmethod
    def _apply_each(
        operators: Sequence[OperatorModule], element: Element, stats: dict[str, Any] | None
    ) -> list[Element]:
        """Apply every operator to the same record, each with its own statistics."""
        return [
            operator.apply_record(element, statistics_for_child(stats, index))
            for index, operator in enumerate(operators)
        ]

    @abc.abstractmethod
    def apply(
        self,
        operators: Sequence[OperatorModule],
        element: Element,
        stats: dict[str, Any] | None,
    ) -> Element:
        """Combine the operators for one record.

        Args:
            operators: The composite's children, in order.
            element: The record.
            stats: One entry per child (``child_statistics``), or ``None``.

        Returns:
            The record.
        """
        ...
