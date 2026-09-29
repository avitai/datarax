"""Branching composition strategy."""

import logging
from collections.abc import Callable, Sequence
from typing import Any

import jax
from jaxtyping import PyTree

from datarax.core.element_batch import Element
from datarax.core.operator import apply_selected, OperatorModule
from datarax.operators.strategies.base import CompositionStrategyImpl


logger = logging.getLogger(__name__)


class BranchingStrategy(CompositionStrategyImpl):
    """Routes each record to one operator with an integer router."""

    def __init__(self, router: Callable[[PyTree], int | jax.Array]) -> None:
        """Initialize branching strategy.

        Args:
            router: Returns the index of the operator for a record's data.
        """
        self.router = router

    def apply(
        self,
        operators: Sequence[OperatorModule],
        element: Element,
        stats: dict[str, Any] | None,
    ) -> Element:
        """Route the record to exactly one operator via ``jax.lax.switch``.

        Only the selected branch runs (``apply_selected``).

        Args:
            operators: Candidate operators, indexed by the router's output.
            element: The record.
            stats: One entry per child, or ``None``.

        Returns:
            The record from the selected branch.
        """
        return apply_selected(operators, self.router(element.data), element, stats)
