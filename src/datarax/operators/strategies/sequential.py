"""Sequential composition strategies."""

import logging
from collections.abc import Callable, Sequence
from typing import Any

import jax
from jaxtyping import PyTree

from datarax.core.element_batch import Element
from datarax.core.operator import apply_where, OperatorModule, statistics_for_child
from datarax.operators.strategies.base import CompositionStrategyImpl


logger = logging.getLogger(__name__)


class SequentialStrategy(CompositionStrategyImpl):
    """Applies operators in order to one record, piping each output to the next."""

    def apply(
        self,
        operators: Sequence[OperatorModule],
        element: Element,
        stats: dict[str, Any] | None,
    ) -> Element:
        """Apply the operators in order.

        Args:
            operators: Ordered operators to chain.
            element: The record.
            stats: One entry per child, or ``None``.

        Returns:
            The record after every operator has run.
        """
        for index, operator in enumerate(operators):
            element = operator.apply_record(element, statistics_for_child(stats, index))
        return element


class ConditionalSequentialStrategy(CompositionStrategyImpl):
    """Applies operators in order, each only where its condition holds for the record."""

    def __init__(self, conditions: Sequence[Callable[[PyTree], bool | jax.Array]]) -> None:
        """Initialize ConditionalSequentialStrategy.

        Args:
            conditions: One per operator, each reading the record's data as it reaches that
                operator.
        """
        self.conditions = conditions

    def apply(
        self,
        operators: Sequence[OperatorModule],
        element: Element,
        stats: dict[str, Any] | None,
    ) -> Element:
        """Apply the operators in order, skipping those whose condition is False.

        Uses ``jax.lax.cond`` so the condition may depend on traced values.

        Args:
            operators: Operators to apply (as many as there are conditions).
            element: The record.
            stats: One entry per child, or ``None``.

        Returns:
            The record after the conditional chain.

        Raises:
            ValueError: If the operator count does not match the condition count.
        """
        if len(operators) != len(self.conditions):
            raise ValueError(
                f"Number of operators ({len(operators)}) does not match "
                f"number of conditions ({len(self.conditions)})"
            )
        for index, (operator, condition) in enumerate(zip(operators, self.conditions, strict=True)):
            element = apply_where(
                operator, condition(element.data), element, statistics_for_child(stats, index)
            )
        return element
