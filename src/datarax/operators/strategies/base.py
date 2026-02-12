"""Base class for composition strategies."""

import abc
from typing import Any
from collections.abc import Callable

from jaxtyping import PyTree
from dataclasses import dataclass

from datarax.core.operator import OperatorModule


@dataclass
class StrategyContext:
    """Context passed to strategy application."""

    data: PyTree
    state: PyTree
    metadata: dict[str, Any]
    random_params: dict[str, Any] | None = None
    extra_params: dict[str, Any] | None = None
    stats_callback: Callable[[int, dict[str, Any]], None] | None = None


class CompositionStrategyImpl(abc.ABC):
    """Abstract base class for composition strategies."""

    def _execute_operators(
        self,
        operators: list[OperatorModule],
        context: StrategyContext,
    ) -> tuple[list, list, list]:
        """Execute operators and collect outputs (shared loop).

        Args:
            operators: List of operators to apply
            context: Execution context with data, state, metadata, random_params

        Returns:
            Tuple of (outputs, states, metadatas) lists
        """
        outputs, states, metadatas = [], [], []
        for i, operator in enumerate(operators):
            op_random_params = None
            if context.random_params and f"operator_{i}" in context.random_params:
                op_random_params = context.random_params[f"operator_{i}"]
            out_data, out_state, out_metadata = operator.apply(
                context.data, context.state, context.metadata, op_random_params
            )
            outputs.append(out_data)
            states.append(out_state)
            metadatas.append(out_metadata)
            if context.stats_callback and hasattr(operator, "statistics") and operator.statistics:
                context.stats_callback(i, operator.statistics)
        return outputs, states, metadatas

    @abc.abstractmethod
    def apply(
        self,
        operators: list[OperatorModule],
        context: StrategyContext,
    ) -> tuple[PyTree, PyTree, dict[str, Any]]:
        """Apply the composition strategy.

        Args:
            operators: List of operators to compose
            context: Execution context containing data, state, metadata, etc.

        Returns:
            Tuple of (result_data, result_state, result_metadata)
        """
        ...
