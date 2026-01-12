"""Base class for composition strategies."""

import abc
from typing import Any, Callable

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
