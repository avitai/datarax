"""Base class for composition strategies."""

import abc
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
from jaxtyping import PyTree

from datarax.core.operator import OperatorModule


logger = logging.getLogger(__name__)


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

    @staticmethod
    def _extract_operator_random_params(
        random_params: dict[str, Any] | None, operator_index: int
    ) -> Any | None:
        """Extract random params payload for one operator index."""
        if random_params is None:
            return None
        return random_params.get(f"operator_{operator_index}")

    @staticmethod
    def _apply_operator_conditionally(
        operator: OperatorModule,
        should_apply: bool | jax.Array,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any],
        random_params: Any | None,
    ) -> tuple[PyTree, PyTree, dict[str, Any]]:
        """Apply operator with JAX control flow (cond) for trace compatibility."""

        def apply_fn(
            operands: tuple[PyTree, PyTree, dict[str, Any], Any | None],
        ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
            d, s, m, rp = operands
            return operator.apply(d, s, m, rp)

        def noop_fn(
            operands: tuple[PyTree, PyTree, dict[str, Any], Any | None],
        ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
            d, s, m, _ = operands
            return d, s, m

        return jax.lax.cond(
            should_apply,
            apply_fn,
            noop_fn,
            (data, state, metadata, random_params),
        )

    @staticmethod
    def _emit_operator_statistics(
        operator: OperatorModule,
        operator_index: int,
        stats_callback: Callable[[int, dict[str, Any]], None] | None,
    ) -> None:
        """Send operator statistics to callback when available."""
        stats = getattr(operator, "statistics", None)
        if stats_callback and stats:
            stats_callback(operator_index, stats)

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
            op_random_params = self._extract_operator_random_params(context.random_params, i)
            out_data, out_state, out_metadata = operator.apply(
                context.data, context.state, context.metadata, op_random_params
            )
            outputs.append(out_data)
            states.append(out_state)
            metadatas.append(out_metadata)
            self._emit_operator_statistics(operator, i, context.stats_callback)
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
