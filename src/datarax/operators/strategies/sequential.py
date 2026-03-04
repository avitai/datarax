"""Sequential composition strategies."""

from typing import Any
from collections.abc import Callable

import jax
from jaxtyping import PyTree

from datarax.operators.strategies.base import CompositionStrategyImpl, StrategyContext
from datarax.core.operator import OperatorModule


class SequentialStrategy(CompositionStrategyImpl):
    """Applies operators in sequence (chain)."""

    def apply(
        self,
        operators: list[OperatorModule],
        context: StrategyContext,
    ) -> tuple[PyTree, PyTree, dict[str, Any]]:
        """Apply operators sequentially, piping each output to the next.

        Args:
            operators: Ordered list of operators to chain.
            context: Execution context with input data, state, and RNG params.

        Returns:
            Tuple of (data, state, metadata) after all operators have run.
        """
        result_data, result_state, result_metadata = context.data, context.state, context.metadata

        for i, operator in enumerate(operators):
            op_random_params = self._extract_operator_random_params(context.random_params, i)

            # Apply operator
            result_data, result_state, result_metadata = operator.apply(
                result_data, result_state, result_metadata, op_random_params
            )

            # Track statistics
            self._emit_operator_statistics(operator, i, context.stats_callback)

        return result_data, result_state, result_metadata


class ConditionalSequentialStrategy(CompositionStrategyImpl):
    """Applies operators sequentially with conditions.

    Only applies operators where condition evaluates to True.
    """

    def __init__(self, conditions: list[Callable[[PyTree], bool | jax.Array]]):
        """Initialize ConditionalSequentialStrategy.

        Args:
            conditions: List of callables that determine whether each operator is applied.
        """
        self.conditions = conditions

    def apply(
        self,
        operators: list[OperatorModule],
        context: StrategyContext,
    ) -> tuple[PyTree, PyTree, dict[str, Any]]:
        """Apply operators sequentially, skipping those whose condition is False.

        Uses ``jax.lax.cond`` for vmap-compatible conditional execution.

        Args:
            operators: Operators to apply (must match length of conditions).
            context: Execution context with input data, state, and RNG params.

        Returns:
            Tuple of (data, state, metadata) after conditional execution.

        Raises:
            ValueError: If operator count doesn't match condition count.
        """
        result_data, result_state, result_metadata = context.data, context.state, context.metadata

        if len(operators) != len(self.conditions):
            raise ValueError(
                f"Number of operators ({len(operators)}) does not match "
                f"number of conditions ({len(self.conditions)})"
            )

        for i, (operator, condition) in enumerate(zip(operators, self.conditions)):
            op_random_params = self._extract_operator_random_params(context.random_params, i)
            result_data, result_state, result_metadata = self._apply_operator_conditionally(
                operator,
                condition(result_data),
                result_data,
                result_state,
                result_metadata,
                op_random_params,
            )

            # Track statistics
            self._emit_operator_statistics(operator, i, context.stats_callback)

        return result_data, result_state, result_metadata
