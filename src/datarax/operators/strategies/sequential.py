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
            # Extract random params for this operator
            op_random_params = None
            if context.random_params and f"operator_{i}" in context.random_params:
                op_random_params = context.random_params[f"operator_{i}"]

            # Apply operator
            result_data, result_state, result_metadata = operator.apply(
                result_data, result_state, result_metadata, op_random_params
            )

            # Track statistics
            if context.stats_callback and hasattr(operator, "statistics") and operator.statistics:
                context.stats_callback(i, operator.statistics)

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
            # Extract random params for this operator
            op_random_params = None
            if context.random_params and f"operator_{i}" in context.random_params:
                op_random_params = context.random_params[f"operator_{i}"]

            # Use JAX control flow for vmap compatibility
            def apply_fn(operands):
                d, s, m, rp = operands
                return operator.apply(d, s, m, rp)

            def noop_fn(operands):
                d, s, m, _ = operands
                return d, s, m

            # Evaluate condition on data *before* this operator?
            # Original implementation: condition(result_data) i.e. current data
            # Evaluate condition and apply operator conditionally using jax.lax.cond
            result_data, result_state, result_metadata = jax.lax.cond(
                condition(result_data),
                apply_fn,
                noop_fn,
                (result_data, result_state, result_metadata, op_random_params),
            )

            # Track statistics
            if context.stats_callback and hasattr(operator, "statistics") and operator.statistics:
                context.stats_callback(i, operator.statistics)

        return result_data, result_state, result_metadata
