"""Parallel composition strategies."""

from typing import Any, Callable

import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from datarax.operators.strategies.base import CompositionStrategyImpl, StrategyContext
from datarax.operators.strategies.merging import merge_outputs, merge_outputs_conditional
from datarax.core.operator import OperatorModule


class ParallelStrategy(CompositionStrategyImpl):
    """Applies operators in parallel and merges outputs.

    This strategy executes all child operators on the same input data
    and merges their results according to the specified strategy.

    Attributes:
        merge_strategy: How to merge outputs ('concat', 'stack', 'sum', 'mean').
        merge_axis: Axis along which to merge (for concat/stack).
        merge_fn: Custom merging function.

    Examples:
        Example usage:

        ```python
        strategy = ParallelStrategy(merge_strategy='concat', merge_axis=-1)
        # op1 returns shape (B, 10), op2 returns shape (B, 5)
        # result shape will be (B, 15)
        ```
    """

    def __init__(
        self,
        merge_strategy: str | None = None,
        merge_axis: int = 0,
        merge_fn: Callable | None = None,
    ):
        """Initialize parallel strategy.

        Args:
            merge_strategy: String identifier for merge strategy. available:

                - 'concat': Concatenate along axis.
                - 'stack': Stack along (new) axis.
                - 'sum': Sum outputs (element-wise).
                - 'mean': Average outputs (element-wise).
            merge_axis: Axis for concatenation or stacking. Defaults to 0.
            merge_fn: Optional custom callable to merge outputs.
        """
        self.merge_strategy = merge_strategy
        self.merge_axis = merge_axis
        self.merge_fn = merge_fn

    def apply(
        self,
        operators: list[OperatorModule],
        context: StrategyContext,
    ) -> tuple[PyTree, PyTree, dict[str, Any]]:
        outputs = []
        states = []
        metadatas = []

        for i, operator in enumerate(operators):
            # Extract random params for this operator
            op_random_params = None
            if context.random_params and f"operator_{i}" in context.random_params:
                op_random_params = context.random_params[f"operator_{i}"]

            # Apply operator
            out_data, out_state, out_metadata = operator.apply(
                context.data, context.state, context.metadata, op_random_params
            )
            outputs.append(out_data)
            states.append(out_state)
            metadatas.append(out_metadata)

            # Track statistics
            if context.stats_callback and hasattr(operator, "statistics") and operator.statistics:
                context.stats_callback(i, operator.statistics)

        # Merge outputs
        merged_data = merge_outputs(
            outputs,
            self.merge_strategy,
            self.merge_axis,
            self.merge_fn,
        )

        # For state and metadata, take the last one for now (as per original logic)
        merged_state = states[-1] if states else context.state
        merged_metadata = metadatas[-1] if metadatas else context.metadata

        return merged_data, merged_state, merged_metadata


class WeightedParallelStrategy(CompositionStrategyImpl):
    """Applies operators in parallel and merges with weights."""

    def apply(
        self,
        operators: list[OperatorModule],
        context: StrategyContext,
    ) -> tuple[PyTree, PyTree, dict[str, Any]]:
        outputs = []
        states = []
        metadatas = []

        for i, operator in enumerate(operators):
            # Extract random params for this operator
            op_random_params = None
            if context.random_params and f"operator_{i}" in context.random_params:
                op_random_params = context.random_params[f"operator_{i}"]

            # Apply operator
            out_data, out_state, out_metadata = operator.apply(
                context.data, context.state, context.metadata, op_random_params
            )
            outputs.append(out_data)
            states.append(out_state)
            metadatas.append(out_metadata)

            # Track statistics
            if context.stats_callback and hasattr(operator, "statistics") and operator.statistics:
                context.stats_callback(i, operator.statistics)

        # Get weights from context
        if not context.extra_params or "weights" not in context.extra_params:
            raise ValueError("WeightedParallelStrategy requires 'weights' in extra_params")

        weights = context.extra_params["weights"]

        # Apply weights and summ
        # Stack outputs first, then multiply by weights and sum
        weighted_data = jax.tree.map(
            lambda *args: jnp.sum(
                jnp.stack(args, axis=0) * weights.reshape(-1, *([1] * (args[0].ndim))), axis=0
            ),
            *outputs,
        )

        merged_state = states[-1] if states else context.state
        merged_metadata = metadatas[-1] if metadatas else context.metadata

        return weighted_data, merged_state, merged_metadata


class ConditionalParallelStrategy(CompositionStrategyImpl):
    """Applies operators in parallel with conditions (vmap-compatible)."""

    def __init__(
        self,
        conditions: list[Callable[[PyTree], bool | jax.Array]],
        merge_strategy: str | None = None,
        merge_axis: int = 0,
        merge_fn: Callable | None = None,
    ):
        self.conditions = conditions
        self.merge_strategy = merge_strategy
        self.merge_axis = merge_axis
        self.merge_fn = merge_fn

    def apply(
        self,
        operators: list[OperatorModule],
        context: StrategyContext,
    ) -> tuple[PyTree, PyTree, dict[str, Any]]:
        outputs = []
        states = []
        metadatas = []
        condition_results = []

        # First pass: evaluate all conditions
        for condition in self.conditions:
            condition_results.append(condition(context.data))

        # Second pass: apply operators with jax.lax.cond
        for i, (operator, cond_result) in enumerate(zip(operators, condition_results)):
            # Extract random params
            op_random_params = None
            if context.random_params and f"operator_{i}" in context.random_params:
                op_random_params = context.random_params[f"operator_{i}"]

            def apply_fn(operands):
                d, s, m, rp = operands
                return operator.apply(d, s, m, rp)

            def noop_fn(operands):
                d, s, m, _ = operands
                return d, s, m

            out_data, out_state, out_metadata = jax.lax.cond(
                cond_result,
                apply_fn,
                noop_fn,
                (context.data, context.state, context.metadata, op_random_params),
            )

            outputs.append(out_data)
            states.append(out_state)
            metadatas.append(out_metadata)

            if context.stats_callback and hasattr(operator, "statistics") and operator.statistics:
                context.stats_callback(i, operator.statistics)

        if not outputs:
            return context.data, context.state, context.metadata

        merged_data = merge_outputs_conditional(
            outputs,
            condition_results,
            self.merge_strategy,
            self.merge_axis,
            self.merge_fn,
        )

        merged_state = states[-1] if states else context.state
        merged_metadata = metadatas[-1] if metadatas else context.metadata

        return merged_data, merged_state, merged_metadata
