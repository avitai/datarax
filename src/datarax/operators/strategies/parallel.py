"""Parallel composition strategies."""

import logging
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from datarax.core.field_paths import get_field, set_field
from datarax.core.operator import OperatorModule
from datarax.operators.strategies.base import CompositionStrategyImpl, StrategyContext
from datarax.operators.strategies.merging import merge_output_sequence, merge_outputs_conditional


logger = logging.getLogger(__name__)


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
    ) -> None:
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
        """Apply all operators on the same input and merge their outputs.

        Args:
            operators: Operators to execute in parallel on identical input.
            context: Execution context with input data, state, and RNG params.

        Returns:
            Tuple of (merged_data, last_state, last_metadata).
        """
        outputs, states, metadatas = self._execute_operators(operators, context)

        # Merge outputs
        merged_data = merge_output_sequence(
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
    """Applies operators in parallel and combines the fields they write with weights.

    Only the named ``mix_fields`` come from the operators' outputs, each as the weighted
    sum ``sum_i weights[i] * output_i[field]``. Every other field of the input passes
    through unchanged, as Kornia's ``data_keys`` and torchvision's AugMix leave inputs they
    do not transform alone.
    """

    def __init__(self, mix_fields: Sequence[str]) -> None:
        """Initialize with the fields to combine.

        Args:
            mix_fields: Dotted paths of the data fields to combine.
        """
        self.mix_fields = tuple(mix_fields)

    def apply(
        self,
        operators: list[OperatorModule],
        context: StrategyContext,
    ) -> tuple[PyTree, PyTree, dict[str, Any]]:
        """Apply operators in parallel and combine their ``mix_fields`` with weights.

        Args:
            operators: Operators to execute in parallel.
            context: Must include ``extra_params["weights"]`` JAX array.

        Returns:
            Tuple of (the input data with each mix field replaced by its weighted sum,
            last_state, last_metadata).

        Raises:
            ValueError: If ``weights`` not found in ``context.extra_params``.
        """
        outputs, states, metadatas = self._execute_operators(operators, context)

        if not context.extra_params or "weights" not in context.extra_params:
            raise ValueError("WeightedParallelStrategy requires 'weights' in extra_params")

        weights = jnp.asarray(context.extra_params["weights"])
        mixed_data = context.data
        for path in self.mix_fields:
            stacked = jnp.stack([get_field(output, path) for output in outputs])
            mixed_data = set_field(mixed_data, path, jnp.tensordot(weights, stacked, axes=1))

        merged_state = states[-1] if states else context.state
        merged_metadata = metadatas[-1] if metadatas else context.metadata

        return mixed_data, merged_state, merged_metadata


class ConditionalParallelStrategy(CompositionStrategyImpl):
    """Applies operators in parallel with conditions (vmap-compatible)."""

    def __init__(
        self,
        conditions: Sequence[Callable[[PyTree], bool | jax.Array]],
        merge_strategy: str | None = None,
        merge_axis: int = 0,
        merge_fn: Callable | None = None,
    ) -> None:
        """Initialize ConditionalParallelStrategy.

        Args:
            conditions: List of callables that determine whether each operator is applied.
            merge_strategy: Strategy for merging active outputs (e.g. 'concat', 'stack').
            merge_axis: Axis along which to merge outputs.
            merge_fn: Custom merge function, overrides merge_strategy if provided.
        """
        self.conditions = conditions
        self.merge_strategy = merge_strategy
        self.merge_axis = merge_axis
        self.merge_fn = merge_fn

    def apply(
        self,
        operators: list[OperatorModule],
        context: StrategyContext,
    ) -> tuple[PyTree, PyTree, dict[str, Any]]:
        """Apply operators conditionally in parallel and merge active outputs.

        Uses ``jax.lax.cond`` per operator for vmap/JIT compatibility.

        Args:
            operators: Operators to evaluate (must match length of conditions).
            context: Execution context with input data, state, and RNG params.

        Returns:
            Tuple of (merged_data, last_state, last_metadata).
        """
        outputs = []
        # First pass: evaluate all conditions
        condition_results = [condition(context.data) for condition in self.conditions]

        # Second pass: apply operators with jax.lax.cond, each with its own key
        outputs, states, metadatas = [], [], []
        for (operator, key), cond_result in zip(
            self._with_keys(operators, context), condition_results, strict=False
        ):
            out_data, out_state, out_metadata = self._apply_operator_conditionally(
                operator, cond_result, context.data, context.state, context.metadata, key
            )
            outputs.append(out_data)
            states.append(out_state)
            metadatas.append(out_metadata)

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
