"""Parallel composition strategies."""

import logging
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from flax.nnx.nn.dtypes import promote_dtype
from flax.typing import Dtype
from jaxtyping import PyTree

from datarax.core.element_batch import Element
from datarax.core.field_paths import get_field, set_field
from datarax.core.maybe import refuse_maybe
from datarax.core.operator import apply_where, OperatorModule, statistics_for_child
from datarax.operators.strategies.base import CompositionStrategyImpl
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
        operators: Sequence[OperatorModule],
        element: Element,
        stats: dict[str, Any] | None,
    ) -> Element:
        """Apply all operators to the same record and merge their data.

        Args:
            operators: Operators to apply to the record.
            element: The record.
            stats: One entry per child, or ``None``.

        Returns:
            The record with the merged data and the last operator's state.
        """
        outputs = self._apply_each(operators, element, stats)
        merged = merge_output_sequence(
            [output.data for output in outputs],
            self.merge_strategy,
            self.merge_axis,
            self.merge_fn,
        )
        return element.replace(data=merged, state=outputs[-1].state)


type MixtureWeights = jax.Array | tuple[float, ...]
"""A weighted parallel's weights: fixed Python numbers, or an array (learned or per record)."""


class WeightedParallelStrategy(CompositionStrategyImpl):
    """Applies operators in parallel and combines the fields they write with weights.

    Only the named ``mix_fields`` come from the operators' outputs, each as the weighted
    sum ``sum_i weights[i] * output_i[field]``. Every other field of the input passes
    through unchanged, as Kornia's ``data_keys`` and torchvision's AugMix leave inputs they
    do not transform alone.
    """

    def __init__(
        self,
        mix_fields: Sequence[str],
        weights: MixtureWeights,
        dtype: Dtype | None = None,
    ) -> None:
        """Initialize with the fields to combine and the weights to combine them with.

        Args:
            mix_fields: Dotted paths of the data fields to combine.
            weights: One weight per operator, for this call: fixed Python numbers, or an array
                (the softmax of learnable logits, or read from the record).
            dtype: The dtype array weights mix in, ``None`` for the promotion of the field and
                the weights, as ``nnx.Linear``'s ``dtype``.
        """
        self.mix_fields = tuple(mix_fields)
        self.weights = weights
        self.dtype = dtype

    def apply(
        self,
        operators: Sequence[OperatorModule],
        element: Element,
        stats: dict[str, Any] | None,
    ) -> Element:
        """Apply the operators to the same record and combine their ``mix_fields`` with weights.

        Args:
            operators: Operators to apply to the record.
            element: The record.
            stats: One entry per child, or ``None``.

        Returns:
            The record with each mix field replaced by its weighted sum and the last operator's
            state.
        """
        outputs = self._apply_each(operators, element, stats)
        mixed = element.data
        for path in self.mix_fields:
            fields = [get_field(output.data, path) for output in outputs]
            for field in fields:
                refuse_maybe(
                    field, "WEIGHTED_PARALLEL mixes its fields as arrays", f"data field {path!r}"
                )
            stacked = jnp.stack(fields)
            if isinstance(self.weights, tuple):
                # Fixed weights are constants, not parameters: they mix in the field's dtype
                weights = jnp.asarray(self.weights, stacked.dtype)
            else:
                # Learnable or per-record weights mix as a Flax layer does
                stacked, weights = promote_dtype((stacked, self.weights), dtype=self.dtype)
            mixed = set_field(mixed, path, jnp.tensordot(weights, stacked, axes=1))
        return element.replace(data=mixed, state=outputs[-1].state)


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
        operators: Sequence[OperatorModule],
        element: Element,
        stats: dict[str, Any] | None,
    ) -> Element:
        """Apply each operator where its condition holds for the record, and merge the outputs.

        Every condition reads the record as it reached the composite; ``jax.lax.cond`` keeps
        each traceable.

        Args:
            operators: Operators to evaluate (as many as there are conditions).
            element: The record.
            stats: One entry per child, or ``None``.

        Returns:
            The record with the merged data of the operators whose condition held and the last
            operator's state.
        """
        conditions = [condition(element.data) for condition in self.conditions]
        outputs = [
            apply_where(operator, condition, element, statistics_for_child(stats, index))
            for index, (operator, condition) in enumerate(zip(operators, conditions, strict=True))
        ]
        merged = merge_outputs_conditional(
            [output.data for output in outputs],
            conditions,
            self.merge_strategy,
            self.merge_axis,
            self.merge_fn,
        )
        return element.replace(data=merged, state=outputs[-1].state)
