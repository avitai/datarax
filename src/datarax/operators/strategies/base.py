"""Base class for composition strategies."""

import abc
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import jax
from jaxtyping import PyTree

from datarax.core.operator import OperatorModule, statistics_for_child


logger = logging.getLogger(__name__)


@dataclass
class StrategyContext:
    """Context passed to strategy application."""

    data: PyTree
    state: PyTree
    metadata: dict[str, Any]
    key: jax.Array | None = None
    stats: dict[str, Any] | None = None
    extra_params: dict[str, Any] | None = None


class CompositionStrategyImpl(abc.ABC):
    """Abstract base class for composition strategies."""

    def describe(self) -> dict[str, Any]:
        """Return a serializable description of this strategy."""
        return {"strategy": type(self).__name__}

    @staticmethod
    def _key_for_operator(key: jax.Array | None, operator_index: int) -> jax.Array | None:
        """Derive one child's key by folding its position into the record's key.

        Each child gets an independent key that is still a function of the record alone, so a
        child draws the same values wherever its parent sits in a batch. A deterministic
        parent has no key to fold, and hands each child ``None``.

        Args:
            key: The record's key, or ``None`` for a deterministic parent.
            operator_index: The child's position in the composition.

        Returns:
            The child's key, or ``None``.
        """
        if key is None:
            return None
        return jax.random.fold_in(key, operator_index)

    @staticmethod
    def _apply_operator_conditionally(
        operator: OperatorModule,
        should_apply: bool | jax.Array,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any],
        key: jax.Array | None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any]]:
        """Apply operator with JAX control flow (cond) for trace compatibility."""

        def apply_fn(
            operands: tuple[PyTree, PyTree, dict[str, Any], jax.Array | None],
        ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
            d, s, m, k = operands
            return operator.apply(d, s, m, k, stats)

        def noop_fn(
            operands: tuple[PyTree, PyTree, dict[str, Any], jax.Array | None],
        ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
            d, s, m, _ = operands
            return d, s, m

        return jax.lax.cond(
            should_apply,
            apply_fn,
            noop_fn,
            (data, state, metadata, key),
        )

    def _execute_operators(
        self,
        operators: list[OperatorModule],
        context: StrategyContext,
    ) -> tuple[list, list, list]:
        """Execute operators and collect outputs (shared loop).

        Args:
            operators: List of operators to apply
            context: Execution context with data, state, metadata and the record's key

        Returns:
            Tuple of (outputs, states, metadatas) lists
        """
        outputs, states, metadatas = [], [], []
        for operator, key, stats in self._with_key_and_stats(operators, context):
            out_data, out_state, out_metadata = operator.apply(
                context.data, context.state, context.metadata, key, stats
            )
            outputs.append(out_data)
            states.append(out_state)
            metadatas.append(out_metadata)
        return outputs, states, metadatas

    def _with_key_and_stats(
        self,
        operators: list[OperatorModule],
        context: StrategyContext,
    ) -> Iterator[tuple[OperatorModule, jax.Array | None, dict[str, Any] | None]]:
        """Yield each operator with the key and the statistics belonging to its position.

        Both are a function of the child's position: the key is folded from the record's, and
        the statistics are the entry the composition computed for that child on its own input.

        Args:
            operators: The composition's operators, in order.
            context: Execution context carrying the record's key and the children's statistics.

        Yields:
            Each operator with its own key and its own statistics.
        """
        for index, operator in enumerate(operators):
            yield (
                operator,
                self._key_for_operator(context.key, index),
                statistics_for_child(context.stats, index),
            )

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
