"""Each child of a wrapper receives the statistics it computed on the wrapper's input.

A wrapper -- a composite, a selector, a probabilistic wrapper -- applies its children inside one
vectorized call, so a child cannot compute statistics of its own while it runs: by then the batch
is gone. The wrapper computes each child's statistics once per batch, before the batch is
vectorized, and gives each child its own.

What that cannot give a child is statistics of the input it actually sees. A sequential
composite's second child runs on the first child's output, which does not exist when the
statistics are computed, so every child sees statistics of the wrapper's input. Exact per-stage
statistics come from separate Pipeline stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import PyTree

from datarax.core.config import OperatorConfig
from datarax.core.element_batch import Batch, Element
from datarax.core.operator import OperatorModule
from datarax.operators.composite_operator import (
    CompositeOperatorConfig,
    CompositeOperatorModule,
    CompositionStrategy,
)
from datarax.operators.probabilistic_operator import (
    ProbabilisticOperator,
    ProbabilisticOperatorConfig,
)
from datarax.operators.selector_operator import SelectorOperator, SelectorOperatorConfig


# The batch every test applies: mean 2.0, max 3.0, min 1.0, so a child's own statistic is
# visible in the output and no two children can be confused for one another.
VALUES = [1.0, 2.0, 3.0]
MEAN, MAXIMUM, MINIMUM = 2.0, 3.0, 1.0

_REDUCERS = {"mean": jnp.mean, "max": jnp.max, "min": jnp.min}


@dataclass(frozen=True)
class AddStatisticConfig(OperatorConfig):
    """Configuration naming which statistic of the batch a child adds."""

    statistic: str = field(default="mean", kw_only=True)


class AddStatistic(OperatorModule):
    """Adds a named statistic of the whole batch to every record.

    ``apply`` uses exactly the statistics it is handed and keeps no fallback to the stored
    ones, so a child given the wrong statistics, or none, fails here rather than quietly
    applying something else.
    """

    config: AddStatisticConfig  # pyright: ignore[reportIncompatibleVariableOverride]

    def compute_statistics(self, batch_data: PyTree) -> dict[str, Any] | None:
        """Reduce the batch to this child's own statistic."""
        return {"offset": _REDUCERS[self.config.statistic](batch_data["value"])}

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Add this child's statistic to the record."""
        del key
        if stats is None:
            raise AssertionError(f"the {self.config.statistic} child received no statistics")
        return {**data, "value": data["value"] + stats["offset"]}, state, metadata


class PassThrough(OperatorModule):
    """A child that computes no statistics of its own and needs none."""

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Return the record unchanged."""
        del key, stats
        return data, state, metadata


def _adds(statistic: str) -> AddStatistic:
    """A child adding the named statistic of the batch."""
    return AddStatistic(AddStatisticConfig(stochastic=False, statistic=statistic))


def _batch() -> Batch:
    """The three-record batch every test applies."""
    return Batch([Element(data={"value": jnp.array([value])}) for value in VALUES])


def _values(batch: Batch) -> list[float]:
    """The batch's ``value`` field as plain floats."""
    return [float(v) for v in batch.get_data()["value"].reshape(-1)]


def _composite(strategy: CompositionStrategy, operators: list[OperatorModule]) -> OperatorModule:
    """A composite over the given children."""
    return CompositeOperatorModule(
        CompositeOperatorConfig(strategy=strategy, operators=operators), rngs=nnx.Rngs(0)
    )


def test_each_sequential_child_receives_the_statistic_it_computed() -> None:
    """Two children reducing the same batch differently each get their own result."""
    composite = _composite(CompositionStrategy.SEQUENTIAL, [_adds("mean"), _adds("max")])

    result = composite(_batch())

    # Both statistics are of the composite's input, so each record gains mean + max.
    assert _values(result) == [value + MEAN + MAXIMUM for value in VALUES]


def test_a_child_does_not_receive_the_statistics_of_a_sibling() -> None:
    """A single child's statistic is its own, not one shared across the composition."""
    composite = _composite(CompositionStrategy.SEQUENTIAL, [_adds("min")])

    result = composite(_batch())

    assert _values(result) == [value + MINIMUM for value in VALUES]


def test_the_selected_child_receives_the_statistic_it_computed() -> None:
    """A selector with one child always applies it, with that child's own statistics."""
    selector = SelectorOperator(SelectorOperatorConfig(operators=[_adds("max")]), rngs=nnx.Rngs(0))

    result = selector(_batch())

    assert _values(result) == [value + MAXIMUM for value in VALUES]


def test_an_always_applied_child_receives_the_statistic_it_computed() -> None:
    """A probability of one reaches the child, which draws on its own statistics."""
    wrapper = ProbabilisticOperator(
        ProbabilisticOperatorConfig(operator=_adds("mean"), probability=1.0)
    )

    result = wrapper(_batch())

    assert _values(result) == [value + MEAN for value in VALUES]


def test_a_wrapper_whose_children_compute_none_passes_none() -> None:
    """Nothing is invented for children that have no statistics of their own."""
    deterministic = OperatorConfig(stochastic=False)
    composite = _composite(
        CompositionStrategy.SEQUENTIAL,
        [PassThrough(deterministic), PassThrough(deterministic)],
    )

    assert composite.compute_statistics({"value": jnp.asarray(VALUES).reshape(3, 1)}) is None


def test_a_wrapper_computes_statistics_for_the_children_that_have_them() -> None:
    """A composite mixing both kinds applies the one statistic that exists."""
    composite = _composite(
        CompositionStrategy.SEQUENTIAL,
        [PassThrough(OperatorConfig(stochastic=False)), _adds("mean")],
    )

    result = composite(_batch())

    assert _values(result) == [value + MEAN for value in VALUES]
