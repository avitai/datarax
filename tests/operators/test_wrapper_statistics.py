"""Each child of a wrapper receives the statistics it computed, on the input it sees.

A wrapper deciding per record -- a selector, a probabilistic wrapper, a parallel composite --
applies its children inside one vectorized call, so a child cannot compute statistics of its own
while it runs: by then the batch is gone. The wrapper computes each child's statistics once per
batch, on its own input (which is each child's input), and gives each child its own.

A sequential composite calls each child on the whole batch in turn, so each child computes its
statistics on the input it actually receives: the second child's are of the first child's output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from datarax.core import batch_ops
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

    def compute_statistics(self, batch: Batch) -> dict[str, Any] | None:
        """Reduce the batch to this child's own statistic."""
        return {"offset": _REDUCERS[self.config.statistic](batch.data["value"])}

    def apply(
        self,
        element: Element,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> Element:
        """Add this child's statistic to the record."""
        data = element.data
        del key
        if stats is None:
            raise AssertionError(f"the {self.config.statistic} child received no statistics")
        return element.replace(data={**data, "value": data["value"] + stats["offset"]})


class PassThrough(OperatorModule):
    """A child that computes no statistics of its own and needs none."""

    def apply(
        self,
        element: Element,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> Element:
        """Return the record unchanged."""
        del key, stats
        return element


def _adds(statistic: str) -> AddStatistic:
    """A child adding the named statistic of the batch."""
    return AddStatistic(AddStatisticConfig(stochastic=False, statistic=statistic))


def _batch() -> Batch:
    """The three-record batch every test applies."""
    return batch_ops.from_stacked(
        batch_ops.stack([Element(data={"value": jnp.array([value])}) for value in VALUES])
    )


def _values(batch: Batch) -> list[float]:
    """The batch's ``value`` field as plain floats."""
    return [float(v) for v in batch.data["value"].reshape(-1)]


def _composite(strategy: CompositionStrategy, operators: list[OperatorModule]) -> OperatorModule:
    """A composite over the given children."""
    return CompositeOperatorModule(
        CompositeOperatorConfig(
            strategy=strategy,
        ),
        operators=operators,
    )


def test_each_sequential_child_fits_its_statistic_on_its_own_input() -> None:
    """A later child's statistic describes the earlier child's output, not the chain's input."""
    composite = _composite(CompositionStrategy.SEQUENTIAL, [_adds("mean"), _adds("max")])

    result = composite(_batch())

    # The first child adds the input's mean; the second adds the max of that output,
    # MAXIMUM + MEAN.
    assert _values(result) == [value + MEAN + (MAXIMUM + MEAN) for value in VALUES]


def test_a_child_does_not_receive_the_statistics_of_a_sibling() -> None:
    """A single child's statistic is its own, not one shared across the composition."""
    composite = _composite(CompositionStrategy.SEQUENTIAL, [_adds("min")])

    result = composite(_batch())

    assert _values(result) == [value + MINIMUM for value in VALUES]


def test_the_selected_child_receives_the_statistic_it_computed() -> None:
    """A selector with one child always applies it, with that child's own statistics."""
    selector = SelectorOperator(
        SelectorOperatorConfig(), operators=[_adds("max")], rngs=nnx.Rngs(0)
    )

    result = selector(_batch())

    assert _values(result) == [value + MAXIMUM for value in VALUES]


def test_an_always_applied_child_receives_the_statistic_it_computed() -> None:
    """A probability of one reaches the child, which draws on its own statistics."""
    wrapper = ProbabilisticOperator(
        ProbabilisticOperatorConfig(probability=1.0),
        operator=_adds("mean"),
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

    assert composite.compute_statistics(_batch()) is None


def test_a_wrapper_computes_statistics_for_the_children_that_have_them() -> None:
    """A composite mixing both kinds applies the one statistic that exists."""
    composite = _composite(
        CompositionStrategy.SEQUENTIAL,
        [PassThrough(OperatorConfig(stochastic=False)), _adds("mean")],
    )

    result = composite(_batch())

    assert _values(result) == [value + MEAN for value in VALUES]
