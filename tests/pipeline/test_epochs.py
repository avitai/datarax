"""How a pipeline's records divide into batches and epochs, decided in one place.

``EpochPlan`` holds the rule. A batch never holds padding: under ``drop_last`` an epoch's
records that do not fill a batch are skipped and the next batch starts the next epoch;
otherwise a batch that reaches the epoch's end is completed from the head of the next
epoch's order, as ``repeat().batch()`` does in tf.data and Grain. The pipeline's length,
``batches_left``, iteration sessions and the compiled step all read the plan, so they cannot
disagree, and the pipeline builds it from its source's current length.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule
from datarax.pipeline import Pipeline, PipelineIterator
from datarax.pipeline.epochs import EpochPlan


_GRID = [
    (length, batch_size, drop_last)
    for length in (1, 7, 8, 9, 16, 17)
    for batch_size in (1, 3, 8)
    for drop_last in (False, True)
    if batch_size <= length or not drop_last
]


def _plan(
    length: int | None, batch_size: int, *, drop_last: bool = False, num_epochs: int | None = 1
) -> EpochPlan:
    return EpochPlan(
        length=length, batch_size=batch_size, drop_last=drop_last, num_epochs=num_epochs
    )


def _served(plan: EpochPlan, position: int, epoch: int) -> list[tuple[int, int]]:
    """The ``(epoch, position)`` of every record a session starting here serves, by the rule."""
    extent = plan.run_extent(position)
    assert extent is not None
    batches, final_rows = extent
    records = []
    for index in range(batches):
        size = final_rows if index == batches - 1 else plan.batch_size
        start, epoch = plan.batch_start(position, epoch)
        assert plan.length is not None
        for offset in range(size):
            row = start + offset
            records.append((epoch + row // plan.length, row % plan.length))
        position, epoch = plan.advance(start, epoch, size)
    return records


class TestExhausted:
    @pytest.mark.parametrize(("length", "batch_size", "drop_last"), _GRID)
    def test_an_epoch_is_exhausted_when_it_cannot_start_another_batch(
        self, length: int, batch_size: int, drop_last: bool
    ) -> None:
        plan = _plan(length, batch_size, drop_last=drop_last)
        for position in range(length + batch_size + 1):
            left = length - position
            expected = left < batch_size if drop_last else left <= 0
            assert plan.exhausted(position) == expected

    def test_a_source_without_a_length_is_never_exhausted(self) -> None:
        assert not _plan(None, 4).exhausted(100)


class TestBatchStart:
    def test_a_batch_starts_where_the_last_one_ended(self) -> None:
        assert _plan(10, 4).batch_start(8, 3) == (8, 3)

    def test_an_exhausted_epoch_rolls_into_the_next(self) -> None:
        assert _plan(10, 4).batch_start(10, 3) == (0, 4)

    def test_under_drop_last_the_records_short_of_a_batch_are_skipped(self) -> None:
        assert _plan(10, 4, drop_last=True).batch_start(8, 3) == (0, 4)

    def test_a_source_without_a_length_never_rolls_over(self) -> None:
        assert _plan(None, 4).batch_start(100, 3) == (100, 3)


class TestAdvance:
    def test_a_batch_inside_the_epoch_advances_the_position(self) -> None:
        assert _plan(10, 4).advance(4, 3, 4) == (8, 3)

    def test_a_batch_crossing_the_end_continues_in_the_next_epoch(self) -> None:
        assert _plan(10, 4).advance(8, 3, 4) == (2, 4)

    def test_a_batch_ending_exactly_at_the_end_stays_in_its_epoch(self) -> None:
        # The epoch is then exhausted, so the next batch starts the next epoch.
        plan = _plan(8, 4)
        assert plan.advance(4, 3, 4) == (8, 3)
        assert plan.batch_start(8, 3) == (0, 4)

    def test_a_short_batch_advances_by_its_own_size(self) -> None:
        assert _plan(10, 4).advance(8, 3, 2) == (10, 3)

    def test_the_next_epoch_starts_at_position_zero(self) -> None:
        assert _plan(10, 4).next_epoch(3) == (0, 4)

    @pytest.mark.parametrize("drop_last", [False, True])
    def test_host_integers_and_traced_arrays_follow_one_rule(self, drop_last: bool) -> None:
        plan = _plan(10, 4, drop_last=drop_last)

        def after_batch(position: Any, epoch: Any) -> Any:
            return plan.advance(*plan.batch_start(position, epoch), plan.batch_size)

        for position, epoch in ((0, 0), (6, 1), (8, 2), (9, 5), (10, 3)):
            traced = jax.jit(after_batch)(jnp.int32(position), jnp.int32(epoch))
            assert tuple(int(value) for value in traced) == after_batch(position, epoch)
            assert all(value.dtype == jnp.int32 for value in traced)
            assert not any(value.weak_type for value in traced)


class TestRunExtent:
    @pytest.mark.parametrize(("length", "batch_size", "drop_last"), _GRID)
    @pytest.mark.parametrize("num_epochs", [1, 2, 3])
    def test_a_run_serves_its_epochs_and_nothing_else(
        self, length: int, batch_size: int, drop_last: bool, num_epochs: int
    ) -> None:
        plan = _plan(length, batch_size, drop_last=drop_last, num_epochs=num_epochs)
        records = _served(plan, 0, 0)

        if drop_last:
            kept = length // batch_size * batch_size
            expected = [(epoch, row) for epoch in range(num_epochs) for row in range(kept)]
        else:
            expected = [(epoch, row) for epoch in range(num_epochs) for row in range(length)]
        assert records == expected

    @pytest.mark.parametrize(("length", "batch_size", "drop_last"), _GRID)
    def test_the_batch_count_is_the_standard_one(
        self, length: int, batch_size: int, drop_last: bool
    ) -> None:
        plan = _plan(length, batch_size, drop_last=drop_last, num_epochs=3)
        extent = plan.run_extent(0)
        assert extent is not None
        if drop_last:
            assert extent == (3 * (length // batch_size), batch_size)
        else:
            batches = math.ceil(3 * length / batch_size)
            assert extent == (batches, 3 * length - (batches - 1) * batch_size)

    def test_a_run_from_mid_epoch_finishes_that_epoch_first(self) -> None:
        plan = _plan(10, 4, num_epochs=2)
        assert _served(plan, 6, 0)[:4] == [(0, 6), (0, 7), (0, 8), (0, 9)]
        assert len(_served(plan, 6, 0)) == 4 + 10

    def test_an_exhausted_epoch_counts_as_one_served(self) -> None:
        assert _plan(10, 4, num_epochs=1).run_extent(10) == (0, 0)
        assert _plan(10, 4, num_epochs=2).run_extent(10) == (3, 2)

    def test_a_stream_or_a_source_without_a_length_has_no_extent(self) -> None:
        assert _plan(10, 4, num_epochs=None).run_extent(0) is None
        assert _plan(None, 4).run_extent(0) is None


class TestValidation:
    @pytest.mark.parametrize("num_epochs", [0, -1])
    def test_num_epochs_is_at_least_one_or_none(self, num_epochs: int) -> None:
        with pytest.raises(ValueError, match="num_epochs"):
            _plan(10, 4, num_epochs=num_epochs)

    def test_batch_size_is_positive(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            _plan(10, 0)

    def test_drop_last_refuses_a_batch_larger_than_an_epoch(self) -> None:
        with pytest.raises(ValueError, match="batch_size <= len"):
            _plan(3, 4, drop_last=True)

    def test_a_batch_larger_than_an_epoch_spans_several(self) -> None:
        plan = _plan(3, 7, num_epochs=3)
        assert plan.epochs_touched(7) == 3  # from position 2: rows 2..8, epochs 0, 1, 2
        assert _served(plan, 0, 0) == [(epoch, row) for epoch in range(3) for row in range(3)]
        assert plan.advance(2, 0, 7) == (3, 2)

    @pytest.mark.parametrize("drop_last", [False, True])
    def test_a_source_without_records_cannot_serve_a_batch(self, drop_last: bool) -> None:
        with pytest.raises(ValueError, match="no records"):
            _plan(0, 1, drop_last=drop_last)


@dataclass(frozen=True)
class _Config(StructuralConfig):
    pass


class _Resizable(DataSourceModule):
    """Indexed source whose length can change after the pipeline is built."""

    def __init__(self, rows: int) -> None:
        super().__init__(_Config())
        self.rows = rows

    def __len__(self) -> int:
        return self.rows

    def get_records(self, indices: jax.Array) -> dict[str, jax.Array]:
        return {"x": jnp.asarray(indices, jnp.float32)[:, None]}

    def element_spec(self) -> dict[str, jax.ShapeDtypeStruct]:
        return {"x": jax.ShapeDtypeStruct((1,), jnp.float32)}


def _pipeline(rows: int, batch_size: int, *, drop_last: bool = False) -> Pipeline:
    return Pipeline(
        source=_Resizable(rows),
        stages=[],
        batch_size=batch_size,
        drop_last=drop_last,
        rngs=nnx.Rngs(0),
    )


class TestPipelineAgreement:
    @pytest.mark.parametrize(("length", "batch_size", "drop_last"), _GRID)
    def test_length_batches_left_and_iteration_agree(
        self, length: int, batch_size: int, drop_last: bool
    ) -> None:
        pipeline = _pipeline(length, batch_size, drop_last=drop_last)
        extent = _plan(length, batch_size, drop_last=drop_last).run_extent(0)
        assert extent is not None
        assert len(pipeline) == extent[0]
        assert pipeline.batches_left() == extent[0]
        assert sum(1 for _ in pipeline) == extent[0]
        assert pipeline.batches_left() == 0

    def test_the_pipeline_follows_its_source_s_current_length(self) -> None:
        source = _Resizable(8)
        pipeline = Pipeline(source=source, stages=[], batch_size=4, rngs=nnx.Rngs(0))
        source.rows = 12
        assert len(pipeline) == 3
        assert sum(1 for _ in pipeline) == 3


class TestSession:
    def test_a_session_is_a_checkpointable_iterator(self) -> None:
        pipeline = _pipeline(8, 4)
        session = pipeline.session()
        assert isinstance(session, PipelineIterator)
        assert "position" in session.get_state()

    def test_iterating_a_random_access_pipeline_is_its_session(self) -> None:
        assert isinstance(iter(_pipeline(8, 4)), PipelineIterator)
