"""How a pipeline's records divide into batches and epochs, decided in one place.

``EpochPlan`` holds the rule: batches per epoch (floor under ``drop_last``, ceil otherwise),
batches left from a position, when an epoch is exhausted, how a position advances (wrapping
into the next epoch for a continuous stream) and which rows of a batch are real records. The
pipeline's length, ``batches_left``, iteration and the compiled step all read it, so they
cannot disagree, and the pipeline builds it from its source's current length.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
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
]


def _plan(
    length: int | None, batch_size: int, *, drop_last: bool = False, num_epochs: int | None = 1
) -> EpochPlan:
    return EpochPlan(
        length=length, batch_size=batch_size, drop_last=drop_last, num_epochs=num_epochs
    )


class TestBatchCounts:
    @pytest.mark.parametrize(("length", "batch_size", "drop_last"), _GRID)
    def test_an_epoch_holds_floor_or_ceil_of_its_records(
        self, length: int, batch_size: int, drop_last: bool
    ) -> None:
        expected = length // batch_size if drop_last else math.ceil(length / batch_size)
        assert _plan(length, batch_size, drop_last=drop_last).batches_per_epoch() == expected

    def test_a_source_without_a_length_has_no_batches_per_epoch(self) -> None:
        with pytest.raises(TypeError, match="no length"):
            _plan(None, 4).batches_per_epoch()

    @pytest.mark.parametrize(("length", "batch_size", "drop_last"), _GRID)
    def test_batches_left_count_the_records_after_the_position(
        self, length: int, batch_size: int, drop_last: bool
    ) -> None:
        plan = _plan(length, batch_size, drop_last=drop_last)
        for position in range(length + batch_size + 1):
            remaining = max(length - position, 0)
            expected = remaining // batch_size if drop_last else math.ceil(remaining / batch_size)
            assert plan.batches_left(position) == expected

    @pytest.mark.parametrize(("length", "batch_size", "drop_last"), _GRID)
    def test_an_epoch_is_exhausted_exactly_when_no_batch_is_left(
        self, length: int, batch_size: int, drop_last: bool
    ) -> None:
        plan = _plan(length, batch_size, drop_last=drop_last)
        for position in range(length + batch_size + 1):
            assert plan.exhausted(position) == (plan.batches_left(position) == 0)

    def test_a_continuous_stream_is_never_exhausted_and_has_no_batches_left(self) -> None:
        plan = _plan(10, 4, num_epochs=None)
        assert plan.batches_left(8) is None
        assert not plan.exhausted(8)

    def test_a_source_without_a_length_is_never_exhausted(self) -> None:
        plan = _plan(None, 4)
        assert plan.batches_left(100) is None
        assert not plan.exhausted(100)


class TestAdvance:
    def test_a_bounded_epoch_advances_the_position_only(self) -> None:
        assert _plan(10, 4).advance(8, 3) == (12, 3)

    def test_a_continuous_stream_wraps_into_the_next_epoch(self) -> None:
        plan = _plan(10, 4, num_epochs=None)
        assert plan.advance(4, 0) == (8, 0)
        assert plan.advance(8, 0) == (2, 1)

    def test_the_position_after_a_batch_is_the_position_advance_gives(self) -> None:
        for plan in (_plan(10, 4), _plan(10, 4, num_epochs=None)):
            for position in range(10):
                assert plan.position_after(position) == plan.advance(position, 7)[0]

    def test_the_next_epoch_starts_at_position_zero(self) -> None:
        assert _plan(10, 4).next_epoch(3) == (0, 4)

    def test_host_integers_and_traced_arrays_advance_alike(self) -> None:
        plan = _plan(10, 4, num_epochs=None)
        for position, epoch in ((0, 0), (8, 2), (9, 5)):
            traced = jax.jit(plan.advance)(jnp.int32(position), jnp.int32(epoch))
            assert tuple(int(value) for value in traced) == plan.advance(position, epoch)

    def test_valid_rows_mark_the_records_before_the_epoch_ends(self) -> None:
        np.testing.assert_array_equal(
            np.asarray(_plan(10, 4).valid_rows(8)), [True, True, False, False]
        )

    def test_every_row_is_valid_without_a_length(self) -> None:
        np.testing.assert_array_equal(np.asarray(_plan(None, 3).valid_rows(50)), [True] * 3)


class TestValidation:
    @pytest.mark.parametrize("num_epochs", [0, -1])
    def test_num_epochs_is_at_least_one_or_none(self, num_epochs: int) -> None:
        with pytest.raises(ValueError, match="num_epochs"):
            _plan(10, 4, num_epochs=num_epochs)

    def test_batch_size_is_positive(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            _plan(10, 0)

    def test_a_continuous_batch_fits_in_one_epoch(self) -> None:
        with pytest.raises(ValueError, match="batch_size <= len"):
            _plan(3, 4, num_epochs=None)


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

    def get_batch_at(self, start: Any, size: int, key: Any = None) -> dict[str, jax.Array]:
        del key
        positions = jnp.asarray(start, jnp.int32) + jnp.arange(size, dtype=jnp.int32)
        return {"x": (positions % self.rows).astype(jnp.float32)[:, None]}

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
        expected = _plan(length, batch_size, drop_last=drop_last).batches_per_epoch()
        assert len(pipeline) == expected
        assert pipeline.batches_left() == expected
        assert sum(1 for _ in pipeline) == expected
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
