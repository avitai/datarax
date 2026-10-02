"""The stream contract every stream builds on (D4, DD-DEC-3B, DD-DEC-12; W2b-12, W2b-14, W2b-54).

A stream returns host ``Batch``es named by itself: ``STREAM_IDS`` by the ids it reports, ``ARRIVAL``
by arrival ordinals that are never reset, ``epochs`` the pass from 0. An empty ``Batch`` ends a
pass and the next pull starts the next. The pipeline's key orders a pass; without it the stream
serves its own order. Strings travel beside the batch, never in it. Where the stream is lives
outside NNX state, so no Variable holds a Python value and the graph definition does not move.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import MappingProxyType
from typing import Any

import jax
import numpy as np
import pytest
from flax import nnx

from datarax.core.data_source import RecordIdentity
from datarax.core.element_batch import Batch
from datarax.core.index_words import from_words
from datarax.sources import StreamChunk, StreamingSourceBase
from tests.test_common.streams import (
    graph_definitions_across_a_pass,
    non_array_state_leaves,
    RecordStream,
    second_pulls_after_a_tree_round_trip,
)


_N = 10


def _stream(kind: RecordIdentity = RecordIdentity.STREAM_IDS, **kwargs: Any) -> RecordStream:
    columns = {
        "x": np.arange(_N, dtype=np.float32)[:, None] * np.ones((1, 2), np.float32),
        "label": np.arange(_N, dtype=np.int64) % 3,
    }
    return RecordStream(columns, kind=kind, texts=[f"t{i}" for i in range(_N)], **kwargs)


def _pass(stream: StreamingSourceBase, size: int, **kwargs: object) -> list[Batch]:
    batches = []
    while (batch := stream.get_batch(size, **kwargs)).batch_size:  # type: ignore[arg-type]
        batches.append(batch)
    return batches


def _names(batches: list[Batch]) -> list[int]:
    return [int(v) for b in batches for v in from_words(np.asarray(b.indices))]


class TestTheStreamBatch:
    def test_a_pull_is_a_host_batch_named_by_the_stream(self) -> None:
        batch = _stream().get_batch(4)

        assert isinstance(batch, Batch)
        assert all(isinstance(leaf, np.ndarray) for leaf in jax.tree.leaves(batch))
        assert batch.indices.dtype == np.uint32
        assert batch.indices.shape == (4, 2)
        np.testing.assert_array_equal(from_words(np.asarray(batch.indices)), [0, 1, 2, 3])
        np.testing.assert_array_equal(batch.epochs, np.zeros(4, np.int32))
        np.testing.assert_array_equal(batch.draws, np.zeros(4, np.int32))
        np.testing.assert_array_equal(batch["x"][:, 0], [0, 1, 2, 3])
        assert "text" not in batch

    def test_an_empty_batch_ends_a_pass_and_the_next_pull_starts_the_next(self) -> None:
        stream = _stream()

        first = _pass(stream, 4)
        second = _pass(stream, 4)

        assert [b.batch_size for b in first] == [4, 4, 2]
        assert _names(first) == _names(second) == list(range(_N))
        assert {int(e) for b in first for e in b.epochs} == {0}
        assert {int(e) for b in second for e in b.epochs} == {1}
        assert stream.pass_index == 2

    def test_pulls_of_any_size_take_the_records_in_order(self) -> None:
        stream = _stream(chunk=4)

        sizes = [b.batch_size for b in (stream.get_batch(n) for n in (1, 5, 2, 7))]

        assert sizes == [1, 5, 2, 2]

    def test_reset_starts_the_first_pass_again(self) -> None:
        stream = _stream()
        _pass(stream, 4)
        stream.get_batch(3)

        stream.reset()

        assert stream.pass_index == 0
        assert _names(_pass(stream, 4)) == list(range(_N))

    def test_the_spec_is_the_first_record_s_array_part_as_the_device_holds_it(self) -> None:
        stream = _stream()

        spec = stream.element_spec()

        assert spec == {
            "x": jax.ShapeDtypeStruct((2,), np.float32),
            "label": jax.ShapeDtypeStruct((), np.int32),
        }
        assert stream.pass_index == 0
        assert stream.get_batch(2).batch_size == 2

    def test_records_without_a_numeric_field_are_refused(self) -> None:
        class _TextOnly(RecordStream):
            def _open_pass(
                self, pass_index: int, key: jax.Array | None, read_size: int
            ) -> Iterator[StreamChunk]:
                for chunk in super()._open_pass(pass_index, key, read_size):
                    yield chunk._replace(columns={})

        with pytest.raises(ValueError, match="no numeric field"):
            _TextOnly({"x": np.zeros(3, np.float32)}, texts=["a", "b", "c"]).get_batch(2)

    def test_a_stream_is_built_from_the_public_base(self) -> None:
        import datarax.sources as sources  # noqa: PLC0415

        assert {"StreamingSourceBase", "StreamChunk"} <= set(sources.__all__)
        assert sources.StreamChunk is StreamChunk


class TestTheOrder:
    def test_without_a_key_every_pass_is_the_stream_s_own_order(self) -> None:
        stream = _stream()

        assert _names(_pass(stream, 4)) == _names(_pass(stream, 4)) == list(range(_N))

    def test_the_key_orders_each_pass_reproducibly(self) -> None:
        key = jax.random.key(7)
        first, second = _stream(), _stream()

        orders = [_names(_pass(first, 4, key=key)) for _ in range(3)]
        again = [_names(_pass(second, 4, key=key)) for _ in range(3)]

        assert orders == again
        assert all(sorted(order) == list(range(_N)) for order in orders)
        assert len({tuple(order) for order in orders}) == 3

    def test_another_seed_gives_another_order(self) -> None:
        one = _names(_pass(_stream(), 10, key=jax.random.key(0)))
        other = _names(_pass(_stream(), 10, key=jax.random.key(1)))

        assert one != other

    def test_the_key_is_read_once_per_pass(self) -> None:
        stream = _stream()
        key = jax.random.key(3)

        _pass(stream, 4, key=key)
        _pass(stream, 2, key=key)

        assert [index for index, _ in stream.opened] == [0, 1]
        assert all(given is key for _, given in stream.opened)


class TestTheReadSize:
    """A pass is read in chunks of the size it is opened with, whatever the pulls ask for."""

    def test_a_pass_is_opened_with_the_read_size_given(self) -> None:
        stream = _stream(chunk=None)

        first = stream.get_batch(2, read_size=4)
        rest = _pass(stream, 3, read_size=4)

        assert stream.read_sizes == [4]
        assert first.batch_size == 2
        assert [b.batch_size for b in rest] == [3, 3, 2]

    def test_without_one_a_pass_is_read_in_the_first_pull_s_size(self) -> None:
        stream = _stream(chunk=None)

        _pass(stream, 3)

        assert stream.read_sizes == [3]

    def test_a_read_size_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="read_size"):
            _stream().get_batch(2, read_size=0)


class TestAClone:
    """A clone has the stream's state and a position of its own (``DataraxModule.clone``)."""

    def test_a_clone_reads_from_its_own_position(self) -> None:
        stream = _stream()
        clone = stream.clone()

        assert _names([stream.get_batch(4)]) == [0, 1, 2, 3]
        assert _names([clone.get_batch(4)]) == [0, 1, 2, 3]
        assert _names([stream.get_batch(4)]) == [4, 5, 6, 7]

    def test_a_clone_starts_where_the_stream_is_between_passes(self) -> None:
        stream = _stream(RecordIdentity.ARRIVAL)
        _pass(stream, 4)

        clone = stream.clone()
        batch = clone.get_batch(4)

        assert clone.pass_index == 1
        assert _names([batch]) == [_N, _N + 1, _N + 2, _N + 3]
        assert {int(e) for e in batch.epochs} == {1}
        assert stream.pass_index == 1
        assert _names([stream.get_batch(4)]) == [_N, _N + 1, _N + 2, _N + 3]

    def test_a_stream_mid_pass_is_refused_a_clone(self) -> None:
        stream = _stream()
        stream.get_batch(4)

        with pytest.raises(ValueError, match="between passes"):
            stream.clone()

        assert _names([stream.get_batch(4)]) == [4, 5, 6, 7]


class _Failing(RecordStream):
    """A stream whose pass reader raises after its first chunk, on the passes named."""

    def __init__(self, failing: set[int]) -> None:
        super().__init__({"x": np.arange(_N, dtype=np.float32)}, chunk=3)
        self.failing = failing

    def _open_pass(
        self, pass_index: int, key: jax.Array | None, read_size: int
    ) -> Iterator[StreamChunk]:
        chunks = super()._open_pass(pass_index, key, read_size)
        yield next(chunks)
        if pass_index in self.failing:
            raise OSError("the connection was reset")
        yield from chunks


class TestAReadThatFails:
    """An error inside a pass never reads as the pass's end (F9)."""

    def test_the_error_reaches_the_pull_that_met_it(self) -> None:
        stream = _Failing({0})
        stream.get_batch(3)

        with pytest.raises(OSError, match="connection was reset"):
            stream.get_batch(3)

    def test_the_next_pull_is_refused_naming_the_pass_and_the_error(self) -> None:
        stream = _Failing({0})
        stream.get_batch(3)
        with pytest.raises(OSError):
            stream.get_batch(3)

        with pytest.raises(RuntimeError, match="pass 0") as refused:
            stream.get_batch(3)

        assert isinstance(refused.value.__cause__, OSError)
        assert stream.pass_index == 0

    def test_reset_starts_the_first_pass_again_after_a_failure(self) -> None:
        stream = _Failing({0})
        stream.get_batch(3)
        with pytest.raises(OSError):
            stream.get_batch(3)
        stream.failing.clear()

        stream.reset()

        assert _names(_pass(stream, 4)) == list(range(_N))


class TestArrivalNames:
    def test_ordinals_count_on_across_passes_and_never_repeat(self) -> None:
        stream = _stream(RecordIdentity.ARRIVAL)

        first, second = _names(_pass(stream, 4)), _names(_pass(stream, 4))

        assert first == list(range(_N))
        assert second == list(range(_N, 2 * _N))

    def test_ordinals_keep_counting_past_a_reset(self) -> None:
        stream = _stream(RecordIdentity.ARRIVAL)
        _pass(stream, 4)

        stream.reset()

        assert _names(_pass(stream, 4))[0] == _N

    def test_ordinals_cross_the_32_bit_boundary_as_distinct_words(self) -> None:
        stream = _stream(RecordIdentity.ARRIVAL)
        stream._position.value.arrived = 2**32 - 2

        batch = stream.get_batch(4)

        np.testing.assert_array_equal(
            np.asarray(batch.indices), [[0, 2**32 - 2], [0, 2**32 - 1], [1, 0], [1, 1]]
        )


class TestProvenanceBesideTheBatch:
    @pytest.mark.parametrize("kind", [RecordIdentity.STREAM_IDS, RecordIdentity.ARRIVAL])
    def test_the_sidecar_is_aligned_row_for_row(self, kind: RecordIdentity) -> None:
        stream = _stream(kind, chunk=3)
        key = jax.random.key(5)

        served = []
        while True:
            batch, provenance = stream.get_batch(4, key=key, with_provenance=True)
            if not batch.batch_size:
                assert provenance == ()
                break
            served.append((batch, provenance))

        for batch, provenance in served:
            assert len(provenance) == batch.batch_size
            assert all(isinstance(p, MappingProxyType) for p in provenance)
            assert [p["text"] for p in provenance] == [
                f"t{int(x)}" for x in np.asarray(batch["x"][:, 0])
            ]


class TestNnxHygiene:
    def test_no_variable_holds_a_python_value(self) -> None:
        stream = _stream()
        _pass(stream, 4)

        assert non_array_state_leaves(stream) == []

    def test_the_graph_definition_is_hashable_and_does_not_move_as_the_stream_advances(
        self,
    ) -> None:
        before, after = graph_definitions_across_a_pass(_stream(), 4)

        assert before == after
        assert hash(before) == hash(after)

    def test_a_tree_mode_split_and_merge_round_trip_keeps_reading(self) -> None:
        merged, twin = second_pulls_after_a_tree_round_trip(_stream, 4)

        assert from_words(merged[0]).tolist() == [4, 5, 6, 7]
        for ours, theirs in zip(merged, twin, strict=True):
            np.testing.assert_array_equal(ours, theirs)

    def test_streams_differing_only_in_their_strings_share_one_graph_definition(self) -> None:
        one = _stream()
        other = RecordStream(
            {"x": np.zeros((_N, 2), np.float32), "label": np.zeros(_N, np.int64)},
            texts=[f"another {i}" for i in range(_N)],
        )

        assert nnx.graphdef(one) == nnx.graphdef(other)


class _Watched(RecordStream):
    """A stream whose pass reader records when it is closed."""

    closed: list[int] = []

    def _open_pass(
        self, pass_index: int, key: jax.Array | None, read_size: int
    ) -> Iterator[StreamChunk]:
        try:
            yield from super()._open_pass(pass_index, key, read_size)
        finally:
            _Watched.closed.append(pass_index)


class TestTheOpenPassReaderIsClosed:
    """A stream stopped mid-pass closes the pass reader it holds, while the interpreter runs.

    A backend's reader left suspended until interpreter shutdown is finalized during module
    teardown, where HuggingFace's Parquet reader hangs (``uoft-cs/cifar10``).
    """

    def test_reset_closes_the_reader_of_a_pass_stopped_midway(self) -> None:
        _Watched.closed.clear()
        stream = _Watched({"x": np.arange(_N, dtype=np.float32)}, chunk=2)
        stream.get_batch(2)

        stream.reset()

        assert _Watched.closed == [0]

    def test_a_finished_pass_leaves_nothing_to_close(self) -> None:
        _Watched.closed.clear()
        stream = _Watched({"x": np.arange(4, dtype=np.float32)}, chunk=2)
        _pass(stream, 2)

        stream.reset()

        assert _Watched.closed == [0]  # closed once, when the pass ended

    def test_an_unreferenced_stream_closes_its_reader(self) -> None:
        import gc  # noqa: PLC0415

        _Watched.closed.clear()
        stream = _Watched({"x": np.arange(_N, dtype=np.float32)}, chunk=2)
        stream.get_batch(2)

        del stream
        gc.collect()

        assert _Watched.closed == [0]


# A stream stopped mid-pass and still referenced at exit: its reader's cleanup writes a line. It
# must run while the interpreter is alive (its module globals intact), not during teardown.
_STOPPED_AT_EXIT = """
import sys
import numpy as np
from tests.test_common.streams import (
    graph_definitions_across_a_pass,
    non_array_state_leaves,
    RecordStream,
    second_pulls_after_a_tree_round_trip,
)

LOG = open(sys.argv[1], "w")

class Stopped(RecordStream):
    def _open_pass(self, pass_index, key, read_size):
        try:
            yield from super()._open_pass(pass_index, key, read_size)
        finally:
            LOG.write("closed with globals intact\\n")
            LOG.flush()

stream = Stopped({"x": np.arange(10, dtype=np.float32)}, chunk=2)
stream.get_batch(2)
"""


def test_a_reader_still_open_at_exit_is_closed_before_teardown(tmp_path: Path) -> None:
    from substrax.testing import run_python  # noqa: PLC0415

    log = tmp_path / "closed.txt"
    result = run_python(
        _STOPPED_AT_EXIT,
        str(log),
        timeout=120.0,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert result.returncode == 0, result.stderr
    assert "Exception ignored" not in result.stderr
    assert log.read_text() == "closed with globals intact\n"
