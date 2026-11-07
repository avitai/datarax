"""The stream contract every stream builds on (D4, DD-DEC-3B, DD-DEC-12; W2b-12, W2b-14, W2b-54).

A stream is read through ``iter_batches(batch_size, ...)``: one resumable iterator over its passes
returning host ``Batch``es named by the stream, ``STREAM_IDS`` by the ids it reports, ``ARRIVAL`` by
arrival ordinals that never repeat within the iterator, ``epochs`` the pass from 0. A pass's last
records are completed from the next pass's head, each row keeping its pass, unless ``drop_last``
skips them. The pipeline's key orders each pass; without it the stream serves its own order.
Strings travel beside the batch, never in it. The iterator holds where it is; the stream holds no
position, so no Variable holds a Python value and the graph definition never moves.
"""

from __future__ import annotations

import itertools
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
from datarax.sources import StreamChunk
from tests.test_common.streams import (
    batches_after_a_split_and_merge,
    graph_definitions_across_a_pass,
    non_array_state_leaves,
    RecordStream,
)


_N = 10


def _stream(kind: RecordIdentity = RecordIdentity.STREAM_IDS, **kwargs: Any) -> RecordStream:
    columns = {
        "x": np.arange(_N, dtype=np.float32)[:, None] * np.ones((1, 2), np.float32),
        "label": np.arange(_N, dtype=np.int64) % 3,
    }
    return RecordStream(columns, kind=kind, texts=[f"t{i}" for i in range(_N)], **kwargs)


def _names(batches: list[Batch]) -> list[int]:
    return [int(v) for b in batches for v in from_words(np.asarray(b.indices))]


def _epochs(batches: list[Batch]) -> list[int]:
    return [int(e) for b in batches for e in np.asarray(b.epochs)]


def _by_pass(batches: list[Batch]) -> list[list[int]]:
    """The names each pass served, in order."""
    passes: dict[int, list[int]] = {}
    for name, epoch in zip(_names(batches), _epochs(batches), strict=True):
        passes.setdefault(epoch, []).append(name)
    return [passes[p] for p in sorted(passes)]


class TestTheStreamBatch:
    def test_a_batch_is_a_host_batch_named_by_the_stream(self) -> None:
        batch = next(_stream().iter_batches(4))

        assert isinstance(batch, Batch)
        assert all(isinstance(leaf, np.ndarray) for leaf in jax.tree.leaves(batch))
        assert batch.indices.dtype == np.uint32
        assert batch.indices.shape == (4, 2)
        np.testing.assert_array_equal(from_words(np.asarray(batch.indices)), [0, 1, 2, 3])
        np.testing.assert_array_equal(batch.epochs, np.zeros(4, np.int32))
        np.testing.assert_array_equal(batch.draws, np.zeros(4, np.int32))
        np.testing.assert_array_equal(batch["x"][:, 0], [0, 1, 2, 3])
        assert "text" not in batch

    def test_one_pass_serves_every_record_once_then_ends(self) -> None:
        batches = list(_stream().iter_batches(4))

        assert [b.batch_size for b in batches] == [4, 4, 2]
        assert _names(batches) == list(range(_N))
        assert set(_epochs(batches)) == {0}

    def test_a_pass_s_tail_is_completed_from_the_next_pass_each_row_keeping_its_pass(
        self,
    ) -> None:
        batches = list(_stream().iter_batches(4, num_passes=2))

        assert [b.batch_size for b in batches] == [4] * 5
        assert _names(batches) == [*range(_N), *range(_N)]
        assert _epochs(batches) == [0] * _N + [1] * _N

    def test_drop_last_skips_each_pass_s_short_tail(self) -> None:
        batches = list(_stream().iter_batches(4, num_passes=2, drop_last=True))

        assert [b.batch_size for b in batches] == [4] * 4
        assert _by_pass(batches) == [list(range(8)), list(range(8))]

    def test_without_a_pass_count_the_stream_never_ends(self) -> None:
        batches = list(itertools.islice(_stream().iter_batches(4, num_passes=None), 12))

        assert [b.batch_size for b in batches] == [4] * 12
        assert _epochs(batches)[-1] == 4

    def test_chunks_of_any_size_are_cut_into_the_batches_asked_for(self) -> None:
        batches = list(_stream(chunk=4).iter_batches(3))

        assert [b.batch_size for b in batches] == [3, 3, 3, 1]
        assert _names(batches) == list(range(_N))

    @pytest.mark.parametrize(
        ("arguments", "named"),
        [
            ({"batch_size": 0}, "batch_size"),
            ({"batch_size": 2, "read_size": 0}, "read_size"),
            ({"batch_size": 2, "num_passes": 0}, "num_passes"),
        ],
    )
    def test_a_size_or_count_below_one_is_refused(
        self, arguments: dict[str, Any], named: str
    ) -> None:
        with pytest.raises(ValueError, match=named):
            _stream().iter_batches(**arguments)

    def test_the_spec_is_the_first_record_s_array_part_as_the_device_holds_it(self) -> None:
        stream = _stream()

        spec = stream.element_spec()

        assert spec == {
            "x": jax.ShapeDtypeStruct((2,), np.float32),
            "label": jax.ShapeDtypeStruct((), np.int32),
        }
        assert _names([next(stream.iter_batches(2))]) == [0, 1]

    def test_records_without_a_numeric_field_are_refused(self) -> None:
        class _TextOnly(RecordStream):
            def _open_pass(
                self, pass_index: int, key: np.ndarray | None, read_size: int
            ) -> Iterator[StreamChunk]:
                for chunk in super()._open_pass(pass_index, key, read_size):
                    yield chunk._replace(columns={})

        stream = _TextOnly({"x": np.zeros(3, np.float32)}, texts=["a", "b", "c"])
        with pytest.raises(ValueError, match="no numeric field"):
            next(stream.iter_batches(2))

    def test_a_stream_naming_records_by_id_without_ids_is_refused(self) -> None:
        class _NoIds(RecordStream):
            def _open_pass(
                self, pass_index: int, key: np.ndarray | None, read_size: int
            ) -> Iterator[StreamChunk]:
                for chunk in super()._open_pass(pass_index, key, read_size):
                    yield chunk._replace(ids=None)

        stream = _NoIds({"x": np.zeros(3, np.float32)}, kind=RecordIdentity.STREAM_IDS)
        with pytest.raises(TypeError, match="carries the id"):
            next(stream.iter_batches(2))

    def test_an_empty_chunk_is_passed_over(self) -> None:
        class _WithEmpty(RecordStream):
            def _open_pass(
                self, pass_index: int, key: np.ndarray | None, read_size: int
            ) -> Iterator[StreamChunk]:
                for chunk in super()._open_pass(pass_index, key, read_size):
                    yield chunk._replace(
                        columns={k: v[:0] for k, v in chunk.columns.items()},
                        provenance=(),
                        ids=None if chunk.ids is None else chunk.ids[:0],
                    )
                    yield chunk

        assert _names(list(_WithEmpty({"x": np.arange(_N, dtype=np.float32)}).iter_batches(4))) == (
            list(range(_N))
        )

    def test_a_stream_is_built_from_the_public_base(self) -> None:
        import datarax.sources as sources  # noqa: PLC0415

        assert {"StreamingSourceBase", "StreamChunk"} <= set(sources.__all__)
        assert sources.StreamChunk is StreamChunk


class TestTheOrder:
    def test_without_a_key_every_pass_is_the_stream_s_own_order(self) -> None:
        batches = list(_stream().iter_batches(4, num_passes=2))

        assert _by_pass(batches) == [list(range(_N))] * 2

    def test_the_key_orders_each_pass_reproducibly(self) -> None:
        key = jax.random.key(7)

        orders = _by_pass(list(_stream().iter_batches(4, num_passes=3, key=key)))
        again = _by_pass(list(_stream().iter_batches(4, num_passes=3, key=key)))

        assert orders == again
        assert all(sorted(order) == list(range(_N)) for order in orders)
        assert len({tuple(order) for order in orders}) == 3

    def test_another_seed_gives_another_order(self) -> None:
        one = _names(list(_stream().iter_batches(10, key=jax.random.key(0))))
        other = _names(list(_stream().iter_batches(10, key=jax.random.key(1))))

        assert one != other

    def test_the_key_is_read_once_per_pass_as_its_host_words(self) -> None:
        stream = _stream()
        key = jax.random.key(3)

        list(stream.iter_batches(4, num_passes=2, key=key))

        assert [index for index, _ in stream.opened] == [0, 1]
        words = np.asarray(jax.random.key_data(key), np.uint32)
        for _, given in stream.opened:
            assert isinstance(given, np.ndarray) and given.dtype == np.uint32
            np.testing.assert_array_equal(given, words)


class TestTheReadSize:
    """A pass is read in chunks of the size it is opened with, whatever the batch size."""

    def test_a_pass_is_opened_with_the_read_size_given(self) -> None:
        stream = _stream(chunk=None)

        batches = list(stream.iter_batches(3, read_size=4))

        assert stream.read_sizes == [4]
        assert [b.batch_size for b in batches] == [3, 3, 3, 1]

    def test_without_one_a_pass_is_read_in_the_batch_size(self) -> None:
        stream = _stream(chunk=None)

        list(stream.iter_batches(3))

        assert stream.read_sizes == [3]


class _Failing(RecordStream):
    """A stream whose pass reader raises after its first chunk, on the passes named."""

    def __init__(self, failing: set[int]) -> None:
        super().__init__({"x": np.arange(_N, dtype=np.float32)}, chunk=3)
        self.failing = failing

    def _open_pass(
        self, pass_index: int, key: np.ndarray | None, read_size: int
    ) -> Iterator[StreamChunk]:
        chunks = super()._open_pass(pass_index, key, read_size)
        yield next(chunks)
        if pass_index in self.failing:
            raise OSError("the connection was reset")
        yield from chunks


class TestAReadThatFails:
    """An error inside a pass never reads as the pass's end (F9)."""

    def test_the_error_reaches_the_next_that_met_it(self) -> None:
        batches = _Failing({0}).iter_batches(3)
        next(batches)

        with pytest.raises(OSError, match="connection was reset"):
            next(batches)

    def test_the_next_call_is_refused_naming_the_pass_and_the_error(self) -> None:
        batches = _Failing({0}).iter_batches(3)
        next(batches)
        with pytest.raises(OSError):
            next(batches)

        with pytest.raises(RuntimeError, match="pass 0") as refused:
            next(batches)

        assert isinstance(refused.value.__cause__, OSError)

    def test_a_new_iterator_after_a_failure_reads_from_a_clean_start(self) -> None:
        stream = _Failing({0})
        failed = stream.iter_batches(3)
        next(failed)
        with pytest.raises(OSError):
            next(failed)
        stream.failing.clear()

        assert _names(list(stream.iter_batches(4))) == list(range(_N))


class TestArrivalNames:
    def test_ordinals_count_on_across_passes_and_never_repeat(self) -> None:
        batches = list(_stream(RecordIdentity.ARRIVAL).iter_batches(4, num_passes=2))

        assert _names(batches) == list(range(2 * _N))

    def test_an_iterator_restored_from_a_state_continues_the_ordinals(self) -> None:
        stream = _stream(RecordIdentity.ARRIVAL)
        first = stream.iter_batches(4, num_passes=3)
        list(itertools.islice(first, 3))

        resumed = stream.iter_batches(4, num_passes=3)
        resumed.set_state(first.get_state())

        assert _names(list(resumed))[0] == 12

    def test_ordinals_cross_the_32_bit_boundary_as_distinct_words(self) -> None:
        batches = _stream(RecordIdentity.ARRIVAL).iter_batches(4)
        batches.set_state({"pass": 0, "records": 0, "arrived": 2**32 - 2, "passes_left": 1})

        np.testing.assert_array_equal(
            np.asarray(next(batches).indices), [[0, 2**32 - 2], [0, 2**32 - 1], [1, 0], [1, 1]]
        )


class TestNamesKeepTheTopBitClear:
    """A stream's names are record indices: below ``2**63``, the top bit marking no record."""

    def test_ordinals_up_to_the_last_record_index_are_served(self) -> None:
        batches = _stream(RecordIdentity.ARRIVAL).iter_batches(2)
        batches.set_state({"pass": 0, "records": 0, "arrived": (1 << 63) - 2, "passes_left": 1})

        assert _names([next(batches)]) == [(1 << 63) - 2, (1 << 63) - 1]

    def test_an_ordinal_past_the_last_record_index_is_refused(self) -> None:
        batches = _stream(RecordIdentity.ARRIVAL).iter_batches(4)
        batches.set_state({"pass": 0, "records": 0, "arrived": (1 << 63) - 2, "passes_left": 1})

        with pytest.raises(ValueError, match=r"2\*\*63"):
            next(batches)

    def test_an_id_with_the_top_bit_set_is_refused(self) -> None:
        class _HighIds(RecordStream):
            def _open_pass(
                self, pass_index: int, key: np.ndarray | None, read_size: int
            ) -> Iterator[StreamChunk]:
                for chunk in super()._open_pass(pass_index, key, read_size):
                    assert chunk.ids is not None  # a STREAM_IDS stream reports ids
                    yield chunk._replace(ids=chunk.ids + np.uint64(1 << 63))

        stream = _HighIds({"x": np.arange(_N, dtype=np.float32)})

        with pytest.raises(ValueError, match=r"2\*\*63"):
            next(stream.iter_batches(4))


class TestState:
    """The iterator's state is where it is: the pass, the records served in it, the ordinals
    served, and the passes left; an iterator set to it serves exactly what the original would."""

    def test_the_state_names_the_pass_the_records_the_arrivals_and_the_passes_left(
        self,
    ) -> None:
        batches = _stream(RecordIdentity.ARRIVAL).iter_batches(4, num_passes=3)
        list(itertools.islice(batches, 3))

        assert batches.get_state() == {"pass": 1, "records": 2, "arrived": 12, "passes_left": 2}

    def test_after_a_run_s_short_final_batch_the_state_is_in_its_last_pass(self) -> None:
        """The state after a batch is just past its last row: a run's last pass, not the next."""
        batches = _stream(RecordIdentity.ARRIVAL).iter_batches(4)
        list(batches)

        assert batches.get_state() == {"pass": 0, "records": _N, "arrived": _N, "passes_left": 1}
        ended = _stream(RecordIdentity.ARRIVAL).iter_batches(4)
        ended.set_state(batches.get_state())
        assert list(ended) == []

    @pytest.mark.parametrize(
        "state",
        [
            {"pass": 0, "records": 0, "arrived": 0},
            {"pass": -1, "records": 0, "arrived": 0, "passes_left": 1},
            {"pass": 0, "records": 0, "arrived": 0, "passes_left": -1},
        ],
        ids=["a-key-missing", "a-negative-pass", "negative-passes-left"],
    )
    def test_a_malformed_state_is_refused_and_the_iterator_reads_on(
        self, state: dict[str, Any]
    ) -> None:
        batches = _stream().iter_batches(4)
        next(batches)

        with pytest.raises(ValueError, match="state"):
            batches.set_state(state)

        assert _names([next(batches)]) == [4, 5, 6, 7]

    def test_a_state_past_its_pass_s_end_resumes_at_the_next_pass(self) -> None:
        """A saved count the pass no longer holds is replayed to the pass's end, which is served."""
        batches = _stream().iter_batches(4, num_passes=2)
        batches.set_state({"pass": 0, "records": _N + 5, "arrived": 0, "passes_left": 2})

        served = list(batches)

        assert _names(served) == list(range(_N))
        assert set(_epochs(served)) == {1}

    def test_a_stream_without_end_has_no_passes_left(self) -> None:
        batches = _stream().iter_batches(4, num_passes=None)
        next(batches)

        assert batches.get_state()["passes_left"] is None

    @pytest.mark.parametrize("kind", [RecordIdentity.STREAM_IDS, RecordIdentity.ARRIVAL])
    @pytest.mark.parametrize("drop_last", [False, True])
    def test_resuming_after_any_batch_serves_the_rest_of_the_uninterrupted_run(
        self, kind: RecordIdentity, drop_last: bool
    ) -> None:
        key = jax.random.key(11)

        def run() -> Any:
            return _stream(kind).iter_batches(4, num_passes=3, key=key, drop_last=drop_last)

        whole = list(run())
        for done in range(len(whole)):
            first = run()
            list(itertools.islice(first, done))
            resumed = run()
            resumed.set_state(first.get_state())
            rest = list(resumed)

            assert _names(rest) == _names(whole[done:])
            assert _epochs(rest) == _epochs(whole[done:])


class TestProvenanceBesideTheBatch:
    @pytest.mark.parametrize("kind", [RecordIdentity.STREAM_IDS, RecordIdentity.ARRIVAL])
    def test_the_sidecar_is_aligned_row_for_row(self, kind: RecordIdentity) -> None:
        stream = _stream(kind, chunk=3)

        served = list(stream.iter_batches(4, key=jax.random.key(5), with_provenance=True))

        assert sum(batch.batch_size for batch, _ in served) == _N
        for batch, provenance in served:
            assert len(provenance) == batch.batch_size
            assert all(isinstance(p, MappingProxyType) for p in provenance)
            assert [p["text"] for p in provenance] == [
                f"t{int(x)}" for x in np.asarray(batch["x"][:, 0])
            ]

    def test_a_text_column_is_refused_pointing_at_provenance(self) -> None:
        """A device holds no text, so the test stream takes text only as provenance."""
        with pytest.raises(TypeError, match=r"'name'.*<U2.*texts="):
            RecordStream(
                {"x": np.arange(3, dtype=np.float32), "name": np.array(["r0", "r1", "r2"])}
            )


class TestNnxHygiene:
    def test_no_variable_holds_a_python_value(self) -> None:
        stream = _stream()
        list(stream.iter_batches(4))

        assert non_array_state_leaves(stream) == []

    def test_the_graph_definition_is_hashable_and_does_not_move_as_the_stream_is_read(
        self,
    ) -> None:
        before, after = graph_definitions_across_a_pass(_stream(), 4)

        assert before == after
        assert hash(before) == hash(after)

    @pytest.mark.parametrize("graph", [True, False], ids=["graph-mode", "tree-mode"])
    def test_a_stream_split_and_merged_reads_as_the_original(self, graph: bool) -> None:
        merged, original = batches_after_a_split_and_merge(_stream, 4, graph=graph)

        assert from_words(merged[0]).tolist() == [0, 1, 2, 3]
        for ours, theirs in zip(merged, original, strict=True):
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
        self, pass_index: int, key: np.ndarray | None, read_size: int
    ) -> Iterator[StreamChunk]:
        try:
            yield from super()._open_pass(pass_index, key, read_size)
        finally:
            _Watched.closed.append(pass_index)


class TestTheOpenPassReaderIsClosed:
    """An iterator stopped mid-pass closes the pass reader it holds, while the interpreter runs.

    A backend's reader left suspended until interpreter shutdown is finalized during module
    teardown, where HuggingFace's Parquet reader hangs (``uoft-cs/cifar10``).
    """

    def test_closing_the_iterator_closes_the_reader_of_a_pass_stopped_midway(self) -> None:
        _Watched.closed.clear()
        batches = _Watched({"x": np.arange(_N, dtype=np.float32)}, chunk=2).iter_batches(2)
        next(batches)

        batches.close()

        assert _Watched.closed == [0]

    def test_a_finished_pass_closes_its_reader_once(self) -> None:
        _Watched.closed.clear()
        batches = _Watched({"x": np.arange(4, dtype=np.float32)}, chunk=2).iter_batches(2)
        list(batches)

        batches.close()

        assert _Watched.closed == [0]

    def test_an_unreferenced_iterator_closes_its_reader(self) -> None:
        import gc  # noqa: PLC0415

        _Watched.closed.clear()
        batches = _Watched({"x": np.arange(_N, dtype=np.float32)}, chunk=2).iter_batches(2)
        next(batches)

        del batches
        gc.collect()

        assert _Watched.closed == [0]


# An iterator stopped mid-pass and still referenced at exit: its reader's cleanup writes a line.
# It must run while the interpreter is alive (its module globals intact), not during teardown.
_STOPPED_AT_EXIT = """
import sys
import numpy as np
from tests.test_common.streams import RecordStream

LOG = open(sys.argv[1], "w")

class Stopped(RecordStream):
    def _open_pass(self, pass_index, key, read_size):
        try:
            yield from super()._open_pass(pass_index, key, read_size)
        finally:
            LOG.write("closed with globals intact\\n")
            LOG.flush()

batches = Stopped({"x": np.arange(10, dtype=np.float32)}, chunk=2).iter_batches(2)
next(batches)
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
