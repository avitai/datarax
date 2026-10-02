"""The stream contract every stream builds on (D4, DD-DEC-3B, DD-DEC-12; W2b-12, W2b-14, W2b-54).

A stream returns host ``Batch``es named by itself: ``STREAM_IDS`` by the ids it reports, ``ARRIVAL``
by arrival ordinals that are never reset, ``epochs`` the pass from 0. An empty ``Batch`` ends a
pass and the next pull starts the next. The pipeline's key orders a pass; without it the stream
serves its own order. Strings travel beside the batch, never in it. Where the stream is lives
outside NNX state, so no Variable holds a Python value and the graph definition does not move.
"""

from __future__ import annotations

from collections.abc import Iterator
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
from tests.test_common.streams import RecordStream


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
                self, pass_index: int, key: jax.Array | None, size_hint: int
            ) -> Iterator[StreamChunk]:
                for chunk in super()._open_pass(pass_index, key, size_hint):
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

        leaves = jax.tree.leaves(nnx.state(stream))
        assert all(isinstance(leaf, np.ndarray | jax.Array) for leaf in leaves)

    def test_the_graph_definition_is_hashable_and_does_not_move_as_the_stream_advances(
        self,
    ) -> None:
        stream = _stream()
        before = nnx.graphdef(stream)

        stream.get_batch(4, key=jax.random.key(0))
        _pass(stream, 4)

        after = nnx.graphdef(stream)
        assert before == after
        assert hash(before) == hash(after)

    def test_a_tree_mode_split_and_merge_round_trip_keeps_reading(self) -> None:
        stream = _stream()
        stream.get_batch(4)
        graphdef, state = nnx.split(stream, graph=False)

        merged = nnx.merge(graphdef, state)

        assert _names([merged.get_batch(4)]) == [4, 5, 6, 7]

    def test_streams_differing_only_in_their_strings_share_one_graph_definition(self) -> None:
        one = _stream()
        other = RecordStream(
            {"x": np.zeros((_N, 2), np.float32), "label": np.zeros(_N, np.int64)},
            texts=[f"another {i}" for i in range(_N)],
        )

        assert nnx.graphdef(one) == nnx.graphdef(other)
