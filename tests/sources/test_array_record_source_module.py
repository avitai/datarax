"""``ArrayRecordSourceModule`` is an ``INDEXED`` source read in batches on the host.

ArrayRecord files hold ``bytes`` records. The source names a record by its position in the files
and reads the records a batch names with one batched read (``__getitems__``, the reader's
per-file parallel read), then decodes them with one call of its ``decode``, which turns the
batch's records into one mapping of values per record. Numeric values become the batch's
columns and every other value the record's provenance, as in the eager sources. The order and
the position belong to the pipeline: the source holds no seed, no epoch and no cursor.
"""

from __future__ import annotations

import inspect
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

import jax
import numpy as np
import pytest
from array_record.python.array_record_module import ArrayRecordWriter
from flax import nnx

from datarax.core.data_source import IndexedHostReadWithProvenance, RecordIdentity
from datarax.core.index_words import from_words, to_words
from datarax.pipeline import Pipeline
from datarax.sources import MemorySource, MemorySourceConfig
from datarax.sources.array_record_source import (
    ArrayRecordSourceConfig,
    ArrayRecordSourceModule,
)
from datarax.sources.mixed_source import MixDataSourcesConfig, MixDataSourcesNode


_RECORDS = 12
_SHARD_RECORDS = (5, 7)


def _encode(index: int) -> bytes:
    return np.full(3, index, dtype=np.float32).tobytes() + f"name-{index}".encode()


def _decode_one(record: bytes) -> dict[str, Any]:
    values = np.frombuffer(record[:12], dtype=np.float32)
    return {"x": values, "label": np.int32(values[0]), "name": record[12:].decode()}


def _decode(records: Sequence[bytes]) -> list[dict[str, Any]]:
    return [_decode_one(record) for record in records]


def _numeric(records: Sequence[bytes]) -> list[dict[str, Any]]:
    return [{k: v for k, v in _decode_one(r).items() if k != "name"} for r in records]


@pytest.fixture
def shards(tmp_path: Path) -> list[str]:
    """Two ArrayRecord files holding records 0..4 and 5..11."""
    paths, start = [], 0
    for shard, count in enumerate(_SHARD_RECORDS):
        path = tmp_path / f"records-{shard}.array_record"
        writer = ArrayRecordWriter(str(path), "group_size:1")
        for index in range(start, start + count):
            writer.write(_encode(index))
        writer.close()
        paths.append(str(path))
        start += count
    return paths


def _source(paths: list[str], decode: Any = _decode) -> ArrayRecordSourceModule:
    return ArrayRecordSourceModule(ArrayRecordSourceConfig(), paths, decode=decode)


def _names(indices: Any) -> list[int]:
    return [int(i) for i in from_words(np.asarray(indices))]


class TestTheIndexedRead:
    def test_it_is_an_indexed_source_over_every_record_of_its_files(
        self, shards: list[str]
    ) -> None:
        source = _source(shards)

        assert source.record_identity is RecordIdentity.INDEXED
        assert len(source) == _RECORDS

    def test_a_batch_holds_the_named_records_decoded_named_with_the_given_words(
        self, shards: list[str]
    ) -> None:
        rows = np.array([9, 0, 4, 5], np.uint64)

        batch = _source(shards).get_batch(to_words(rows), epochs=3)

        assert [int(v) for v in batch["label"]] == [9, 0, 4, 5]
        np.testing.assert_array_equal(batch["x"][0], np.full(3, 9, np.float32))
        assert batch["x"].dtype == np.float32
        assert _names(batch.indices) == [9, 0, 4, 5]
        assert [int(e) for e in batch.epochs] == [3, 3, 3, 3]
        assert "name" not in batch.data

    def test_one_batched_read_and_one_decode_call_serve_a_batch(self, shards: list[str]) -> None:
        calls: list[int] = []

        def counted(records: Sequence[bytes]) -> list[dict[str, Any]]:
            calls.append(len(records))
            return _decode(records)

        source = _source(shards, decode=counted)
        reader = type(source._records.source)
        with patch.object(
            reader, "__getitems__", autospec=True, side_effect=reader.__getitems__
        ) as read:
            source.get_batch(to_words(np.array([1, 8, 2], np.uint64)))

        assert read.call_count == 1
        assert calls == [3]

    def test_a_contiguous_read_names_its_rows_as_the_run_and_refuses_a_gap(
        self, shards: list[str]
    ) -> None:
        source = _source(shards)

        batch = source.get_batch(to_words(np.arange(3, 7, dtype=np.uint64)), contiguous=True)

        assert _names(batch.indices) == [3, 4, 5, 6]
        with pytest.raises(ValueError, match="not a run"):
            source.get_batch(to_words(np.array([3, 5], np.uint64)), contiguous=True)

    def test_an_index_outside_the_files_is_refused(self, shards: list[str]) -> None:
        with pytest.raises(IndexError, match="outside"):
            _source(shards).get_batch(to_words(np.array([_RECORDS], np.uint64)))

    def test_provenance_holds_each_named_record_s_other_values(self, shards: list[str]) -> None:
        source = _source(shards)

        provenance = source.provenance(to_words(np.array([7, 2], np.uint64)))

        assert [dict(p) for p in provenance] == [{"name": "name-7"}, {"name": "name-2"}]

    def test_the_element_spec_describes_a_decoded_record_s_numbers(self, shards: list[str]) -> None:
        assert _source(shards).element_spec() == {
            "x": jax.ShapeDtypeStruct((3,), np.float32),
            "label": jax.ShapeDtypeStruct((), np.int32),
        }

    def test_a_decoder_whose_records_disagree_is_refused_naming_the_field(
        self, shards: list[str]
    ) -> None:
        def ragged(records: Sequence[bytes]) -> list[dict[str, Any]]:
            decoded = _numeric(records)
            decoded[-1]["x"] = decoded[-1]["x"][:2]
            return decoded

        with pytest.raises(ValueError, match="'x'"):
            _source(shards, decode=ragged).get_batch(to_words(np.arange(3, dtype=np.uint64)))


class TestThePipelineOwnsTheOrder:
    def test_a_shuffled_pipeline_serves_the_order_of_a_memory_source_of_the_same_records(
        self, shards: list[str]
    ) -> None:
        columns = {
            "x": np.stack([np.full(3, i, np.float32) for i in range(_RECORDS)]),
            "label": np.arange(_RECORDS, dtype=np.int32),
        }
        memory = MemorySource(MemorySourceConfig(), columns)
        source = _source(shards, decode=_numeric)

        def served(pipe_source: Any) -> list[tuple[list[int], list[int]]]:
            pipe = Pipeline(
                source=pipe_source,
                stages=[],
                batch_size=4,
                rngs=nnx.Rngs(5),
                shuffle=True,
                num_epochs=2,
            )
            out = [([int(v) for v in b["label"]], _names(b.indices)) for b in pipe]
            pipe.close()
            return out

        assert served(source) == served(memory)

    def test_the_source_holds_no_order_epoch_or_cursor_of_its_own(self, shards: list[str]) -> None:
        source = _source(shards)
        absent = (
            "current_index",
            "current_epoch",
            "total_records",
            "prefetch_cache",
            "iterator_initialized",
            "shuffled_indices",
            "_initialize_shuffle",
            "__iter__",
            "__next__",
            "_epochs_exhausted",
            "_start_next_epoch",
            "grain_source",
            "get_state",
            "set_state",
            "_getitems",
            "__getitem__",
        )
        own = set(vars(ArrayRecordSourceModule)) | set(vars(source))

        assert [name for name in absent if name in own] == []
        assert "rngs" not in inspect.signature(ArrayRecordSourceModule).parameters
        assert {f for f in ArrayRecordSourceConfig.__dataclass_fields__} >= {"local_files_only"}
        assert not {"seed", "num_epochs", "shuffle_files"} & set(
            ArrayRecordSourceConfig.__dataclass_fields__
        )
        assert jax.tree.leaves(nnx.state(source)) == []

    def test_the_traced_read_is_refused_naming_get_records(self, shards: list[str]) -> None:
        pipe = Pipeline(source=_source(shards), stages=[], batch_size=4, rngs=nnx.Rngs(0))

        with pytest.raises(NotImplementedError, match="get_records"):
            pipe.step()

    def test_a_tree_mode_split_and_merge_reads_the_same_records(self, shards: list[str]) -> None:
        source = _source(shards)
        graphdef, state = nnx.split(source, graph=False)
        merged = nnx.merge(graphdef, state)
        words = to_words(np.array([10, 1], np.uint64))

        assert [int(v) for v in merged.get_batch(words)["label"]] == [10, 1]

    def test_a_pickled_copy_reads_the_same_records_and_carries_none(
        self, shards: list[str]
    ) -> None:
        source = _source(shards, decode=_numeric)
        words = to_words(np.array([11, 3], np.uint64))
        source.get_batch(words)  # its readers are open

        # Round-trips the test's own object: no untrusted payload is loaded.
        payload = pickle.dumps(source)
        copy = pickle.loads(payload)

        assert len(payload) < 4096
        assert [int(v) for v in copy.get_batch(words)["label"]] == [11, 3]

    def test_a_mix_with_an_array_record_child_iterates(self, shards: list[str]) -> None:
        columns = {
            "x": np.zeros((6, 3), np.float32),
            "label": np.full(6, 100, np.int32),
        }
        mix = MixDataSourcesNode(
            MixDataSourcesConfig(weights=(0.5, 0.5)),
            [_source(shards, decode=_numeric), MemorySource(MemorySourceConfig(), columns)],
        )
        pipe = Pipeline(source=mix, stages=[], batch_size=4, rngs=nnx.Rngs(1), num_epochs=1)

        labels = [int(v) for b in pipe for v in b["label"]]
        pipe.close()

        assert labels
        assert set(labels) - {100} <= set(range(_RECORDS))
        assert set(labels) - {100}


class TestOneReadWithProvenance:
    """A batch and its records' provenance come from one batched read and one decode call."""

    def test_the_read_with_provenance_equals_get_batch_and_provenance_from_one_decode(
        self, shards: list[str]
    ) -> None:
        calls: list[int] = []

        def counted(records: Sequence[bytes]) -> list[dict[str, Any]]:
            calls.append(len(records))
            return _decode(records)

        source = _source(shards, decode=counted)
        words = to_words(np.array([1, 8, 2], np.uint64))
        assert isinstance(source, IndexedHostReadWithProvenance)
        batch, provenance = source.read_with_provenance(words, epochs=2)
        assert calls == [3]
        expected = _source(shards).get_batch(words, epochs=2)
        np.testing.assert_array_equal(batch.indices, expected.indices)
        np.testing.assert_array_equal(batch.epochs, expected.epochs)
        for field in ("x", "label"):
            np.testing.assert_array_equal(batch[field], expected[field])
        assert provenance == _source(shards).provenance(words)

    def test_raw_batches_with_provenance_decode_each_batch_once(self, shards: list[str]) -> None:
        def run(with_provenance: bool) -> tuple[list[Any], list[int]]:
            calls: list[int] = []

            def counted(records: Sequence[bytes]) -> list[dict[str, Any]]:
                calls.append(len(records))
                return _decode(records)

            pipe = Pipeline(
                source=_source(shards, decode=counted),
                stages=[],
                batch_size=4,
                rngs=nnx.Rngs(0),
                shuffle=True,
                num_epochs=2,
            )
            calls.clear()
            served = list(pipe.raw_batches(with_provenance=with_provenance))
            return served, calls

        plain, plain_calls = run(False)
        pairs, pair_calls = run(True)
        assert pair_calls == plain_calls == [4] * 6
        for batch, (paired, provenance) in zip(plain, pairs, strict=True):
            np.testing.assert_array_equal(batch.indices, paired.indices)
            np.testing.assert_array_equal(batch["x"], paired["x"])
            assert [p["name"] for p in provenance] == [f"name-{i}" for i in _names(paired.indices)]

    def test_an_eager_source_serves_pairs_by_its_lookup(self) -> None:
        source = MemorySource(
            MemorySourceConfig(), [{"x": np.float32(i), "name": f"r{i}"} for i in range(8)]
        )
        assert not isinstance(source, IndexedHostReadWithProvenance)
        pipe = Pipeline(source=source, stages=[], batch_size=4, rngs=nnx.Rngs(0), shuffle=True)
        for batch, provenance in pipe.raw_batches(with_provenance=True):
            assert [p["name"] for p in provenance] == [f"r{i}" for i in _names(batch.indices)]


class TestFiles:
    def test_local_files_only_refuses_a_missing_path_by_name(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.array_record"

        with pytest.raises(FileNotFoundError, match=r"(?s)local_files_only.*missing\.array_record"):
            ArrayRecordSourceModule(
                ArrayRecordSourceConfig(local_files_only=True), str(missing), decode=_decode
            )

    def test_local_files_only_reads_present_files(self, shards: list[str]) -> None:
        source = ArrayRecordSourceModule(
            ArrayRecordSourceConfig(local_files_only=True), shards, decode=_decode
        )

        assert len(source) == _RECORDS

    def test_the_repr_names_the_files_and_the_record_count(self, shards: list[str]) -> None:
        text = repr(_source(shards))

        assert "ArrayRecordSourceModule" in text
        assert shards[0] in text
        assert f"num_records={_RECORDS}" in text

    def test_close_releases_the_readers_and_may_repeat(self, shards: list[str]) -> None:
        source = _source(shards)
        source.get_batch(to_words(np.array([0], np.uint64)))

        source.close()
        source.close()

        assert [int(v) for v in source.get_batch(to_words(np.array([6], np.uint64)))["label"]] == [
            6
        ]

    def test_the_context_manager_closes_the_readers(self, shards: list[str]) -> None:
        reader = type(_source(shards)._records.source)
        with patch.object(reader, "__exit__", autospec=True) as exited:
            with _source(shards):
                pass

        exited.assert_called_once()
