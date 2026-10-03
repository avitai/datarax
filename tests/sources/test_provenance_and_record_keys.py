"""Provenance by index and the record keys a table uses, by source kind (D3, D7, E-16, E-17).

A source that names records stably (``INDEXED``, ``STREAM_IDS``) serves each record's provenance by
its index, ``source.provenance(indices)``, and hands out the keys a per-record table is keyed by,
``source.record_keys(batch)``: the batch's indices. A source that names records by arrival cannot
look a past record up, so both are refused, naming the kind. A bare ``Batch`` has no source and so
no record keys. Provenance never enters a ``Batch`` or a trace.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from substrax.testing.compiles import expect_compiles

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule, RecordIdentity
from datarax.core.element_batch import Batch, PADDING_INDEX
from datarax.core.index_words import to_words
from datarax.sources import (
    MemorySource,
    MemorySourceConfig,
    MixDataSourcesConfig,
    MixDataSourcesNode,
    StreamingDiskSource,
    StreamingDiskSourceConfig,
)
from tests.test_common.tfds_fixture import FIXTURE, TFDSFixture


def _words(rows: list[int]) -> np.ndarray:
    return to_words(np.asarray(rows, dtype=np.uint64))


def _texts(prefix: str, count: int = 6) -> list[dict[str, Any]]:
    return [
        {
            "x": np.float32(i),
            "label": np.int32(i % 2),
            "text": f"{prefix}{i}",
            "meta": {"file": f"f{i}"},
        }
        for i in range(count)
    ]


def _memory(prefix: str = "r", count: int = 6) -> MemorySource:
    return MemorySource(MemorySourceConfig(), _texts(prefix, count))


@dataclass(frozen=True)
class _Config(StructuralConfig):
    pass


class _Kind(DataSourceModule):
    """A source of a given kind that holds no provenance."""

    def __init__(self, kind: RecordIdentity) -> None:
        super().__init__(_Config())
        self._kind = kind

    @property
    def record_identity(self) -> RecordIdentity:
        """The kind this source was built with."""
        return self._kind

    def __iter__(self) -> Iterator[Any]:
        return iter(())


class TestProvenanceByIndex:
    def test_each_named_record_s_provenance_in_the_order_named(self) -> None:
        provenance = _memory().provenance(_words([4, 0, 4, 2]))

        assert [p["text"] for p in provenance] == ["r4", "r0", "r4", "r2"]
        assert [p["meta/file"] for p in provenance] == ["f4", "f0", "f4", "f2"]

    def test_the_mappings_are_immutable(self) -> None:
        (record,) = _memory().provenance(_words([1]))

        assert isinstance(record, MappingProxyType)
        with pytest.raises(TypeError):
            record["text"] = "changed"  # type: ignore[index]

    def test_a_source_holding_none_serves_an_empty_mapping_per_record(self, tmp_path: Path) -> None:
        path = tmp_path / "rows.npy"
        np.save(path, np.zeros((5, 2), np.float32))
        numeric = MemorySource(MemorySourceConfig(), {"x": np.arange(5.0)})
        disk = StreamingDiskSource(StreamingDiskSourceConfig(path=str(path)))

        for source in (numeric, disk, _Kind(RecordIdentity.STREAM_IDS)):
            provenance = source.provenance(_words([3, 1]))
            assert len(provenance) == 2
            assert all(dict(p) == {} for p in provenance)

    @pytest.mark.parametrize(
        ("words", "match"),
        [
            (np.full((1, 2), PADDING_INDEX, np.uint32), "padding"),
            (_words([6]), "outside"),
            (np.asarray([[1, 0]], np.uint32), "outside"),
        ],
        ids=["padding", "past-the-end", "high-word"],
    )
    def test_words_naming_no_record_are_refused(self, words: np.ndarray, match: str) -> None:
        with pytest.raises(IndexError, match=match):
            _memory().provenance(words)

    def test_indices_that_are_not_words_are_refused(self) -> None:
        with pytest.raises(ValueError, match="uint32"):
            _memory().provenance(np.asarray([1, 2], np.int64))

    def test_the_provenance_of_a_served_batch_names_its_rows(self) -> None:
        source = _memory()
        batch = source.get_batch(_words([5, 3]))

        assert [p["text"] for p in source.provenance(batch.indices)] == ["r5", "r3"]

    def test_a_mix_serves_each_record_s_provenance_from_the_source_that_owns_it(self) -> None:
        mix = MixDataSourcesNode(
            MixDataSourcesConfig(weights=(0.5, 0.5)),
            [_memory("a", 3), _memory("b", 4)],
        )

        provenance = mix.provenance(_words([0, 3, 2, 6]))

        assert [p["text"] for p in provenance] == ["a0", "b0", "a2", "b3"]


class TestRecordKeys:
    @pytest.mark.parametrize("kind", [RecordIdentity.INDEXED, RecordIdentity.STREAM_IDS])
    def test_a_stably_named_source_keys_records_by_the_batch_s_indices(
        self, kind: RecordIdentity
    ) -> None:
        source = _memory() if kind is RecordIdentity.INDEXED else _Kind(kind)
        batch = _memory().get_batch(_words([4, 1, 2]))

        keys = source.record_keys(batch)

        np.testing.assert_array_equal(np.asarray(keys), np.asarray(batch.indices))

    def test_record_keys_compile_once_under_jit_and_nnx_jit(self) -> None:
        source = _memory()
        batches = [source.get_batch(_words(rows)) for rows in ([0, 1], [5, 2], [3, 3])]
        keyed = jax.jit(lambda batch: source.record_keys(batch))

        @nnx.jit
        def nnx_keyed(module: MemorySource, batch: Batch) -> jax.Array:
            return module.record_keys(batch)

        with expect_compiles(1):
            first = keyed(batches[0])
        with expect_compiles(0):
            rest = [keyed(batch) for batch in batches[1:]]
        for got, batch in zip((first, *rest), batches, strict=True):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(batch.indices))
        with expect_compiles(1):
            nnx_keyed(source, batches[0])
        with expect_compiles(0):
            for batch in batches[1:]:
                np.testing.assert_array_equal(
                    np.asarray(nnx_keyed(source, batch)), np.asarray(batch.indices)
                )

    def test_a_bare_batch_has_no_record_keys(self) -> None:
        assert not hasattr(Batch, "record_keys")
        assert not hasattr(_memory().get_batch(_words([0])), "record_keys")


class TestAnArrivalSourceRefusesTables:
    def test_provenance_by_index_is_refused_naming_the_kind(self) -> None:
        with pytest.raises(TypeError, match="ARRIVAL") as raised:
            _Kind(RecordIdentity.ARRIVAL).provenance(_words([0]))
        assert "beside each batch" in str(raised.value)

    def test_record_keys_are_refused_naming_the_kind(self) -> None:
        batch = _memory().get_batch(_words([0, 1]))

        with pytest.raises(TypeError, match="ARRIVAL"):
            _Kind(RecordIdentity.ARRIVAL).record_keys(batch)

    def test_the_refusal_raises_while_tracing(self) -> None:
        source = _Kind(RecordIdentity.ARRIVAL)
        batch = _memory().get_batch(_words([0, 1]))

        with pytest.raises(TypeError, match="ARRIVAL"):
            jax.jit(lambda b: source.record_keys(b))(batch)


class TestProvenanceStaysOutOfEveryTrace:
    def test_a_batch_holds_arrays_only(self) -> None:
        batch = _memory().get_batch(_words([0, 1, 2]))

        assert all(isinstance(leaf, np.ndarray) for leaf in jax.tree.leaves(batch))
        assert "text" not in batch

    def test_sources_differing_only_in_provenance_share_one_graphdef_and_one_program(
        self,
    ) -> None:
        first, second = _memory("a"), _memory("zz")
        assert nnx.graphdef(first) == nnx.graphdef(second)

        @nnx.jit
        def read(source: MemorySource, indices: jax.Array) -> jax.Array:
            return source.get_records(indices)["x"]

        words = jnp.asarray(_words([1, 4]))
        with expect_compiles(1):
            read(first, words)
        with expect_compiles(0):
            np.testing.assert_array_equal(np.asarray(read(second, words)), [1.0, 4.0])


@pytest.mark.tfds
def test_the_eager_tfds_source_serves_its_text_by_index(tfds_fixture: TFDSFixture) -> None:
    from datarax.sources import TFDSEagerConfig, TFDSEagerSource  # noqa: PLC0415

    source = TFDSEagerSource(
        TFDSEagerConfig(name=FIXTURE, split="train", data_dir=str(tfds_fixture.array_record))
    )
    rows = [7, 0, 19]

    provenance = source.provenance(_words(rows))

    records = source._provenance.value  # the stored mapping of each row
    assert [p["name"] for p in provenance] == [records[r]["name"] for r in rows]
    assert all(p["name"].startswith(b"record_0_") for p in provenance)
