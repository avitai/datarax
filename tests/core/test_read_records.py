"""``read_records``: an indexed source's records, and their provenance when asked, from one read.

A source implementing ``IndexedHostReadWithProvenance`` (ArrayRecord, whose provenance would cost a
second decode) reads both at once; any other reads its batch with ``get_batch`` and looks its
records' provenance up by index. Every reader of an indexed source (the host stage's unit read,
a build writing a decoded copy) holds this rule once.
"""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
import pytest

from datarax.core.data_source import IndexedHostReadWithProvenance, read_records
from datarax.core.index_words import to_words
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from tests.test_common.worker_reads import png_source, served, write_png_records


@pytest.fixture(scope="module")
def png_paths(tmp_path_factory: pytest.TempPathFactory) -> list[str]:
    """8 PNG records in two ArrayRecord files."""
    return write_png_records(tmp_path_factory.mktemp("png"), 8)


def _words(*values: int) -> np.ndarray:
    return to_words(np.asarray(values, dtype=np.uint64))


class _Counting(MemorySource):
    """A memory source counting its reads and lookups."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.calls: list[str] = []

    def get_batch(self, indices: Any, *, epochs: Any = 0, contiguous: bool = False) -> Any:
        self.calls.append(f"get_batch contiguous={contiguous}")
        return super().get_batch(indices, epochs=epochs, contiguous=contiguous)

    def provenance(self, indices: Any) -> Any:
        self.calls.append("provenance")
        return super().provenance(indices)


def _memory() -> _Counting:
    return _Counting(
        MemorySourceConfig(),
        {
            "x": np.arange(10 * 3, dtype=np.float32).reshape(10, 3),
            "name": [f"r{i}" for i in range(10)],
        },
    )


class TestASourceWithoutAJointRead:
    def test_with_provenance_reads_the_batch_and_looks_provenance_up(self) -> None:
        source = _memory()
        words = _words(4, 5, 6)
        batch, provenance = read_records(
            source, words, epochs=1, contiguous=True, with_provenance=True
        )
        assert source.calls == ["get_batch contiguous=True", "provenance"]
        expected = source.get_batch(words, epochs=1, contiguous=True)
        assert served(batch) == served(expected)
        assert provenance is not None
        assert [dict(record) for record in provenance] == [
            dict(record) for record in source.provenance(words)
        ]

    def test_without_provenance_reads_the_batch_alone(self) -> None:
        source = _memory()
        words = _words(7, 2)
        batch, provenance = read_records(
            source,
            words,
            epochs=np.asarray([0, 3], np.int32),
            contiguous=False,
            with_provenance=False,
        )
        assert source.calls == ["get_batch contiguous=False"]
        assert provenance is None
        assert served(batch) == served(
            source.get_batch(words, epochs=np.asarray([0, 3], np.int32), contiguous=False)
        )


class TestASourceWithAJointRead:
    def test_with_provenance_reads_both_at_once(self, png_paths: list[str]) -> None:
        source = png_source(png_paths)
        assert isinstance(source, IndexedHostReadWithProvenance)
        words = _words(1, 2, 3)
        batch, provenance = read_records(
            source, words, epochs=2, contiguous=True, with_provenance=True
        )
        expected, expected_provenance = source.read_with_provenance(
            words, epochs=2, contiguous=True
        )
        assert served(batch) == served(expected)
        assert provenance is not None
        assert [dict(r) for r in provenance] == [dict(r) for r in expected_provenance]
        assert [r["name"] for r in provenance] == ["record-1", "record-2", "record-3"]

    def test_without_provenance_its_batch_equals_get_batch(self, png_paths: list[str]) -> None:
        source = png_source(png_paths)
        words = _words(6, 0)
        batch, provenance = read_records(
            source, words, epochs=0, contiguous=False, with_provenance=False
        )
        assert provenance is None
        assert served(batch) == served(source.get_batch(words, epochs=0, contiguous=False))
        assert np.asarray(jax.device_get(batch["label"])).tolist() == [6, 0]
