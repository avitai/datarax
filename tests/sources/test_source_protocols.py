"""A source supports a Python protocol only by implementing it.

``DataSourceModule`` gives no default ``__getitem__``, ``__len__``, ``__iter__`` or ``__next__``:
with them a stream passed ``isinstance(stream, grain.sources.RandomAccessDataSource)`` and
``stream[0]`` returned ``None``. A source that reads records by position (the eager sources)
implements both ``__len__`` and ``__getitem__`` and is a random-access source; a stream is not,
and indexing, sizing or iterating it is refused by Python. A source without a length still names
its records by position, and refuses only a shuffled order, which needs a length.
"""

from __future__ import annotations

from dataclasses import dataclass

import grain
import jax
import numpy as np
import pytest

from datarax.core.config import StructuralConfig
from datarax.core.data_source import (
    DataSourceModule,
    has_indexed_host_read,
    IndexedHostRead,
    RecordIdentity,
    SliceableStreamRun,
    StreamHostRead,
)
from datarax.core.element_batch import Batch
from datarax.sources import MemorySource, MemorySourceConfig
from datarax.sources.tfds_source import TFDSStreamingSource
from tests.test_common.streams import RecordStream


def _stream() -> RecordStream:
    return RecordStream({"x": np.arange(8, dtype=np.float32)})


@dataclass(frozen=True)
class _Config(StructuralConfig):
    pass


class _Unsized(DataSourceModule):
    """An indexed source with no length: it implements no container protocol at all."""

    @property
    def record_identity(self) -> RecordIdentity:
        return RecordIdentity.INDEXED

    def __init__(self) -> None:
        super().__init__(_Config())


def test_the_base_declares_no_container_protocol() -> None:
    for name in ("__getitem__", "__len__", "__iter__", "__next__"):
        assert name not in vars(DataSourceModule), name


def test_a_stream_is_not_a_random_access_source_and_refuses_indexing() -> None:
    stream = _stream()

    assert not isinstance(stream, grain.sources.RandomAccessDataSource)
    with pytest.raises(TypeError):
        stream[0]  # type: ignore[index]
    with pytest.raises(TypeError):
        next(stream)  # type: ignore[call-overload]


def test_an_eager_source_is_a_random_access_source() -> None:
    source = MemorySource(MemorySourceConfig(), {"x": np.arange(8, dtype=np.float32)})

    assert isinstance(source, grain.sources.RandomAccessDataSource)
    assert float(source[3]["x"]) == 3.0
    assert len(source) == 8


def test_a_source_without_a_length_names_positions_and_refuses_only_a_shuffle() -> None:
    source = _Unsized()

    with pytest.raises(TypeError):
        len(source)  # type: ignore[arg-type]
    np.testing.assert_array_equal(np.asarray(source.record_indices_at(2, 3))[:, 1], [2, 3, 4])
    with pytest.raises(ValueError, match="has no length"):
        source.record_indices_at(0, 3, jax.random.key(0))


def _memory() -> MemorySource:
    return MemorySource(MemorySourceConfig(), {"x": np.arange(8, dtype=np.float32)})


def test_an_indexed_source_reads_by_index_and_a_stream_reads_forward() -> None:
    """The two host reads have different names, so each kind satisfies only its own protocol."""
    assert isinstance(_memory(), IndexedHostRead)
    assert not isinstance(_memory(), StreamHostRead)
    assert isinstance(_stream(), StreamHostRead)
    assert not isinstance(_stream(), IndexedHostRead)


class _ForwardRead(DataSourceModule):
    """An indexed source whose ``get_batch`` reads forward: the name, not the read."""

    @property
    def record_identity(self) -> RecordIdentity:
        return RecordIdentity.INDEXED

    def get_batch(self, batch_size: int) -> Batch:
        raise AssertionError("never read")


def test_the_indexed_read_is_the_one_taking_epochs_and_contiguous() -> None:
    """A protocol checks names only; the host read's keywords decide (one predicate for all)."""
    forward = _ForwardRead(_Config())

    assert has_indexed_host_read(_memory())
    assert isinstance(forward, IndexedHostRead)
    assert not has_indexed_host_read(forward)
    assert not has_indexed_host_read(_stream())


def test_only_a_stream_reading_its_run_as_one_dataset_offers_a_sliceable_run() -> None:
    assert issubclass(TFDSStreamingSource, SliceableStreamRun)
    assert not isinstance(_stream(), SliceableStreamRun)
