"""T1: every source declares what its record index means, and the pipeline routes on it.

``RecordIdentity`` has three kinds: ``INDEXED`` (a stable position in the source),
``STREAM_IDS`` (an id the stream reports) and ``ARRIVAL`` (the arrival ordinal).
``DataSourceModule.record_identity`` is abstract, so a source that does not declare its kind is
refused at construction. The access predicates and ``get_batch_at`` are gone: the declared kind
is the one source of truth. ``for batch in pipeline`` reads every kind on the host stage: an
``INDEXED`` source with its stateless host read, a stream pass by pass at a position of the stage's
own; the compiled session (``pipeline.session()``) serves ``INDEXED`` sources only.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest
from flax import nnx

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule, RecordIdentity
from datarax.pipeline import iteration
from datarax.pipeline.pipeline import Pipeline
from datarax.sources import EagerSource, MemorySource
from datarax.sources.eager_source import read_host_batch
from datarax.sources.mixed_source import MixDataSourcesNode
from datarax.sources.streaming_disk_source import StreamingDiskSource
from tests.test_common.streams import RecordStream


@dataclass(frozen=True)
class _Config(StructuralConfig):
    pass


def test_the_kinds_are_exactly_three() -> None:
    assert {kind.name for kind in RecordIdentity} == {"INDEXED", "STREAM_IDS", "ARRIVAL"}


def test_every_exported_source_declares_its_kind() -> None:
    from datarax.sources.array_record_source import ArrayRecordSourceModule
    from datarax.sources.hf_source import HFEagerSource, HFStreamingSource
    from datarax.sources.tfds_source import TFDSEagerSource, TFDSStreamingSource

    expected = {
        EagerSource: RecordIdentity.INDEXED,
        MemorySource: RecordIdentity.INDEXED,
        TFDSEagerSource: RecordIdentity.INDEXED,
        HFEagerSource: RecordIdentity.INDEXED,
        StreamingDiskSource: RecordIdentity.INDEXED,
        MixDataSourcesNode: RecordIdentity.INDEXED,
        ArrayRecordSourceModule: RecordIdentity.INDEXED,
        TFDSStreamingSource: RecordIdentity.STREAM_IDS,
        HFStreamingSource: RecordIdentity.ARRIVAL,
    }
    for source_class, kind in expected.items():
        # Each declares a constant: its property's getter returns the kind for any instance.
        assert source_class.record_identity.fget(None) is kind, source_class.__name__


def test_a_source_without_a_kind_is_refused_at_construction() -> None:
    class Undeclared(DataSourceModule):
        def __len__(self) -> int:
            return 1

    with pytest.raises(TypeError, match=r"Undeclared.*record_identity"):
        Undeclared(_Config())  # pyright: ignore[reportAbstractUsage] - the refusal under test


def test_a_property_declares_the_kind() -> None:
    class Declared(DataSourceModule):
        @property
        def record_identity(self) -> RecordIdentity:
            """What this source's record index means: ARRIVAL."""
            return RecordIdentity.ARRIVAL

    assert Declared(_Config()).record_identity is RecordIdentity.ARRIVAL


def test_the_predicates_and_get_batch_at_are_gone() -> None:
    for name in ("supports_indexed_access", "supports_streaming", "get_batch_at"):
        assert not hasattr(DataSourceModule, name), name
        assert not hasattr(MemorySource, name), name


class _Indexed(DataSourceModule):
    """An indexed source over a column, with get_records."""

    @property
    def record_identity(self) -> RecordIdentity:
        """What this source's record index means: INDEXED."""
        return RecordIdentity.INDEXED

    def __init__(self) -> None:
        super().__init__(_Config())
        self.data = nnx.data({"x": np.arange(8, dtype=np.float32)})

    def __len__(self) -> int:
        return 8

    def get_records(self, indices: jax.Array) -> dict[str, jax.Array]:
        return {"x": jax.numpy.take(jax.numpy.asarray(self.data["x"]), indices[:, 1])}

    def get_batch(self, indices: Any, *, epochs: Any = 0, contiguous: bool = False) -> Any:
        """The host read the host stage serves the source with."""
        return read_host_batch(self.data, 8, indices, epochs=epochs, contiguous=contiguous)

    def element_spec(self) -> Any:
        return {"x": jax.ShapeDtypeStruct((), np.float32)}


def _stream(kind: RecordIdentity) -> RecordStream:
    """A stream of eight records."""
    return RecordStream({"x": np.arange(8, dtype=np.float32)}, kind=kind, chunk=4)


def _served(source: DataSourceModule) -> tuple[list[float], bool]:
    """The records a pipeline serves, and whether a compiled session served them."""
    pipeline = Pipeline(source=source, stages=[], batch_size=4, rngs=nnx.Rngs(0))
    batches: Iterator[Any] = iter(pipeline)
    in_session = isinstance(batches, iteration.PipelineIterator)
    values = [float(v) for batch in batches for v in np.asarray(batch["x"])]
    return values, in_session


def test_an_indexed_source_is_served_by_the_host_stage() -> None:
    values, in_session = _served(_Indexed())
    assert not in_session
    assert values == [float(i) for i in range(8)]
    session = Pipeline(source=_Indexed(), stages=[], batch_size=4, rngs=nnx.Rngs(0)).session()
    assert [float(v) for batch in session for v in np.asarray(batch["x"])] == values


@pytest.mark.parametrize("kind", [RecordIdentity.STREAM_IDS, RecordIdentity.ARRIVAL])
def test_a_stream_is_served_by_the_host_stage_at_a_position_of_its_own(
    kind: RecordIdentity,
) -> None:
    source = _stream(kind)
    values, in_session = _served(source)
    assert not in_session
    assert values == [float(i) for i in range(8)]
    assert source.pass_index == 0  # the stream's own position is the standalone API's


def test_a_session_of_a_stream_is_refused_naming_its_kind() -> None:
    pipeline = Pipeline(
        source=_stream(RecordIdentity.ARRIVAL), stages=[], batch_size=4, rngs=nnx.Rngs(0)
    )
    with pytest.raises(TypeError, match="ARRIVAL"):
        pipeline.session()


def test_a_memory_mapped_source_is_indexed(tmp_path: Path) -> None:
    from datarax.sources.streaming_disk_source import StreamingDiskSourceConfig

    path = tmp_path / "x.npy"
    np.save(path, np.arange(8, dtype=np.float32))
    source = StreamingDiskSource(StreamingDiskSourceConfig(path=str(path)))
    values, in_session = _served(source)
    assert not in_session
    assert values == [float(i) for i in range(8)]
