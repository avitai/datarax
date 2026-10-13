"""No compiled program reads a kind-equal host value.

A ``HostValue`` compares equal to every holder of its kind, so two modules differing only in what
it holds share one graphdef and one compiled program. That is correct only while nothing a program
runs reads the holder: a program that took one source's holder off the module, or captured it in a
host callback, would serve that source's content to every other source of the structure, with no
compile and no error. The holder a program does capture (``HostArray``, the memory map
``StreamingDiskSource``'s host callback reads) compares equal only to itself, and is exempt.

:class:`~tests.test_common.holder_reads.HolderRecorder` notes every kind-equal holder taken off a
module, read, or captured in a host callback while JAX traces. Every source kind runs through
every path that serves it: the traced paths for a source with a traced read, and iteration for
every source.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import datasets
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from datarax.core.data_source import RecordIdentity
from datarax.pipeline.pipeline import Pipeline
from datarax.sources import (
    TFDSEagerConfig,
    TFDSEagerSource,
    TFDSStreamingConfig,
    TFDSStreamingSource,
)
from datarax.sources.array_record_source import ArrayRecordSourceConfig, ArrayRecordSourceModule
from datarax.sources.hf_source import (
    HFEagerConfig,
    HFEagerSource,
    HFStreamingConfig,
    HFStreamingSource,
)
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from datarax.sources.mixed_source import MixDataSourcesConfig, MixDataSourcesNode
from datarax.sources.streaming_disk_source import StreamingDiskSource, StreamingDiskSourceConfig
from tests.test_common.holder_reads import HolderRecorder
from tests.test_common.streams import RecordStream
from tests.test_common.tfds_fixture import FIXTURE, TFDSFixture
from tests.test_common.worker_reads import write_records


_RECORDS = 16
_BATCH = 4

Make = Callable[[Path, pytest.MonkeyPatch], Any]


def _rows() -> list[dict[str, Any]]:
    return [{"x": np.full((3,), i, np.float32), "id": f"r{i}"} for i in range(_RECORDS)]


def _memory(directory: Path, monkeypatch: pytest.MonkeyPatch) -> MemorySource:
    return MemorySource(MemorySourceConfig(), _rows())


def _disk(directory: Path, monkeypatch: pytest.MonkeyPatch) -> StreamingDiskSource:
    path = directory / "rows.npy"
    if not path.exists():
        np.save(path, np.arange(_RECORDS * 3, dtype=np.float32).reshape(_RECORDS, 3))
    return StreamingDiskSource(StreamingDiskSourceConfig(path=str(path)))


def _decode(records: Sequence[bytes]) -> list[dict[str, np.ndarray]]:
    return [{"x": np.frombuffer(record, np.float32).copy()} for record in records]


def _array_records(directory: Path, monkeypatch: pytest.MonkeyPatch) -> ArrayRecordSourceModule:
    rows = [np.full((3,), i, np.float32).tobytes() for i in range(_RECORDS)]
    shards = directory / f"shards_{len(list(directory.iterdir()))}"
    shards.mkdir()
    paths = write_records(shards, "r", rows, 2)
    return ArrayRecordSourceModule(ArrayRecordSourceConfig(), paths, decode=_decode)


def _hf_eager(directory: Path, monkeypatch: pytest.MonkeyPatch) -> HFEagerSource:
    table = datasets.Dataset.from_dict(
        {"x": [[float(i)] * 3 for i in range(_RECORDS)], "id": [f"r{i}" for i in range(_RECORDS)]}
    )
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: table)
    return HFEagerSource(HFEagerConfig(name="local", split="train"))


def _mix(*children: Make) -> Make:
    def make(directory: Path, monkeypatch: pytest.MonkeyPatch) -> MixDataSourcesNode:
        config = MixDataSourcesConfig(weights=tuple(1.0 for _ in children))
        return MixDataSourcesNode(config, [child(directory, monkeypatch) for child in children])

    return make


def _pipeline(source: Any, *, shuffle: bool = False) -> Pipeline:
    return Pipeline(source=source, stages=[], batch_size=_BATCH, rngs=nnx.Rngs(0), shuffle=shuffle)


def _total(batch: Any) -> jax.Array:
    return jnp.sum(
        jnp.stack([jnp.sum(jnp.asarray(leaf, jnp.float32)) for leaf in jax.tree.leaves(batch.data)])
    )


def _iterate(source: Any, *, shuffle: bool = True) -> None:
    pipeline = _pipeline(source, shuffle=shuffle)
    next(iter(pipeline))
    pipeline.close()


_TRACED_PATHS: dict[str, Callable[[Any], Any]] = {
    "step": lambda source: _pipeline(source).step(),
    "step, shuffled": lambda source: _pipeline(source, shuffle=True).step(),
    "session": lambda source: next(iter(_pipeline(source).session())),
    "scan": lambda source: _pipeline(source).scan(_total, length=2),
    "nnx.jit, graph mode": lambda source: nnx.jit(lambda p: p.step())(_pipeline(source)),
    "nnx.jit, tree mode": lambda source: nnx.jit(lambda p: p.step(), graph=False)(
        _pipeline(source)
    ),
    "scan in nnx.jit": lambda source: nnx.jit(lambda p: p.scan(_total, length=2))(
        _pipeline(source)
    ),
}

# Sources with a traced read (``get_records``).
_TRACED_SOURCES: dict[str, Make] = {
    "memory": _memory,
    "streaming disk": _disk,
    "hf eager": _hf_eager,
    "mix of memory sources": _mix(_memory, _memory),
    "mix of hf eager sources": _mix(_hf_eager, _hf_eager),
}

# Sources read only on the host: a Python decode, or a mix over one.
_HOST_SOURCES: dict[str, Make] = {
    "array record": _array_records,
    "mix of memory and array record": _mix(_memory, _array_records),
    "mix of memory and streaming disk": _mix(_memory, _disk),
}


def _recorded(run: Callable[[Any], Any], source: Any) -> list[tuple[str, str]]:
    with HolderRecorder() as recorder:
        run(source)
    return recorder.taken


@pytest.mark.parametrize("path", list(_TRACED_PATHS))
@pytest.mark.parametrize("kind", list(_TRACED_SOURCES))
def test_no_traced_path_reads_a_kind_equal_holder(
    kind: str, path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _TRACED_SOURCES[kind](tmp_path, monkeypatch)
    assert _recorded(_TRACED_PATHS[path], source) == []


@pytest.mark.parametrize("kind", [*_TRACED_SOURCES, *_HOST_SOURCES])
def test_iteration_reads_no_kind_equal_holder_in_a_program(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = {**_TRACED_SOURCES, **_HOST_SOURCES}[kind](tmp_path, monkeypatch)
    assert _recorded(_iterate, source) == []


def _record_stream(kind: RecordIdentity) -> Make:
    def make(directory: Path, monkeypatch: pytest.MonkeyPatch) -> RecordStream:
        columns = {"x": np.arange(_RECORDS * 3, dtype=np.float32).reshape(_RECORDS, 3)}
        return RecordStream(columns, kind=kind, texts=[f"r{i}" for i in range(_RECORDS)])

    return make


def _hf_stream(directory: Path, monkeypatch: pytest.MonkeyPatch) -> HFStreamingSource:
    table = datasets.Dataset.from_dict(
        {"x": [[float(i)] * 3 for i in range(_RECORDS)], "id": [f"r{i}" for i in range(_RECORDS)]}
    )
    monkeypatch.setattr(
        datasets, "load_dataset", lambda *args, **kwargs: table.to_iterable_dataset(num_shards=2)
    )
    return HFStreamingSource(HFStreamingConfig(name="local", split="train"))


# Streams, read forward on the host; one without a length has no order to shuffle.
_STREAMS: dict[str, tuple[Make, bool]] = {
    "record stream named by ids": (_record_stream(RecordIdentity.STREAM_IDS), True),
    "record stream named by arrival": (_record_stream(RecordIdentity.ARRIVAL), True),
    "record stream in its own order": (_record_stream(RecordIdentity.STREAM_IDS), False),
    "hf stream": (_hf_stream, False),
}


@pytest.mark.parametrize("kind", list(_STREAMS))
def test_iterating_a_stream_reads_no_kind_equal_holder_in_a_program(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make, shuffle = _STREAMS[kind]
    source = make(tmp_path, monkeypatch)
    assert _recorded(lambda s: _iterate(s, shuffle=shuffle), source) == []


def _tfds(fixture: TFDSFixture) -> TFDSEagerSource:
    config = TFDSEagerConfig(name=FIXTURE, split="train", data_dir=str(fixture.array_record))
    return TFDSEagerSource(config)


@pytest.mark.tfds
@pytest.mark.parametrize("path", [*_TRACED_PATHS, "iteration"])
def test_no_path_over_a_tfds_source_reads_a_kind_equal_holder(
    path: str, tfds_fixture: TFDSFixture
) -> None:
    run = _iterate if path == "iteration" else _TRACED_PATHS[path]
    assert _recorded(run, _tfds(tfds_fixture)) == []
    mix = MixDataSourcesNode(
        MixDataSourcesConfig(weights=(1.0, 1.0)), [_tfds(tfds_fixture), _tfds(tfds_fixture)]
    )
    assert _recorded(run, mix) == []


@pytest.mark.tfds
def test_iterating_a_tfds_stream_reads_no_kind_equal_holder_in_a_program(
    tfds_fixture: TFDSFixture,
) -> None:
    config = TFDSStreamingConfig(name=FIXTURE, split="train", data_dir=str(tfds_fixture.tfrecord))
    assert _recorded(_iterate, TFDSStreamingSource(config)) == []


class _TakesProvenance(MemorySource):
    """A source whose traced read takes its provenance holder off itself."""

    def get_records(self, indices: jax.Array) -> Any:
        """Take the holder, then gather as a memory source does."""
        held = self._provenance
        del held
        return super().get_records(indices)


class _ReadsProvenance(MemorySource):
    """A source whose traced read reads what its provenance holder holds."""

    def get_records(self, indices: jax.Array) -> Any:
        """Read the holder, then gather as a memory source does."""
        rows = self._provenance.value
        del rows
        return super().get_records(indices)


class _CapturesProvenance(MemorySource):
    """A source whose traced read captures its provenance holder in a host callback."""

    def get_records(self, indices: jax.Array) -> Any:
        """Capture the holder in a debug callback, then gather as a memory source does."""
        held = self._provenance

        def keep() -> None:
            _ = held

        jax.debug.callback(keep)
        return super().get_records(indices)


class _CapturesItself(MemorySource):
    """A source whose traced read captures itself, and with it its holder, in a host callback."""

    def get_records(self, indices: jax.Array) -> Any:
        """Capture the source in a debug callback, then gather as a memory source does."""

        def keep() -> None:
            _ = self

        jax.debug.callback(keep)
        return super().get_records(indices)


@pytest.mark.parametrize(
    ("source_type", "event"),
    [
        (_TakesProvenance, ("taken", "HostProvenance")),
        (_ReadsProvenance, ("read", "HostProvenance")),
        (_CapturesProvenance, ("captured", "HostProvenance")),
        (_CapturesItself, ("captured", "HostProvenance")),
    ],
)
@pytest.mark.parametrize("path", list(_TRACED_PATHS))
def test_a_program_reading_a_holder_is_recorded(
    source_type: type[MemorySource], event: tuple[str, str], path: str
) -> None:
    source = source_type(MemorySourceConfig(), _rows())
    assert event in _recorded(_TRACED_PATHS[path], source)


def test_a_holder_read_on_the_host_is_not_recorded() -> None:
    source = MemorySource(MemorySourceConfig(), _rows())
    with HolderRecorder() as recorder:
        provenance = source.provenance(source.record_indices_at(0, 2))
    assert [dict(record) for record in provenance] == [{"id": "r0"}, {"id": "r1"}]
    assert recorder.taken == []


def test_holders_of_a_kind_share_a_program_and_memory_maps_do_not(tmp_path: Path) -> None:
    """Kind equality shares the graphdef; the memory map a program reads keeps its own."""
    one, other = _rows(), _rows()
    other[0]["id"] = "another"
    memory = [MemorySource(MemorySourceConfig(), rows) for rows in (one, other)]
    assert nnx.graphdef(memory[0]) == nnx.graphdef(memory[1])
    disk = [_disk(tmp_path, pytest.MonkeyPatch()) for _ in range(2)]
    assert nnx.graphdef(disk[0]) != nnx.graphdef(disk[1])
