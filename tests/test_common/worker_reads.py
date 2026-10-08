"""Sources whose host read holds the GIL, read in Grain worker processes by the tests.

Every callable here is defined at module level, so it pickles by reference into a worker
process, which imports this module to call it.

* :func:`write_png_records` writes ArrayRecord files of PNG images (encoded with Pillow from
  NumPy), each record also holding its index as its label and a text name; :class:`PngDecoder`
  decodes a batch of them, optionally logging every record it decodes, with the decoding
  process's id, to a file (a read counter that crosses processes).
* :func:`write_tensor_records` writes ArrayRecord files of TFDS-serialized examples holding a
  16,384-value float ``Tensor`` (a non-image kind); :class:`TensorDecoder` decodes them with
  TFDS's NumPy decoder, as ``from_tfds(..., in_memory=False)`` does.
* :class:`ProcessReadMemorySource` is an in-memory source declaring its read GIL-bound, so a
  budget sends it to worker processes: the positive control of the per-worker O(batch) check.
"""

from __future__ import annotations

import functools
import io
import os
import struct
import threading
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np
import psutil
from array_record.python.array_record_module import ArrayRecordWriter
from PIL import Image

from datarax.core.data_source import DataSourceModule, HostRead
from datarax.core.host_resources import available_cpus, HostResources
from datarax.pipeline import host_workers
from datarax.pipeline.host_workers import JaxSettings
from datarax.sources.array_record_source import ArrayRecordSourceConfig, ArrayRecordSourceModule
from datarax.sources.memory_source import MemorySource


IMAGE_SHAPE = (8, 8, 3)
TENSOR_WIDTH = 16_384
WORKER_BYTES = 300 << 20
"""A worker's own memory passed to the plan, so a test does not start the footprint probe."""
BUDGET = 32 << 30

WORKER_COUNTS = tuple(k for k in (1, 2, 3, 8) if k <= available_cpus())
"""Worker counts the process tests read with: 1 to 3 (CI runners have 4 CPUs), and 8 where the
machine has the CPUs for it."""

_HEADER = struct.Struct("<iH")


def resources(
    workers: int, *, budget: int = BUDGET, worker_bytes: int = WORKER_BYTES
) -> HostResources:
    """A budget whose plan reads with ``workers`` processes (threads, for a GIL-free read).

    Args:
        workers: The cap, which an ample budget reaches.
        budget: The RAM budget.
        worker_bytes: A worker's own memory, passed so no probe worker starts.

    Returns:
        The resources.
    """
    return HostResources(ram_budget_bytes=budget, max_workers=workers, worker_bytes=worker_bytes)


def _image(index: int, shape: tuple[int, ...] = IMAGE_SHAPE) -> np.ndarray:
    return np.random.default_rng(index).integers(0, 256, shape, dtype=np.uint8)


def _png(image: np.ndarray) -> bytes:
    out = io.BytesIO()
    Image.fromarray(image).save(out, format="PNG")
    return out.getvalue()


def _write(directory: Path, prefix: str, records: Iterable[bytes], shards: int) -> list[str]:
    """``records`` split into ``shards`` ArrayRecord files, in order."""
    records = list(records)
    paths, start = [], 0
    for shard, count in enumerate(np.array_split(np.arange(len(records)), shards)):
        path = directory / f"{prefix}-{shard:05d}.array_record"
        writer = ArrayRecordWriter(str(path), "group_size:1")
        for index in range(start, start + len(count)):
            writer.write(records[index])
        writer.close()
        paths.append(str(path))
        start += len(count)
    return paths


def write_png_records(
    directory: Path, count: int, *, shards: int = 2, shape: tuple[int, ...] = IMAGE_SHAPE
) -> list[str]:
    """ArrayRecord files of ``count`` PNG records: record ``i`` is image ``i``, label ``i``.

    Args:
        directory: Where the files go.
        count: The records.
        shards: The files they are split into, in order.
        shape: Each image's shape (random pixels, so PNG barely compresses them).

    Returns:
        The files' paths.
    """
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(count):
        name = f"record-{index}".encode()
        records.append(_HEADER.pack(index, len(name)) + name + _png(_image(index, shape)))
    return _write(directory, "png", records, shards)


def expected_image(index: int) -> np.ndarray:
    """Record ``index``'s image, as :class:`PngDecoder` decodes it."""
    return _image(index)


@dataclass(frozen=True, slots=True)
class PngDecoder:
    """Decodes a batch of :func:`write_png_records` records with Pillow, one call per batch.

    Attributes:
        names: Whether each record keeps its name (a string, so its provenance).
        float64: Whether each record also holds ``scale``, a float64 value.
        log: A file every decoded record's label is appended to, with the process id, or
            ``None``.
        unpicklable: A label whose record's provenance holds a value no pickler can copy, or
            ``None``.
    """

    names: bool = True
    float64: bool = False
    log: str | None = None
    unpicklable: int | None = None

    def __call__(self, records: Sequence[bytes]) -> list[dict[str, Any]]:
        """One mapping per record: ``image``, ``label`` and what the options add."""
        decoded = [self._one(record) for record in records]
        if self.log is not None:
            with Path(self.log).open("a") as log:
                log.writelines(f"{os.getpid()} {int(r['label'])}\n" for r in decoded)
        return decoded

    def _one(self, record: bytes) -> dict[str, Any]:
        label, size = _HEADER.unpack_from(record)
        start = _HEADER.size
        name = record[start : start + size].decode()
        image = np.asarray(Image.open(io.BytesIO(record[start + size :])))
        values: dict[str, Any] = {"image": image, "label": np.int32(label)}
        if self.names:
            values["name"] = name
        if self.float64:
            values["scale"] = np.float64(label) / 3.0
        if self.unpicklable == label:
            values["handle"] = _Unpicklable()
        return values


def worker_facts() -> dict[str, Any]:
    """What the decoding process runs with: its platform, backend, GPU visibility, precision mode,
    its global JAX settings, whether TensorFlow is imported, how it was started and whether its
    parent watcher runs."""
    import sys  # noqa: PLC0415

    return {
        "pid": os.getpid(),
        "jax_platforms": jax.config.jax_platforms,
        "backend": jax.default_backend(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "x64": bool(jax.config.read("jax_enable_x64")),
        "global": _on_a_new_thread(_jax_settings),
        "tensorflow": "tensorflow" in sys.modules,
        "spawned": "spawn_main" in " ".join(sys.orig_argv),
        "watcher": any(
            thread.name == PARENT_WATCH and thread.is_alive() for thread in threading.enumerate()
        ),
    }


PARENT_WATCH = "datarax-parent-watch"
"""The name of the thread ending a worker whose training process is gone."""


def _jax_settings() -> dict[str, Any]:
    return {
        "x64": bool(jax.config.jax_enable_x64),
        "threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
    }


def _on_a_new_thread(read: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """``read()`` on a new thread: the process's global JAX settings, no context manager's."""
    found: dict[str, Any] = {}
    thread = threading.Thread(target=lambda: found.update(read()))
    thread.start()
    thread.join()
    return found


@dataclass(frozen=True, slots=True)
class WorkerFactsDecoder:
    """A :class:`PngDecoder` adding :func:`worker_facts` as each record's ``worker`` provenance."""

    def __call__(self, records: Sequence[bytes]) -> list[dict[str, Any]]:
        """The records as :class:`PngDecoder` decodes them, each with the process's facts."""
        import json  # noqa: PLC0415

        facts = json.dumps(worker_facts())
        return [{**values, "worker": facts} for values in PngDecoder()(records)]


class _Unpicklable:
    """A value whose pickling fails, as an open file or a lock would."""

    def __reduce__(self) -> Any:
        raise TypeError("this value cannot be pickled")


def png_source(
    paths: Sequence[str], decoder: Callable[[Sequence[bytes]], Any] | None = None
) -> ArrayRecordSourceModule:
    """An ``ArrayRecordSourceModule`` over PNG records, decoded per batch (a GIL-bound read)."""
    return ArrayRecordSourceModule(
        ArrayRecordSourceConfig(), list(paths), decode=decoder or PngDecoder()
    )


@functools.cache
def _tensor_features(width: int) -> Any:
    import tensorflow_datasets as tfds  # noqa: PLC0415 - TFDS's decoder, imported where it decodes

    return tfds.features.FeaturesDict(
        {
            "x": tfds.features.Tensor(shape=(width,), dtype=np.float32),
            "label": tfds.features.Scalar(dtype=np.int64),
            "name": tfds.features.Text(),
        }
    )


def tensor_values(index: int, width: int = TENSOR_WIDTH) -> np.ndarray:
    """Record ``index``'s ``x``."""
    return np.random.default_rng(index).standard_normal(width).astype(np.float32)


def write_tensor_records(
    directory: Path, count: int, *, width: int = TENSOR_WIDTH, shards: int = 2
) -> list[str]:
    """ArrayRecord files of ``count`` TFDS-serialized examples with a ``width``-value float tensor.

    Args:
        directory: Where the files go.
        count: The records.
        width: Values in each record's ``x``.
        shards: The files they are split into, in order.

    Returns:
        The files' paths.
    """
    directory.mkdir(parents=True, exist_ok=True)
    features = _tensor_features(width)
    records = [
        features.serialize_example(
            {"x": tensor_values(index, width), "label": index, "name": f"example-{index}"}
        )
        for index in range(count)
    ]
    return _write(directory, "tensor", records, shards)


@dataclass(frozen=True, slots=True)
class TensorDecoder:
    """Decodes a batch of :func:`write_tensor_records` examples with TFDS's NumPy decoder.

    Attributes:
        width: Values in each record's ``x``.
    """

    width: int = TENSOR_WIDTH

    def __call__(self, records: Sequence[bytes]) -> list[dict[str, Any]]:
        """One mapping per record: ``x``, ``label`` and the ``name`` (provenance)."""
        features = _tensor_features(self.width)
        return [features.deserialize_example_np(record) for record in records]


class ProcessReadMemorySource(MemorySource):
    """An in-memory source declaring its read GIL-bound, so a budget reads it in processes.

    Its read carries the whole dataset into every worker: the case the O(batch) check refuses.
    """

    @property
    def host_read(self) -> HostRead:
        """Declared GIL-bound."""
        return HostRead.GIL_BOUND


type Served = tuple[bytes, bytes, bytes, tuple[tuple[str, bytes], ...]]


def served(batch: Any) -> Served:
    """A host-read batch as bytes: its indices, epochs, draws and every data leaf."""
    host = jax.device_get(batch)
    leaves = jax.tree_util.tree_flatten_with_path(host.data)[0]
    return (
        np.asarray(host.indices).tobytes(),
        np.asarray(host.epochs).tobytes(),
        np.asarray(host.draws).tobytes(),
        tuple((jax.tree_util.keystr(path), np.asarray(leaf).tobytes()) for path, leaf in leaves),
    )


def stream_of(batches: Iterator[Any]) -> list[Served]:
    """Every batch an iterator serves, as :func:`served` gives it."""
    return [served(batch) for batch in batches]


def logged(log: Path) -> list[tuple[int, int]]:
    """``(process id, label)`` of every record a :class:`PngDecoder` logged."""
    if not log.exists():
        return []
    return [
        (int(pid), int(label))
        for pid, label in (line.split() for line in log.read_text().splitlines() if line)
    ]


def worker_children() -> list[Any]:
    """The live ``multiprocessing`` children Grain started as prefetch workers."""
    import multiprocessing  # noqa: PLC0415

    return [p for p in multiprocessing.active_children() if "grain-process-prefetch" in p.name]


def provenance_records(source: DataSourceModule, labels: Sequence[int]) -> list[str]:
    """The names a PNG source holds for ``labels``, as its provenance serves them."""
    from datarax.core.index_words import to_words  # noqa: PLC0415

    found = source.provenance(to_words(np.asarray(labels, dtype=np.uint64)))
    return [str(record.get("name")) for record in found]


def proc_kib(pid: int | str, field: str) -> int:
    """A ``/proc/<pid>/status`` memory field (``VmRSS``, ``VmHWM``) in KiB (Linux)."""
    with Path(f"/proc/{pid}/status").open() as status:
        return next(int(line.split()[1]) for line in status if line.startswith(f"{field}:"))


def reset_peak(pid: int | str) -> None:
    """Restart the kernel's high-water mark of ``pid`` at its current resident size (Linux)."""
    Path(f"/proc/{pid}/clear_refs").write_text("5")


def hold_units_in_a_worker(
    payload: bytes, keep: int, units: int, settings: JaxSettings, report: Any
) -> None:
    """The by-hand control of a worker: read ``units`` units holding the last ``keep``.

    Runs in a spawned process with datarax's worker set-up, as a Grain worker does; the read is
    unpickled and its first unit read before the window, which starts at the resident size then.
    Sends the window's peak growth in KiB through ``report``.

    Args:
        payload: The read, pickled with ``cloudpickle``.
        keep: Units held at once.
        units: Units read in the window.
        settings: The run's JAX settings.
        report: The sending end of a pipe.
    """
    import collections  # noqa: PLC0415
    import gc  # noqa: PLC0415

    import cloudpickle  # noqa: PLC0415

    host_workers.WorkerSetUp(settings=settings)(0, 1)
    read = cloudpickle.loads(payload)  # the test's own read, pickled by its parent process
    read(0)
    gc.collect()
    reset_peak("self")
    start = proc_kib("self", "VmRSS")
    held: collections.deque[Any] = collections.deque(maxlen=keep)
    for ordinal in range(1, units + 1):
        held.append(read(ordinal))
    report.send(proc_kib("self", "VmHWM") - start)
    report.close()


@dataclass(frozen=True, slots=True)
class UnwatchedSetUp:
    """datarax's worker set-up without its parent watcher: the parent watcher's positive control.

    A training script puts it in place of ``host_workers.WorkerSetUp``; in the worker it turns the
    watcher off, then sets the worker up as datarax does.

    Attributes:
        settings: The run's JAX settings.
    """

    settings: JaxSettings

    def __call__(self, index: int, count: int) -> None:
        """Set the worker up as datarax does, starting no parent watcher."""
        host_workers._watch_parent = _no_watch  # noqa: SLF001 - this worker's own module
        _DATARAX_SET_UP(settings=self.settings)(index, count)


def _no_watch() -> None:
    """Start nothing."""


def spawned_children(pid: int) -> list[int]:
    """The process ids of ``pid``'s children other than multiprocessing's resource tracker.

    A child just forked and not yet executing its spawned interpreter still shows its parent's
    command line, so children are kept by what they are not.
    """
    found = []
    for child in psutil.Process(pid).children():
        try:
            command = " ".join(child.cmdline())
        except psutil.NoSuchProcess:
            continue
        if "resource_tracker" not in command:
            found.append(child.pid)
    return sorted(found)


def is_gone(pid: int) -> bool:
    """Whether process ``pid`` has exited (reaped, or a zombie its reaper has yet to collect)."""
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def shm_names_of(pids: Iterable[int]) -> set[str]:
    """The ``/dev/shm`` names the processes ``pids`` hold open or mapped (Linux)."""
    names: set[str] = set()
    for pid in pids:
        try:
            maps = Path(f"/proc/{pid}/maps").read_text().splitlines()
            descriptors = list(Path(f"/proc/{pid}/fd").iterdir())
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        names.update(
            Path(line.split()[-1]).name for line in maps if line.split()[-1].startswith(_SHM)
        )
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except (FileNotFoundError, ProcessLookupError):
                continue
            if target.startswith(_SHM) and not target.endswith(_UNLINKED):
                names.add(Path(target).name)
    return names


_SHM = "/dev/shm/"  # nosec B108 - read only: the names a process holds
_UNLINKED = " (deleted)"
"""What ``/proc`` appends to a file already unlinked, whose name is gone."""
_DATARAX_SET_UP = host_workers.WorkerSetUp
