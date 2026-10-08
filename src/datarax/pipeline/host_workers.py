"""The host stage's Grain worker processes: their set-up, what crosses back, what they cost.

Grain starts a worker by spawning it (``mp_prefetch``), so a worker inherits the environment but
none of the JAX settings the training process made in code, and imports the training script's
main module before any set-up runs. A read thread in the training process does not inherit a
setting held in a context manager either. :class:`JaxSettings` holds what a run is read under
(precision, and the threefry setting that changes the order a key gives), set in each worker and
entered on each read thread. :class:`WorkerSetUp` runs first in every worker: it hides the GPU,
starts JAX on the CPU platform only (a worker reads and names records on the host; it never
places a batch) under the run's settings, refuses a worker whose JAX was already started by the
main module, naming the guard that prevents it, and ends the worker when the training process
dies (Grain leaves it running, google/grain#1423).

Grain's process queue loses an element it cannot pickle without raising in the training process
(google/grain#1415), so a worker sends provenance as plain dicts (:func:`crossing_provenance`),
the training process restores the read-only mappings (:func:`received_provenance`), and it
counts the units arriving (:func:`check_arrival`, :func:`check_all_arrived`).
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import multiprocessing
import os
import platform
import resource
import signal
import threading
from collections.abc import Callable, Iterator
from multiprocessing import connection
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from pickle import PicklingError  # nosec B403 - the error a pickling check catches
from types import MappingProxyType
from typing import Any

import cloudpickle
import grain
import jax
import jax.numpy as jnp
import psutil
from absl import flags

# The check that JAX started its backends: JAX exposes it nowhere public, and
# jax.distributed.initialize refuses a late start with the same call.
from jax._src import xla_bridge

from datarax.core.data_source import Provenance
from datarax.core.prng import host_device


GRAIN_WORKER_FLAG = "grain_enable_multiprocess_worker_profiling"
"""The absl flag Grain reads as it starts a worker, which absl refuses to read before parsing."""

_SHM = Path("/dev/shm")  # nosec B108 - only measured (statvfs), never written
_GUARD = 'if __name__ == "__main__":'
_PARENT_WATCH = "datarax-parent-watch"


def mark_grain_worker_flag() -> None:
    """Mark Grain's worker-profiling flag present when absl flags are unparsed.

    Grain 0.2.17 and later read ``--grain_enable_multiprocess_worker_profiling`` as they start a
    worker; outside ``absl.app.run`` absl refuses the read (``UnparsedFlagAccessError``) once
    jaxlib's profiler is loaded. Marking that one flag present makes the read return its
    default, as Grain's own workers mark their flags; a later ``FLAGS(argv)`` still sets it.
    Grain's fix (a3b351858a) is in no release yet; a tripwire test fails once it is.
    """
    if not flags.FLAGS.is_parsed():
        flags.FLAGS[GRAIN_WORKER_FLAG].present = 1


def _backends_started() -> bool:
    """Whether JAX started its backends in this process (the check ``jax.distributed`` uses)."""
    return xla_bridge.backends_are_initialized()


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class JaxSettings:
    """The JAX settings a run is read under, which a worker process or a read thread lacks.

    A spawned worker starts without any setting the training process made in code, and a new
    thread reads the global value of a setting the consumer holds in a context manager. Read on
    the consumer's thread when a run opens (:meth:`current`), they are set in each worker
    (:meth:`apply`) and entered around each read on a read thread (:meth:`entered`), so every
    reader names and decodes as the consumer would. Static and hashable: part of a run's
    identity, so a change of any opens a new run.

    ``jax_default_prng_impl`` is not among them: record naming wraps its order key as
    :data:`~datarax.core.prng.NAMING_PRNG_IMPL` whatever the caller's default, and nothing else a
    reader runs draws from a key.

    Attributes:
        x64: ``jax_enable_x64``, the precision mode.
        threefry_partitionable: ``jax_threefry_partitionable``, which changes the bits a
            threefry key gives, so the order the naming key gives.
    """

    x64: bool
    threefry_partitionable: bool

    @classmethod
    def current(cls) -> JaxSettings:
        """The settings in force on the calling thread, a context manager's override included.

        Read by attribute: ``jax.config.read`` refuses ``jax_threefry_partitionable``, which has
        a context manager.

        Returns:
            The settings.
        """
        return cls(
            x64=bool(jax.config.jax_enable_x64),
            threefry_partitionable=bool(jax.config.jax_threefry_partitionable),
        )

    def apply(self) -> None:
        """Make these the process's global settings (in a worker, before it reads)."""
        jax.config.update("jax_enable_x64", self.x64)
        jax.config.update("jax_threefry_partitionable", self.threefry_partitionable)

    @contextlib.contextmanager
    def entered(self) -> Iterator[None]:
        """Hold these settings on the calling thread (a read thread) while the block runs.

        Yields:
            Nothing; the settings hold until the block ends.
        """
        with jax.enable_x64(self.x64), jax.threefry_partitionable(self.threefry_partitionable):
            yield


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class WorkerSetUp:
    """Each worker's first step: JAX on the CPU only, under the run's settings, parent watched.

    Grain calls it as ``worker_init_fn(index, count)`` before it unpickles the read.

    Attributes:
        settings: The run's JAX settings, which a spawned worker does not inherit.
    """

    settings: JaxSettings

    def __call__(self, index: int, count: int) -> None:
        """Set the worker up.

        Args:
            index: The worker's index.
            count: The run's workers.

        Raises:
            RuntimeError: If the worker's JAX started before its set-up (the training script's
                main module starts it at import, which a spawned worker repeats), or reads on
                another platform than the CPU.
        """
        del index, count
        _watch_parent()
        if _backends_started():
            raise RuntimeError(
                "a host-stage worker process started JAX before datarax set it up: the training "
                "script's main module starts JAX when imported, and every spawned worker imports "
                f"it. Put the script's work under `{_GUARD}`, so a worker runs nothing of it"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        jax.config.update("jax_platforms", "cpu")
        self.settings.apply()
        backend = jax.default_backend()
        if backend != "cpu":
            raise RuntimeError(
                f"a host-stage worker reads on the CPU, but JAX's default backend is {backend!r}"
            )


def _watch_parent() -> None:
    """End this process when the process that spawned it dies; nothing outside a spawned child.

    Grain leaves a worker running when the training process is killed (google/grain#1423). The
    spawn pipe's read end (``multiprocessing.parent_process().sentinel``) reads end-of-file
    when the parent process dies, whichever of its threads spawned the child, so a daemon thread
    waits on it. ``PR_SET_PDEATHSIG`` fires instead when the spawning thread ends, which Grain
    ends at every ``close`` and ``set_state``.
    """
    parent = multiprocessing.parent_process()
    if parent is None:
        return
    threading.Thread(
        target=_end_with_parent, args=(parent.sentinel,), name=_PARENT_WATCH, daemon=True
    ).start()


def _end_with_parent(sentinel: int) -> None:
    """Wait until the parent process is gone, then end this one."""
    connection.wait([sentinel])
    os.kill(os.getpid(), signal.SIGTERM)


def in_workers(
    dataset: grain.IterDataset, *, workers: int, worker_buffer: int, settings: JaxSettings
) -> grain.IterDataset:
    """``dataset`` read by ``workers`` worker processes, each set up as :class:`WorkerSetUp` says.

    Grain gives worker ``i`` of ``k`` the units ``i, i + k, ...`` and interleaves them from
    worker 0, so the units arrive in the run's order whatever ``k``.

    Args:
        dataset: The run's dataset, which pickles and slices.
        workers: Worker processes.
        worker_buffer: Units each worker puts in its queue ahead of the training process.
        settings: The run's JAX settings.

    Returns:
        The dataset read in the workers.
    """
    mark_grain_worker_flag()
    return dataset.mp_prefetch(
        grain.MultiprocessingOptions(num_workers=workers, per_worker_buffer_size=worker_buffer),
        worker_init_fn=WorkerSetUp(settings=settings),
    )


def check_arrival(expected: int, unit: int) -> None:
    """Refuse a unit arriving out of turn: the one expected was lost on the way.

    Args:
        expected: The unit due next.
        unit: The unit that arrived.

    Raises:
        RuntimeError: If ``unit`` is not ``expected``, naming the lost unit.
    """
    if unit != expected:
        raise RuntimeError(
            f"unit {expected} of the run never arrived from its worker process (unit {unit} came "
            "next): Grain's process queue drops an element it cannot pickle without raising "
            "(google/grain#1415). The run ends at the units delivered"
        )


def check_all_arrived(received: int, total: int | None) -> None:
    """Refuse a run of known length ending before its last unit arrived.

    Args:
        received: The units that arrived.
        total: The run's units, or ``None`` for a run without an end.

    Raises:
        RuntimeError: If fewer than ``total`` units arrived, naming the first missing one.
    """
    if total is not None and received < total:
        raise RuntimeError(
            f"the run ended after {received} of its {total} units: unit {received} never arrived "
            "from its worker process (Grain's process queue drops an element it cannot pickle, "
            "google/grain#1415)"
        )


def crossing_provenance(provenance: Provenance | None, unit: int) -> Provenance | None:
    """A unit's provenance as plain dicts a worker can send, checked to pickle.

    Args:
        provenance: One read-only mapping per record, or ``None``.
        unit: The unit's place in the run, named when a value cannot cross.

    Returns:
        One ``dict`` per record, or ``None``.

    Raises:
        TypeError: If a record's provenance holds a value no pickler copies, naming the unit.
    """
    if provenance is None:
        return None
    plain = tuple(dict(record) for record in provenance)
    try:
        ForkingPickler.dumps(plain)  # as the worker's queue pickles it
    except (TypeError, AttributeError, PicklingError) as error:
        raise TypeError(
            f"unit {unit} of the run holds provenance a worker process cannot send to the "
            f"training process ({error}): a source read in processes keeps provenance values "
            "that pickle (strings, numbers, bytes)"
        ) from error
    return plain


def received_provenance(provenance: Provenance | None) -> Provenance | None:
    """A unit's provenance from a worker, as the read-only mappings a thread read serves.

    Args:
        provenance: One mapping per record, as sent, or ``None``.

    Returns:
        One read-only mapping per record, or ``None``.
    """
    if provenance is None:
        return None
    return tuple(MappingProxyType(dict(record)) for record in provenance)


class _ByteCounter(io.RawIOBase):
    """A file counting what is written to it and keeping none of it."""

    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def writable(self) -> bool:
        """Always writable."""
        return True

    def write(self, data: Any) -> int:
        """Count ``data``'s bytes.

        Args:
            data: A bytes-like object.

        Returns:
            Its size.
        """
        size = memoryview(data).nbytes
        self.count += size
        return size


def pickled_bytes(value: Any) -> int:
    """The bytes ``value`` pickles to, as Grain sends it to a worker, counted without keeping them.

    Args:
        value: What a worker is sent.

    Returns:
        The size of its ``cloudpickle`` stream.
    """
    with _ByteCounter() as counter:
        cloudpickle.dump(value, counter)
        return counter.count


def _zero() -> jax.Array:
    return jnp.zeros((), jnp.float32)


def resident_bytes() -> int:
    """The training process's resident memory, read once JAX's runtime and compiler run.

    A first jitted program on the CPU device starts them (once a process; no host transfer, so a
    caller's transfer guard is untouched), so the reading holds them as every later one will.

    Returns:
        The resident set size in bytes.
    """
    jax.block_until_ready(jax.jit(_zero, out_shardings=host_device())())
    return psutil.Process().memory_info().rss


def shm_free_bytes() -> int | None:
    """Free bytes of ``/dev/shm``, where Grain's shared-memory units live, or ``None`` without one.

    Returns:
        The free bytes, or ``None`` where there is no ``/dev/shm`` (macOS keeps POSIX shared
        memory elsewhere).
    """
    if not _SHM.is_dir():
        return None
    stats = os.statvfs(_SHM)
    return stats.f_bavail * stats.f_frsize


def peak_resident_bytes() -> int:
    """This process's peak resident size, in bytes.

    Returns:
        ``ru_maxrss``, which Linux reports in KiB and macOS in bytes.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if platform.system() == "Darwin" else peak * 1024


def _probe(payload: bytes, settings: JaxSettings, report: Any) -> None:
    """A probe worker: set up as a worker, read one unit, report the peak resident size."""
    try:
        WorkerSetUp(settings=settings)(0, 1)
        read = cloudpickle.loads(payload)  # the host stage's own read, sent by its parent
        read()
        report.send(peak_resident_bytes())
    except Exception as error:  # noqa: BLE001 - sent back to the training process, raised there
        report.send(f"{type(error).__name__}: {error}")
    finally:
        report.close()


def probe_worker_bytes(read: Callable[[], Any], *, settings: JaxSettings) -> int:
    """A worker's own memory ``F``: one spawned worker set up as the run's reads one unit.

    The probe's peak resident size holds the interpreter, the imports, JAX's CPU runtime, the
    unpickled read, its naming compile and one unit's read; resident size counts pages shared
    with other processes too, so it errs toward fewer workers. It costs one worker start.

    Args:
        read: Reads one unit of the run when called; it pickles.
        settings: The run's JAX settings.

    Returns:
        The probe's peak resident size, in bytes.

    Raises:
        RuntimeError: If the probe failed to read, naming its error.
    """
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    probe = context.Process(
        target=_probe,
        args=(cloudpickle.dumps(read), settings, send),
        name="datarax-footprint-probe",
        daemon=True,
    )
    probe.start()
    send.close()
    try:
        found = receive.recv()
    except EOFError as error:
        raise RuntimeError(
            f"the footprint probe worker exited (code {probe.exitcode}) before reporting"
        ) from error
    finally:
        receive.close()
        probe.join()
    if isinstance(found, str):
        raise RuntimeError(f"the footprint probe worker could not read a unit: {found}")
    return int(found)


__all__ = [
    "GRAIN_WORKER_FLAG",
    "JaxSettings",
    "WorkerSetUp",
    "check_all_arrived",
    "check_arrival",
    "crossing_provenance",
    "in_workers",
    "mark_grain_worker_flag",
    "peak_resident_bytes",
    "pickled_bytes",
    "probe_worker_bytes",
    "received_provenance",
    "resident_bytes",
    "shm_free_bytes",
]
