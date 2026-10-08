"""How a run's reads are sized from a caller's RAM budget, as Grain sizes its workers.

A training process says what it may spend on its host stage
(:class:`~datarax.core.host_resources.HostResources`): a RAM budget and a cap on read threads or
worker processes. Grain's own sizing (``grain.experimental.pick_performance_config``) divides the
budget by an element's size, so it returns every CPU for any budget its elements fit, and a
worker of a datarax read costs its own interpreter and JAX runtime besides (about 300 MB) before
its first element. The plan here is Grain's arithmetic with every term a worker costs. In host
bytes:

- ``E``: one unit (a batch, or a ``(K, B, ...)`` chunk) as the read returns it;
- ``d``: placed units waiting on the device ahead of the consumer's, so ``d + 1`` units at the
  device boundary;
- ``b``: units the training process reads ahead on threads; ``r`` and ``o``: a worker's read
  buffer and output buffer, its queues holding about ``2o + 2`` units in shared memory (``o`` in
  the worker's queue and one waiting to be put, ``o`` in the training process's buffer of the
  worker and one in its producer's hand);
- ``F``: a worker's own memory; ``P``: what a worker is sent, the read; ``S``: the source's
  working set in a worker; ``O``: the order's working set; ``C``: the decoded columns a build
  holds in the training process (0 for a pipeline's run); ``M_main``: the training process's
  resident memory when the plan is made, with JAX's runtime and compiler started (the source's
  data, the model and everything else the process holds).

Workers ``k`` for a GIL-bound read is the largest count with::

    M_main + O + C + (d + 1) E + k (F + P + r E + S) + k (2o + 2) E  <=  ram_budget_bytes
    k (2o + 2) E + (d + 1) E                                          <=  free bytes of /dev/shm
    k  <=  max_workers  <=  the CPUs available

A GIL-free read runs on threads (fewer than the CPUs, Grain's advice for a GIL-free bottleneck,
and at most the cap) reading ``b`` units ahead, the largest ``M_main + O + C + (b + 1 + d + 1) E``
admits, at least two and at most Grain's 1,000. A budget below ``M_main + O + C + (d + 1) E`` is
refused on either path, and one admitting no worker, naming the terms; nothing is rounded up to
one worker. The terms are measured from a source's values (:func:`measure_terms`), so a pipeline's
host stage and a build plan alike.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from collections.abc import Callable
from typing import Any

import jax
import numpy as np

from datarax.core.data_source import DataSourceModule, HostRead
from datarax.core.host_resources import available_cpus, HostResources, WorkerFootprint
from datarax.core.spec import declared_spec
from datarax.pipeline import host_workers
from datarax.pipeline.host_workers import JaxSettings


READ_THREADS = 1
"""Read threads without a budget: one host thread gathers millions of in-memory records a second;
a budget sets more (:func:`budget_plan`)."""
READ_BUFFER = 2
"""Units read ahead without a budget, and the fewest a budget sets."""
MAX_READ_BUFFER = 1000
"""The most units read ahead: Grain's ``pick_performance_config`` buffer bound."""
WORKER_BUFFER = 2
"""A worker's output buffer ``o``: units it puts in its queue ahead of the training process."""
IDENTITY_BYTES = 2 * 4 + 4 + 4
"""Host bytes a record's identity adds to its batch: its index words, epoch and draw."""


class HostReadPath(enum.Enum):
    """Where a run's units are read."""

    THREADS = "threads"
    """Grain read threads in the training process, each reading whole units."""
    PROCESSES = "processes"
    """Grain worker processes (``mp_prefetch``), each reading every ``k``-th unit."""
    ONE_THREAD = "one thread"
    """A stream's one producer thread in the training process."""


class ReadAccess(enum.Enum):
    """How a source's run can be read, which bounds what can read it in parallel."""

    BY_UNIT = "by unit"
    """Any unit read alone (an indexed source): by threads, or by processes."""
    SLICEABLE_STREAM = "sliceable stream"
    """One dataset of the run that Grain can slice across processes (a TFDS stream)."""
    STREAM = "stream"
    """A stream read pass by pass (HuggingFace), on its one thread."""


def host_read_path(
    resources: HostResources | None, read: HostRead, access: ReadAccess
) -> HostReadPath:
    """Where a run is read: processes for a GIL-bound read with a budget, else threads.

    Args:
        resources: The caller's budget and cap, or ``None``.
        read: Whether the source's host read holds the GIL.
        access: How the source's run can be read.

    Returns:
        The path.
    """
    if access is ReadAccess.STREAM:
        return HostReadPath.ONE_THREAD
    if resources is not None and read is HostRead.GIL_BOUND:
        return HostReadPath.PROCESSES
    if access is ReadAccess.BY_UNIT:
        return HostReadPath.THREADS
    return HostReadPath.ONE_THREAD


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class HostTerms:
    """What a run's reads cost, in host bytes, as measured for the plan.

    Attributes:
        unit_bytes: ``E``, one unit as the read returns it.
        device_buffer: ``d``, placed units staged on the device ahead of the consumer's.
        main_bytes: ``M_main``, the training process's resident memory at plan time.
        order_bytes: ``O``, the order's working set; 0 for the global order.
        sink_bytes: ``C``, the decoded columns a build holds in the training process; 0 for a
            pipeline's run.
        worker_bytes: ``F``, a worker's own memory.
        read_bytes: ``P``, what a worker is sent: the read, as it pickles.
        working_bytes: ``S``, the source's working set in a worker beyond its buffers.
        worker_read_buffer: ``r``, units a worker reads ahead of its queue.
        worker_buffer: ``o``, a worker's output buffer.
        shm_free_bytes: Free bytes of ``/dev/shm``, or ``None`` where there is none to measure.
        cpus: The CPUs this process may run on.
    """

    unit_bytes: int
    device_buffer: int
    main_bytes: int = 0
    order_bytes: int = 0
    sink_bytes: int = 0
    worker_bytes: int = 0
    read_bytes: int = 0
    working_bytes: int = 0
    worker_read_buffer: int = 0
    worker_buffer: int = WORKER_BUFFER
    shm_free_bytes: int | None = None
    cpus: int = 1


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class HostPlan:
    """How a run's units are read: its path, threads, workers and buffers, and what it costs.

    Attributes:
        path: Where the units are read.
        threads: Read threads in the training process; 0 when workers read.
        workers: Worker processes; 0 on threads.
        read_buffer: ``b``, units the training process reads ahead on its threads; 0 when
            workers read (their buffers hold the units read ahead).
        worker_buffer: ``o``, each worker's output buffer; 0 on threads.
        worker_read_buffer: ``r``, units each worker reads ahead of its queue; 0 on threads.
        device_buffer: ``d``, placed units staged on the device ahead of the consumer's.
        terms: What the budget was charged with, or ``None`` without a budget.
        accounted_bytes: The budget's left side: the bytes the plan holds, or ``None``.
        shm_bytes: The ``/dev/shm`` bound's left side under processes, else ``None``.
    """

    path: HostReadPath
    threads: int
    workers: int
    read_buffer: int
    worker_buffer: int
    worker_read_buffer: int
    device_buffer: int
    terms: HostTerms | None = None
    accounted_bytes: int | None = None
    shm_bytes: int | None = None


def default_plan(path: HostReadPath, *, device_buffer: int) -> HostPlan:
    """The plan without a budget: one read thread, two units ahead.

    Args:
        path: Where the run is read; never processes, which need a budget.
        device_buffer: ``d`` for the default device's platform.

    Returns:
        The plan.

    Raises:
        ValueError: If ``path`` is processes.
    """
    if path is HostReadPath.PROCESSES:
        raise ValueError(
            "worker processes are sized from a RAM budget: pass Pipeline(host_resources="
            "HostResources(ram_budget_bytes=..., max_workers=...))"
        )
    return HostPlan(
        path=path,
        threads=READ_THREADS,
        workers=0,
        read_buffer=READ_BUFFER,
        worker_buffer=0,
        worker_read_buffer=0,
        device_buffer=device_buffer,
    )


def budget_plan(resources: HostResources, path: HostReadPath, terms: HostTerms) -> HostPlan:
    """The plan a budget gives: worker processes, or threads and their read buffer.

    Args:
        resources: The caller's budget and cap.
        path: Where the run is read.
        terms: What the reads cost.

    Returns:
        The plan.

    Raises:
        ValueError: If the budget is below what the run holds before reading ahead,
            ``M_main + O + C + (d + 1) E``, naming the terms.
    """
    fixed = _fixed_bytes(terms)
    budget = resources.ram_budget_bytes
    if budget < fixed:
        raise ValueError(
            f"a RAM budget of {budget} bytes is below the {fixed} bytes the run holds before "
            f"reading ahead: M_main={terms.main_bytes} (the training process now) + "
            f"O={terms.order_bytes} (the order) + C={terms.sink_bytes} (the decoded columns a "
            f"build holds) + d={terms.device_buffer} + 1 units of E={terms.unit_bytes} bytes at "
            "the device: raise ram_budget_bytes"
        )
    if path is HostReadPath.PROCESSES:
        return _worker_plan(resources, terms)
    return _thread_plan(resources, path, terms)


def _fixed_bytes(terms: HostTerms) -> int:
    """``M_main + O + C + (d + 1) E``: what a run holds whatever reads it."""
    at_device = (terms.device_buffer + 1) * terms.unit_bytes
    return terms.main_bytes + terms.order_bytes + terms.sink_bytes + at_device


def _thread_plan(resources: HostResources, path: HostReadPath, terms: HostTerms) -> HostPlan:
    """Threads below the CPUs and within the cap; the read buffer the budget holds."""
    e, d = terms.unit_bytes, terms.device_buffer
    held = (resources.ram_budget_bytes - _fixed_bytes(terms)) // max(e, 1)
    read_buffer = min(MAX_READ_BUFFER, max(READ_BUFFER, held - 1))
    threads = 1
    if path is HostReadPath.THREADS:
        threads = max(1, min(resources.max_workers, terms.cpus - 1))
    return HostPlan(
        path=path,
        threads=threads,
        workers=0,
        read_buffer=read_buffer,
        worker_buffer=0,
        worker_read_buffer=0,
        device_buffer=d,
        terms=terms,
        accounted_bytes=_fixed_bytes(terms) + (read_buffer + 1) * e,
    )


def _worker_plan(resources: HostResources, terms: HostTerms) -> HostPlan:
    """The largest worker count meeting the RAM bound, the ``/dev/shm`` bound and the cap.

    Args:
        resources: The caller's budget and cap.
        terms: What the reads cost.

    Returns:
        The plan.

    Raises:
        ValueError: If no worker fits the budget or ``/dev/shm``, naming the terms.
    """
    e, d = terms.unit_bytes, terms.device_buffer
    o, r = terms.worker_buffer, terms.worker_read_buffer
    at_device = (d + 1) * e
    in_queues = (2 * o + 2) * e
    per_worker = terms.worker_bytes + terms.read_bytes + r * e + terms.working_bytes + in_queues
    fixed = _fixed_bytes(terms)
    budget = resources.ram_budget_bytes
    k = (budget - fixed) // per_worker
    if k < 1:
        raise ValueError(
            f"a RAM budget of {budget} bytes admits no worker process: each costs "
            f"F={terms.worker_bytes} (its own memory) + P={terms.read_bytes} (its copy of the "
            f"read) + S={terms.working_bytes} + {r + 2 * o + 2} units of E={e} bytes in its "
            f"buffers, beside M_main={terms.main_bytes}, O={terms.order_bytes}, "
            f"C={terms.sink_bytes} and d={d} + 1 units at the device: raise ram_budget_bytes, "
            "or omit host_resources to read on one thread"
        )
    if terms.shm_free_bytes is not None:
        room = (terms.shm_free_bytes - at_device) // in_queues
        if room < 1:
            raise ValueError(
                f"/dev/shm has {terms.shm_free_bytes} bytes free, below the "  # nosec B108
                f"{in_queues + at_device} bytes one worker's units take there ({2 * o + 2} units "
                f"of E={e} bytes in its queues and d={d} + 1 at the device): enlarge /dev/shm "
                "(Docker: --shm-size) or read smaller units"
            )
        k = min(k, room)
    k = min(k, resources.max_workers)
    return HostPlan(
        path=HostReadPath.PROCESSES,
        threads=0,
        workers=k,
        read_buffer=0,
        worker_buffer=o,
        worker_read_buffer=r,
        device_buffer=d,
        terms=terms,
        accounted_bytes=fixed + k * per_worker,
        shm_bytes=k * in_queues + at_device,
    )


def spec_bytes(spec: Any) -> int:
    """The bytes of one record described by ``spec``.

    Args:
        spec: A pytree of ``jax.ShapeDtypeStruct``.

    Returns:
        The sum of its leaves' bytes.
    """
    return sum(
        math.prod(leaf.shape) * np.dtype(leaf.dtype).itemsize for leaf in jax.tree.leaves(spec)
    )


def host_batch_bytes(source: DataSourceModule, batch_size: int) -> int:
    """The bytes of one batch on the host: its data at the stored dtypes, and its identity.

    A float64 column is 8 bytes a value on the host whatever the precision mode; the spec read
    with 64-bit types on states the stored dtypes.

    Args:
        source: The source the batch is read from.
        batch_size: Records per batch.

    Returns:
        The batch's host bytes.
    """
    with jax.enable_x64(True):
        spec = declared_spec(source)
    return batch_size * (spec_bytes(spec) + IDENTITY_BYTES)


def measure_terms(  # noqa: PLR0913 - what one run's measurement needs, each given once
    *,
    unit_bytes: int,
    depth: int,
    path: HostReadPath,
    resources: HostResources,
    read: Callable[[], Any],
    first_unit: Callable[[Any], Callable[[], Any]],
    settings: JaxSettings,
) -> HostTerms:
    """What a run's reads cost, measured: ``E`` and ``M_main``, and for processes ``P, S, F``.

    ``M_main`` is the training process's resident memory now, the source's data and the
    process's runtime included. Under processes the workers' read is built first, so what it
    shares with them (a stream's offset index, mapped by the source) is resident in the training
    process once and never added on top; ``P`` is that read as it pickles, ``S`` what it declares
    (:class:`~datarax.core.host_resources.WorkerFootprint`), and ``F``, unless the caller gives
    it, the peak of one probe worker reading the run's first unit under the run's settings. A
    thread plan builds and pickles nothing.

    Args:
        unit_bytes: ``E``, one unit's host bytes.
        depth: ``d``, placed units staged on the device ahead of the consumer's.
        path: Where the run is read.
        resources: The caller's budget and cap.
        read: Builds the read worker processes run, called only under processes.
        first_unit: Turns that read into a call reading the run's first unit, which the probe
            worker makes; called once, only when the probe runs.
        settings: The run's JAX settings, which the probe worker is set up with.

    Returns:
        The terms.
    """
    terms = HostTerms(unit_bytes=unit_bytes, device_buffer=depth, cpus=available_cpus())
    if path is not HostReadPath.PROCESSES:
        return dataclasses.replace(terms, main_bytes=host_workers.resident_bytes())
    worker_read = read()
    read_bytes = host_workers.pickled_bytes(worker_read)
    footprint = worker_read if isinstance(worker_read, WorkerFootprint) else None
    main_bytes = host_workers.resident_bytes()
    worker_bytes = resources.worker_bytes
    if worker_bytes is None:
        worker_bytes = host_workers.probe_worker_bytes(first_unit(worker_read), settings=settings)
    return dataclasses.replace(
        terms,
        main_bytes=main_bytes,
        read_bytes=read_bytes,
        working_bytes=0 if footprint is None else footprint.working_bytes(),
        worker_bytes=worker_bytes,
        worker_buffer=WORKER_BUFFER,
        shm_free_bytes=host_workers.shm_free_bytes(),
    )


__all__ = [
    "IDENTITY_BYTES",
    "MAX_READ_BUFFER",
    "READ_BUFFER",
    "READ_THREADS",
    "WORKER_BUFFER",
    "HostPlan",
    "HostReadPath",
    "HostTerms",
    "ReadAccess",
    "budget_plan",
    "default_plan",
    "host_batch_bytes",
    "host_read_path",
    "measure_terms",
    "spec_bytes",
]
