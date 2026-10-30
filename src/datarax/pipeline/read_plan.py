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
  memory when the plan is made (its resident size, on macOS its physical footprint, which also
  counts compressed pages), with JAX's runtime and compiler started (the source's data, the
  model and everything else the process holds).

Workers ``k`` for a GIL-bound read is the largest count with::

    M_main + O + C + (d + 1) E + k (F + P + r E + S) + k (2o + 2) E  <=  ram_budget_bytes
    k (2o + 2) E + (d + 1) E                                          <=  the /dev/shm bytes
    k  <=  max_workers  <=  the CPUs available

where the ``/dev/shm`` bytes are the smaller of the caller's share
(``HostResources.shm_budget_bytes``) and the free bytes measured, or the free bytes without a share.

A GIL-free read runs on ``min(4, max_workers, CPUs - 1)`` threads reading
``b = max(16, 2 x threads)`` units ahead, whatever the budget: read threads that outrun the
consumer fill a deep buffer, and taking batches written long ago doubles the consumer's cost per
batch (:data:`THREAD_READ_BUFFER`, :data:`MAX_READ_THREADS`). A budget below
``M_main + O + C + (d + 1) E`` is refused on either path, one below
``M_main + O + C + (b + 1 + d + 1) E`` on threads, and one admitting no worker, naming the terms;
nothing is rounded up to one worker or down to a smaller buffer. Without a budget a run reads on
one thread two units ahead (:data:`READ_THREADS`, :data:`READ_BUFFER`). The terms are measured from
a source's values (:func:`measure_terms`), so a pipeline's host stage and a build plan alike.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
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


logger = logging.getLogger(__name__)

READ_THREADS = 1
"""Read threads without a budget. One host thread gathers millions of in-memory records a second,
but it can leave a fast train step waiting (a quarter to two thirds of an H100's step on the 8.2 GB
set): pass ``host_resources`` to read on up to :data:`MAX_READ_THREADS` (:func:`budget_plan`)."""
READ_BUFFER = 2
"""Units read ahead without a budget."""
MAX_READ_THREADS = 4
"""The most read threads a budget gives a GIL-free read. Their calls serialise on the GIL, so more
threads add contention, not supply: on an H100 train step over an 8.2 GB in-memory set, four fed
the step best (0.569 of the pre-placed floor against 0.524 at two and 0.484 at eight, each with a
``2 x threads`` buffer)."""
THREAD_READ_BUFFER = 16
"""The fewest units a budget's thread path reads ahead; ``max(16, 2 x threads)`` keeps the buffer at
least the thread count, below which Grain limits the threads it runs. Not sized from the budget:
eight threads filling a 1,000-unit buffer read 0.397 of an H100's floor against 0.484 with 16
(Mann-Whitney p = 0.008, five runs each) and held 27.7 against 10.4 GiB."""
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


def warn_one_thread(source: DataSourceModule, call: str) -> None:
    """Log that ``source``'s GIL-bound read runs on one thread for want of a budget.

    Args:
        source: The source whose read decodes in Python.
        call: What takes the budget that would read it in worker processes (``"Pipeline"``).
    """
    logger.warning(
        "%s decodes in Python on each read, on one thread: pass %s(host_resources="
        "HostResources(ram_budget_bytes=..., max_workers=...)) to read it in worker processes",
        type(source).__name__,
        call,
    )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class HostTerms:
    """What a run's reads cost, in host bytes, as measured for the plan.

    Attributes:
        unit_bytes: ``E``, one unit as the read returns it.
        device_buffer: ``d``, placed units staged on the device ahead of the consumer's.
        main_bytes: ``M_main``, the training process's memory at plan time.
        order_bytes: ``O``, the order's working set; 0 for the global order.
        sink_bytes: ``C``, the decoded columns a build holds in the training process; 0 for a
            pipeline's run.
        worker_bytes: ``F``, a worker's own memory.
        read_bytes: ``P``, what a worker is sent: the read, as it pickles.
        working_bytes: ``S``, the source's working set in a worker beyond its buffers.
        worker_read_buffer: ``r``, units a worker reads ahead of its queue.
        worker_buffer: ``o``, a worker's output buffer.
        shm_free_bytes: The ``/dev/shm`` bytes the workers' units may take: the smaller of the
            caller's share (``HostResources.shm_budget_bytes``) and the free bytes measured, or
            ``None`` with no share and none to measure.
        shm_share: Whether ``shm_free_bytes`` is the caller's share, the smaller bound, rather
            than the free bytes measured.
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
    shm_share: bool = False
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
        device_buffer: ``d`` for the platform the run places on.

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


def budget_plan(  # noqa: DOC502 - refuse_below_fixed and the plans raise
    resources: HostResources, path: HostReadPath, terms: HostTerms
) -> HostPlan:
    """The plan a budget gives: worker processes, or threads and their read buffer.

    Args:
        resources: The caller's budget and cap.
        path: Where the run is read.
        terms: What the reads cost.

    Returns:
        The plan.

    Raises:
        ValueError: If the budget is below what the run holds before reading ahead,
            ``M_main + O + C + (d + 1) E``, below that and a thread plan's read buffer, or
            admits no worker, naming the terms.
    """
    refuse_below_fixed(resources, terms)
    if path is HostReadPath.PROCESSES:
        return _worker_plan(resources, terms)
    return _thread_plan(resources, path, terms)


def refuse_below_fixed(resources: HostResources, terms: HostTerms) -> None:
    """Refuse a budget below what the run holds before reading ahead, naming the terms.

    A build decoding on the calling thread reads nothing ahead, so this is its whole check.

    Args:
        resources: The caller's budget and cap.
        terms: What the reads cost.

    Raises:
        ValueError: If the budget is below ``M_main + O + C + (d + 1) E``.
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


def _fixed_bytes(terms: HostTerms) -> int:
    """``M_main + O + C + (d + 1) E``: what a run holds whatever reads it."""
    at_device = (terms.device_buffer + 1) * terms.unit_bytes
    return terms.main_bytes + terms.order_bytes + terms.sink_bytes + at_device


def _thread_plan(resources: HostResources, path: HostReadPath, terms: HostTerms) -> HostPlan:
    """At most four threads, below the CPUs and within the cap, reading ``max(16, 2T)`` ahead.

    Args:
        resources: The caller's budget and cap.
        path: Where the run is read: threads, or a stream's one producer thread.
        terms: What the reads cost.

    Returns:
        The plan.

    Raises:
        ValueError: If the budget is below the run and its read buffer, naming them.
    """
    e, d = terms.unit_bytes, terms.device_buffer
    threads = 1
    if path is HostReadPath.THREADS:
        threads = max(1, min(MAX_READ_THREADS, resources.max_workers, terms.cpus - 1))
    read_buffer = max(THREAD_READ_BUFFER, 2 * threads)
    fixed = _fixed_bytes(terms)
    needed = fixed + (read_buffer + 1) * e
    if resources.ram_budget_bytes < needed:
        raise ValueError(
            f"a RAM budget of {resources.ram_budget_bytes} bytes is below the {needed} bytes a run "
            f"on {threads} read thread(s) holds: the {fixed} bytes it holds before reading ahead "
            f"+ b={read_buffer} + 1 units read ahead of E={e} bytes: raise ram_budget_bytes, or "
            f"omit host_resources to read on {READ_THREADS} thread {READ_BUFFER} units ahead"
        )
    return HostPlan(
        path=path,
        threads=threads,
        workers=0,
        read_buffer=read_buffer,
        worker_buffer=0,
        worker_read_buffer=0,
        device_buffer=d,
        terms=terms,
        accounted_bytes=needed,
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
            if terms.shm_share:
                held = f"this process's /dev/shm share, shm_budget_bytes={terms.shm_free_bytes},"
                way_out = "raise HostResources.shm_budget_bytes"
            else:
                held = f"/dev/shm has {terms.shm_free_bytes} bytes free,"  # nosec B108
                way_out = "enlarge /dev/shm (Docker: --shm-size)"
            raise ValueError(
                f"{held} below the {in_queues + at_device} bytes one worker's units take there "
                f"({2 * o + 2} units of E={e} bytes in its queues and d={d} + 1 at the device): "
                f"{way_out} or read smaller units"
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
    order_bytes: int,
    depth: int,
    path: HostReadPath,
    resources: HostResources,
    read: Callable[[], Any],
    first_unit: Callable[[Any], Callable[[], Any]],
    settings: JaxSettings,
    spans_processes: bool,
) -> HostTerms:
    """What a run's reads cost, measured: ``E``, ``O``, ``M_main``, and for processes ``P, S, F``.

    ``M_main`` is the training process's memory now (:func:`host_workers.process_bytes`), the
    source's data and the process's runtime included. Under processes the workers' read is built
    first, so what it shares with them (a stream's offset index, mapped by the source) is resident
    in the training process once and never added on top; ``P`` is that read as it pickles, ``S``
    what it declares (:class:`~datarax.core.host_resources.WorkerFootprint`), and ``F``, unless
    the caller gives it, the peak of one probe worker reading the run's first unit under the
    run's settings. A thread plan builds and pickles nothing. The workers' units in ``/dev/shm``
    are bounded by the smaller of the caller's share and the free bytes measured, or by the free
    bytes without a share; under a mesh spanning processes the share is required, since every
    process on a machine measures the same free bytes and would plan against all of them.

    Args:
        unit_bytes: ``E``, one unit's host bytes.
        order_bytes: ``O``, what the run's order keeps resident
            (:func:`~datarax.core.host_resources.order_working_bytes`); 0 for the global order.
        depth: ``d``, placed units staged on the device ahead of the consumer's.
        path: Where the run is read.
        resources: The caller's budget and cap.
        read: Builds the read worker processes run, called only under processes.
        first_unit: Turns that read into a call reading the run's first unit, which the probe
            worker makes; called once, only when the probe runs.
        settings: The run's JAX settings, which the probe worker is set up with.
        spans_processes: Whether the run's sharding spans other processes' devices.

    Returns:
        The terms.

    Raises:
        ValueError: If worker processes read under a mesh spanning processes without a
            ``/dev/shm`` share.
    """
    terms = HostTerms(
        unit_bytes=unit_bytes,
        order_bytes=order_bytes,
        device_buffer=depth,
        cpus=available_cpus(),
    )
    if path is not HostReadPath.PROCESSES:
        return dataclasses.replace(terms, main_bytes=host_workers.process_bytes())
    share = resources.shm_budget_bytes
    if share is None and spans_processes:
        raise ValueError(
            "worker processes read this run under a sharding spanning other processes' "
            "devices, and every process on a machine sees the same free /dev/shm: give each "
            "process its share, HostResources(shm_budget_bytes=...), e.g. the free bytes over "
            "the processes on the machine"
        )
    worker_read = read()
    read_bytes = host_workers.pickled_bytes(worker_read)
    footprint = worker_read if isinstance(worker_read, WorkerFootprint) else None
    main_bytes = host_workers.process_bytes()
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
        **_shm_bound(share),
    )


def _shm_bound(share: int | None) -> dict[str, Any]:
    """The ``/dev/shm`` bound of a worker plan: the smaller of ``share`` and the free bytes.

    Args:
        share: The caller's share of ``/dev/shm``, or ``None``.

    Returns:
        ``shm_free_bytes`` and ``shm_share`` for :class:`HostTerms`.
    """
    free = host_workers.shm_free_bytes()
    if share is None or (free is not None and free < share):
        return {"shm_free_bytes": free, "shm_share": False}
    return {"shm_free_bytes": share, "shm_share": True}


__all__ = [
    "IDENTITY_BYTES",
    "MAX_READ_THREADS",
    "READ_BUFFER",
    "READ_THREADS",
    "THREAD_READ_BUFFER",
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
    "refuse_below_fixed",
    "spec_bytes",
    "warn_one_thread",
]
