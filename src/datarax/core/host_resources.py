"""What a training process may spend on its host stage: a RAM budget, and threads or processes.

A training process says what it may spend on its host stage, :class:`HostResources`: a RAM
budget and a cap on read threads or worker processes, at most the CPUs it may run on
(:func:`available_cpus`). The plan a budget gives, Grain's arithmetic with every term a worker
costs, is the pipeline's (:mod:`datarax.pipeline.read_plan`); a source states here what a run of
it read in worker processes costs beyond the read (:class:`WorkerFootprint`) and what a block
order keeps resident in the page cache (:class:`OrderWorkingSet`), so a source declares both
without importing the pipeline.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Protocol, runtime_checkable

from datarax.core.block_order import BlockOrder


def available_cpus() -> int:
    """The CPUs this process may run on: its affinity, or the machine's where none is set.

    Returns:
        The CPU count.
    """
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        return len(affinity(0))
    return os.cpu_count() or 1


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class HostResources:
    """What a training process may spend on its host stage: RAM, and threads or processes.

    The budget and the cap are the training process's own: with one process per GPU, each
    process passes its share. Static and hashable, so a pipeline holding it keeps one graph.

    Attributes:
        ram_budget_bytes: Host RAM the training process and its host stage may use: the
            process's resident memory when a run is planned, the units in flight and every
            worker it starts.
        max_workers: The most read threads or worker processes, at most the CPUs available.
        worker_bytes: A worker's own memory (interpreter, imports, JAX's CPU runtime, its
            decoder), or ``None`` for one probe worker to measure it when a run first needs it.
        shm_budget_bytes: This process's share of ``/dev/shm``, where its workers' units wait,
            or ``None`` to bound them by the free space ``/dev/shm`` reports. Required for
            worker processes under a sharding spanning processes: every process on the machine
            would read the same free space and plan against all of it.
    """

    ram_budget_bytes: int
    max_workers: int
    worker_bytes: int | None = None
    shm_budget_bytes: int | None = None

    def __post_init__(self) -> None:
        """Refuse a budget, cap or footprint no plan can honour.

        Raises:
            ValueError: If the budget is below one byte, the cap below one or above the CPUs
                available, ``worker_bytes`` negative, or ``shm_budget_bytes`` below one byte.
        """
        if self.ram_budget_bytes < 1:
            raise ValueError(f"ram_budget_bytes is at least 1; got {self.ram_budget_bytes}")
        if self.max_workers < 1:
            raise ValueError(f"max_workers is at least 1; got {self.max_workers}")
        cpus = available_cpus()
        if self.max_workers > cpus:
            raise ValueError(
                f"max_workers={self.max_workers} is above the {cpus} CPUs this process may run "
                "on: a worker or thread past them only waits for a CPU"
            )
        if self.worker_bytes is not None and self.worker_bytes < 0:
            raise ValueError(f"worker_bytes is at least 0; got {self.worker_bytes}")
        if self.shm_budget_bytes is not None and self.shm_budget_bytes < 1:
            raise ValueError(f"shm_budget_bytes is at least 1; got {self.shm_budget_bytes}")


@runtime_checkable
class WorkerFootprint(Protocol):
    """What a run's dataset read in worker processes costs beyond the read each worker is sent.

    A stream's run dataset implements it (``TFDSStreamDataset``); the plan charges
    :meth:`shared_bytes` once to the training process and :meth:`working_bytes` to every worker.
    """

    def shared_bytes(self) -> int:
        """Bytes held once on the host and read by every worker, as a shared read-only copy."""
        ...

    def working_bytes(self) -> int:
        """Bytes each worker holds while reading, beyond its buffers and its copy of the read."""
        ...


@runtime_checkable
class OrderWorkingSet(Protocol):
    """What a block order keeps resident while a run of a source is read in it.

    Every ``INDEXED`` source implements it: a source read from storage reports one buffer's
    records' stored bytes (:func:`~datarax.core.block_order.buffer_bytes`), an in-memory source
    0, since its records are already the training process's resident memory, and a mix the sum
    of its children's. The plan charges it to the budget as the order's term, ``O``.
    """

    def order_working_bytes(self, order: BlockOrder) -> int:
        """Bytes ``order`` keeps resident while a run of this source is read in it."""
        ...


def order_working_bytes(source: Any, order: BlockOrder | None) -> int:
    """The bytes a run of ``source`` read in ``order`` keeps resident: 0 for the global order.

    Args:
        source: The source the run reads.
        order: The run's block order, or ``None`` for the global order.

    Returns:
        The bytes.

    Raises:
        TypeError: If a block order is given for a source that reports no working set.
    """
    if order is None:
        return 0
    if not isinstance(source, OrderWorkingSet):
        raise TypeError(
            f"{type(source).__name__} reports no working set for a block order: implement "
            "OrderWorkingSet.order_working_bytes(order), the stored bytes a buffer of its "
            "records keeps resident, so a RAM budget can be planned for it"
        )
    return source.order_working_bytes(order)


__all__ = [
    "HostResources",
    "OrderWorkingSet",
    "WorkerFootprint",
    "available_cpus",
    "order_working_bytes",
]
