"""What a training process may spend on its host stage: a RAM budget, and threads or processes.

A training process says what it may spend on its host stage, :class:`HostResources`: a RAM
budget and a cap on read threads or worker processes, at most the CPUs it may run on
(:func:`available_cpus`). The plan a budget gives, Grain's arithmetic with every term a worker
costs, is the pipeline's (:mod:`datarax.pipeline.read_plan`); a source states here what a run of
it read in worker processes costs beyond the read (:class:`WorkerFootprint`), so a source
declares it without importing the pipeline.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Protocol, runtime_checkable


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
    """

    ram_budget_bytes: int
    max_workers: int
    worker_bytes: int | None = None

    def __post_init__(self) -> None:
        """Refuse a budget, cap or footprint no plan can honour.

        Raises:
            ValueError: If the budget is below one byte, the cap below one or above the CPUs
                available, or ``worker_bytes`` negative.
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


__all__ = [
    "HostResources",
    "WorkerFootprint",
    "available_cpus",
]
