"""Budgets whose host plan reads a GIL-free source with a chosen thread count, for the tests.

A pipeline's read options come only from its plan (``pipe.host_plan``). With a budget a GIL-free
read takes ``min(4, max_workers, CPUs - 1)`` threads and reads ``max(16, 2 x threads)`` units
ahead whatever the budget (``read_plan.THREAD_READ_BUFFER``), so a test asks for the threads by the
cap and for a budget that holds the run and its buffer. :data:`STILL_MAIN_BYTES` is what the
``still_main_bytes`` fixture (``tests/conftest.py``) pins a plan's ``M_main`` at, for tests that
compute a budget from it.
"""

from __future__ import annotations

from datarax.core.host_resources import available_cpus, HostResources


STILL_MAIN_BYTES = 1 << 30
"""What the ``still_main_bytes`` fixture pins a plan's ``M_main`` at."""


def reading_with(threads: int) -> HostResources:
    """Resources under which a GIL-free read plans ``threads`` read threads (within the CPUs).

    The budget holds any test's run and its read buffer; the plan caps the threads at four and
    below the CPUs (:func:`~datarax.pipeline.read_plan.budget_plan`).

    Args:
        threads: Read threads asked for, from 1.

    Returns:
        The resources.
    """
    return HostResources(ram_budget_bytes=1 << 40, max_workers=min(threads, available_cpus()))
