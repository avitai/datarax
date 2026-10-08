"""Budgets whose host plan reads with a chosen thread count and read buffer, for the tests.

A pipeline's read options come only from its plan (``pipe.host_plan``). A test that reads with
``n`` threads ``b`` units ahead asks for the budget the plan turns into exactly that: the plan's
terms for the source (what the training process holds of it, ``M_main``, and one unit's bytes
``E``) give ``M_main + O + (b + 1 + d + 1) E``. ``M_main`` is the process's resident size when the
plan is made, which moves between two plans: the read buffer is exactly ``b`` only while it holds
still, and a budget below ``M_main + O + (d + 1) E`` is refused. So a test reading with a chosen
plan holds it still with the ``still_resident`` fixture (``tests/conftest.py``), which pins
``host_workers.resident_bytes`` at :data:`STILL_RESIDENT`; :func:`reading_with` refuses to run
without it.
"""

from __future__ import annotations

from flax import nnx

from datarax.core.data_source import DataSourceModule
from datarax.core.host_resources import available_cpus, HostResources
from datarax.pipeline import host_workers, Pipeline


STILL_RESIDENT = 1 << 30
"""The resident size the ``still_resident`` fixture pins a plan's ``M_main`` at."""


def reading_with(
    source: DataSourceModule,
    batch_size: int,
    *,
    threads: int,
    read_buffer: int,
    depth: int | None = None,
) -> HostResources:
    """The resources under which a GIL-free read of ``source`` plans these options.

    Args:
        source: The source the pipeline reads (GIL-free).
        batch_size: The pipeline's batch size.
        threads: Read threads; at most the CPUs below the machine's count, as the plan caps.
        read_buffer: Units read ahead, 2 or more.
        depth: Placed units staged ahead, or ``None`` for the platform's.

    Returns:
        The resources.

    Raises:
        RuntimeError: If the plan's ``M_main`` is not held still (the ``still_resident`` fixture).
    """
    if host_workers.resident_bytes() != STILL_RESIDENT:
        raise RuntimeError(
            "reading_with needs the plan's M_main held still: request the still_resident fixture"
        )
    cap = min(threads, available_cpus())
    probe = Pipeline(
        source=source,
        stages=[],
        batch_size=batch_size,
        rngs=nnx.Rngs(0),
        host_resources=HostResources(ram_budget_bytes=1 << 40, max_workers=cap),
    )
    if depth is not None:
        probe.host_stage._device_buffer = depth  # noqa: SLF001 - the depth the tests pin
    terms = probe.host_plan.terms
    assert terms is not None  # noqa: S101 - a budget's plan holds its terms
    # Half a unit of slack: a run opened elsewhere (a resumed cursor) pickles a few bytes apart.
    budget = (
        terms.main_bytes
        + terms.order_bytes
        + (read_buffer + 1 + terms.device_buffer + 1) * terms.unit_bytes
        + terms.unit_bytes // 2
    )
    return HostResources(ram_budget_bytes=budget, max_workers=cap)
