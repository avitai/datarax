"""What a training process may spend on its host stage (``HostResources``), refused where unusable.

``HostResources(ram_budget_bytes, max_workers)`` is what the training process may spend on the host
stage and the most threads or worker processes it may start; the plan it gives lives in the
pipeline layer (``tests/pipeline/test_read_plan.py``).
"""

from __future__ import annotations

import dataclasses
import os

import pytest

from datarax.core.host_resources import available_cpus, HostResources


class TestHostResources:
    """The caller's budget and cap, refused where no plan could honour them."""

    def test_holds_the_budget_and_the_cap_static_and_hashable(self) -> None:
        resources = HostResources(ram_budget_bytes=8 << 30, max_workers=1)
        assert resources.ram_budget_bytes == 8 << 30
        assert resources.max_workers == 1
        assert resources.worker_bytes is None
        assert hash(resources) == hash(HostResources(ram_budget_bytes=8 << 30, max_workers=1))
        with pytest.raises(dataclasses.FrozenInstanceError):
            resources.max_workers = 2  # type: ignore[misc]

    @pytest.mark.parametrize("budget", [0, -1])
    def test_a_budget_below_one_byte_is_refused(self, budget: int) -> None:
        with pytest.raises(ValueError, match="ram_budget_bytes"):
            HostResources(ram_budget_bytes=budget, max_workers=1)

    def test_a_cap_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="max_workers"):
            HostResources(ram_budget_bytes=1 << 30, max_workers=0)

    def test_a_cap_above_the_cpus_available_is_refused_naming_both(self) -> None:
        cpus = available_cpus()
        with pytest.raises(ValueError, match=rf"max_workers={cpus + 1}.*{cpus} CPUs"):
            HostResources(ram_budget_bytes=1 << 30, max_workers=cpus + 1)

    def test_a_negative_worker_footprint_is_refused(self) -> None:
        with pytest.raises(ValueError, match="worker_bytes"):
            HostResources(ram_budget_bytes=1 << 30, max_workers=1, worker_bytes=-1)

    def test_the_cpus_available_are_the_process_s_affinity(self) -> None:
        if hasattr(os, "sched_getaffinity"):
            assert available_cpus() == len(os.sched_getaffinity(0))
        else:  # pragma: no cover - macOS
            assert available_cpus() == os.cpu_count()
