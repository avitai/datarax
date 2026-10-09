"""The plan a RAM budget gives the host stage's reads (``datarax.pipeline.read_plan``).

The plan is Grain's arithmetic (``pick_performance_config``: workers and buffers from a RAM
budget) with the terms a worker really costs: its own memory ``F``, its copy of the read ``P``,
the source's working set ``S``, the units in flight in its queues (``2o + 2`` of ``E`` bytes, plus
its read buffer ``r``), beside the training process's ``M_main``, the order's ``O``, the decoded
columns a build holds ``C`` and the ``d + 1`` units at the device boundary. Every bound is a
refusal naming its terms, never a rounded-up worker; ``/dev/shm`` is a second bound. GIL-free
reads run on threads, GIL-bound reads in processes, a stream with no run dataset on its one
producer thread. The terms are measured by :func:`measure_terms` from a bare source's values.
"""

from __future__ import annotations

import dataclasses
import multiprocessing
from collections.abc import Callable
from typing import Any

import grain
import jax
import numpy as np
import pytest
from flax import nnx

from datarax.core import host_resources as core_resources
from datarax.core.data_source import HostRead
from datarax.core.host_resources import available_cpus, HostResources
from datarax.core.spec import declared_spec
from datarax.pipeline import host_workers, Pipeline
from datarax.pipeline.host_stage import HostStage
from datarax.pipeline.host_workers import JaxSettings
from datarax.pipeline.read_plan import (
    budget_plan,
    default_plan,
    host_batch_bytes,
    host_read_path,
    HostPlan,
    HostReadPath,
    HostTerms,
    IDENTITY_BYTES,
    MAX_READ_BUFFER,
    measure_terms,
    READ_BUFFER,
    READ_THREADS,
    ReadAccess,
    spec_bytes,
    WORKER_BUFFER,
)
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from tests.test_common.worker_reads import png_source, resources, write_png_records


_MiB = 1 << 20


def _terms(**changes: int | None) -> HostTerms:
    """Terms of a worker read: E = 1 MiB, F = 300 MiB, P = 1 MiB, d = 1, 16 CPUs, ample shm."""
    base = HostTerms(
        unit_bytes=_MiB,
        device_buffer=1,
        main_bytes=10 * _MiB,
        order_bytes=0,
        worker_bytes=300 * _MiB,
        read_bytes=_MiB,
        working_bytes=0,
        worker_read_buffer=0,
        worker_buffer=WORKER_BUFFER,
        shm_free_bytes=64 << 30,
        cpus=16,
    )
    return dataclasses.replace(base, **changes)


def _workers_by_hand(budget: int, terms: HostTerms, cap: int) -> int:
    """The largest k meeting the RAM, ``/dev/shm`` and cap bounds, searched one k at a time."""
    best = 0
    o, r = terms.worker_buffer, terms.worker_read_buffer
    e, d = terms.unit_bytes, terms.device_buffer
    for k in range(1, cap + 1):
        ram = (
            terms.main_bytes
            + terms.order_bytes
            + terms.sink_bytes
            + (d + 1) * e
            + k * (terms.worker_bytes + terms.read_bytes + r * e + terms.working_bytes)
            + k * (2 * o + 2) * e
        )
        shm = k * (2 * o + 2) * e + (d + 1) * e
        if ram <= budget and (terms.shm_free_bytes is None or shm <= terms.shm_free_bytes):
            best = k
    return best


class TestReadPath:
    """Threads or processes by the read's kind; a stream without a run dataset on one thread."""

    def test_without_a_budget_every_read_runs_on_threads(self) -> None:
        for read in HostRead:
            assert host_read_path(None, read, ReadAccess.BY_UNIT) is HostReadPath.THREADS
            for access in (ReadAccess.SLICEABLE_STREAM, ReadAccess.STREAM):
                assert host_read_path(None, read, access) is HostReadPath.ONE_THREAD

    def test_with_a_budget_a_gil_bound_read_runs_in_processes(self) -> None:
        resources = HostResources(ram_budget_bytes=8 << 30, max_workers=1)
        for access in (ReadAccess.BY_UNIT, ReadAccess.SLICEABLE_STREAM):
            path = host_read_path(resources, HostRead.GIL_BOUND, access)
            assert path is HostReadPath.PROCESSES

    def test_with_a_budget_a_gil_free_read_stays_on_threads(self) -> None:
        resources = HostResources(ram_budget_bytes=8 << 30, max_workers=1)
        assert host_read_path(resources, HostRead.GIL_FREE, ReadAccess.BY_UNIT) is (
            HostReadPath.THREADS
        )
        assert host_read_path(resources, HostRead.GIL_FREE, ReadAccess.SLICEABLE_STREAM) is (
            HostReadPath.ONE_THREAD
        )

    def test_a_stream_read_pass_by_pass_keeps_one_thread_whatever_its_read(self) -> None:
        """HuggingFace streams stay on threads (one download), budget or not."""
        resources = HostResources(ram_budget_bytes=8 << 30, max_workers=1)
        for read in HostRead:
            assert host_read_path(resources, read, ReadAccess.STREAM) is HostReadPath.ONE_THREAD


class TestDefaultPlan:
    """Without a budget the host stage reads as C5a-4 left it: one thread, two units ahead."""

    @pytest.mark.parametrize("path", [HostReadPath.THREADS, HostReadPath.ONE_THREAD])
    @pytest.mark.parametrize("depth", [0, 1])
    def test_one_thread_two_units_ahead_and_the_platform_s_depth(
        self, path: HostReadPath, depth: int
    ) -> None:
        plan = default_plan(path, device_buffer=depth)
        assert (plan.path, plan.threads, plan.workers) == (path, READ_THREADS, 0)
        assert (plan.read_buffer, plan.device_buffer) == (READ_BUFFER, depth)
        assert (plan.worker_buffer, plan.worker_read_buffer) == (0, 0)
        assert plan.terms is None and plan.accounted_bytes is None and plan.shm_bytes is None

    def test_processes_need_a_budget(self) -> None:
        with pytest.raises(ValueError, match="HostResources"):
            default_plan(HostReadPath.PROCESSES, device_buffer=0)

    def test_c5a4_s_constants(self) -> None:
        assert (READ_THREADS, READ_BUFFER, MAX_READ_BUFFER, WORKER_BUFFER) == (1, 2, 1000, 2)


class TestWorkerPlan:
    """k is the largest count meeting the RAM bound, the ``/dev/shm`` bound and the cap."""

    @pytest.mark.parametrize("budget_gib", [1, 2, 4, 16])
    @pytest.mark.parametrize("cap", [1, 3, 8])
    def test_k_is_the_largest_meeting_every_bound(self, budget_gib: int, cap: int) -> None:
        budget = budget_gib << 30
        terms = _terms()
        expected = _workers_by_hand(budget, terms, cap)
        resources = HostResources(ram_budget_bytes=budget, max_workers=min(cap, available_cpus()))
        if expected == 0 or cap > available_cpus():
            pytest.skip("the case needs more CPUs than this machine has, or admits no worker")
        plan = budget_plan(resources, HostReadPath.PROCESSES, terms)
        assert plan.workers == expected
        assert plan.workers <= resources.max_workers

    def test_the_plan_accounts_every_term(self) -> None:
        terms = _terms()
        plan = budget_plan(
            HostResources(ram_budget_bytes=4 << 30, max_workers=min(3, available_cpus())),
            HostReadPath.PROCESSES,
            terms,
        )
        k, e, o = plan.workers, terms.unit_bytes, terms.worker_buffer
        per_worker = terms.worker_bytes + terms.read_bytes + terms.working_bytes + (2 * o + 2) * e
        assert plan.accounted_bytes == terms.main_bytes + (1 + 1) * e + k * per_worker
        assert plan.shm_bytes == k * (2 * o + 2) * e + 2 * e
        assert (plan.path, plan.threads, plan.read_buffer) == (HostReadPath.PROCESSES, 0, 0)
        assert (plan.worker_buffer, plan.worker_read_buffer) == (2, 0)
        assert plan.device_buffer == terms.device_buffer
        assert plan.terms == terms

    def test_the_order_s_working_set_and_the_source_s_are_charged(self) -> None:
        budget = 2 << 30
        cap = min(8, available_cpus())
        plain = budget_plan(
            HostResources(ram_budget_bytes=budget, max_workers=cap),
            HostReadPath.PROCESSES,
            _terms(),
        )
        heavier = budget_plan(
            HostResources(ram_budget_bytes=budget, max_workers=cap),
            HostReadPath.PROCESSES,
            _terms(order_bytes=200 * _MiB, working_bytes=200 * _MiB),
        )
        assert heavier.workers < plain.workers or plain.workers == 1

    def test_a_budget_below_one_worker_is_refused_naming_the_terms(self) -> None:
        resources = HostResources(ram_budget_bytes=200 * _MiB, max_workers=1)
        with pytest.raises(ValueError) as refused:
            budget_plan(resources, HostReadPath.PROCESSES, _terms())
        message = str(refused.value)
        for term in ("F=", "P=", "E=", "d=", str(200 * _MiB)):
            assert term in message
        assert "no worker" in message

    def test_a_dev_shm_below_one_worker_s_units_is_refused_naming_both(self) -> None:
        terms = _terms(shm_free_bytes=4 * _MiB)  # one worker needs 6 + 2 units of 1 MiB
        resources = HostResources(ram_budget_bytes=16 << 30, max_workers=1)
        with pytest.raises(ValueError, match=r"/dev/shm") as refused:
            budget_plan(resources, HostReadPath.PROCESSES, terms)
        assert str(4 * _MiB) in str(refused.value)
        assert str(8 * _MiB) in str(refused.value)

    def test_a_small_dev_shm_lowers_k(self) -> None:
        cap = min(8, available_cpus())
        if cap < 2:  # pragma: no cover - a one-CPU machine
            pytest.skip("needs two CPUs")
        terms = _terms(shm_free_bytes=(8 + 6) * _MiB)  # room for two workers' units
        plan = budget_plan(
            HostResources(ram_budget_bytes=16 << 30, max_workers=cap),
            HostReadPath.PROCESSES,
            terms,
        )
        assert plan.workers == 2

    def test_no_dev_shm_to_measure_is_no_bound(self) -> None:
        cap = min(2, available_cpus())
        plan = budget_plan(
            HostResources(ram_budget_bytes=16 << 30, max_workers=cap),
            HostReadPath.PROCESSES,
            _terms(shm_free_bytes=None),
        )
        assert plan.workers == cap
        assert plan.shm_bytes is not None


class TestThreadPlan:
    """GIL-free reads: threads below the CPUs and the cap, the read buffer from the budget."""

    def test_the_read_buffer_is_the_largest_the_budget_holds(self) -> None:
        terms = _terms(unit_bytes=_MiB, main_bytes=10 * _MiB, device_buffer=1)
        budget = 10 * _MiB + (8 + 1 + 1 + 1) * _MiB  # M_main + (b + 1 + d + 1) E with b = 8
        plan = budget_plan(
            HostResources(ram_budget_bytes=budget, max_workers=1), HostReadPath.THREADS, terms
        )
        assert plan.read_buffer == 8
        assert plan.accounted_bytes == budget
        assert (plan.workers, plan.worker_buffer, plan.shm_bytes) == (0, 0, None)

    def test_the_read_buffer_is_never_below_c5a4_s_two(self) -> None:
        """A budget holding the fixed part and no unit more still reads two units ahead."""
        terms = _terms()
        fixed = terms.main_bytes + (terms.device_buffer + 1) * terms.unit_bytes
        plan = budget_plan(
            HostResources(ram_budget_bytes=fixed, max_workers=1), HostReadPath.THREADS, terms
        )
        assert plan.read_buffer == READ_BUFFER

    def test_the_read_buffer_is_capped_at_grain_s_thousand(self) -> None:
        plan = budget_plan(
            HostResources(ram_budget_bytes=1 << 40, max_workers=1),
            HostReadPath.THREADS,
            _terms(unit_bytes=1),
        )
        assert plan.read_buffer == MAX_READ_BUFFER

    @pytest.mark.parametrize("cap", [1, 2, 4])
    def test_threads_stay_below_the_cpus_and_within_the_cap(self, cap: int) -> None:
        if cap > available_cpus():
            pytest.skip("needs more CPUs")
        for cpus in (1, 2, 3, 16):
            plan = budget_plan(
                HostResources(ram_budget_bytes=1 << 30, max_workers=cap),
                HostReadPath.THREADS,
                _terms(cpus=cpus),
            )
            assert plan.threads == max(1, min(cap, cpus - 1))

    def test_a_stream_keeps_one_producer_thread(self) -> None:
        plan = budget_plan(
            HostResources(ram_budget_bytes=1 << 30, max_workers=min(4, available_cpus())),
            HostReadPath.ONE_THREAD,
            _terms(),
        )
        assert plan.threads == 1
        assert plan.read_buffer >= READ_BUFFER


def test_a_plan_is_static_and_hashable() -> None:
    plan = budget_plan(
        HostResources(ram_budget_bytes=4 << 30, max_workers=1), HostReadPath.PROCESSES, _terms()
    )
    assert isinstance(plan, HostPlan)
    assert hash(plan) == hash(dataclasses.replace(plan))
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.workers = 3  # type: ignore[misc]


def test_positive_control_grain_s_own_pick_ignores_what_a_worker_costs() -> None:
    """Grain's function returns every CPU and a 1,000-element buffer whatever the budget.

    Elements of 0.75 MiB (a CIFAR batch of 256): 16 GiB and 64 GiB both give ``cpu_count()``
    workers, though each worker of a datarax read costs about 300 MB before its first element.
    """
    element = np.zeros(int(0.75 * _MiB), np.uint8)
    dataset = grain.MapDataset.source([element] * 8).to_iter_dataset()
    for budget_mib in (16 << 10, 64 << 10):
        config = grain.experimental.pick_performance_config(
            dataset, ram_budget_mb=budget_mib, max_workers=None, max_buffer_size=None
        )
        assert config.multiprocessing_options is not None and config.read_options is not None
        assert config.multiprocessing_options.num_workers == multiprocessing.cpu_count()
        assert config.read_options.prefetch_buffer_size == 1000


_PLANNED_PATHS = [HostReadPath.PROCESSES, HostReadPath.THREADS, HostReadPath.ONE_THREAD]


class TestTheSinkTerm:
    """``C``, the decoded columns a build holds in RAM, is charged by both plans' fixed part."""

    @pytest.mark.parametrize("path", _PLANNED_PATHS)
    def test_the_accounted_bytes_rise_by_exactly_c(self, path: HostReadPath) -> None:
        resources = HostResources(ram_budget_bytes=1 << 40, max_workers=1)
        sink = 300 * _MiB
        plain = budget_plan(resources, path, _terms())
        sunk = budget_plan(resources, path, _terms(sink_bytes=sink))
        assert (sunk.workers, sunk.read_buffer) == (plain.workers, plain.read_buffer)
        assert plain.accounted_bytes is not None and sunk.accounted_bytes is not None
        assert sunk.accounted_bytes - plain.accounted_bytes == sink

    def test_the_sink_lowers_the_worker_count(self) -> None:
        """A sink of one worker's bytes costs one worker, below the cap on any machine.

        The budget admits exactly ``cap`` workers without the sink, so the cap the machine's
        CPUs set never hides what the sink takes.
        """
        cap = min(8, available_cpus())
        plain = _terms()
        e, o, r = plain.unit_bytes, plain.worker_buffer, plain.worker_read_buffer
        per_worker = (
            plain.worker_bytes + plain.read_bytes + r * e + plain.working_bytes + (2 * o + 2) * e
        )
        fixed = plain.main_bytes + plain.order_bytes + (plain.device_buffer + 1) * e
        budget = fixed + cap * per_worker
        resources = HostResources(ram_budget_bytes=budget, max_workers=cap)
        sunk = _terms(sink_bytes=per_worker)
        assert budget_plan(resources, HostReadPath.PROCESSES, plain).workers == cap
        assert _workers_by_hand(budget, plain, cap) == cap
        if cap == 1:
            with pytest.raises(ValueError, match="admits no worker"):
                budget_plan(resources, HostReadPath.PROCESSES, sunk)
            return
        plan = budget_plan(resources, HostReadPath.PROCESSES, sunk)
        assert plan.workers == _workers_by_hand(budget, sunk, cap) == cap - 1

    def test_the_sink_lowers_the_read_buffer(self) -> None:
        terms = _terms(device_buffer=1, sink_bytes=4 * _MiB)
        budget = terms.main_bytes + terms.sink_bytes + (8 + 1 + 1 + 1) * terms.unit_bytes
        plan = budget_plan(
            HostResources(ram_budget_bytes=budget, max_workers=1), HostReadPath.THREADS, terms
        )
        assert plan.read_buffer == 8
        assert plan.accounted_bytes == budget

    @pytest.mark.parametrize("path", _PLANNED_PATHS)
    def test_a_budget_below_the_fixed_part_is_refused_naming_c(self, path: HostReadPath) -> None:
        """``M_main + O + C + (d + 1) E`` above the budget: no plan, threads included."""
        terms = _terms(sink_bytes=300 * _MiB, order_bytes=_MiB)
        fixed = (
            terms.main_bytes
            + terms.order_bytes
            + terms.sink_bytes
            + (terms.device_buffer + 1) * terms.unit_bytes
        )
        resources = HostResources(ram_budget_bytes=fixed - 1, max_workers=1)
        with pytest.raises(ValueError) as refused:
            budget_plan(resources, path, terms)
        message = str(refused.value)
        for term in (f"C={300 * _MiB}", f"M_main={10 * _MiB}", f"O={_MiB}", f"E={_MiB}"):
            assert term in message, message
        assert str(fixed - 1) in message


class TestHostBytes:
    """A unit's host bytes: its data at the stored dtypes and each record's identity."""

    def test_a_record_s_identity_is_its_words_epoch_and_draw(self) -> None:
        assert IDENTITY_BYTES == 2 * 4 + 4 + 4

    def test_a_spec_s_bytes_sum_its_leaves(self) -> None:
        spec = {
            "a": jax.ShapeDtypeStruct((3, 2), np.float32),
            "b": jax.ShapeDtypeStruct((), np.int16),
        }
        assert spec_bytes(spec) == 3 * 2 * 4 + 2

    def test_a_float64_column_counts_eight_bytes_with_x64_off(self) -> None:
        assert not jax.config.read("jax_enable_x64")
        source = MemorySource(MemorySourceConfig(), {"x": np.zeros((10, 6), np.float64)})
        assert spec_bytes(declared_spec(source)) == 6 * 4  # as the device holds it
        assert host_batch_bytes(source, 4) == 4 * (6 * 8 + IDENTITY_BYTES)


class _Footprint:
    """A worker read declaring its working set, as a TFDS stream's run dataset does."""

    def shared_bytes(self) -> int:
        return 0

    def working_bytes(self) -> int:
        return 7 * _MiB


class TestMeasureTerms:
    """The terms of a run measured from values: no pipeline is needed."""

    def _measure(
        self,
        path: HostReadPath,
        resources: HostResources,
        read: Callable[[], Any],
        first_unit: Callable[[Any], Callable[[], Any]],
    ) -> HostTerms:
        return measure_terms(
            unit_bytes=3 * _MiB,
            depth=1,
            path=path,
            resources=resources,
            read=read,
            first_unit=first_unit,
            settings=JaxSettings.current(),
        )

    @pytest.mark.parametrize("path", [HostReadPath.THREADS, HostReadPath.ONE_THREAD])
    def test_threads_build_no_read_and_start_no_probe(self, path: HostReadPath) -> None:
        built: list[int] = []
        firsts: list[Any] = []

        def read() -> Any:
            built.append(1)
            return _Footprint()

        def first_unit(value: Any) -> Callable[[], Any]:
            firsts.append(value)
            return lambda: None

        terms = self._measure(
            path, HostResources(ram_budget_bytes=1 << 40, max_workers=1), read, first_unit
        )
        assert (built, firsts) == ([], [])
        assert (terms.unit_bytes, terms.device_buffer, terms.cpus) == (
            3 * _MiB,
            1,
            available_cpus(),
        )
        assert (terms.read_bytes, terms.worker_bytes, terms.working_bytes) == (0, 0, 0)
        assert terms.main_bytes > 0

    def test_processes_read_the_first_unit_once_through_the_probe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        settings = JaxSettings.current()
        probed: list[tuple[Any, JaxSettings]] = []
        firsts: list[Any] = []
        read_value = _Footprint()

        def probe(first: Callable[[], Any], *, settings: JaxSettings) -> int:
            probed.append((first, settings))
            return 11 * _MiB

        def first_unit(value: Any) -> Callable[[], Any]:
            firsts.append(value)
            return probe_target

        def probe_target() -> None:
            return None

        monkeypatch.setattr(host_workers, "probe_worker_bytes", probe)
        terms = self._measure(
            HostReadPath.PROCESSES,
            HostResources(ram_budget_bytes=1 << 40, max_workers=1),
            lambda: read_value,
            first_unit,
        )
        assert firsts == [read_value]
        assert probed == [(probe_target, settings)]
        assert terms.worker_bytes == 11 * _MiB
        assert terms.working_bytes == 7 * _MiB
        assert terms.read_bytes == host_workers.pickled_bytes(read_value) > 0
        assert terms.worker_buffer == WORKER_BUFFER
        assert terms.shm_free_bytes == host_workers.shm_free_bytes()

    def test_a_caller_s_worker_bytes_reads_no_first_unit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refused(*_args: Any, **_kwargs: Any) -> int:
            raise AssertionError("a caller's worker_bytes starts no probe worker")

        def first_unit(value: Any) -> Callable[[], Any]:
            raise AssertionError(f"no first unit is read for a caller's F ({value!r})")

        monkeypatch.setattr(host_workers, "probe_worker_bytes", refused)
        terms = self._measure(
            HostReadPath.PROCESSES,
            HostResources(ram_budget_bytes=1 << 40, max_workers=1, worker_bytes=5 * _MiB),
            _Footprint,
            first_unit,
        )
        assert terms.worker_bytes == 5 * _MiB


@pytest.fixture(scope="module")
def png_paths(tmp_path_factory: pytest.TempPathFactory) -> list[str]:
    """12 PNG records in two ArrayRecord files."""
    return write_png_records(tmp_path_factory.mktemp("png"), 12)


def _same_terms_but_resident(measured: HostTerms, planned: HostTerms) -> None:
    """Every term equal but ``M_main``, the resident size read at a different moment."""
    assert dataclasses.replace(measured, main_bytes=0) == dataclasses.replace(planned, main_bytes=0)


class TestHostStageMeasuresFromABareSource:
    """``HostStage._terms`` and ``_read`` take a source and values, never a ``Pipeline``."""

    def test_measure_terms_gives_the_terms_the_stage_plans_with(self, png_paths: list[str]) -> None:
        budget = resources(2)
        pipe = Pipeline(
            source=png_source(png_paths),
            stages=[],
            batch_size=4,
            rngs=nnx.Rngs(0),
            shuffle=True,
            host_resources=budget,
        )
        planned = pipe.host_plan.terms
        assert planned is not None and pipe.host_plan.path is HostReadPath.PROCESSES
        stage = HostStage(end_epoch=pipe.epoch_plan.num_epochs)
        settings = JaxSettings.current()
        source = pipe.source
        key = np.asarray(jax.random.key_data(pipe._epoch_key_base.get_value()), np.uint32)  # noqa: SLF001

        def read() -> Any:
            return stage._read(  # noqa: SLF001
                source, pipe.epoch_plan, shuffle=True, key=key, chunk=None, settings=settings
            )

        measured = measure_terms(
            unit_bytes=host_batch_bytes(source, 4),
            depth=planned.device_buffer,
            path=HostReadPath.PROCESSES,
            resources=budget,
            read=read,
            first_unit=lambda value: value,
            settings=JaxSettings.current(),
        )
        _same_terms_but_resident(measured, planned)
        direct = stage._terms(  # noqa: SLF001
            source,
            batch_size=4,
            epoch_plan=pipe.epoch_plan,
            shuffle=True,
            key=key,
            chunk=None,
            depth=planned.device_buffer,
            path=HostReadPath.PROCESSES,
            resources=budget,
            settings=JaxSettings.current(),
        )
        _same_terms_but_resident(direct, planned)
        pipe.close()

    def test_a_thread_plan_s_terms_from_a_bare_source(self) -> None:
        source = MemorySource(MemorySourceConfig(), {"x": np.zeros((40, 6), np.float32)})
        budget = HostResources(ram_budget_bytes=1 << 40, max_workers=1)
        pipe = Pipeline(
            source=source, stages=[], batch_size=4, rngs=nnx.Rngs(0), host_resources=budget
        )
        planned = pipe.host_plan.terms
        assert planned is not None
        direct = HostStage(end_epoch=1)._terms(  # noqa: SLF001
            source,
            batch_size=4,
            epoch_plan=pipe.epoch_plan,
            shuffle=False,
            key=None,
            chunk=None,
            depth=planned.device_buffer,
            path=HostReadPath.THREADS,
            resources=budget,
            settings=JaxSettings.current(),
        )
        _same_terms_but_resident(direct, planned)


def test_the_moved_names_are_gone_from_core() -> None:
    """The plan lives in the pipeline layer; core keeps the budget and the CPU count only."""
    for name in ("HostPlan", "HostTerms", "budget_plan", "default_plan", "host_read_path"):
        assert not hasattr(core_resources, name), name
    assert set(core_resources.__all__) == {"HostResources", "WorkerFootprint", "available_cpus"}
