"""``pipe.host_plan``: how a pipeline's host stage reads, the only source of its read options.

The plan comes from the pipeline's ``host_resources`` and what the stage measures: one unit's host
bytes ``E`` from the source's spec at the stored dtypes, the depth ``d`` of the default device's
platform, what the training process holds of the source ``M_main`` and what a worker is sent
``P`` (the read, as it pickles), and a worker's own memory ``F`` (measured by one probe worker,
unless the caller passes it). Its threads, buffers and workers are what the run's Grain datasets
are built with, and a different plan opens a new run.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import cloudpickle
import grain
import jax
import numpy as np
import psutil
import pytest
from flax import nnx

from datarax.core.data_source import HostRead
from datarax.core.host_resources import available_cpus, HostResources
from datarax.pipeline import Pipeline
from datarax.pipeline.host_stage import default_device_buffer
from datarax.pipeline.read_plan import (
    HostPlan,
    HostReadPath,
    HostTerms,
    READ_BUFFER,
    READ_THREADS,
    WORKER_BUFFER,
)
from datarax.sources import TFDSStreamingConfig, TFDSStreamingSource
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from tests.test_common.tfds_fixture import FIXTURE, TFDSFixture
from tests.test_common.worker_reads import png_source, resources, write_png_records


_MiB = 1 << 20


def _memory(length: int = 40, width: int = 6, dtype: Any = np.float32) -> MemorySource:
    return MemorySource(
        MemorySourceConfig(), {"x": np.arange(length * width, dtype=dtype).reshape(length, width)}
    )


def _pipeline(source: Any, host_resources: HostResources | None = None, **options: Any) -> Pipeline:
    return Pipeline(
        source=source,
        stages=[],
        batch_size=4,
        rngs=nnx.Rngs(0),
        host_resources=host_resources,
        **options,
    )


def _terms(pipe: Pipeline) -> HostTerms:
    """The terms of a budget's plan."""
    return _terms_of(pipe.host_plan)


def _terms_of(plan: HostPlan) -> HostTerms:
    terms = plan.terms
    assert terms is not None
    return terms


@pytest.fixture(scope="module")
def png_paths(tmp_path_factory: pytest.TempPathFactory) -> list[str]:
    """12 PNG records in two ArrayRecord files."""
    return write_png_records(tmp_path_factory.mktemp("png"), 12)


class TestDefaultPlan:
    def test_without_resources_the_stage_reads_as_c5a4_left_it(self) -> None:
        pipe = _pipeline(_memory())
        plan = pipe.host_plan
        assert (plan.path, plan.threads, plan.read_buffer, plan.workers) == (
            HostReadPath.THREADS,
            READ_THREADS,
            READ_BUFFER,
            0,
        )
        assert plan.device_buffer == default_device_buffer(jax.default_backend())
        assert plan.terms is None

    def test_a_gil_bound_read_without_resources_logs_one_line_naming_them(
        self, png_paths: list[str], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="datarax.pipeline.host_stage"):
            pipe = _pipeline(png_source(png_paths))
            list(pipe.raw_batches())
            list(pipe.raw_batches())  # the same run's plan: not logged again
        lines = [r.getMessage() for r in caplog.records if "HostResources" in r.getMessage()]
        assert len(lines) == 1, lines
        assert pipe.host_plan.path is HostReadPath.THREADS


class TestPlanTerms:
    def test_a_unit_s_bytes_are_host_bytes_at_the_stored_dtype(self) -> None:
        """A float64 column counts 8 bytes a value with x64 off, where the device holds 4."""
        assert not jax.config.read("jax_enable_x64")
        pipe = _pipeline(_memory(width=6, dtype=np.float64), resources(1))
        terms = pipe.host_plan.terms
        assert terms is not None
        identity = 2 * 4 + 4 + 4  # index words, epoch and draw, a record
        assert terms.unit_bytes == 4 * (6 * 8 + identity)

    def test_a_chunk_s_unit_is_k_batches(self) -> None:
        pipe = _pipeline(_memory(), resources(1))
        single = _terms(pipe).unit_bytes
        next(iter(pipe.raw_batches(2)))
        assert _terms(pipe).unit_bytes == 2 * single
        pipe.close()

    def test_the_depth_is_the_default_device_s_platform_s(self) -> None:
        plan = _pipeline(_memory(), resources(1)).host_plan
        assert plan.device_buffer == default_device_buffer(jax.default_backend())

    def test_the_global_order_holds_nothing_per_pipeline(self) -> None:
        assert _terms(_pipeline(_memory(), resources(1))).order_bytes == 0

    def test_m_main_is_the_training_process_s_resident_memory_at_plan_time(self) -> None:
        """A source holding 256 MiB more raises ``M_main`` by about 256 MiB, unpickled.

        Both plans are made in this process, the smaller pipeline kept alive, after collecting:
        between the two readings the resident size moves only by what the allocator returns or
        keeps of other objects, which P3 measured at about 1 MB; 16 MiB of slack is far above
        that and far below the 256 MiB the property is about.
        """
        held = 256 << 20
        small_pipe = _pipeline(_memory(length=40), resources(1))
        gc.collect()
        small = _terms(small_pipe).main_bytes
        column = np.ones((held // (6 * 4), 6), np.float32)  # every page written: resident
        large_pipe = _pipeline(MemorySource(MemorySourceConfig(), {"x": column}), resources(1))
        gc.collect()
        large = _terms(large_pipe).main_bytes
        assert large - small >= held - (16 << 20), (small, large)
        assert large >= psutil.Process().memory_info().rss - (16 << 20)

    def test_a_thread_plan_pickles_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``P`` is a worker's read; threads send none, so a thread plan never serializes."""
        calls: list[int] = []

        def counted(*args: Any, **kwargs: Any) -> Any:
            calls.append(1)
            return original(*args, **kwargs)

        original = cloudpickle.dump
        monkeypatch.setattr(cloudpickle, "dump", counted)
        monkeypatch.setattr(cloudpickle, "dumps", counted)
        plan = _pipeline(_memory(length=4000), resources(1)).host_plan
        assert plan.path is HostReadPath.THREADS
        assert calls == []
        assert _terms_of(plan).read_bytes == 0

    def test_a_worker_plan_measures_p_by_pickling_the_read(
        self, png_paths: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from datarax.pipeline import host_workers  # noqa: PLC0415

        sizes: list[int] = []
        original = host_workers.pickled_bytes

        def counted(value: Any) -> int:
            sizes.append(original(value))
            return sizes[-1]

        monkeypatch.setattr(host_workers, "pickled_bytes", counted)
        plan = _pipeline(png_source(png_paths), resources(2)).host_plan
        assert plan.path is HostReadPath.PROCESSES
        assert len(sizes) == 1
        assert _terms_of(plan).read_bytes == sizes[0] > 0

    def test_a_worker_read_takes_no_grain_default(self, png_paths: list[str]) -> None:
        plan = _pipeline(png_source(png_paths), resources(2)).host_plan
        assert (plan.path, plan.workers) == (HostReadPath.PROCESSES, 2)
        assert (plan.worker_read_buffer, plan.worker_buffer) == (0, WORKER_BUFFER)
        assert (plan.threads, plan.read_buffer) == (0, 0)


class TestTheRunReadsWithThePlan:
    def test_a_gil_free_read_s_threads_and_buffer_reach_its_read_options(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        built: list[grain.ReadOptions] = []
        original = grain.MapDataset.to_iter_dataset

        def spied(self: Any, read_options: Any = None, **kwargs: Any) -> Any:
            built.append(read_options)
            return original(self, read_options, **kwargs)

        monkeypatch.setattr(grain.MapDataset, "to_iter_dataset", spied)
        cap = min(4, available_cpus())
        pipe = _pipeline(_memory(), resources(cap))
        list(pipe.raw_batches())
        plan = pipe.host_plan
        assert plan.threads == max(1, min(cap, available_cpus() - 1))
        assert [(o.num_threads, o.prefetch_buffer_size) for o in built] == [
            (plan.threads, plan.read_buffer)
        ]

    def test_without_resources_the_run_reads_one_thread_two_ahead(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        built: list[grain.ReadOptions] = []
        original = grain.MapDataset.to_iter_dataset

        def spied(self: Any, read_options: Any = None, **kwargs: Any) -> Any:
            built.append(read_options)
            return original(self, read_options, **kwargs)

        monkeypatch.setattr(grain.MapDataset, "to_iter_dataset", spied)
        list(_pipeline(_memory()).raw_batches())
        assert [(o.num_threads, o.prefetch_buffer_size) for o in built] == [
            (READ_THREADS, READ_BUFFER)
        ]

    def test_a_worker_read_s_options_reach_mp_prefetch(
        self, png_paths: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reads: list[grain.ReadOptions] = []
        workers: list[Any] = []
        to_iter, mp_prefetch = grain.MapDataset.to_iter_dataset, grain.IterDataset.mp_prefetch

        def spied_read(self: Any, read_options: Any = None, **kwargs: Any) -> Any:
            reads.append(read_options)
            return to_iter(self, read_options, **kwargs)

        def spied_workers(self: Any, options: Any = None, *args: Any, **kwargs: Any) -> Any:
            workers.append(options)
            return mp_prefetch(self, options, *args, **kwargs)

        monkeypatch.setattr(grain.MapDataset, "to_iter_dataset", spied_read)
        monkeypatch.setattr(grain.IterDataset, "mp_prefetch", spied_workers)
        pipe = _pipeline(png_source(png_paths), resources(2))
        list(pipe.raw_batches())
        # The run's read, then Grain's own interleave of the workers' shards, which makes their
        # iterators on as many threads as workers (``multiprocess_prefetch``).
        assert [(o.num_threads, o.prefetch_buffer_size) for o in reads] == [(0, 0), (2, 2)]
        assert [(o.num_workers, o.per_worker_buffer_size) for o in workers] == [(2, WORKER_BUFFER)]

    def test_a_plan_change_opens_a_new_run_naming_the_plan(self) -> None:
        pipe = _pipeline(_memory(), resources(1), num_epochs=None)
        batches = iter(pipe.raw_batches())
        next(batches)
        pipe.host_stage._device_buffer = pipe.host_plan.device_buffer + 1  # noqa: SLF001
        next(iter(pipe.raw_batches()))
        with pytest.raises(RuntimeError, match="plan="):
            next(batches)
        pipe.close()


class TestWorkerFootprint:
    def test_a_probe_worker_measures_f_when_the_caller_does_not_pass_it(
        self, png_paths: list[str]
    ) -> None:
        resources_ = HostResources(ram_budget_bytes=32 << 30, max_workers=1)
        terms = _pipeline(png_source(png_paths), resources_).host_plan.terms
        assert terms is not None
        assert terms.worker_bytes > 0

    def test_a_caller_s_f_starts_no_probe(
        self, png_paths: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from datarax.pipeline import host_workers  # noqa: PLC0415

        def refused(*_args: Any, **_kwargs: Any) -> int:
            raise AssertionError("a caller's worker_bytes starts no probe worker")

        monkeypatch.setattr(host_workers, "probe_worker_bytes", refused)
        terms = _pipeline(png_source(png_paths), resources(1, worker_bytes=123 * _MiB)).host_plan
        assert terms.terms is not None and terms.terms.worker_bytes == 123 * _MiB

    def test_sources_declare_their_read(self, png_paths: list[str]) -> None:
        assert _memory().host_read is HostRead.GIL_FREE
        assert png_source(png_paths).host_read is HostRead.GIL_BOUND


@pytest.mark.tfds
def test_a_tfds_stream_sends_its_workers_the_index_by_reference(tfds_fixture: TFDSFixture) -> None:
    """One shared copy of the offset index: a worker's read does not carry it."""
    config = TFDSStreamingConfig(name=FIXTURE, split="train", data_dir=str(tfds_fixture.tfrecord))
    pipe = _pipeline(TFDSStreamingSource(config), resources(2), shuffle=True)
    list(pipe.raw_batches())
    terms = pipe.host_plan.terms
    assert pipe.host_plan.path is HostReadPath.PROCESSES
    assert terms is not None
    records = len(pipe.source)
    logging.getLogger(__name__).info(
        "TFDS stream: %d records, offset index %d bytes held once on the host; a worker is sent "
        "%d bytes",
        records,
        16 * records,
        terms.read_bytes,
    )
    # The shared copy is resident in this process (the source reads it), so the resident size
    # M_main is taken from holds it once: its mapping's pages are all in.
    source = pipe.source
    assert isinstance(source, TFDSStreamingSource)
    place = source._shared_index.value[0].shared  # noqa: SLF001
    assert place is not None
    path = place[0]
    assert _resident_kib_of(path) * 1024 >= 16 * records
    pipe.close()


def _resident_kib_of(path: str) -> int:
    """The resident KiB of this process's mappings of ``path`` (``/proc/self/smaps``)."""
    total, inside = 0, False
    for line in Path("/proc/self/smaps").read_text().splitlines():
        fields = line.split()
        if fields and "-" in fields[0] and len(fields) >= 5:
            inside = fields[-1] == path
        elif inside and fields[:1] == ["Rss:"]:
            total += int(fields[1])
    return total
