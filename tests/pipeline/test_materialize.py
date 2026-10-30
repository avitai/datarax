"""``materialize`` into RAM: the decoded columns, by the calling thread or the budget's workers.

A source whose read decodes every record (``HostRead.GIL_BOUND``) is decoded once by the worker
processes a RAM budget admits (two at least: one is slower than the calling thread), or on the
calling thread without a budget; a GIL-free read on the calling thread whatever the budget. The
result is a ``MemorySource`` holding the columns out of NNX state, record ``r`` the source's
record ``r``, equal bit for bit whatever decoded it. A budget below what the build holds
(``M_main + C + E``) is refused before a worker starts, naming the ways out; it is stricter than
``from_tfds(..., in_memory=True, host_resources=...)``, which checks the decoded size alone.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest
from flax import nnx

from datarax.core.data_source import known_length, RecordIdentity
from datarax.core.host_resources import available_cpus, HostResources
from datarax.pipeline import materialization, materialize, Pipeline
from datarax.pipeline.read_plan import WORKER_BUFFER
from datarax.sources import from_tfds, MemorySource, MemorySourceConfig, open_decoded_copy
from datarax.sources.decoded_copy._units import source_layout
from tests.test_common.decoded_copies import every_record, write_copy
from tests.test_common.host_plans import STILL_MAIN_BYTES
from tests.test_common.tfds_fixture import FIXTURE, TFDSFixture
from tests.test_common.tfds_golden import digests, EAGER_BUILD
from tests.test_common.worker_reads import (
    png_source,
    resources,
    shm_names_of,
    spawned_children,
    stream_of,
    write_png_records,
)


_RECORDS = 40
_UNIT = 4
PROCESS_COUNTS = tuple(k for k in (2, 3) if k <= available_cpus())
"""Worker processes a build is planned with: from two (one never runs), within the CPUs."""
COUNTS = (0, 1, *PROCESS_COUNTS)
_SEGMENT = "psm_"
"""The prefix of a ``multiprocessing.shared_memory`` segment, which a unit crosses in; the
queues' semaphores and heap arenas beside them hold no unit."""


@pytest.fixture(scope="module")
def png_paths(tmp_path_factory: pytest.TempPathFactory) -> list[str]:
    return write_png_records(tmp_path_factory.mktemp("png"), _RECORDS, shards=3)


class _Workers:
    """The worker counts the build started (each ``in_workers`` call)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.counts: list[int] = []
        start = materialization.in_workers

        def counted(dataset: Any, *, workers: int, **kwargs: Any) -> Any:
            self.counts.append(workers)
            return start(dataset, workers=workers, **kwargs)

        monkeypatch.setattr(materialization, "in_workers", counted)


@pytest.fixture
def started(monkeypatch: pytest.MonkeyPatch) -> _Workers:
    return _Workers(monkeypatch)


def _budget(workers: int) -> HostResources | None:
    return resources(workers) if workers else None


def _same(left: Any, right: Any) -> None:
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        assert np.asarray(a).dtype == np.asarray(b).dtype
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


class TestIntoMemory:
    @pytest.mark.parametrize("workers", COUNTS)
    def test_equals_the_source_record_for_record_for_every_worker_count(
        self, png_paths: list[str], started: _Workers, workers: int
    ) -> None:
        source = png_source(png_paths)

        result = materialize(source, host_resources=_budget(workers), unit_records=_UNIT)

        assert isinstance(result, MemorySource)
        assert result.record_identity is RecordIdentity.INDEXED
        assert started.counts == ([workers] if workers >= 2 else [])
        data, provenance = every_record(png_source(png_paths))
        held, held_provenance = every_record(result)
        _same(held, data)
        assert [dict(p) for p in held_provenance] == [dict(p) for p in provenance]

    @pytest.mark.tfds
    @pytest.mark.parametrize("workers", COUNTS)
    @pytest.mark.parametrize("case", ["train", "slice", "filtered"])
    def test_a_tfds_split_equals_the_recorded_eager_build(
        self, tfds_fixture: TFDSFixture, workers: int, case: str
    ) -> None:
        options, columns, names = EAGER_BUILD[case]
        source = from_tfds(
            FIXTURE,
            options.get("split", "train"),
            data_dir=str(tfds_fixture.array_record),
            in_memory=False,
            exclude_keys=options.get("exclude_keys"),
        )

        result = materialize(source, host_resources=_budget(workers), unit_records=_UNIT)

        assert digests(every_record(result)[0]) == columns
        provenance = every_record(result)[1]
        assert [p["name"].decode() for p in provenance if p] == names

    def test_a_pipeline_over_it_serves_the_pipeline_over_the_source(
        self, png_paths: list[str]
    ) -> None:
        workers = PROCESS_COUNTS[0] if PROCESS_COUNTS else 0
        result = materialize(png_source(png_paths), host_resources=_budget(workers))

        def epoch(source: Any) -> list[Any]:
            pipe = Pipeline(
                source=source,
                stages=[],
                batch_size=6,
                rngs=nnx.Rngs(5),
                shuffle=True,
                num_epochs=2,
            )
            served = stream_of(pipe.raw_batches())
            pipe.close()
            return served

        assert epoch(result) == epoch(png_source(png_paths))

    def test_the_argument_is_unchanged(self, png_paths: list[str]) -> None:
        source = png_source(png_paths)
        graphdef, state = nnx.split(source)
        before = [np.asarray(leaf).tobytes() for leaf in jax.tree.leaves(state)]

        materialize(source, host_resources=_budget(PROCESS_COUNTS[0] if PROCESS_COUNTS else 0))

        after_graphdef, after_state = nnx.split(source)
        assert after_graphdef == graphdef
        assert [np.asarray(leaf).tobytes() for leaf in jax.tree.leaves(after_state)] == before

    @pytest.mark.skipif(not Path("/dev/shm").is_dir(), reason="no /dev/shm to inspect (not Linux)")  # nosec B108
    @pytest.mark.skipif(not PROCESS_COUNTS, reason="fewer than two CPUs: no worker processes")
    def test_shared_memory_holds_units_never_the_columns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The decoded units cross ``/dev/shm`` a few at a time, as the plan counts them.

        Sampled from the shared-memory files the training process and its workers hold (only this
        build's, whatever else shares ``/dev/shm``), at every unit the training process takes.
        """
        workers = PROCESS_COUNTS[0]
        source = png_source(write_png_records(tmp_path / "png", 20 * _RECORDS))
        layout = source_layout(source)
        unit = _UNIT * sum(c.row_bytes for c in layout.columns)
        bound = workers * (2 * WORKER_BUFFER + 2) * unit + unit
        peaks: list[int] = []
        take = materialization.check_arrival

        def sampled(expected: int, arrived: int) -> None:
            pids = [os.getpid(), *spawned_children(os.getpid())]
            peaks.append(
                sum(
                    Path("/dev/shm", name).stat().st_size  # nosec B108
                    for name in shm_names_of(pids)
                    if name.startswith(_SEGMENT) and Path("/dev/shm", name).exists()  # nosec B108
                )
            )
            take(expected, arrived)

        monkeypatch.setattr(materialization, "check_arrival", sampled)

        materialize(source, host_resources=resources(workers), unit_records=_UNIT)

        assert peaks and max(peaks) <= bound, (peaks, bound)
        assert bound < layout.nbytes / 5, "the bound is far below the columns, which never cross"

    def test_positive_control_the_columns_held_in_shared_memory_exceed_the_bound(
        self, png_paths: list[str]
    ) -> None:
        if not Path("/dev/shm").is_dir():  # nosec B108
            pytest.skip("no /dev/shm to inspect (not Linux)")
        from multiprocessing import shared_memory  # noqa: PLC0415 - the control's own segment

        layout = source_layout(png_source(png_paths))
        unit = _UNIT * sum(c.row_bytes for c in layout.columns)
        bound = 2 * (2 * WORKER_BUFFER + 2) * unit + unit
        segment = shared_memory.SharedMemory(create=True, size=layout.nbytes * 8)
        try:
            held = sum(
                Path("/dev/shm", name).stat().st_size  # nosec B108
                for name in shm_names_of([os.getpid()])
                if name.startswith(_SEGMENT)
            )
        finally:
            segment.close()
            segment.unlink()
        assert held > bound


class TestWorkersByReadKind:
    @pytest.mark.parametrize("budget", [False, True])
    def test_a_gil_free_source_is_decoded_on_the_calling_thread(
        self, tmp_path: Path, started: _Workers, budget: bool
    ) -> None:
        memory = MemorySource(MemorySourceConfig(), {"x": np.arange(30.0)})
        write_copy(memory, tmp_path / "copy", identity={"source": {"k": 1}, "decoder": {}})
        for source in (memory, open_decoded_copy(tmp_path / "copy")):
            cap = min(2, available_cpus())
            result = materialize(
                source, host_resources=resources(cap) if budget else None, unit_records=_UNIT
            )
            _same(every_record(result)[0], every_record(source)[0])
        assert started.counts == []

    @pytest.mark.usefixtures("still_main_bytes")
    def test_a_gil_free_build_reads_nothing_ahead_so_its_budget_holds_no_read_buffer(
        self, started: _Workers
    ) -> None:
        """On the calling thread a build holds its unit, not a pipeline's 16-unit read buffer."""
        rows = np.zeros((4 * _UNIT, 1 << 16), np.float32)
        unit = _UNIT * rows[0].nbytes
        source = MemorySource(MemorySourceConfig(), {"x": rows})
        budget = HostResources(
            ram_budget_bytes=STILL_MAIN_BYTES + rows.nbytes + 4 * unit, max_workers=1
        )

        result = materialize(source, host_resources=budget, unit_records=_UNIT)

        _same(every_record(result)[0], every_record(source)[0])
        assert started.counts == []

    @pytest.mark.parametrize("workers", PROCESS_COUNTS)
    def test_a_gil_bound_source_with_a_budget_takes_the_planned_workers(
        self, png_paths: list[str], started: _Workers, workers: int
    ) -> None:
        materialize(png_source(png_paths), host_resources=resources(workers), unit_records=_UNIT)

        assert started.counts == [workers]

    def test_one_admitted_worker_builds_on_the_calling_thread_saying_why(
        self, png_paths: list[str], started: _Workers, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger=materialization.__name__):
            materialize(png_source(png_paths), host_resources=resources(1), unit_records=_UNIT)

        assert started.counts == []
        assert "slower than the calling thread" in caplog.text

    def test_without_a_budget_a_gil_bound_source_is_decoded_here_naming_the_budget(
        self, png_paths: list[str], started: _Workers, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            materialize(png_source(png_paths), unit_records=_UNIT)

        assert started.counts == []
        assert [r.getMessage() for r in caplog.records].count(
            "ArrayRecordSourceModule decodes in Python on each read, on one thread: pass "
            "materialize(host_resources=HostResources(ram_budget_bytes=..., max_workers=...)) to "
            "read it in worker processes"
        ) == 1


class TestRefusals:
    def test_a_budget_below_what_the_build_holds_is_refused_before_a_worker_starts(
        self, png_paths: list[str], started: _Workers
    ) -> None:
        budget = HostResources(ram_budget_bytes=1 << 20, max_workers=1, worker_bytes=1)

        with pytest.raises(ValueError, match=r"C=\d+.*per batch.*directory="):
            materialize(png_source(png_paths), host_resources=budget, unit_records=_UNIT)
        assert started.counts == []

    @pytest.mark.tfds
    def test_a_stream_is_refused(self, tfds_fixture: TFDSFixture) -> None:
        stream = from_tfds(FIXTURE, "train", data_dir=str(tfds_fixture.tfrecord))

        with pytest.raises(TypeError, match="INDEXED"):
            materialize(stream)

    def test_a_shard_is_refused(self) -> None:
        shard = MemorySource(MemorySourceConfig(num_workers=2, shard_id=0), {"x": np.arange(10.0)})

        with pytest.raises(TypeError, match="shard"):
            materialize(shard)

    def test_a_copy_of_a_source_without_an_identity_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="identity"):
            materialize(
                MemorySource(MemorySourceConfig(), {"x": np.arange(4.0)}), directory=tmp_path
            )


@pytest.mark.tfds
class TestTheTwoRefusals:
    def _decoded_size(self, fixture: TFDSFixture) -> int:
        source = from_tfds(FIXTURE, "train", data_dir=str(fixture.array_record), in_memory=False)
        return source_layout(source).nbytes

    def test_a_budget_between_them_builds_eagerly_and_is_refused_by_materialize(
        self, tfds_fixture: TFDSFixture, started: _Workers
    ) -> None:
        budget = HostResources(
            ram_budget_bytes=self._decoded_size(tfds_fixture) + 1, max_workers=1, worker_bytes=1
        )

        eager = from_tfds(
            FIXTURE, "train", data_dir=str(tfds_fixture.array_record), host_resources=budget
        )
        per_batch = from_tfds(
            FIXTURE, "train", data_dir=str(tfds_fixture.array_record), in_memory=False
        )
        with pytest.raises(ValueError, match=r"M_main=\d+.*C=\d+.*per batch.*directory="):
            materialize(per_batch, host_resources=budget)
        assert known_length(eager)
        assert started.counts == []

    def test_a_budget_below_the_decoded_size_refuses_both_each_by_its_rule(
        self, tfds_fixture: TFDSFixture
    ) -> None:
        budget = HostResources(
            ram_budget_bytes=self._decoded_size(tfds_fixture) - 1, max_workers=1, worker_bytes=1
        )

        with pytest.raises(ValueError, match=r"decodes to \d+ bytes in memory"):
            from_tfds(
                FIXTURE, "train", data_dir=str(tfds_fixture.array_record), host_resources=budget
            )
        per_batch = from_tfds(
            FIXTURE, "train", data_dir=str(tfds_fixture.array_record), in_memory=False
        )
        with pytest.raises(ValueError, match="before reading ahead"):
            materialize(per_batch, host_resources=budget)
