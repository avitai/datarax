"""GIL-bound reads in Grain worker processes: the thread stream, resumable, and nothing lost.

With ``Pipeline(host_resources=...)`` a source whose host read holds the GIL (a Python decode per
batch: ArrayRecord with a decode callable, TFDS's per-batch read and stream, a mix of them) is
read in Grain worker processes (``mp_prefetch``), as many as the budget admits. Workers read and
decode; the consumer places, so everything after the read happens where it did on threads. The
stream does not depend on the worker count: every unit, its records, epochs, draws, data and
provenance equal the thread run's bit for bit (W2b-21), a run resumes exactly from the cursor
under another worker count (W2b-22), a record's draws do not depend on the count (W2b-07), and
every unit crosses the process boundary or the run fails loudly naming it (#1415).
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import grain
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from datarax.core.config import OperatorConfig
from datarax.core.data_source import DataSourceModule, HostRead
from datarax.core.element_batch import Element
from datarax.core.index_words import from_words
from datarax.core.operator import OperatorModule, require_key
from datarax.pipeline import Pipeline
from datarax.pipeline.read_plan import HostReadPath
from datarax.pipeline.run_units import RunUnits
from datarax.sources import from_tfds, TFDSStreamingConfig, TFDSStreamingSource
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from datarax.sources.mixed_source import MixDataSourcesConfig, MixDataSourcesNode
from datarax.sources.streaming_disk_source import StreamingDiskSource, StreamingDiskSourceConfig
from tests.test_common.streams import RecordStream
from tests.test_common.tfds_fixture import FIXTURE, TFDSFixture
from tests.test_common.worker_reads import (
    logged,
    png_source,
    PngDecoder,
    resources,
    stream_of,
    TensorDecoder,
    WORKER_COUNTS,
    write_png_records,
    write_tensor_records,
)


_RECORDS = 22
_BATCH = 4
_SECOND = 3 if 3 in WORKER_COUNTS else 1


@pytest.fixture(scope="module")
def png_paths(tmp_path_factory: pytest.TempPathFactory) -> list[str]:
    """22 PNG records in two ArrayRecord files."""
    return write_png_records(tmp_path_factory.mktemp("png"), _RECORDS)


@pytest.fixture(scope="module")
def tensor_paths(tmp_path_factory: pytest.TempPathFactory) -> list[str]:
    """10 TFDS-serialized examples of 16,384 floats in two ArrayRecord files."""
    return write_tensor_records(tmp_path_factory.mktemp("tensor"), 10)


def _pipeline(
    source: DataSourceModule,
    *,
    workers: int = 0,
    shuffle: bool = True,
    drop_last: bool = False,
    num_epochs: int | None = 2,
    batch_size: int = _BATCH,
    stages: Sequence[nnx.Module] = (),
) -> Pipeline:
    return Pipeline(
        source=source,
        stages=list(stages),
        batch_size=batch_size,
        rngs=nnx.Rngs(5),
        shuffle=shuffle,
        drop_last=drop_last,
        num_epochs=num_epochs,
        host_resources=resources(workers) if workers else None,
    )


def _check_processes(pipe: Pipeline, workers: int) -> None:
    plan = pipe.host_plan
    assert (plan.path, plan.workers) == (HostReadPath.PROCESSES, workers)


def _same_stream(
    make: Callable[[], DataSourceModule],
    workers: int,
    *,
    chunk: int | None = None,
    **options: Any,
) -> None:
    """The stream ``workers`` processes serve equals the thread stream, unit for unit."""
    reference = stream_of(_pipeline(make(), **options).raw_batches(chunk))
    pipe = _pipeline(make(), workers=workers, **options)
    served = stream_of(pipe.raw_batches(chunk))
    _check_processes(pipe, workers)
    assert served == reference
    pipe.close()


def _rule(shuffle: bool, drop_last: bool, *, slow: bool) -> Any:
    marks = [pytest.mark.slow] if slow else []
    return pytest.param(
        shuffle, drop_last, id=f"shuffle={shuffle}-drop_last={drop_last}", marks=marks
    )


_COUNTS = [
    pytest.param(k, id=f"workers={k}", marks=[pytest.mark.slow] if k > 2 else [])
    for k in WORKER_COUNTS
]
"""The fast lane reads with 1 and 2 workers; 3 and 8 run in the long-running job."""


class TestTheStreamDoesNotDependOnTheWorkers:
    """W2b-21: processes serve the thread run's units, bit for bit, whatever their count."""

    @pytest.mark.parametrize("workers", _COUNTS)
    @pytest.mark.parametrize(
        ("shuffle", "drop_last"),
        [
            _rule(True, False, slow=False),
            _rule(False, True, slow=True),
            _rule(True, True, slow=True),
        ],
    )
    def test_an_array_record_decode(
        self, png_paths: list[str], workers: int, shuffle: bool, drop_last: bool
    ) -> None:
        _same_stream(lambda: png_source(png_paths), workers, shuffle=shuffle, drop_last=drop_last)

    @pytest.mark.parametrize("workers", _COUNTS)
    def test_chunks_reshaped_in_the_worker(
        self, png_paths: list[str], workers: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reference = stream_of(_pipeline(png_source(png_paths)).raw_batches(3))

        def refused(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("a chunk is reshaped in the worker, not in the training process")

        # Patched in this process only: a worker imports the module afresh.
        monkeypatch.setattr("datarax.pipeline.host_stage.batch_ops.as_chunk", refused)
        pipe = _pipeline(png_source(png_paths), workers=workers)
        assert stream_of(pipe.raw_batches(3)) == reference
        pipe.close()

    def test_a_single_batch_run(self, png_paths: list[str]) -> None:
        _same_stream(lambda: png_source(png_paths[:1]), 2, batch_size=11, num_epochs=1)

    @pytest.mark.parametrize("names", [True, False])
    def test_provenance_equals_the_thread_run_s(self, png_paths: list[str], names: bool) -> None:
        def make() -> DataSourceModule:
            return png_source(png_paths, PngDecoder(names=names))

        reference = list(_pipeline(make()).raw_batches(with_provenance=True))
        pipe = _pipeline(make(), workers=2)
        served = list(pipe.raw_batches(with_provenance=True))
        assert [stream_of(iter([b])) for b, _ in served] == [
            stream_of(iter([b])) for b, _ in reference
        ]
        assert [[dict(r) for r in p] for _, p in served] == [
            [dict(r) for r in p] for _, p in reference
        ]
        for _, provenance in served:
            assert all(isinstance(record, MappingProxyType) for record in provenance)
        pipe.close()

    def test_under_x64_a_float64_column_is_placed_as_float64(self, png_paths: list[str]) -> None:
        """A worker reads in the caller's precision mode, which it does not inherit."""
        with jax.enable_x64(True):
            decoder = PngDecoder(float64=True)
            reference = list(_pipeline(png_source(png_paths, decoder)).raw_batches())
            pipe = _pipeline(png_source(png_paths, PngDecoder(float64=True)), workers=2)
            served = list(pipe.raw_batches())
            assert {b["scale"].dtype for b in served} == {jnp.dtype("float64")}
            assert stream_of(iter(served)) == stream_of(iter(reference))
            pipe.close()

    def test_a_mix_of_array_record_children(self, png_paths: list[str]) -> None:
        """A mix reads as its strictest child: a GIL-bound child sends it to processes."""

        def make() -> DataSourceModule:
            return MixDataSourcesNode(
                MixDataSourcesConfig(weights=(1.0, 2.0)),
                [png_source(png_paths[:1]), png_source(png_paths[1:])],
            )

        assert make().host_read is HostRead.GIL_BOUND
        _same_stream(make, 2)

    def test_a_mix_of_in_memory_children_stays_on_threads(self) -> None:
        mix = MixDataSourcesNode(
            MixDataSourcesConfig(weights=(1.0, 1.0)),
            [
                MemorySource(MemorySourceConfig(), {"x": np.arange(8, dtype=np.float32)}),
                MemorySource(MemorySourceConfig(), {"x": np.arange(6, dtype=np.float32)}),
            ],
        )
        assert mix.host_read is HostRead.GIL_FREE
        pipe = _pipeline(mix, workers=2)
        list(pipe.raw_batches())
        assert pipe.host_plan.path is HostReadPath.THREADS
        pipe.close()

    def test_a_stream_read_pass_by_pass_plans_one_thread(self) -> None:
        def make() -> RecordStream:
            return RecordStream({"x": np.arange(10, dtype=np.float32)})

        reference = stream_of(_pipeline(make(), shuffle=False).raw_batches())
        pipe = _pipeline(make(), shuffle=False, workers=2)
        assert stream_of(pipe.raw_batches()) == reference
        assert (pipe.host_plan.path, pipe.host_plan.threads) == (HostReadPath.ONE_THREAD, 1)
        pipe.close()


class TestNonImageAndLargeRecords:
    """A non-image kind (TFDS-serialized float tensors) through processes, and large records."""

    @pytest.mark.parametrize("workers", [1, 2])
    def test_the_tensor_kind_through_processes_equals_threads(
        self, tensor_paths: list[str], workers: int
    ) -> None:
        def make() -> DataSourceModule:
            return png_source(tensor_paths, TensorDecoder())

        _same_stream(make, workers, batch_size=3)

    def test_a_large_record_memmap_plans_threads(self, tmp_path: Path) -> None:
        path = tmp_path / "rows.npy"
        np.save(path, np.zeros((16, 256, 1024), np.float32))  # 1 MiB records
        source = StreamingDiskSource(StreamingDiskSourceConfig(path=str(path), feature_key="x"))
        assert source.host_read is HostRead.GIL_FREE
        pipe = _pipeline(source, workers=2)
        next(iter(pipe.raw_batches()))
        assert pipe.host_plan.path is HostReadPath.THREADS
        pipe.close()


def _wait_until(predicate: Callable[[], bool], seconds: float = 60.0) -> None:
    deadline = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < deadline, "the workers never read ahead"
        threading.Event().wait(0.05)


class TestExactResume:
    """W2b-22: a run cut with workers reading ahead resumes exactly, under another count."""

    @pytest.mark.parametrize("cut", [1, 3, 6])
    @pytest.mark.parametrize("depth", [0, 1])
    def test_head_and_tail_equal_the_uninterrupted_stream(
        self, png_paths: list[str], tmp_path: Path, cut: int, depth: int
    ) -> None:
        whole = stream_of(_pipeline(png_source(png_paths)).raw_batches())
        log = tmp_path / f"reads-{cut}-{depth}.log"
        pipe = _pipeline(png_source(png_paths, PngDecoder(log=str(log))), workers=2)
        pipe.host_stage._device_buffer = depth  # the deque's staged units are not served
        batches = iter(pipe.raw_batches())
        head = [next(batches) for _ in range(cut)]
        served_records = cut * _BATCH
        _wait_until(lambda: len(logged(log)) > served_records + (depth + 1) * _BATCH)
        state = pipe.get_state()
        pipe.close()
        assert set(state) == {
            "version",
            "kind",
            "epoch",
            "position",
            "run_end_epoch",
            "stream",
            "fingerprint",
        }
        assert state["version"] == 3

        resumed = _pipeline(png_source(png_paths), workers=_SECOND)
        resumed.set_state(state)
        tail = stream_of(resumed.raw_batches())
        assert stream_of(iter(head)) + tail == whole
        resumed.close()

    def test_a_resume_from_threads_to_processes_and_back(self, png_paths: list[str]) -> None:
        whole = stream_of(_pipeline(png_source(png_paths)).raw_batches())
        threads = _pipeline(png_source(png_paths))
        batches = iter(threads.raw_batches())
        head = [next(batches) for _ in range(4)]
        state = threads.get_state()
        threads.close()
        processes = _pipeline(png_source(png_paths), workers=2)
        processes.set_state(state)
        batches = iter(processes.raw_batches())
        middle = [next(batches) for _ in range(3)]
        state = processes.get_state()
        processes.close()
        back = _pipeline(png_source(png_paths))
        back.set_state(state)
        assert stream_of(iter(head + middle)) + stream_of(back.raw_batches()) == whole


class _Jitter(OperatorModule):
    """A stochastic operator adding a per-record draw to ``image`` (as float)."""

    def __init__(self) -> None:
        super().__init__(
            OperatorConfig(stochastic=True, stream_name="augment"), rngs=nnx.Rngs(augment=11)
        )

    def apply(self, element: Element, key: jax.Array | None = None, stats: Any = None) -> Element:
        del stats
        noise = jax.random.uniform(require_key(key, self), ())
        return element.update_data({"image": element.data["image"].astype(jnp.float32) + noise})


def _per_record(batches: Any) -> dict[tuple[int, int], bytes]:
    """Each served row's output, by its record and epoch."""
    rows = {}
    for batch in batches:
        host = jax.device_get(batch)
        names = from_words(np.asarray(host.indices))
        for row, (name, epoch) in enumerate(zip(names, np.asarray(host.epochs), strict=True)):
            rows[int(name), int(epoch)] = np.asarray(host["image"][row]).tobytes()
    return rows


class TestDrawsDoNotDependOnTheWorkers:
    """W2b-07: a record's draws depend on its identity, never on the worker count."""

    @pytest.mark.slow
    def test_tier_a_and_the_dag_in_a_step_draw_alike_for_0_2_and_4_workers(
        self, png_paths: list[str]
    ) -> None:
        counts = [0, 2] + ([4] if 4 in range(1, os.cpu_count() or 1) else [])
        tier_a, in_step = {}, {}
        for workers in counts:
            pipe = _pipeline(png_source(png_paths), workers=workers, stages=[_Jitter()])
            tier_a[workers] = _per_record(iter(pipe))
            pipe.close()
            pipe = _pipeline(png_source(png_paths), workers=workers, stages=[_Jitter()])
            step = jax.jit(lambda dag, batch: dag(batch))
            in_step[workers] = _per_record(step(pipe.dag, b) for b in pipe.raw_batches())
            pipe.close()
        for workers in counts[1:]:
            assert tier_a[workers] == tier_a[0]
            assert in_step[workers] == in_step[0]
        assert tier_a[0] == in_step[0]

    def test_draws_survive_a_resume_under_another_count(self, png_paths: list[str]) -> None:
        whole = _per_record(iter(_pipeline(png_source(png_paths), stages=[_Jitter()])))
        pipe = _pipeline(png_source(png_paths), workers=2, stages=[_Jitter()])
        batches = iter(pipe)
        head = [next(batches) for _ in range(3)]
        state = pipe.get_state()
        pipe.close()
        resumed = _pipeline(png_source(png_paths), workers=_SECOND, stages=[_Jitter()])
        resumed.set_state(state)
        assert {**_per_record(iter(head)), **_per_record(iter(resumed))} == whole
        resumed.close()


class TestNothingIsLostCrossingTheBoundary:
    """#1415: Grain's queue drops an element it cannot pickle; datarax never serves a short run."""

    @pytest.mark.parametrize("names", [True, False])
    def test_every_unit_arrives_with_its_provenance(
        self, png_paths: list[str], names: bool
    ) -> None:
        pipe = _pipeline(png_source(png_paths, PngDecoder(names=names)), workers=2, num_epochs=3)
        units = RunUnits(run=pipe.host_stage._run(pipe.epoch_plan)).units()  # noqa: SLF001
        served = list(pipe.raw_batches(with_provenance=True))
        assert len(served) == units
        for batch, provenance in served:
            assert len(provenance) == batch.batch_size
            if names:
                labels = np.asarray(jax.device_get(batch["label"]))
                assert [r["name"] for r in provenance] == [f"record-{v}" for v in labels]
            else:
                assert all(dict(record) == {} for record in provenance)
        pipe.close()

    def test_an_element_that_cannot_cross_stops_the_run_naming_its_unit(
        self, png_paths: list[str]
    ) -> None:
        decoder = PngDecoder(unpicklable=9)
        pipe = _pipeline(png_source(png_paths, decoder), workers=2, shuffle=False, num_epochs=1)
        served = []
        with pytest.raises((TypeError, RuntimeError), match=r"unit 2\b"):
            for batch in pipe.raw_batches(with_provenance=True):
                served.append(batch)
        assert len(served) == 2  # units 0 and 1 arrived; unit 2 holds record 9
        pipe.close()

    def test_a_run_ending_short_is_refused_naming_the_missing_unit(
        self, png_paths: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The training process counts units: a unit lost on the way is never a short run."""
        original = grain.IterDataset.mp_prefetch

        def dropping(self: Any, *args: Any, **kwargs: Any) -> Any:
            return _Dropping(original(self, *args, **kwargs), drop=3)

        monkeypatch.setattr(grain.IterDataset, "mp_prefetch", dropping)
        pipe = _pipeline(png_source(png_paths), workers=2, num_epochs=1)
        with pytest.raises(RuntimeError, match=r"unit 3\b"):
            list(pipe.raw_batches())
        pipe.close()


class _Dropping(grain.IterDataset):
    """A dataset losing one element, as Grain's process queue loses one it cannot pickle."""

    def __init__(self, parent: grain.IterDataset, *, drop: int) -> None:
        super().__init__(parent)
        self._drop = drop

    def __iter__(self) -> _DroppingIterator:
        return _DroppingIterator(iter(self._parent), self._drop)


class _DroppingIterator(grain.DatasetIterator):
    def __init__(self, parent: grain.DatasetIterator, drop: int) -> None:
        super().__init__(parent)
        self._drop, self._seen = drop, 0

    def __next__(self) -> Any:
        element = next(self._parent)
        self._seen += 1
        return next(self._parent) if self._seen - 1 == self._drop else element

    def get_state(self) -> dict[str, Any]:
        return {}

    def set_state(self, state: dict[str, Any]) -> None:
        del state


@pytest.mark.tfds
class TestTFDSThroughProcesses:
    """The per-batch TFDS read and the TFDS stream, through processes, as on threads."""

    def test_the_per_batch_read_equals_threads(self, tfds_fixture: TFDSFixture) -> None:
        def make() -> DataSourceModule:
            return from_tfds(
                FIXTURE, "train", data_dir=str(tfds_fixture.array_record), in_memory=False
            )

        assert make().host_read is HostRead.GIL_BOUND
        _same_stream(make, 2)

    @pytest.mark.parametrize("workers", WORKER_COUNTS)
    def test_the_stream_through_option_e_equals_threads(
        self, tfds_fixture: TFDSFixture, workers: int
    ) -> None:
        def make() -> DataSourceModule:
            config = TFDSStreamingConfig(
                name=FIXTURE, split="train", data_dir=str(tfds_fixture.tfrecord)
            )
            return TFDSStreamingSource(config)

        assert make().host_read is HostRead.GIL_BOUND
        _same_stream(make, workers, batch_size=3)

    def test_stream_provenance_equals_threads(self, tfds_fixture: TFDSFixture) -> None:
        def make() -> DataSourceModule:
            config = TFDSStreamingConfig(
                name=FIXTURE, split="train", data_dir=str(tfds_fixture.tfrecord)
            )
            return TFDSStreamingSource(config)

        reference = list(_pipeline(make(), batch_size=3).raw_batches(with_provenance=True))
        pipe = _pipeline(make(), workers=2, batch_size=3)
        served = list(pipe.raw_batches(with_provenance=True))
        assert [[dict(r) for r in p] for _, p in served] == [
            [dict(r) for r in p] for _, p in reference
        ]
        pipe.close()

    def test_a_stream_resumed_mid_pass_reads_nothing_before_the_cut(
        self, tfds_fixture: TFDSFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cut at unit 5 (not a multiple of 2 or 3): no payload before it is read again.

        One pass, so every record the resumed run reads is of the pass the cut stopped in (a later
        pass reads every record again, by design).
        """

        def make() -> DataSourceModule:
            config = TFDSStreamingConfig(
                name=FIXTURE, split="train", data_dir=str(tfds_fixture.tfrecord)
            )
            return TFDSStreamingSource(config)

        whole = stream_of(_pipeline(make(), batch_size=3, num_epochs=1).raw_batches())
        pipe = _pipeline(make(), workers=2, batch_size=3, num_epochs=1)
        batches = iter(pipe.raw_batches())
        head = [next(batches) for _ in range(5)]
        state = pipe.get_state()
        pipe.close()

        log = tmp_path / "payloads.log"
        spy = Path(__file__).resolve().parents[1] / "test_common" / "payload_spy"
        monkeypatch.setenv("DATARAX_TEST_PAYLOAD_LOG", str(log))
        monkeypatch.setenv("PYTHONPATH", f"{spy}{os.pathsep}{os.environ.get('PYTHONPATH', '')}")
        resumed = _pipeline(make(), workers=_SECOND, batch_size=3, num_epochs=1)
        resumed.set_state(state)
        tail = stream_of(resumed.raw_batches())
        resumed.close()
        assert stream_of(iter(head)) + tail == whole

        files = TFDSStreamingSource(
            TFDSStreamingConfig(name=FIXTURE, split="train", data_dir=str(tfds_fixture.tfrecord))
        ).shard_files
        cut = {
            (files[int(shard)], int(offset))
            for batch in head
            for (shard, offset), epoch in zip(
                np.asarray(jax.device_get(batch.indices)),
                np.asarray(jax.device_get(batch.epochs)),
                strict=True,
            )
            if int(epoch) == state["stream"]["pass"]
        }
        assert cut, "the cut serves records of the pass it stopped in"
        read = logged_payloads(log)
        assert read, "the spy saw no payload read: the instrument is broken"
        assert not cut & set(read)


def logged_payloads(log: Path) -> list[tuple[str, int]]:
    """``(shard file, offset)`` of every payload the worker spy logged."""
    if not log.exists():
        return []
    pairs = (line.split() for line in log.read_text().splitlines() if line)
    return [(name, int(offset)) for name, offset in pairs]
