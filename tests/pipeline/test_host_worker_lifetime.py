"""Grain worker processes live once per run, and leave nothing behind.

A run's workers start with its first unit and serve every epoch or pass of the run: a second
epoch reuses them, so it waits for no start-up (0.7-1.8 s a run). The run's end, ``close()``,
dropping its iterators, ``reset()`` and a ``set_state`` that moves the cursor stop them, and no
shared-memory segment of theirs outlives the run. While iterating, each worker holds at most the
units its queues hold in shared memory.
"""

from __future__ import annotations

import gc
import threading
import time
from pathlib import Path
from typing import Any

import jax
import pytest
from flax import nnx
from substrax.testing import run_python

from datarax.pipeline import Pipeline
from datarax.pipeline.read_plan import HostReadPath
from tests.test_common.worker_reads import (
    png_source,
    resources,
    worker_children,
    write_png_records,
)


_ROOT = Path(__file__).resolve().parents[2]
_SHM = Path("/dev/shm")
_SEGMENT = "psm_"
"""The name prefix of Python's shared-memory segments, which hold Grain's units."""

linux_only = pytest.mark.skipif(not _SHM.is_dir(), reason="no /dev/shm to inspect (not Linux)")


@pytest.fixture(scope="module")
def png_paths(tmp_path_factory: pytest.TempPathFactory) -> list[str]:
    """40 PNG records in two ArrayRecord files."""
    return write_png_records(tmp_path_factory.mktemp("png"), 40)


def _pipeline(paths: list[str], *, workers: int = 2, num_epochs: int | None = 2) -> Pipeline:
    return Pipeline(
        source=png_source(paths),
        stages=[],
        batch_size=4,
        rngs=nnx.Rngs(1),
        shuffle=True,
        num_epochs=num_epochs,
        host_resources=resources(workers),
    )


def _segments(prefix: str = "") -> set[str]:
    """Names in ``/dev/shm``: shared-memory segments (``psm_``) and multiprocessing's semaphores."""
    if not _SHM.is_dir():
        return set()
    return {path.name for path in _SHM.iterdir() if path.name.startswith(prefix)}


def _settled(check: Any, seconds: float = 30.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        gc.collect()
        if check():
            return True
        threading.Event().wait(0.05)
    return False


class TestWorkersLiveOncePerRun:
    """W2b-22's runs are long: workers serve every epoch of the run, started once."""

    def test_the_second_epoch_reuses_the_workers_without_their_start_up(
        self, png_paths: list[str]
    ) -> None:
        pipe = _pipeline(png_paths)
        per_epoch = len(pipe) // 2
        started = time.perf_counter()
        batches = iter(pipe.raw_batches())
        next(batches)
        start_up = time.perf_counter() - started
        first = {child.pid for child in worker_children()}
        for _ in range(per_epoch - 1):
            next(batches)
        waited = time.perf_counter()
        next(batches)  # the second epoch's first unit
        epoch_two = time.perf_counter() - waited
        second = {child.pid for child in worker_children()}
        assert len(first) == 2
        assert second == first
        assert epoch_two < start_up / 2, (epoch_two, start_up)
        pipe.close()

    def test_two_calls_of_iter_continue_one_run_with_the_same_workers(
        self, png_paths: list[str]
    ) -> None:
        pipe = _pipeline(png_paths)
        batches = iter(pipe)
        next(batches)
        first = {child.pid for child in worker_children()}
        next(iter(pipe))
        assert {child.pid for child in worker_children()} == first
        assert pipe.host_plan.path is HostReadPath.PROCESSES
        pipe.close()


class TestNothingOutlivesTheRun:
    """No worker and no shared-memory segment survives what ends a run."""

    @pytest.mark.parametrize("ending", ["exhausted", "close", "dropped", "reset", "set_state"])
    @linux_only
    def test_workers_and_segments_go_with_the_run(self, png_paths: list[str], ending: str) -> None:
        before = _segments()
        pipe = _pipeline(png_paths, num_epochs=1)
        state = pipe.get_state()
        batches = iter(pipe.raw_batches())
        held = [next(batches) for _ in range(3)]
        assert len(worker_children()) == 2
        if ending == "exhausted":
            held.extend(batches)
        elif ending == "close":
            pipe.close()
        elif ending == "dropped":
            del batches
            pipe = None  # the run's last holders
        elif ending == "reset":
            pipe.reset()
        else:
            pipe.set_state(state)
        del held
        assert _settled(lambda: not worker_children()), [c.name for c in worker_children()]
        assert _settled(lambda: not _segments() - before), sorted(_segments() - before)

    @linux_only
    def test_positive_control_a_segment_not_yet_unlinked_is_seen(self) -> None:
        from multiprocessing import shared_memory  # noqa: PLC0415

        before = _segments()
        segment = shared_memory.SharedMemory(create=True, size=4096)
        try:
            assert segment.name in _segments() - before
        finally:
            segment.close()
            segment.unlink()
        assert segment.name not in _segments()

    @linux_only
    def test_steady_state_segments_are_the_units_the_queues_hold(
        self, png_paths: list[str]
    ) -> None:
        """At most ``2o + 2`` units per worker in shared memory, plus ``d + 1`` at the device.

        Counted as Python's shared-memory segments (``psm_``): the run's multiprocessing queues,
        events and locks keep POSIX semaphores in ``/dev/shm`` too (``sem.mp-*``, about 25 a
        worker, measured), which hold no unit.
        """
        before = _segments(_SEGMENT)
        pipe = _pipeline(png_paths, num_epochs=None)
        batches = iter(pipe.raw_batches())
        first = next(batches)
        leaves = len(jax.tree.leaves(first))
        del first  # the consumer holds one unit: the one it takes next
        plan = pipe.host_plan
        bound = (plan.workers * (2 * plan.worker_buffer + 2) + plan.device_buffer + 1) * leaves
        highest = 0
        for _ in range(30):
            next(batches)
            threading.Event().wait(0.02)  # a step's time, in which the queues fill
            highest = max(highest, len(_segments(_SEGMENT) - before))
        pipe.close()
        assert 0 < highest <= bound, (highest, bound)


_EXIT_WITH_A_RUN_OPEN = """
import sys
from flax import nnx
from datarax.pipeline import Pipeline
from tests.test_common.worker_reads import png_source, resources

if __name__ == "__main__":
    pipe = Pipeline(source=png_source(sys.argv[1:]), stages=[], batch_size=4, rngs=nnx.Rngs(0),
                    num_epochs=None, host_resources=resources(2))
    batches = iter(pipe.raw_batches())
    for _ in range(5):
        next(batches)
    print("exiting with the run open")
"""


def test_exiting_with_a_run_open_prints_no_ignored_exception(
    png_paths: list[str], tmp_path: Path
) -> None:
    script = tmp_path / "exit_open.py"
    script.write_text(_EXIT_WITH_A_RUN_OPEN)
    result = run_python(script, *png_paths, timeout=300, cwd=_ROOT)
    assert result.returncode == 0, result.stderr[-3000:]
    assert "exiting with the run open" in result.stdout
    assert "Exception ignored" not in result.stderr, result.stderr[-3000:]
    assert "Traceback" not in result.stderr, result.stderr[-3000:]
