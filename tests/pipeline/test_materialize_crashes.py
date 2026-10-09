"""A build that dies leaves no copy a reader takes for finished, and its rerun starts afresh.

A copy is finished only when its manifest is written, last; a writer killed at any point leaves
``.partial`` files or files without a manifest, which every reader refuses, and the next build
writes every file anew (new inodes), so a writer still holding a map of an old file cannot reach
the new copy. Workers a killed training process leaves behind end with it (the parent watcher).
Grain's shared-memory defects (google/grain#1420) are recorded as they show, never suppressed:
the in-place disk build sends no rows through shared memory and shows none; the RAM build, whose
units cross shared memory, shows at most one leaked segment per column and worker.

Every process here is started by the test and killed only by the process id it reports.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import signal
import subprocess  # nosec B404 - the tests start their own builder processes
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from datarax.core.host_resources import available_cpus
from datarax.pipeline import materialize
from datarax.sources import open_decoded_copy
from tests.test_common.decoded_copies import write_copy
from tests.test_common.worker_reads import (
    is_gone,
    png_source,
    write_png_records,
)


_RECORDS = 40
_UNIT = 4
_WORKERS = 2
_LEAK = "leaked shared_memory objects"

pytestmark = pytest.mark.skipif(
    available_cpus() < _WORKERS, reason="fewer than two CPUs: no worker processes"
)


@pytest.fixture(scope="module")
def png_paths(tmp_path_factory: pytest.TempPathFactory) -> list[str]:
    return write_png_records(tmp_path_factory.mktemp("png"), _RECORDS, shards=2)


def _files(directory: Path) -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.iterdir())
        if p.name.endswith((".npy", ".array_record"))
    }


_PRELUDE = """
import json, os, signal, sys
import psutil
from datarax.pipeline import materialization, materialize
from tests.test_common.worker_reads import png_source, PngDecoder, resources, spawned_children
paths, directory, workers, unit = json.loads(sys.argv[1])

def report(**facts):
    print(json.dumps(facts), flush=True)

def die():
    os.kill(os.getpid(), signal.SIGKILL)
"""

_SCRIPTS = {
    "worker": """
source = png_source(paths, PngDecoder(kill_at=13))
materialize(source, host_resources=resources(workers), directory=directory, unit_records=unit)
""",
    "before manifest": """
replace = os.replace
def replacing(src, dst):
    if str(dst).endswith("manifest.json"):
        die()
    replace(src, dst)
os.replace = replacing
materialize(png_source(paths), host_resources=resources(workers), directory=directory,
            unit_records=unit)
""",
    "mid": """
arrival = materialization.check_arrival
def arriving(expected, unit_):
    if unit_ == 3:
        children = spawned_children(os.getpid())
        rss = {pid: psutil.Process(pid).memory_info().rss for pid in children}
        report(workers=children, rss=rss)
        die()
    arrival(expected, unit_)
materialization.check_arrival = arriving
materialize(png_source(paths), host_resources=resources(workers), directory=directory,
            unit_records=unit)
""",
}


def _run(
    script: str, paths: list[str], directory: Path, logs: Path
) -> subprocess.CompletedProcess[str]:
    """Run a builder script; its output goes to files, which a surviving worker cannot hold open."""
    arguments = json.dumps([paths, str(directory), _WORKERS, _UNIT])
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "stdout").open("w") as stdout, (logs / "stderr").open("w") as stderr:
        run = subprocess.run(  # nosec B603 - fixed arguments
            [sys.executable, "-c", textwrap.dedent(_PRELUDE) + textwrap.dedent(script), arguments],
            stdout=stdout,
            stderr=stderr,
            timeout=300,
            check=False,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
    return subprocess.CompletedProcess(
        run.args, run.returncode, (logs / "stdout").read_text(), (logs / "stderr").read_text()
    )


def _reports(run: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    for line in run.stdout.splitlines():
        if line.startswith("{"):
            facts.update(json.loads(line))
    return facts


_ENDS_WITHIN = 120.0
"""Seconds a watched worker has to end: a liveness bound, never a speed. A worker spawned as its
training process dies ends once it reaches the watcher, after its interpreter starts and imports
(0.9 s on two cores here), so this fails only when the watcher does not end it at all."""

_UNWATCHED_WAIT = 2.0
"""Seconds an unwatched worker is watched for: it never ends by itself, so it is alive after any
wait, on any machine."""


def _end_survivors(pids: list[int], *, within: float) -> list[int]:
    """Every listed process still running after ``within`` seconds, killed by its id.

    The wait ends as soon as every process is gone.

    Args:
        pids: The processes, by the ids they reported.
        within: Seconds to wait for them to end.

    Returns:
        The ids that survived the wait.
    """
    deadline = time.monotonic() + within
    while time.monotonic() < deadline and not all(is_gone(pid) for pid in pids):
        time.sleep(0.05)
    survivors = [pid for pid in pids if not is_gone(pid)]
    for pid in survivors:
        os.kill(pid, signal.SIGKILL)
    return survivors


class TestAKilledBuild:
    @pytest.mark.parametrize("where", list(_SCRIPTS))
    def test_leaves_no_finished_copy_and_its_rerun_writes_fresh_identical_files(
        self, png_paths: list[str], tmp_path: Path, where: str
    ) -> None:
        directory = tmp_path / "copy"
        reference = tmp_path / "reference"
        write_copy(png_source(png_paths), reference, unit_records=_UNIT)

        run = _run(_SCRIPTS[where], png_paths, directory, tmp_path / "logs")

        facts = _reports(run)
        if where == "worker":
            assert run.returncode not in (0, -signal.SIGKILL), run.stderr[-2000:]
            assert "terminated unexpectedly" in run.stderr, run.stderr[-2000:]
        else:
            assert run.returncode == -signal.SIGKILL, run.stderr[-2000:]
        if where == "mid":
            assert facts["workers"], "the build had started its workers"
            survivors = _end_survivors(facts["workers"], within=_ENDS_WITHIN)
            assert survivors == [], f"orphans (rss {facts['rss']})"
        with pytest.raises(FileNotFoundError, match=r"no manifest\.json"):
            open_decoded_copy(directory)
        # A killed process leaves its files; a worker's crash reaches the build as an error, whose
        # exit removes the unfinished files itself.
        leftovers = {p.name: p.stat().st_ino for p in directory.iterdir() if p.name != ".lock"}
        assert bool(leftovers) is (where != "worker"), sorted(leftovers)
        # Held open, the old inodes cannot be reused, so a new file is a new inode; a writer
        # that outlived the build keeps a map of an old file.
        with contextlib.ExitStack() as held:
            for name in leftovers:
                held.enter_context((directory / name).open("r+b"))
            stale = [
                np.memmap(directory / name, mode="r+", dtype=np.uint8)
                for name in leftovers
                if name.endswith(".npy.partial") and (directory / name).stat().st_size
            ]

            materialize(png_source(png_paths), directory=directory, unit_records=_UNIT)

            for old in stale:
                old[-64:] = 0xAB  # writes through an old map reach no file of the new copy
                old.flush()
            renewed = {p.name: p.stat().st_ino for p in directory.iterdir()}
            del stale
        assert _files(directory) == _files(reference)
        assert not [n for n in leftovers if n in renewed and renewed[n] == leftovers[n]]

    def test_positive_control_without_the_watcher_the_killed_build_s_workers_live_on(
        self, png_paths: list[str], tmp_path: Path
    ) -> None:
        script = (
            "from datarax.pipeline import host_workers\n"
            "from tests.test_common.worker_reads import UnwatchedSetUp\n"
            "host_workers.WorkerSetUp = UnwatchedSetUp\n" + _SCRIPTS["mid"]
        )

        run = _run(script, png_paths, tmp_path / "copy", tmp_path / "logs")

        workers = _reports(run)["workers"]
        survivors = _end_survivors(workers, within=_UNWATCHED_WAIT)
        assert survivors == workers, "unwatched workers outlive their parent"


@pytest.mark.skipif(not Path("/dev/shm").is_dir(), reason="no /dev/shm (not Linux)")  # nosec B108
class TestGrainSharedMemorySymptoms:
    """google/grain#1420, recorded as it shows. The expectation goes once Grain fixes it."""

    _BUILD = """
import gc
directory = None if directory == "-" else directory
names = set()
arrival = materialization.check_arrival
def arriving(expected, unit_):
    for p in [os.getpid(), *spawned_children(os.getpid())]:
        try:
            names.update(os.path.basename(m.path) for m in psutil.Process(p).memory_maps()
                         if m.path.startswith("/dev/shm/psm_"))
        except psutil.Error:
            pass
    arrival(expected, unit_)
materialization.check_arrival = arriving
materialize(png_source(paths), host_resources=resources(workers), directory=directory,
            unit_records=unit)
report(names=sorted(names))
"""

    def test_the_in_place_disk_build_shows_none(self, png_paths: list[str], tmp_path: Path) -> None:
        run = _run(self._BUILD, png_paths, tmp_path / "copy", tmp_path / "logs")

        assert run.returncode == 0, run.stderr[-2000:]
        assert run.stderr.count(_LEAK) == 0
        assert _reports(run)["names"] == [], "no unit crossed shared memory (psm_ segments)"

    def test_the_ram_build_shows_at_most_a_unit_per_column_and_worker(
        self, png_paths: list[str], tmp_path: Path
    ) -> None:
        run = _run(self._BUILD, png_paths, Path("-"), tmp_path / "logs")

        assert run.returncode == 0, run.stderr[-2000:]
        leaked = sum(int(n) for n in re.findall(r"There appear to be (\d+) " + _LEAK, run.stderr))
        names = _reports(run)["names"]
        print(f"grain#1420: {leaked} leaked segments reported, {len(names)} seen")  # noqa: T201
        assert names, "the decoded units crossed shared memory"
        assert 1 <= leaked <= 2 * _WORKERS, run.stderr[-2000:]  # 2 columns: image, label
        assert not [n for n in names if Path("/dev/shm", n).exists()], "none outlives the run"  # nosec B108
