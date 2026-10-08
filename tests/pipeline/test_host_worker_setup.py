"""What a Grain worker process starts with, what it is sent, and the one absl flag Grain reads.

A worker is spawned (Grain's start method) and inherits the environment but no JAX setting made
in code. datarax's worker set-up runs before the read is unpickled: it hides the GPU, starts JAX on
the CPU platform only, sets the run's precision mode and refuses a worker whose JAX was started
before it (a main module without the ``if __name__ == "__main__":`` guard). A worker is sent the
read alone, which holds no dataset, and never TensorFlow. Grain reads an absl flag when it starts a
worker, which raises outside ``absl.app.run``; datarax marks that one flag present, and a tripwire
fails once Grain no longer needs it.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import signal
import subprocess  # nosec B404 - the tests start this interpreter on their own scripts
import sys
import sysconfig
import threading
import time
from pathlib import Path
from typing import Any

import cloudpickle
import jax
import numpy as np
import psutil
import pytest
from absl import flags
from flax import nnx
from substrax.testing import restored_jax_config, run_python

from datarax.pipeline import Pipeline
from datarax.pipeline.epochs import EpochPlan, Run
from datarax.pipeline.read_plan import HostReadPath
from datarax.pipeline.run_units import RunUnits
from datarax.sources import TFDSStreamingConfig, TFDSStreamingSource
from tests.test_common.tfds_fixture import FIXTURE, TFDSFixture
from tests.test_common.worker_reads import (
    is_gone,
    png_source,
    ProcessReadMemorySource,
    resources,
    stream_of,
    worker_children,
    WorkerFactsDecoder,
    write_png_records,
)


_ROOT = Path(__file__).resolve().parents[2]
_MiB = 1 << 20

free_threaded = pytest.mark.skipif(
    bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
    reason="free-threaded Python: mp_prefetch reads on threads and skips worker set-up",
)


@pytest.fixture(scope="module")
def png_paths(tmp_path_factory: pytest.TempPathFactory) -> list[str]:
    """12 PNG records in two ArrayRecord files."""
    return write_png_records(tmp_path_factory.mktemp("png"), 12)


def _pipeline(source: Any, workers: int = 2, **options: Any) -> Pipeline:
    return Pipeline(
        source=source,
        stages=[],
        batch_size=3,
        rngs=nnx.Rngs(0),
        shuffle=True,
        host_resources=resources(workers),
        **options,
    )


def _facts(pipe: Pipeline) -> dict[int, dict[str, Any]]:
    """Each worker's :func:`~tests.test_common.worker_reads.worker_facts`, by process id."""
    found = {}
    for _, provenance in pipe.raw_batches(with_provenance=True):
        for record in provenance:
            facts = json.loads(record["worker"])
            found[facts["pid"]] = facts
    return found


@free_threaded
class TestWorkerSetUp:
    """Inside each worker, after its first read."""

    def test_cpu_platform_gpu_hidden_no_tensorflow_spawned(
        self, png_paths: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Inherited by the workers; names no device, so whatever happens no GPU is reached.
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "inherited-not-a-device")
        monkeypatch.setenv("JAX_PLATFORMS", "cuda,cpu")  # the GPU recipe's, inherited
        pipe = _pipeline(png_source(png_paths, WorkerFactsDecoder()))
        facts = _facts(pipe)
        pipe.close()
        assert len(facts) == 2
        for worker in facts.values():
            assert worker["jax_platforms"] == "cpu"
            assert worker["backend"] == "cpu"
            assert worker["cuda_visible_devices"] == ""
            assert worker["x64"] is False
            assert worker["tensorflow"] is False
            assert worker["spawned"] is True

    def test_a_worker_reads_in_the_caller_s_precision_mode(self, png_paths: list[str]) -> None:
        with jax.enable_x64(True):
            pipe = _pipeline(png_source(png_paths, WorkerFactsDecoder()))
            facts = _facts(pipe)
            pipe.close()
        assert {worker["x64"] for worker in facts.values()} == {True}

    @pytest.mark.parametrize("how", ["config.update", "context manager"])
    def test_a_worker_holds_the_caller_s_prng_settings_globally(
        self, png_paths: list[str], how: str
    ) -> None:
        """The settings a reader carries, set in code or by context manager, reach every worker
        as its global settings (read there on a new thread); the default impl is not one of
        them, since naming pins its own."""
        pipe = Pipeline(
            source=png_source(png_paths, WorkerFactsDecoder()),
            stages=[],
            batch_size=3,
            rngs=nnx.Rngs(jax.random.key(0, impl="rbg")),
            shuffle=True,
            host_resources=resources(2),
        )
        if how == "config.update":
            with restored_jax_config():
                jax.config.update("jax_threefry_partitionable", False)
                jax.config.update("jax_default_prng_impl", "rbg")
                facts = _facts(pipe)
        else:
            with jax.threefry_partitionable(False), jax.default_prng_impl("rbg"):
                facts = _facts(pipe)
        pipe.close()
        assert len(facts) == 2
        for worker in facts.values():
            assert worker["global"] == {"x64": False, "threefry_partitionable": False}

    def test_every_worker_watches_its_training_process(self, png_paths: list[str]) -> None:
        pipe = _pipeline(png_source(png_paths, WorkerFactsDecoder()))
        facts = _facts(pipe)
        pipe.close()
        assert len(facts) == 2
        assert {worker["watcher"] for worker in facts.values()} == {True}


_INHERITED_CUDA = """
import json, sys
import jax
jax.config.update("jax_platforms", "cpu")  # set in code: a spawned worker does not inherit it
import grain
from absl import flags
from flax import nnx
from datarax.pipeline import Pipeline
from tests.test_common.worker_reads import png_source, resources, WorkerFactsDecoder

if __name__ == "__main__":
    paths = sys.argv[1:]
    pipe = Pipeline(source=png_source(paths, WorkerFactsDecoder()), stages=[], batch_size=3,
                    rngs=nnx.Rngs(0), host_resources=resources(2))
    facts = sorted({json.loads(r["worker"])["backend"]
                    for _, p in pipe.raw_batches(with_provenance=True) for r in p})
    control = "served"
    read = pipe.host_stage.read_for_workers(pipe)
    pipe.close()
    if not flags.FLAGS.is_parsed():
        flags.FLAGS["grain_enable_multiprocess_worker_profiling"].present = 1
    bare = grain.MapDataset.range(2).map(read).to_iter_dataset().mp_prefetch(
        grain.multiprocessing.MultiprocessingOptions(num_workers=1))
    try:
        list(bare)
    except Exception as error:  # the control: what a worker without datarax's set-up raises
        control = f"raised {type(error).__name__}: {str(error)[:300]}"
    print(json.dumps({"backends": facts, "control": control}))
"""

_UNGUARDED_MAIN = """
import json, sys
import jax
from flax import nnx
from datarax.pipeline import Pipeline
from tests.test_common.worker_reads import png_source, resources

jax.devices()  # a backend started at import: a spawned worker imports this module too

if __name__ == "__main__":
    pipe = Pipeline(source=png_source(sys.argv[1:]), stages=[], batch_size=3, rngs=nnx.Rngs(0),
                    host_resources=resources(1))
    try:
        list(pipe.raw_batches())
        print(json.dumps({"outcome": "served"}))
    except Exception as error:
        print(json.dumps({"outcome": f"{type(error).__name__}: {error}"}))
"""


def _run_script(tmp_path: Path, source: str, *args: str, env: dict[str, str]) -> dict[str, Any]:
    script = tmp_path / "script.py"
    script.write_text(source)
    result = run_python(script, *args, timeout=300, env=env, cwd=_ROOT)
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


@free_threaded
class TestWorkerSetUpPositiveControls:
    """Each guard above fails without datarax's set-up."""

    def test_an_inherited_cuda_platform_list_never_reaches_a_worker(
        self, png_paths: list[str], tmp_path: Path
    ) -> None:
        """The GPU recipe's ``JAX_PLATFORMS=cuda,cpu`` is inherited; datarax's workers read on CPU.

        The same read in a bare Grain worker, without datarax's set-up, starts the listed
        platforms and fails: the control.
        """
        report = _run_script(
            tmp_path,
            _INHERITED_CUDA,
            *png_paths,
            env={"JAX_PLATFORMS": "cuda,cpu", "CUDA_VISIBLE_DEVICES": ""},
        )
        assert report["backends"] == ["cpu"]
        assert report["control"].startswith("raised"), report["control"]

    def test_a_main_module_starting_jax_at_import_is_refused_naming_the_guard(
        self, png_paths: list[str], tmp_path: Path
    ) -> None:
        report = _run_script(tmp_path, _UNGUARDED_MAIN, *png_paths, env={})
        assert 'if __name__ == "__main__":' in report["outcome"], report["outcome"]


class TestWhatGoesToAWorker:
    """A worker is sent the read: the source's files and options, never its records."""

    def test_an_array_record_read_pickles_below_a_mib(self, png_paths: list[str]) -> None:
        pipe = _pipeline(png_source(png_paths))
        next(iter(pipe.raw_batches()))
        terms = pipe.host_plan.terms
        pipe.close()
        assert terms is not None
        read_bytes = terms.read_bytes
        assert read_bytes < _MiB
        direct = len(cloudpickle.dumps(pipe.host_stage.read_for_workers(pipe)))
        assert direct < _MiB

    def test_every_array_an_unpickled_read_holds_is_on_the_cpu(self, png_paths: list[str]) -> None:
        pipe = _pipeline(png_source(png_paths))
        payload = cloudpickle.dumps(pipe.host_stage.read_for_workers(pipe))
        gc.collect()
        before = {id(a) for a in gc.get_objects() if isinstance(a, jax.Array)}
        copy = cloudpickle.loads(payload)
        made = [a for a in gc.get_objects() if isinstance(a, jax.Array) and id(a) not in before]
        assert copy is not None
        assert all(device.platform == "cpu" for a in made for device in a.devices())

    def test_positive_control_an_in_memory_read_carries_its_dataset(self) -> None:
        column = np.zeros((40, 64 * 1024), np.float32)  # 10 MiB
        pipe = _pipeline(ProcessReadMemorySource(config=_memory_config(), data={"x": column}))
        next(iter(pipe.raw_batches()))
        plan = pipe.host_plan
        pipe.close()
        assert plan.path is HostReadPath.PROCESSES
        assert plan.terms is not None
        assert plan.terms.read_bytes > 10 * _MiB


def _memory_config() -> Any:
    from datarax.sources.memory_source import MemorySourceConfig  # noqa: PLC0415

    return MemorySourceConfig()


_OUTSIDE_APP_RUN = """
import json, sys
from flax import nnx
from datarax.pipeline import Pipeline
from tests.test_common.worker_reads import png_source, resources

if __name__ == "__main__":
    pipe = Pipeline(source=png_source(sys.argv[1:]), stages=[], batch_size=3, rngs=nnx.Rngs(0),
                    host_resources=resources(2))
    served = sum(1 for _ in pipe.raw_batches())
    pipe.close()
    from absl import flags
    flags.FLAGS(["script", "--grain_enable_multiprocess_worker_profiling=true"])
    print(json.dumps({"served": served,
                      "flag": flags.FLAGS.grain_enable_multiprocess_worker_profiling}))
"""

_BARE_GRAIN = """
import json
import grain
import jax
import numpy as np

if __name__ == "__main__":
    jax.devices()
    ds = grain.MapDataset.range(4).map(lambda i: np.full(2, i)).to_iter_dataset().mp_prefetch(
        grain.multiprocessing.MultiprocessingOptions(num_workers=1))
    try:
        served = len(list(ds))
        print(json.dumps({"outcome": f"served {served}"}))
    except Exception as error:
        print(json.dumps({"outcome": type(error).__name__}))
"""


class TestTheGrainFlag:
    """Grain reads ``--grain_enable_multiprocess_worker_profiling`` as it starts a worker."""

    def test_the_suite_leaves_absl_flags_unparsed(self) -> None:
        """No conftest marks every flag parsed: each test meets absl as a user's script does."""
        assert not flags.FLAGS.is_parsed()

    def test_a_script_outside_absl_app_run_reads_with_two_workers_to_the_end(
        self, png_paths: list[str], tmp_path: Path
    ) -> None:
        report = _run_script(tmp_path, _OUTSIDE_APP_RUN, *png_paths, env={})
        assert report["served"] == 4

    def test_a_later_flags_parse_still_wins(self, png_paths: list[str], tmp_path: Path) -> None:
        report = _run_script(tmp_path, _OUTSIDE_APP_RUN, *png_paths, env={})
        assert report["flag"] is True

    def test_tripwire_bare_grain_still_raises_outside_app_run(self, tmp_path: Path) -> None:
        """Fails once the installed Grain stops raising: then datarax's mark goes (a3b351858a)."""
        report = _run_script(tmp_path, _BARE_GRAIN, env={})
        assert report["outcome"] == "UnparsedFlagAccessError", (
            f"Grain no longer raises outside absl.app.run ({report['outcome']}): remove datarax's "
            "mark of --grain_enable_multiprocess_worker_profiling and this tripwire"
        )


@pytest.mark.tfds
class TestATFDSStreamSharesItsIndex:
    """One read-only copy of a stream's offset index on the host; a worker maps it."""

    def _source(self, fixture: TFDSFixture) -> TFDSStreamingSource:
        config = TFDSStreamingConfig(name=FIXTURE, split="train", data_dir=str(fixture.tfrecord))
        return TFDSStreamingSource(config)

    def test_a_worker_s_read_carries_the_index_by_reference(
        self, tfds_fixture: TFDSFixture
    ) -> None:
        source = self._source(tfds_fixture)
        schedule = _schedule(source)
        # Before the index is shared, a read carries its own copy: the positive control.
        copy = source.run_dataset(schedule, None)
        offsets = b"".join(index.payloads.tobytes() for index in copy._read.index)  # noqa: SLF001
        assert offsets in cloudpickle.dumps(copy)
        shared = source.run_dataset(schedule, None, for_workers=True)
        assert offsets not in cloudpickle.dumps(shared)
        assert offsets not in cloudpickle.dumps(source.run_dataset(schedule, None))  # one copy
        received = cloudpickle.loads(cloudpickle.dumps(shared))  # this process's own read
        for index in received._read.index:  # noqa: SLF001
            assert isinstance(index.payloads.base, np.memmap) or isinstance(
                index.payloads, np.memmap
            )
            assert index.shared is not None
        assert shared.shared_bytes() == 16 * len(source)

    def test_the_shared_copy_goes_with_its_source(self, tfds_fixture: TFDSFixture) -> None:
        source = self._source(tfds_fixture)
        schedule = _schedule(source)
        shared = source.run_dataset(schedule, None, for_workers=True)
        place = shared._read.index[0].shared  # noqa: SLF001
        assert place is not None
        path = Path(place[0])
        del shared
        assert path.exists()
        del source
        gc.collect()
        assert not path.exists()


def _schedule(source: TFDSStreamingSource) -> RunUnits:
    """One pass of ``source`` in batches of 3."""
    plan = EpochPlan(length=len(source), batch_size=3, drop_last=False, num_epochs=1)
    return RunUnits(run=Run(plan=plan, position=0, epoch=0, end_epoch=1))


_PRIVATE_BACKEND_CHECK = """
from jax._src import xla_bridge
import jax.numpy as jnp

before = xla_bridge.backends_are_initialized()
jnp.zeros(()).block_until_ready()
print(before, xla_bridge.backends_are_initialized())
"""


def _isolated(code: str) -> subprocess.CompletedProcess[str]:
    """``code`` in a fresh ``python -I`` on the CPU platform: no site path, no user module."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("JAX_", "XLA_", "PYTHON"))}
    return subprocess.run(  # noqa: S603  # nosec B603 - this interpreter, the test's own code
        [sys.executable, "-I", "-c", code],
        env={**env, "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_tripwire_jax_s_private_backend_check_still_answers_without_starting_jax() -> None:
    """datarax's worker set-up asks JAX, privately, whether its backends started.

    JAX has no public check that leaves the backends alone (``jax.extend.backend.backends()``
    starts them). This fails once ``jax._src.xla_bridge.backends_are_initialized`` disappears or
    stops answering ``False`` before a first operation and ``True`` after it; re-checked at each
    JAX floor raise.
    """
    result = _isolated(_PRIVATE_BACKEND_CHECK)
    assert result.returncode == 0, result.stderr[-3000:]
    assert result.stdout.split() == ["False", "True"], (
        "jax._src.xla_bridge.backends_are_initialized changed: datarax's worker set-up "
        f"(host_workers._backends_started) needs another check ({result.stdout!r})"
    )


_KILLED_PARENT = """
import json, os, signal, sys, threading, time
from flax import nnx
from datarax.pipeline import Pipeline, host_workers
from tests.test_common.worker_reads import (
    png_source, resources, shm_names_of, spawned_children, UnwatchedSetUp)

if __name__ == "__main__":
    mode, when, *paths = sys.argv[1:]
    if mode == "unwatched":
        host_workers.WorkerSetUp = UnwatchedSetUp
    pipe = Pipeline(source=png_source(paths), stages=[], batch_size=3, rngs=nnx.Rngs(0),
                    num_epochs=None, host_resources=resources(2))
    batches = iter(pipe.raw_batches())
    me = os.getpid()
    if when == "first-unit":
        next(batches)
        deadline = time.monotonic() + 120
        while len(spawned_children(me)) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        workers = spawned_children(me)
        print(json.dumps({"workers": workers, "shm": sorted(shm_names_of([me, *workers]))}),
              flush=True)
        time.sleep(600)  # ended by the test's signal
    else:
        threading.Thread(target=next, args=(batches,), daemon=True).start()
        while not spawned_children(me):
            time.sleep(0.001)
        time.sleep(float(when))
        workers = spawned_children(me)
        print(json.dumps({"workers": workers, "shm": sorted(shm_names_of([me, *workers]))}),
              flush=True)
        os.kill(me, signal.SIGKILL)
"""

_GONE_WITHIN = 2.0
"""Seconds every worker of a dead training process has to exit (measured 0.01-0.06 s)."""


def _until(check: Any, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.01)
    return check()


def _end_by_recorded_pid(pids: list[int]) -> None:
    """Kill what is left of ``pids``, each checked to be a spawned worker first."""
    for pid in pids:
        try:
            command = " ".join(psutil.Process(pid).cmdline())
        except psutil.NoSuchProcess:
            continue
        if "spawn_main" in command:
            os.kill(pid, signal.SIGKILL)
    assert _until(lambda: all(is_gone(pid) for pid in pids), 10.0)


def _killed_parent(
    tmp_path: Path, paths: list[str], mode: str, when: str, kill: int | None
) -> dict[str, Any]:
    """Run :data:`_KILLED_PARENT`; ``kill`` it once it reports its workers, unless it kills itself.

    Returns:
        The workers and ``/dev/shm`` names it reported, and the seconds from its death until
        every worker was gone (``None`` when some outlived :data:`_GONE_WITHIN`).
    """
    script = tmp_path / "killed_parent.py"
    script.write_text(_KILLED_PARENT)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("JAX_", "XLA_"))}
    parent = subprocess.Popen(  # noqa: S603  # nosec B603 - this interpreter, the test's script
        [sys.executable, str(script), mode, when, *paths],
        env={**env, "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""},
        cwd=_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        line = parent.stdout.readline() if parent.stdout is not None else ""
        assert line, parent.stderr.read()[-3000:] if parent.stderr is not None else ""
        report = json.loads(line)
        if kill is not None:
            parent.send_signal(kill)
        parent.wait(timeout=60)
        died = time.monotonic()
        workers: list[int] = report["workers"]
        gone = _until(lambda: all(is_gone(pid) for pid in workers), _GONE_WITHIN)
        report["gone_after"] = time.monotonic() - died if gone else None
        report["alive"] = [pid for pid in workers if not is_gone(pid)]
        logging.getLogger(__name__).info(
            "%s parent, %s, ended %s: workers %s gone after %s s, alive %s",
            mode,
            when,
            "by itself" if kill is None else signal.Signals(kill).name,
            workers,
            report["gone_after"],
            report["alive"],
        )
        return report
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=60)


linux_only = pytest.mark.skipif(
    not Path("/dev/shm").is_dir(), reason="reads /proc and /dev/shm (Linux)"
)


@free_threaded
@linux_only
class TestTheParentWatcher:
    """A worker ends when its training process dies, however it dies (google/grain#1423)."""

    @pytest.mark.parametrize("kill", [signal.SIGKILL, signal.SIGTERM], ids=["SIGKILL", "SIGTERM"])
    def test_every_worker_ends_with_a_killed_training_process(
        self, png_paths: list[str], tmp_path: Path, kill: int
    ) -> None:
        report = _killed_parent(tmp_path, png_paths, "watched", "first-unit", kill)
        try:
            assert len(report["workers"]) == 2
            assert report["gone_after"] is not None, report
            assert _until(
                lambda: not any(Path("/dev/shm", n).exists() for n in report["shm"]), 2.0
            ), [n for n in report["shm"] if Path("/dev/shm", n).exists()]
        finally:
            _end_by_recorded_pid(report["workers"])

    @pytest.mark.parametrize("delay", ["0", "0.2", "0.5", "1"])
    def test_a_training_process_dying_as_its_workers_start_leaves_none(
        self, png_paths: list[str], tmp_path: Path, delay: str
    ) -> None:
        report = _killed_parent(tmp_path, png_paths, "watched", delay, None)
        try:
            assert report["workers"]
            assert report["gone_after"] is not None, report
        finally:
            _end_by_recorded_pid(report["workers"])

    def test_positive_control_workers_without_the_watcher_outlive_it(
        self, png_paths: list[str], tmp_path: Path
    ) -> None:
        report = _killed_parent(tmp_path, png_paths, "unwatched", "first-unit", signal.SIGKILL)
        try:
            assert len(report["workers"]) == 2
            assert report["gone_after"] is None
            assert sorted(report["alive"]) == sorted(report["workers"])
        finally:
            _end_by_recorded_pid(report["workers"])

    def test_a_set_up_in_the_training_process_starts_no_watcher(self) -> None:
        """Outside a spawned worker there is no parent process to watch."""
        result = run_python(
            "import multiprocessing, threading\n"
            "from datarax.pipeline.host_workers import JaxSettings, WorkerSetUp\n"
            "WorkerSetUp(settings=JaxSettings.current())(0, 1)\n"
            "print(multiprocessing.parent_process() is None,"
            " any(t.name == 'datarax-parent-watch' for t in threading.enumerate()))\n",
            timeout=120,
            cwd=_ROOT,
        )
        assert result.returncode == 0, result.stderr[-3000:]
        assert result.stdout.split() == ["True", "False"]


def _on_threads(paths: list[str], *, num_epochs: int) -> Pipeline:
    """The pipeline :func:`_pipeline` builds, without a budget: its reference stream on threads."""
    return Pipeline(
        source=png_source(paths),
        stages=[],
        batch_size=3,
        rngs=nnx.Rngs(0),
        shuffle=True,
        num_epochs=num_epochs,
    )


@free_threaded
class TestTheWatcherChangesNoRun:
    """A watched worker serves a run as before: across epochs, threads, a close and a restore."""

    def test_three_epochs_the_first_from_a_short_lived_thread(self, png_paths: list[str]) -> None:
        reference = stream_of(_on_threads(png_paths, num_epochs=3).raw_batches())
        pipe = _pipeline(png_source(png_paths), num_epochs=3)
        per_epoch = len(reference) // 3
        batches = iter(pipe.raw_batches())
        first: list[Any] = []
        reader = threading.Thread(
            target=lambda: first.extend(next(batches) for _ in range(per_epoch))
        )
        reader.start()
        reader.join()
        workers = {child.pid for child in worker_children()}
        rest = list(batches)
        assert stream_of(iter(first)) + stream_of(iter(rest)) == reference
        assert len(workers) == 2
        pipe.close()

    def test_a_live_restore_and_a_close_serve_the_whole_stream(self, png_paths: list[str]) -> None:
        reference = stream_of(_on_threads(png_paths, num_epochs=4).raw_batches())
        pipe = _pipeline(png_source(png_paths), num_epochs=4)
        batches = iter(pipe.raw_batches())
        head = [next(batches) for _ in range(5)]
        pipe.set_state(pipe.get_state())  # on the live run: it reopens at the cursor
        batches = iter(pipe.raw_batches())
        middle = [next(batches) for _ in range(5)]
        pipe.close()  # after ten units
        tail = list(pipe.raw_batches())
        assert stream_of(iter(head + middle)) + stream_of(iter(tail)) == reference
        pipe.close()
        assert not worker_children()
