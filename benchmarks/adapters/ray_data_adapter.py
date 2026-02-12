"""Ray Data adapter for the benchmark framework.

Wraps Ray Data's actor-based distributed Dataset with the BenchmarkAdapter
lifecycle. Tier 3 -- supports 3 scenarios.

Design ref: Section 14.2 of the benchmark report.

Fork-safety note
~~~~~~~~~~~~~~~~
Libraries with native Rust/C++ extensions (e.g. deeplake v4) can spawn
hundreds of OS-level threads invisible to Python.  ``ray.init()`` internally
uses ``fork()`` (via ``subprocess.Popen``) to start the raylet; if those
threads hold mutexes at fork time, the child deadlocks.  To avoid this, we
start the Ray head node via the CLI (a separate process tree that never
inherits the parent's threads) and connect to it by address.

SkyPilot note
~~~~~~~~~~~~~
SkyPilot runs its own Ray cluster on the default port (6379) for VM
orchestration (auto-shutdown, log streaming, status monitoring).  We start
the benchmark's Ray on a dedicated port (_BENCH_RAY_PORT) so the two
clusters coexist.  Teardown only calls ``ray.shutdown()`` (disconnects the
client); it does NOT call ``ray stop`` which would kill SkyPilot's cluster.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from typing import Any

import numpy as np

from benchmarks.adapters import register
from benchmarks.adapters.base import BenchmarkAdapter, ScenarioConfig

# Dedicated port so we don't collide with SkyPilot's Ray (default 6379).
_BENCH_RAY_PORT = 6399

# Ray 2.53's uv runtime_env hook detects uv-managed venvs and tries to
# recreate them for workers -- but the new venv lacks ray itself, causing
# workers to crash silently and from_numpy()/iter_batches() to hang.
# See: https://github.com/ray-project/ray/issues/59639
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
# Disable tqdm progress bars that can block Ray Data's scheduling loop
# when pytest captures stdout.
os.environ.setdefault("RAY_DATA_DISABLE_PROGRESS_BARS", "1")


def _stop_ray() -> None:
    """Stop the benchmark's Ray cluster (for test cleanup).

    WARNING: ``ray stop --force`` kills ALL local Ray instances.  This is
    safe in local tests but must NEVER be called from ``teardown()`` on
    SkyPilot-managed VMs (it would kill SkyPilot's orchestration cluster).
    The test fixture ``ray_shutdown`` in conftest.py calls this.
    """
    subprocess.run(
        [sys.executable, "-m", "ray.scripts.scripts", "stop", "--force"],
        capture_output=True,
    )


def _start_ray_head(num_cpus: int) -> None:
    """Start a Ray head node via CLI on a dedicated port.

    Uses a non-default port (_BENCH_RAY_PORT) to avoid colliding with
    SkyPilot's orchestration cluster.  Starting via CLI (separate process
    tree) avoids fork-safety issues with native extensions like deeplake.
    """
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ray.scripts.scripts",
            "start",
            "--head",
            f"--port={_BENCH_RAY_PORT}",
            f"--num-cpus={num_cpus}",
            "--num-gpus=0",
            "--include-dashboard=false",
            "--disable-usage-stats",
        ],
        check=True,
        capture_output=True,
    )


@register
class RayDataAdapter(BenchmarkAdapter):
    """BenchmarkAdapter for Ray Data.

    Callers must ensure ``ray.shutdown()`` is called when done (the
    ``ray_shutdown`` test fixture handles this automatically).
    """

    def __init__(self) -> None:
        self._dataset: Any = None
        self._config: ScenarioConfig | None = None
        self._started_head: bool = False

    @property
    def name(self) -> str:
        return "Ray Data"

    @property
    def version(self) -> str:
        import ray

        return ray.__version__

    def is_available(self) -> bool:
        try:
            import ray.data  # noqa: F401

            return True
        except ImportError:
            return False

    def supported_scenarios(self) -> set[str]:
        return {
            "NLP-1",  # No transforms (pure iteration)
            "TAB-1",  # Normalize on float32 (pass-through)
        }

    def setup(self, config: ScenarioConfig, data: Any) -> None:
        import ray
        import ray.data as rd

        if not ray.is_initialized():
            try:
                _start_ray_head(num_cpus=max(config.num_workers, 1))
                self._started_head = True
            except subprocess.CalledProcessError:
                pass  # Head on _BENCH_RAY_PORT may already be running
            ray.init(
                address=f"localhost:{_BENCH_RAY_PORT}",
                ignore_reinit_error=True,
            )

        primary_key = next(iter(data))
        self._dataset = rd.from_numpy(data[primary_key])
        self._config = config

    def _iterate_batches(self) -> Iterator[Any]:
        yield from self._dataset.iter_batches(
            batch_size=self._config.batch_size,
            batch_format="numpy",
        )

    def _materialize_batch(self, batch: Any) -> list[np.ndarray]:
        arr = np.asarray(batch["data"] if "data" in batch else list(batch.values())[0])
        return [arr]

    def teardown(self) -> None:
        self._dataset = None
        self._config = None

        import ray

        if ray.is_initialized():
            ray.shutdown()
        # Note: we intentionally do NOT call `ray stop` here.
        # `ray stop --force` kills ALL local Ray instances, including
        # SkyPilot's orchestration cluster.  The benchmark's cluster on
        # port _BENCH_RAY_PORT is left orphaned — harmless since VMs are
        # terminated after the run.  `ray.shutdown()` above disconnects
        # the client, freeing resources for the next adapter.
