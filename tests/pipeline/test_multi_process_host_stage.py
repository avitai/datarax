"""The host stage runs in every process of a ``jax.distributed`` run.

Under ``jax.distributed`` the CPU platform lists every process's CPU devices, so the host stage
names records and folds keys on its own process's CPU device; on another process's it could not
read the result back. Two real processes, each with two CPU devices, iterate the same pipeline:
each names its batches on its own CPU device, and both name the same records for each batch, the
global order the per-process share of a mesh is cut from.
"""

from __future__ import annotations

import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypedDict

from substrax.runtime import JaxRuntime
from substrax.testing import run_python


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROCESSES = 2
_CHILD_SECONDS = 180.0

# One process of the run: joins it, iterates three batches of a shuffled pipeline and reports.
_PROCESS = """
import json, sys
import jax

port, process = int(sys.argv[1]), int(sys.argv[2])
jax.distributed.initialize(f"localhost:{port}", num_processes=2, process_id=process)

import numpy as np
from flax import nnx

from datarax.core.index_words import from_words
from datarax.core.prng import host_device
from datarax.pipeline import Pipeline
from datarax.sources import MemorySource, MemorySourceConfig

source = MemorySource(MemorySourceConfig(), {"value": np.arange(40, dtype=np.float32)})
pipeline = Pipeline(source=source, stages=[], batch_size=8, rngs=nnx.Rngs(3), shuffle=True)
batches = pipeline.raw_batches()
named = [
    [int(i) for i in from_words(np.asarray(next(batches).indices))] for _ in range(3)
]
print(json.dumps({
    "process": jax.process_index(),
    "host_device_process": next(iter(host_device().device_set)).process_index,
    "local_cpu_devices": len(jax.local_devices(backend="cpu")),
    "named": named,
}), flush=True)
jax.distributed.shutdown()
"""


class ProcessReport(TypedDict):
    """What one process of the run prints."""

    process: int
    host_device_process: int
    local_cpu_devices: int
    named: list[list[int]]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("localhost", 0))
        return probe.getsockname()[1]


def _run_processes(program: str) -> list[ProcessReport]:
    """``program`` in two processes of one ``jax.distributed`` run, each one's last JSON line."""
    port = _free_port()
    runtime = JaxRuntime(platforms=("cpu",), cpu_devices=2, preallocate=False)
    with ThreadPoolExecutor(max_workers=_PROCESSES) as pool:
        children = [
            pool.submit(
                run_python,
                program,
                str(port),
                str(process),
                timeout=_CHILD_SECONDS,
                runtime=runtime,
                cwd=_REPO_ROOT,
            )
            for process in range(_PROCESSES)
        ]
        results = [child.result() for child in children]
    return [result.check().last_json_as(ProcessReport) for result in results]


def test_every_process_names_its_batches_on_its_own_cpu_device() -> None:
    reports = _run_processes(_PROCESS)

    assert [report["process"] for report in reports] == [0, 1]
    assert [report["host_device_process"] for report in reports] == [0, 1]
    assert [report["local_cpu_devices"] for report in reports] == [2, 2]


def test_every_process_names_the_same_records_for_each_batch() -> None:
    first, second = _run_processes(_PROCESS)

    assert first["named"] == second["named"]
    assert [len(batch) for batch in first["named"]] == [8, 8, 8]
    assert len({index for batch in first["named"] for index in batch}) == 24
