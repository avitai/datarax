"""The host stage runs in every process of a ``jax.distributed`` run.

Under ``jax.distributed`` the CPU platform lists every process's CPU devices, so the host stage
names records and folds keys on its own process's CPU device; on another process's it could not
read the result back. Two real processes, each with two CPU devices, iterate the same pipeline:
each names its batches on its own CPU device, and both name the same records for each batch, the
global order the per-process share of a mesh is cut from.
"""

from __future__ import annotations

from typing import TypedDict

from tests.test_common.processes import run_processes


# One process of the run: joins it, iterates three batches of a shuffled pipeline and reports.
_PROCESS = """
import sys
import jax
from substrax.testing import report

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
report({
    "process": jax.process_index(),
    "host_device_process": next(iter(host_device().device_set)).process_index,
    "local_cpu_devices": len(jax.local_devices(backend="cpu")),
    "named": named,
})
jax.distributed.shutdown()
"""


class ProcessReport(TypedDict):
    """What one process of the run reports."""

    process: int
    host_device_process: int
    local_cpu_devices: int
    named: list[list[int]]


def test_every_process_names_its_batches_on_its_own_cpu_device() -> None:
    reports = run_processes(_PROCESS, ProcessReport)

    assert [report["process"] for report in reports] == [0, 1]
    assert [report["host_device_process"] for report in reports] == [0, 1]
    assert [report["local_cpu_devices"] for report in reports] == [2, 2]


def test_every_process_names_the_same_records_for_each_batch() -> None:
    first, second = run_processes(_PROCESS, ProcessReport)

    assert first["named"] == second["named"]
    assert [len(batch) for batch in first["named"]] == [8, 8, 8]
    assert len({index for batch in first["named"] for index in batch}) == 24
