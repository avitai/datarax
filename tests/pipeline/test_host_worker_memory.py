"""Host memory O(batch), never O(dataset), in every worker process and in the training process.

P3's rule applied to worker processes: (1) each worker's peak growth after
its first unit is no more than a by-hand control's (a spawned process with datarax's worker
set-up holding the units a worker's queues hold, ``r + 2o + 2``), and the training process's
growth no more than a control holding the ``d + 1`` units at the device and the ``k o`` units of
the per-worker buffers; (2) a dataset ten times larger leaves each worker's peak and the training
process's growth unchanged. The growth bounds have one unit of slack, the property's unit, as
P3's have; a worker's whole peak is compared across sizes within the units a worker holds.
JAX's runtime and compiler start, in each worker, before its window.

Positive control: an in-memory source sent to processes carries its dataset into every worker,
and its worker peak grows with the dataset by far more than a unit.

Every variant runs in its own process; peaks are the kernel's high-water marks (``VmHWM``,
restarted through ``/proc/<pid>/clear_refs``), so no transient is missed between samples.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from substrax.testing import run_python

from tests.test_common.worker_reads import write_png_records


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not Path("/proc/self/clear_refs").exists(), reason="needs Linux /proc"),
]

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_SHAPE = (64, 64, 3)
_BATCH = 256
_RECORDS = 1024

_FOOTPRINT = """
import collections, gc, json, multiprocessing, sys, threading
import cloudpickle, jax, jax.numpy as jnp, numpy as np
from flax import nnx
from datarax.pipeline import Pipeline
from datarax.pipeline.host_workers import JaxSettings
from datarax.sources.memory_source import MemorySourceConfig
from tests.test_common.worker_reads import (
    hold_units_in_a_worker, png_source, proc_kib, ProcessReadMemorySource, reset_peak, resources,
    worker_children)

def main():
    mode, kind, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
    paths = sys.argv[4:]
    jax.block_until_ready(jax.jit(lambda x: x + 1)(jnp.zeros(())))  # runtime and compiler
    if kind == "memory":
        images = np.random.default_rng(0).integers(0, 256, (int(paths[0]), 64, 64, 3), np.uint8)
        source = ProcessReadMemorySource(MemorySourceConfig(), {"image": images})
    else:
        source = png_source(paths)
    pipe = Pipeline(source=source, stages=[], batch_size={batch}, rngs=nnx.Rngs(0),
                    shuffle=True, num_epochs=None, host_resources=resources(2))
    plan = pipe.host_plan
    unit = plan.terms.unit_bytes
    if mode == "control worker":
        payload = cloudpickle.dumps(pipe.host_stage.read_for_workers(pipe))
        receive, send = multiprocessing.get_context("spawn").Pipe(duplex=False)
        keep = plan.worker_read_buffer + 2 * plan.worker_buffer + 2
        child = multiprocessing.get_context("spawn").Process(
            target=hold_units_in_a_worker,
            args=(payload, keep, steps, JaxSettings.current(), send))
        child.start()
        growth = receive.recv()
        child.join()
        print(json.dumps({"growth_kib": growth, "unit": unit}))
        return
    if mode == "control parent":
        read = pipe.host_stage.read_for_workers(pipe)
        read(0)
        keep = plan.device_buffer + 1 + plan.workers * plan.worker_buffer + 1
        gc.collect()
        reset_peak("self")
        start = proc_kib("self", "VmRSS")
        held = collections.deque(maxlen=keep)
        for ordinal in range(1, steps + 1):
            held.append(jax.device_put(read(ordinal).batch))
        print(json.dumps({"growth_kib": proc_kib("self", "VmHWM") - start, "unit": unit}))
        return
    batches = iter(pipe.raw_batches())
    held = collections.deque(maxlen=1)
    for _ in range(4):  # the workers start, compile their naming and read their first units
        held.append(next(batches))
    threading.Event().wait(1.0)  # their queues fill
    workers = [child.pid for child in worker_children()]
    absolute = {pid: proc_kib(pid, "VmHWM") for pid in workers}
    for pid in workers:
        reset_peak(pid)
    start = {pid: proc_kib(pid, "VmRSS") for pid in workers}
    gc.collect()
    reset_peak("self")
    parent_start = proc_kib("self", "VmRSS")
    for _ in range(steps):
        held.append(next(batches))
        threading.Event().wait(0.02)  # a step's time, in which the queues fill
    growth = {pid: proc_kib(pid, "VmHWM") - start[pid] for pid in workers}
    parent = proc_kib("self", "VmHWM") - parent_start
    pipe.close()
    print(json.dumps({"absolute_kib": list(absolute.values()), "growth_kib": list(growth.values()),
                      "parent_kib": parent, "unit": unit, "workers": plan.workers}))

if __name__ == "__main__":
    main()
""".replace("{batch}", str(_BATCH))


def _measure(tmp_path: Path, mode: str, kind: str, *args: str, steps: int = 24) -> dict[str, Any]:
    script = tmp_path / "footprint.py"
    script.write_text(_FOOTPRINT)
    result = run_python(
        script,
        mode,
        kind,
        str(steps),
        *args,
        timeout=900,
        env={"MALLOC_MMAP_THRESHOLD_": "131072"},
        cwd=_ROOT,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def datasets(tmp_path_factory: pytest.TempPathFactory) -> dict[str, list[str]]:
    """1,024 PNG records and ten times as many, 64x64x3 each."""
    return {
        "1x": write_png_records(tmp_path_factory.mktemp("png1"), _RECORDS, shape=_SHAPE),
        "10x": write_png_records(tmp_path_factory.mktemp("png10"), 10 * _RECORDS, shape=_SHAPE),
    }


def test_each_worker_and_the_training_process_hold_batches_never_the_dataset(
    datasets: dict[str, list[str]], tmp_path: Path
) -> None:
    small = _measure(tmp_path, "pipeline", "png", *datasets["1x"])
    large = _measure(tmp_path, "pipeline", "png", *datasets["10x"])
    worker = _measure(tmp_path, "control worker", "png", *datasets["1x"])
    parent = _measure(tmp_path, "control parent", "png", *datasets["1x"])
    unit_kib = small["unit"] / 1024
    logger.info(
        "T5b: unit %.0f KiB; worker growth %s / %s KiB (1x / 10x), by-hand worker %s KiB; "
        "training process %s / %s KiB, by-hand %s KiB; worker peak %s / %s KiB",
        unit_kib,
        small["growth_kib"],
        large["growth_kib"],
        worker["growth_kib"],
        small["parent_kib"],
        large["parent_kib"],
        parent["growth_kib"],
        small["absolute_kib"],
        large["absolute_kib"],
    )
    assert small["workers"] == large["workers"] == 2
    for variant in (small, large):
        for growth in variant["growth_kib"]:
            assert growth <= worker["growth_kib"] + unit_kib, (growth, worker, unit_kib)
        assert variant["parent_kib"] <= parent["growth_kib"] + unit_kib, (variant, parent)
    # A worker's whole peak also holds the units it is decoding when it is read, which differ
    # from run to run: two runs of one dataset read 710-721 MB at 3 MB units (measured). They
    # are at most the units a worker holds, which the by-hand worker's growth measures.
    in_flight = worker["growth_kib"] + unit_kib
    assert max(large["absolute_kib"]) <= max(small["absolute_kib"]) + in_flight, (small, large)


def test_positive_control_a_read_carrying_its_dataset_grows_each_worker_with_it(
    tmp_path: Path,
) -> None:
    small = _measure(tmp_path, "pipeline", "memory", str(_RECORDS), steps=8)
    large = _measure(tmp_path, "pipeline", "memory", str(10 * _RECORDS), steps=8)
    worker = _measure(tmp_path, "control worker", "memory", str(_RECORDS), steps=8)
    in_flight = worker["growth_kib"] + small["unit"] / 1024
    logger.info(
        "T5b control: worker peak %s / %s KiB (1x / 10x in-memory dataset), units in flight "
        "%.0f KiB",
        small["absolute_kib"],
        large["absolute_kib"],
        in_flight,
    )
    assert max(large["absolute_kib"]) > max(small["absolute_kib"]) + in_flight, (small, large)
