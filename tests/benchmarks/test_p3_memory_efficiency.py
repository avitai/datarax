"""P3: the data path's host memory is O(batch), never O(dataset).

On the CPU backend the device is host memory, so every batch the pipeline holds in flight is host
memory: the units its read threads hold, the batches staged on the device and the one the consumer
holds. The peak host memory a CV-1 pipeline adds while iterating is no more than a by-hand control
holding the same batches in flight, built from the same reads and placements, and a dataset of
another size leaves it unchanged. SPDL's peak is reported beside it, not as a bound.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pytest
from substrax.testing import run_python

from datarax.core.index_words import to_words
from datarax.sources import MemorySource, MemorySourceConfig
from tests.benchmarks.performance_targets import measure_peak_rss_delta_mb


logger = logging.getLogger(__name__)


@pytest.mark.benchmark
class TestP3MemoryEfficiency:
    """P3: host memory O(batch) on CV-1."""

    def test_memory_source_array_views(self):
        """Verify dict data uses array slicing (views) for batching."""
        data = {
            "image": np.random.default_rng(42).integers(0, 255, (1000, 8, 8, 3), dtype=np.uint8)
        }
        config = MemorySourceConfig(prefetch_size=0)
        source = MemorySource(config, data)

        # Iterate through entire source — should use views not copies
        delta_mb = measure_peak_rss_delta_mb(lambda: list(source))

        # RSS increase should be modest. The source data is ~192KB,
        # so RSS delta should be dominated by Python/JAX overhead, not
        # data duplication. After optimization (array views), this should
        # be well under 200MB.
        assert delta_mb < 200, f"RSS increased by {delta_mb:.0f} MB during iteration"

    def test_gather_batch_efficiency(self):
        """The host read gathers with array indexing: one NumPy column, views for a run."""
        data = {"x": np.arange(500)}
        config = MemorySourceConfig(prefetch_size=0)
        source = MemorySource(config, data)

        rows = to_words(np.arange(32, dtype=np.uint64))
        batch = source.get_batch(rows)
        assert isinstance(batch["x"], np.ndarray)
        assert len(batch["x"]) == 32
        assert np.shares_memory(source.get_batch(rows, contiguous=True)["x"], source.data["x"])

    @pytest.mark.parametrize("depth", [0, 1, 2, 3])
    def test_peak_host_memory_is_the_batches_in_flight_whatever_the_dataset(
        self, depth: int
    ) -> None:
        """The peak host memory iterating adds is the batches in flight, never the dataset.

        Each variant runs in its own process (``_PEAK_HOST_MEMORY``), with JAX's runtime and
        compiler, the data and a first run of the same structure (its naming compiled) outside
        the window. The window's peak is the kernel's high-water mark (``VmHWM``, reset at the
        window's start through ``/proc/self/clear_refs``), so no transient is missed between
        samples. glibc's mmap threshold is fixed at 128 KiB so a freed batch returns to the OS:
        with the default dynamic threshold freed batch-sized buffers stay in the heap and the
        peak stops counting batches (measured: the control then steps unevenly).

        Bound: the by-hand control reads the same CV-1 batches with ``source.get_batch`` and
        places them with ``jax.device_put``, holding the read buffer's batches read ahead and
        ``d + 1`` placed (``d`` staged and the consumer's). With ``d = 0`` the pipeline is held to
        it exactly. With ``d > 0`` it may exceed it by one batch: the placement thread refills its
        room as the consumer takes a batch, before or after the consumer drops the one it held
        (the race the GPU peak test measures). A dataset a tenth the size (1,000 records against
        CV-1's 10,000; ten times CV-1 does not fit the 16 GiB process cap) is held to the same
        bound, as the GPU peak test holds both sizes to one range. Measured on one CPU over three
        runs: the pipeline adds one batch per staged batch (43.6, 52.5-52.9, 61.5-62.3,
        70.8-80.2 MB at d = 0..3, B=64 of 224x224x3 uint8, 9.19 MB a batch; the 80.2 is the
        race), the control 44.3-44.7, 53.3-53.9, 62.4-63.3, 72.1-72.5 MB.
        """
        # One batch: its images, and its indices (two words a record), epochs and draws.
        batch_mb = (64 * 224 * 224 * 3 + 64 * (2 * 4 + 4 + 4)) / 2**20

        def peak(*args: str) -> dict:
            result = run_python(
                _PEAK_HOST_MEMORY,
                *args,
                timeout=900,
                env={"JAX_PLATFORMS": "cpu", "MALLOC_MMAP_THRESHOLD_": "131072"},
                cwd=Path(__file__).resolve().parents[2],
            )
            return json.loads(result.check().stdout.strip().splitlines()[-1])

        large = peak("pipeline", "10000", str(depth), "1")
        small = peak("pipeline", "1000", str(depth), "1")
        control = peak("by hand", "10000", str(depth), str(large["read_buffer"]))
        spdl = peak("spdl", "10000", "0", "0")
        logger.info(
            f"P3 d={depth}: pipeline {large['growth_mb']:.1f} MB (1,000 records: "
            f"{small['growth_mb']:.1f} MB), by-hand control {control['growth_mb']:.1f} MB, "
            f"one batch {batch_mb:.2f} MB, SPDL {spdl['growth_mb']} MB (reported, not a bound; "
            "None when spdl is not installed)"
        )
        race = batch_mb if depth else 0.0
        for variant in (large, small):
            assert variant["growth_mb"] <= control["growth_mb"] + race


_PEAK_HOST_MEMORY = """
import collections, gc, json, sys, threading
import jax, jax.numpy as jnp, numpy as np
from flax import nnx
from datarax.core.index_words import to_words
from datarax.pipeline import Pipeline
from datarax.sources import MemorySource, MemorySourceConfig

mode, records, depth, last = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
B, steps, shape = 64, 60, (224, 224, 3)
images = np.random.default_rng(42).integers(0, 255, (records, *shape), dtype=np.uint8)
source = MemorySource(MemorySourceConfig(), {"image": images})
jax.block_until_ready(jax.jit(lambda x: x + 1)(jnp.zeros(())))  # runtime and compiler


def build(over):
    pipe = Pipeline(
        source=over, stages=[], batch_size=B, rngs=nnx.Rngs(42), shuffle=True, num_epochs=None
    )
    if hasattr(pipe, "host_stage"):
        pipe.host_stage._device_buffer = depth
    return pipe


read_buffer = 0
if mode == "pipeline":
    # A first run of the same structure over other records compiles the naming outside the window.
    warm = build(MemorySource(MemorySourceConfig(), {"image": np.zeros_like(images)}))
    jax.block_until_ready(next(iter(warm)))
    if hasattr(warm, "close"):
        warm.close()
    del warm
    for _ in range(200):  # its read threads exit, and what they held is freed, before the window
        gc.collect()
        if not [t for t in threading.enumerate() if "grain" in t.name or "staging" in t.name]:
            break
        threading.Event().wait(0.05)
    pipe = build(source)
    read_buffer = pipe.host_stage._read_buffer if hasattr(pipe, "host_stage") else 0

    def run():
        held = collections.deque(maxlen=last)  # the batches the consumer keeps
        for step, batch in enumerate(pipe):
            held.append(jax.block_until_ready(batch))
            if step == steps:
                break
            threading.Event().wait(0.02)  # a step's time, in which the buffers fill
elif mode == "by hand":
    order = np.random.default_rng(0).permutation(records).astype(np.uint64)
    jax.block_until_ready(jax.device_put(source.get_batch(to_words(order[:B]))))

    def run():
        # ``last`` batches read ahead, the oldest placed when full; ``depth + 1`` placed kept.
        ahead, placed = collections.deque(), collections.deque(maxlen=depth + 1)
        for step in range(steps):
            ahead.append(source.get_batch(to_words(order[step * B % (records - B) :][:B])))
            if len(ahead) > last:
                placed.append(jax.block_until_ready(jax.device_put(ahead.popleft())))
else:
    from benchmarks.adapters.base import ScenarioConfig
    from benchmarks.adapters.spdl_adapter import SpdlAdapter
    from tests.benchmarks.performance_targets import measure_adapter_throughput

    adapter = SpdlAdapter()
    config = ScenarioConfig(
        scenario_id="CV-1", dataset_size=records, element_shape=shape, batch_size=B,
        transforms=[], seed=42,
    )
    if not adapter.is_available():
        print(json.dumps({"growth_mb": None, "read_buffer": 0}))
        sys.exit(0)

    def run():
        measure_adapter_throughput(adapter, config, {"image": images})


def kib(field):
    with open("/proc/self/status") as status:
        return next(int(line.split()[1]) for line in status if line.startswith(field))


gc.collect()
with open("/proc/self/clear_refs", "w") as refs:
    refs.write("5")  # the kernel's high-water mark restarts at the current resident size
start = kib("VmRSS:")
run()
growth = (kib("VmHWM:") - start) / 1024
print(json.dumps({"growth_mb": growth, "read_buffer": read_buffer}))
"""
