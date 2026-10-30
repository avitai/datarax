"""P3: the data path's host memory is O(batch), never O(dataset).

On the CPU backend the device is host memory, so every batch the pipeline holds in flight is host
memory: the units its read threads hold, the batches staged on the device and the one the consumer
holds. The peak host memory a CV-1 pipeline adds while iterating is no more than a by-hand control
holding the same batches in flight, built from the same reads and placements, and a dataset of
another size leaves it unchanged. SPDL's peak is reported beside it, not as a bound.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from substrax.testing import run_python

from datarax.core.index_words import to_words
from datarax.sources import MemorySource, MemorySourceConfig
from tests.benchmarks.performance_targets import measure_peak_rss_delta_mb


logger = logging.getLogger(__name__)

# One batch: its images, and its indices (two words a record), epochs and draws.
_BATCH_MB = (64 * 224 * 224 * 3 + 64 * (2 * 4 + 4 + 4)) / 2**20
# The pipeline's own allocator and thread jitter beside its control. Measured in main's CI, 184
# readings from 2026-10-12 to 2026-10-30: at most 0.4 MB above the control without the extra
# batch, and 0.09 MB below to 0.07 MB above one batch with it; on a 24-CPU Linux 6.8 machine
# pinned to 4 CPUs, 0.39 MB below to 0.31 MB above one batch with it.
_JITTER_MB = 1.0


def _bound_mb(control_mb: float) -> float:
    """The most a pipeline variant's growth may be: its control's, one batch, and the jitter."""
    return control_mb + _BATCH_MB + _JITTER_MB


def test_the_bound_holds_one_extra_batch_with_its_jitter() -> None:
    """main's CI on d22802e, d = 2: one batch above the control, and 0.07 MB of jitter."""
    pipeline_mb, control_mb = 73.4296875, 64.16796875

    assert pipeline_mb <= _bound_mb(control_mb)


def test_the_bound_refuses_two_extra_batches() -> None:
    """A second extra batch is one more staged batch than the property allows."""
    control_mb = 64.16796875

    assert control_mb + 2 * _BATCH_MB > _bound_mb(control_mb)


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
        logger.info(f"P3 array views: RSS grew {delta_mb:.1f} MB over 1,000 records, target 200 MB")
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

        The property is the owner's (OWN-1006-HOST-MEMORY-O-BATCH): host memory O(batch), never
        O(dataset). Single-batch exactness is asserted where it resolves:
        ``tests/pipeline/test_device_staging.py`` counts ``1 + d`` placed batches in
        ``jax.live_arrays()`` after every ``next()`` (on CPU too), and its GPU peak test bounds
        the device's peak by batches.

        Each variant runs in its own process (``_PEAK_HOST_MEMORY``), with JAX's runtime and
        compiler, the data and a first run of the same structure (its naming compiled, its threads
        drained) outside the window. The window's peak is the kernel's high-water mark
        (``VmHWM``, reset at the window's start through ``/proc/self/clear_refs``), so no transient
        is missed between samples. ``MALLOC_MMAP_THRESHOLD_`` is fixed at 128 KiB for the
        measurement only, not as datarax behaviour: under glibc's default dynamic threshold a
        freed batch-sized buffer stays in the heap, and the control did not step by batches
        (36.8, 64.4, 66.2, 82.7 MB keeping 1-4 placed batches); with it fixed the control steps
        by one batch, 9.19 MB.

        Bound, the same for every ``d``: the pipeline's growth is at most a by-hand control's
        plus one batch, the property's unit, plus the pipeline's own allocator and thread jitter
        (:data:`_JITTER_MB`). The control reads the same CV-1 batches with ``source.get_batch``
        and places them with ``jax.device_put``, holding the read buffer's batches read ahead and
        ``d + 1`` placed (``d`` staged and the consumer's). The batch of slack covers an extra
        batch the pipeline's peak holds now and then when it stages (d = 2 and 3), which the
        control never holds; placement runs on the consumer's thread, and what holds the extra
        batch is not identified. In main's CI it appeared in 0 of 152 readings on the runners'
        Linux 6.17 kernel and 7 of 32 on Linux 7.0, and it also appears on a Linux 6.8 machine,
        so a kernel changes how often, not whether. It is one batch to within 0.4 MB each time,
        so the jitter rides on top of the batch rather than inside it. A dataset a tenth the size
        (1,000 records against CV-1's 10,000; ten times CV-1 does not fit the 16 GiB process cap)
        is held to the same bound. Measured on one CPU: the pipeline adds one batch per staged
        batch (43.6, 52.5-52.9, 61.5-62.3, 70.8-80.2 MB at d = 0..3, B=64 of 224x224x3 uint8; the
        80 MB readings are that extra batch).
        """

        def peak(*args: str) -> dict[str, Any]:
            result = run_python(
                _PEAK_HOST_MEMORY,
                *args,
                timeout=900,
                env={"JAX_PLATFORMS": "cpu", "MALLOC_MMAP_THRESHOLD_": "131072"},
                cwd=Path(__file__).resolve().parents[2],
            )
            return result.check().report_as(dict[str, Any])

        large = peak("pipeline", "10000", str(depth), "1")
        small = peak("pipeline", "1000", str(depth), "1")
        control = peak("by hand", "10000", str(depth), str(large["read_buffer"]))
        spdl = peak("spdl", "10000", "0", "0")
        logger.info(
            f"P3 d={depth}: pipeline {large['growth_mb']:.1f} MB (1,000 records: "
            f"{small['growth_mb']:.1f} MB), by-hand control {control['growth_mb']:.1f} MB, "
            f"one batch {_BATCH_MB:.2f} MB, SPDL {spdl['growth_mb']} MB (reported, not a bound; "
            "None when spdl is not installed)"
        )
        for variant in (large, small):
            assert variant["growth_mb"] <= _bound_mb(control["growth_mb"])


_PEAK_HOST_MEMORY = """
import collections, gc, sys, threading
import jax, jax.numpy as jnp, numpy as np
from flax import nnx
from substrax.testing import report
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
        if not [t for t in threading.enumerate() if "grain" in t.name]:
            break
        threading.Event().wait(0.05)
    pipe = build(source)
    read_buffer = pipe.host_plan.read_buffer if hasattr(pipe, "host_plan") else 0

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
        report({"growth_mb": None, "read_buffer": 0})
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
report({"growth_mb": growth, "read_buffer": read_buffer})
"""
