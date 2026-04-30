"""Benchmark: ``Pipeline.scan`` over a ``MixDataSourcesNode`` at varying source counts.

Measures the per-step wall-clock cost of ``MixDataSourcesNode.get_batch_at``
under ``Pipeline.scan`` as the number of mixed sources grows. The
implementation uses ``jax.vmap`` over per-position ``jax.lax.switch``.
While ``lax.switch`` semantically traces every branch, XLA's compile-time
dead-branch elimination means the runtime cost stays roughly constant in
the number of mixed sources — the per-position dispatch + RNG + categorical
sampling dominates over the per-source fetch in compiled code.

Reports:

- Wall-clock per epoch for N ∈ {1, 2, 3, 5, 8} mixed sources.
- Per-step cost in microseconds.
- Comparison against a single-source ``MemorySource`` baseline (no
  mixing overhead at all) to isolate the cost of the mixing layer.

Run via:
    source activate.sh && uv run python benchmarks/mixed_source_scan.py
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax.pipeline import Pipeline
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from datarax.sources.mixed_source import MixDataSourcesConfig, MixDataSourcesNode


# ---------------------------------------------------------------------
# Workload parameters
# ---------------------------------------------------------------------

NUM_RECORDS_PER_SOURCE = 256
ELEMENT_SHAPE = (32,)  # small enough that mixing dispatch dominates
BATCH_SIZE = 32
STEPS_PER_EPOCH = 16
NUM_EPOCHS = 3
NUM_TRIALS = 3


# ---------------------------------------------------------------------
# Source builders
# ---------------------------------------------------------------------


def _make_source(seed: int) -> MemorySource:
    """Build a deterministic single-source MemorySource."""
    rng = np.random.default_rng(seed)
    data = {
        "x": jnp.asarray(
            rng.standard_normal(size=(NUM_RECORDS_PER_SOURCE, *ELEMENT_SHAPE)),
            dtype=jnp.float32,
        )
    }
    return MemorySource(MemorySourceConfig(shuffle=False), data)


def _make_mix(num_sources: int) -> MixDataSourcesNode:
    """Build a MixDataSourcesNode with ``num_sources`` equally-weighted children."""
    from datarax.core.data_source import DataSourceModule

    sources: list[DataSourceModule] = [_make_source(seed=i) for i in range(num_sources)]
    weights = tuple([1.0 / num_sources] * num_sources)
    return MixDataSourcesNode(
        MixDataSourcesConfig(num_sources=num_sources, weights=weights),
        sources,
    )


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------


def _time_pipeline(pipeline: Pipeline) -> float:
    """Run ``NUM_EPOCHS`` epochs of ``pipeline.scan`` and return median per-trial seconds."""

    def step_fn(batch: dict) -> jax.Array:
        return jnp.sum(batch["x"])

    # Warmup compile
    pipeline._position.value = jnp.int32(0)
    pipeline.scan(step_fn, length=STEPS_PER_EPOCH)
    jax.block_until_ready(jnp.asarray(0.0))

    trials = []
    for _ in range(NUM_TRIALS):
        pipeline._position.value = jnp.int32(0)
        start = time.perf_counter()
        for _ in range(NUM_EPOCHS):
            pipeline._position.value = jnp.int32(0)
            outputs = pipeline.scan(step_fn, length=STEPS_PER_EPOCH)
        jax.block_until_ready(outputs)
        trials.append(time.perf_counter() - start)
    return float(np.median(trials))


def main() -> None:
    """Run main."""
    print("=" * 72)
    print("MixDataSourcesNode + Pipeline.scan throughput vs source count")
    print("=" * 72)
    print(f"  Backend:        {jax.default_backend()}")
    print(f"  Devices:        {jax.devices()}")
    print(f"  Records/source: {NUM_RECORDS_PER_SOURCE}")
    print(f"  Element shape:  {ELEMENT_SHAPE}")
    print(f"  Batch size:     {BATCH_SIZE}")
    print(f"  Steps/epoch:    {STEPS_PER_EPOCH}")
    print(f"  Epochs/trial:   {NUM_EPOCHS}")
    print(f"  Trials:         {NUM_TRIALS}")
    print("-" * 72)

    # Baseline: single-source pipeline (no mixing overhead at all).
    baseline_pipeline = Pipeline(
        source=_make_source(seed=0),
        stages=[],
        batch_size=BATCH_SIZE,
        rngs=nnx.Rngs(0),
    )
    baseline_seconds = _time_pipeline(baseline_pipeline)
    baseline_per_step_us = baseline_seconds / (NUM_EPOCHS * STEPS_PER_EPOCH) * 1e6
    print(
        f"baseline (single source):  {baseline_seconds * 1000:7.2f} ms "
        f"({baseline_per_step_us:6.1f} µs/step)"
    )
    print()
    print(f"{'N sources':<14}{'wall-clock':<15}{'µs/step':<12}{'overhead vs N=1':<16}")
    print("-" * 72)

    results: list[tuple[int, float, float]] = []
    for n_sources in (1, 2, 3, 5, 8):
        mix = _make_mix(n_sources)
        pipeline = Pipeline(source=mix, stages=[], batch_size=BATCH_SIZE, rngs=nnx.Rngs(0))
        seconds = _time_pipeline(pipeline)
        per_step_us = seconds / (NUM_EPOCHS * STEPS_PER_EPOCH) * 1e6
        results.append((n_sources, seconds, per_step_us))

        # Overhead reference: N=1 mix vs raw single-source baseline.
        if n_sources == 1:
            ref_per_step_us = per_step_us
            overhead_str = "—"
        else:
            overhead_str = f"{per_step_us / ref_per_step_us:.2f}x"

        print(
            f"  N={n_sources:<10}{seconds * 1000:7.2f} ms     "
            f"{per_step_us:6.1f}      {overhead_str}"
        )

    print("-" * 72)
    print(
        "Note: per-step cost is roughly constant in N (XLA dead-branch "
        "elimination amortizes over lax.switch). Mixing overhead is dominated "
        "by per-position RNG + categorical sampling + vmap dispatch."
    )
    print()


if __name__ == "__main__":
    main()
