"""Shared utilities for TDD performance target tests.

Each optimization priority (P0-P5) writes a test that encodes the performance
target as an assertion. The test MUST fail before optimization and MUST pass
after. This module provides the common measurement and assertion infrastructure.
"""

import time
from collections.abc import Callable, Iterator
from typing import Any

import jax.numpy as jnp
from calibrax.profiling import TimingCollector


def measure_adapter_throughput(
    adapter: Any,
    config: Any,
    data: dict,
    *,
    warmup_batches: int = 5,
    measure_batches: int = 50,
) -> float:
    """Measure adapter throughput in elements/sec.

    Runs the full adapter lifecycle: setup → warmup → iterate → teardown.
    Returns throughput. Reusable across all P0-P5 tests (DRY).

    Args:
        adapter: PipelineAdapter instance.
        config: ScenarioConfig for the benchmark.
        data: Synthetic data dict for the scenario.
        warmup_batches: Number of warmup iterations.
        measure_batches: Number of measured iterations.

    Returns:
        Throughput in elements per second.
    """
    adapter.setup(config, data)
    adapter.warmup(warmup_batches)
    result = adapter.iterate(measure_batches)
    adapter.teardown()
    return result.num_elements / result.wall_clock_sec if result.wall_clock_sec > 0 else 0.0


def measure_pipeline_throughput(
    pipeline_iter: Iterator,
    *,
    warmup_batches: int = 5,
    measure_batches: int = 50,
    count_fn: Callable[[Any], int] | None = None,
    sync_fn: Callable[[Any], None] | None = None,
) -> float:
    """Measure raw pipeline throughput in elements/sec.

    For testing datarax internals directly (not via adapter).

    Args:
        pipeline_iter: Iterator yielding batches.
        warmup_batches: Batches to skip for warmup.
        measure_batches: Batches to measure.
        count_fn: Function to count elements per batch. Default: 1 per batch.
        sync_fn: GPU sync function. Default: JAX block_until_ready.

    Returns:
        Throughput in elements per second.
    """

    def default_sync(_result: Any) -> None:
        jnp.array(0.0).block_until_ready()

    sync = sync_fn or default_sync
    collector = TimingCollector(sync_fn=sync)

    # Warmup: consume and discard warmup_batches
    for i, _ in enumerate(pipeline_iter):
        if i >= warmup_batches:
            break
        sync(None)

    # Measure
    result = collector.measure_iteration(
        pipeline_iter,
        num_batches=measure_batches,
        count_fn=count_fn,
    )
    return result.num_elements / result.wall_clock_sec if result.wall_clock_sec > 0 else 0.0


def measure_peak_rss_delta_mb(
    fn: Callable[[], Any],
    *,
    gc_before: bool = True,
    sample_interval_s: float = 0.01,
) -> float:
    """Measure peak RSS increase during fn() execution.

    Spawns a sampling thread that polls ``psutil`` RSS at
    ``sample_interval_s`` and tracks the max. Returns the delta in MB
    between baseline RSS (after optional GC) and the peak observed
    during execution. The sample thread is daemon-ised so it cannot
    keep the test process alive on errors.

    A naive after-minus-before measurement misses transient
    allocations that the framework releases before fn() returns —
    that is the underlying reason the SPDL adapter (which releases
    its async buffers) showed RSS deltas of 1 MB while Datarax (which
    holds the pipeline alive) showed 28 MB. Polling captures the true
    high-water mark for both.

    Args:
        fn: Callable to measure.
        gc_before: Whether to run garbage collection before measurement.
        sample_interval_s: How often to poll RSS during execution.

    Returns:
        Peak RSS delta in megabytes.
    """
    import gc
    import os
    import threading

    import psutil

    if gc_before:
        gc.collect()
    process = psutil.Process(os.getpid())
    baseline_rss = process.memory_info().rss
    peak_rss = baseline_rss
    stop_sampling = threading.Event()

    def _sample_peak() -> None:
        nonlocal peak_rss
        while not stop_sampling.is_set():
            current = process.memory_info().rss
            if current > peak_rss:
                peak_rss = current
            stop_sampling.wait(sample_interval_s)

    sampler = threading.Thread(target=_sample_peak, daemon=True)
    sampler.start()
    try:
        fn()
    finally:
        # Capture one final RSS reading in case the peak is at exit.
        final_rss = process.memory_info().rss
        if final_rss > peak_rss:
            peak_rss = final_rss
        stop_sampling.set()
        sampler.join(timeout=0.5)

    return (peak_rss - baseline_rss) / (1024 * 1024)


def measure_latency(fn: Callable[[], Any], *, repetitions: int = 5) -> float:
    """Measure median wall-clock latency of fn() in seconds.

    Reusable across P5 checkpoint and any future latency tests (DRY).

    Args:
        fn: Callable to measure.
        repetitions: Number of repetitions (median is returned).

    Returns:
        Median latency in seconds.
    """
    times = []
    for _ in range(repetitions):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Peak-RSS comparison classifier (used by P3 memory-efficiency test)
# ---------------------------------------------------------------------------

# Below this peak-RSS delta SPDL is effectively non-measuring — typically
# because it iterates the in-memory fixture via zero-copy views or because
# the fixture was already paged in and counted toward baseline. Comparing
# Datarax (which holds a Pipeline + buffers at peak) against ~0 produces a
# divergent ratio that says nothing about actual memory regressions.
SPDL_NULL_FLOOR_MB = 1.0

# Below this peak-RSS delta the measurement is dominated by allocator noise
# (Python GC, JAX backend init, kernel page-cache effects) rather than data
# residency. Both-below-this means a ratio assertion is uninformative.
NOISE_FLOOR_MB = 50.0

# Hard ceiling on Datarax peak-RSS regardless of SPDL. CV-1 raw data is
# ~1.5 GB; the cap allows up to ~2.6x for one in-flight copy plus pipeline
# state and JIT-trace overhead. Exceeding this is a memory regression that
# must not be hidden by an SPDL-zero skip.
DATARAX_RSS_ABSOLUTE_CAP_MB = 4000.0


def classify_rss_comparison(
    *,
    datarax_rss: float,
    spdl_rss: float,
    max_ratio: float = 1.5,
) -> tuple[str, str]:
    """Classify a Datarax-vs-SPDL peak-RSS comparison.

    Decision order:

    1. If Datarax exceeds ``DATARAX_RSS_ABSOLUTE_CAP_MB``, fail loud
       regardless of SPDL — this catches regressions that an
       SPDL-effectively-zero skip would otherwise hide.
    2. If SPDL is below ``SPDL_NULL_FLOOR_MB``, skip with a
       SPDL-specific message (zero-copy / pre-loaded fixture).
    3. If either adapter is below ``NOISE_FLOOR_MB``, skip with the
       noise-floor message — the ratio is uninformative.
    4. Otherwise compare ``datarax_rss / spdl_rss`` against
       ``max_ratio``.

    Returns:
        Tuple of ``(verdict, message)`` where ``verdict`` is one of
        ``"pass"``, ``"skip"``, or ``"fail"``.
    """
    if datarax_rss > DATARAX_RSS_ABSOLUTE_CAP_MB:
        return (
            "fail",
            (
                f"Datarax peak RSS {datarax_rss:.0f} MB exceeds absolute cap "
                f"{DATARAX_RSS_ABSOLUTE_CAP_MB:.0f} MB (SPDL={spdl_rss:.0f} MB). "
                "This is a memory regression independent of the ratio comparison."
            ),
        )

    if spdl_rss < SPDL_NULL_FLOOR_MB:
        return (
            "skip",
            (
                f"SPDL allocated {spdl_rss:.3f} MB (below "
                f"{SPDL_NULL_FLOOR_MB:.1f} MB null floor) — likely zero-copy "
                "on the in-memory fixture or fixture pre-loaded into baseline. "
                f"The {max_ratio}x ratio is undefined; absolute cap "
                f"({DATARAX_RSS_ABSOLUTE_CAP_MB:.0f} MB) was satisfied at "
                f"{datarax_rss:.0f} MB."
            ),
        )

    if datarax_rss < NOISE_FLOOR_MB or spdl_rss < NOISE_FLOOR_MB:
        return (
            "skip",
            (
                f"Peak RSS below noise floor ({NOISE_FLOOR_MB} MB): "
                f"Datarax={datarax_rss:.0f} MB, SPDL={spdl_rss:.0f} MB. "
                "Increase scenario.dataset_size for a meaningful comparison."
            ),
        )

    ratio = datarax_rss / spdl_rss
    if ratio > max_ratio:
        return (
            "fail",
            (
                f"Datarax peak RSS ({datarax_rss:.0f} MB) is {ratio:.2f}x "
                f"SPDL ({spdl_rss:.0f} MB), exceeds {max_ratio}x target."
            ),
        )
    return (
        "pass",
        (
            f"Datarax peak RSS ({datarax_rss:.0f} MB) is {ratio:.2f}x "
            f"SPDL ({spdl_rss:.0f} MB), within {max_ratio}x target."
        ),
    )


def assert_within_ratio(
    datarax_value: float,
    alternative_value: float,
    max_ratio: float,
    metric_name: str = "throughput",
) -> None:
    """Assert datarax is within max_ratio of alternative.

    For throughput (higher is better):
        assert_within_ratio(datarax_tp, alternative_tp, 1.2)
        → datarax must be >= alternative / 1.2

    For latency/memory (lower is better), swap args:
        assert_within_ratio(alternative_latency, datarax_latency, 1.5)

    Args:
        datarax_value: The datarax measurement.
        alternative_value: The alternative measurement.
        max_ratio: Maximum acceptable ratio.
        metric_name: Name for error messages.

    Raises:
        AssertionError: If datarax_value < alternative_value / max_ratio.
    """
    min_acceptable = alternative_value / max_ratio
    assert datarax_value >= min_acceptable, (
        f"Datarax {metric_name} ({datarax_value:.0f}) is below "
        f"{max_ratio}x target ({min_acceptable:.0f}) vs alternative ({alternative_value:.0f})"
    )
