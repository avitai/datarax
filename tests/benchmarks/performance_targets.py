"""Shared utilities for TDD performance target tests.

Each optimization priority (P0-P5) writes a test that encodes the performance
target as an assertion. The test MUST fail before optimization and MUST pass
after. This module provides the common measurement and assertion infrastructure.
"""

import time
from collections.abc import Callable
from typing import Any


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
    adapter.warmup(warmup_batches, timed_batches=measure_batches)
    result = adapter.iterate(measure_batches)
    adapter.teardown()
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


def measure_latency(
    fn: Callable[[], Any],
    *,
    repetitions: int = 5,
    warmup: int = 0,
    aggregate: str = "median",
) -> float:
    """Measure wall-clock latency of fn() in seconds.

    Reusable across P5 checkpoint and any future latency tests (DRY).

    Args:
        fn: Callable to measure.
        repetitions: Number of timed repetitions.
        warmup: Number of untimed warmup calls run first to absorb one-time
            costs (filesystem, JIT, Orbax initialization) that would otherwise
            skew the first timed sample.
        aggregate: ``"median"`` (default) or ``"min"``. Use ``"min"`` for
            ratio comparisons where the steady-state best case is the stable
            signal — noise and scheduling jitter only ever add time.

    Returns:
        Aggregated latency in seconds.
    """
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repetitions):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    times.sort()
    if aggregate == "min":
        return times[0]
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Peak-RSS comparison classifier (used by P3 memory-efficiency test)
# ---------------------------------------------------------------------------

# Below this peak-RSS delta the measurement is dominated by allocator noise
# (Python GC, JAX backend init, kernel page-cache effects) rather than data
# residency. A delta below it is known only to be at most the floor.
NOISE_FLOOR_MB = 50.0

# Hard ceiling on Datarax peak-RSS regardless of SPDL. CV-1 raw data is
# ~1.5 GB; the cap allows up to ~2.6x for one in-flight copy plus pipeline
# state and JIT-trace overhead. Exceeding this is a memory regression
# whatever SPDL measured.
DATARAX_RSS_ABSOLUTE_CAP_MB = 4000.0


def classify_rss_comparison(
    *,
    datarax_rss: float,
    spdl_rss: float,
    max_ratio: float = 1.5,
) -> tuple[str, str]:
    """Classify a Datarax-vs-SPDL peak-RSS comparison.

    Decision order:

    1. If Datarax exceeds ``DATARAX_RSS_ABSOLUTE_CAP_MB``, fail regardless
       of SPDL.
    2. If Datarax is below ``NOISE_FLOOR_MB``, skip: its peak is allocator
       noise, and no SPDL value makes that a regression.
    3. Otherwise compare Datarax against ``max_ratio`` times SPDL's peak,
       taken as at least ``NOISE_FLOOR_MB``. An SPDL peak below the floor is
       known only to be at most the floor, so the floor is the largest
       reference it could be; Datarax above ``max_ratio`` times the floor
       exceeds the target whatever SPDL's true peak. Skipping here would
       hide exactly the case where one loader allocates far more.

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

    if datarax_rss < NOISE_FLOOR_MB:
        return (
            "skip",
            (
                f"Datarax peak RSS below noise floor ({NOISE_FLOOR_MB} MB): "
                f"Datarax={datarax_rss:.0f} MB, SPDL={spdl_rss:.0f} MB."
            ),
        )

    reference = max(spdl_rss, NOISE_FLOOR_MB)
    reference_text = (
        f"SPDL ({spdl_rss:.0f} MB)"
        if spdl_rss >= NOISE_FLOOR_MB
        else f"the {NOISE_FLOOR_MB:.0f} MB noise floor (SPDL={spdl_rss:.0f} MB, below it)"
    )
    ratio = datarax_rss / reference
    if ratio > max_ratio:
        return (
            "fail",
            (
                f"Datarax peak RSS ({datarax_rss:.0f} MB) is {ratio:.2f}x "
                f"{reference_text}, exceeds {max_ratio}x target."
            ),
        )
    return (
        "pass",
        (
            f"Datarax peak RSS ({datarax_rss:.0f} MB) is {ratio:.2f}x "
            f"{reference_text}, within {max_ratio}x target."
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
