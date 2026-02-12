"""Shared utilities for TDD performance target tests.

Each optimization priority (P0-P5) writes a test that encodes the performance
target as an assertion. The test MUST fail before optimization and MUST pass
after. This module provides the common measurement and assertion infrastructure.
"""

import time
from typing import Any
from collections.abc import Callable, Iterator

import jax.numpy as jnp

from datarax.benchmarking.timing import TimingCollector


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
        adapter: BenchmarkAdapter instance.
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
    sync_fn: Callable[[], None] | None = None,
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
    sync = sync_fn or (lambda: jnp.array(0.0).block_until_ready())
    collector = TimingCollector(sync_fn=sync)

    # Warmup: consume and discard warmup_batches
    for i, _ in enumerate(pipeline_iter):
        if i >= warmup_batches:
            break
        sync()

    # Measure
    result = collector.measure_iteration(
        pipeline_iter,
        num_batches=measure_batches,
        count_fn=count_fn,
    )
    return result.num_elements / result.wall_clock_sec if result.wall_clock_sec > 0 else 0.0


def measure_peak_rss_delta_mb(fn: Callable[[], Any], *, gc_before: bool = True) -> float:
    """Measure peak RSS increase during fn() execution.

    Returns delta in MB. Reusable across P3 and any future memory tests (DRY).

    Args:
        fn: Callable to measure.
        gc_before: Whether to run garbage collection before measurement.

    Returns:
        Peak RSS delta in megabytes.
    """
    import gc
    import os

    import psutil

    if gc_before:
        gc.collect()
    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss
    fn()
    rss_after = process.memory_info().rss
    return (rss_after - rss_before) / (1024 * 1024)


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
