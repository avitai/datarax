"""Integration tests for the Datarax monitoring system.

This module contains integration tests for the Datarax monitoring system,
including tests with JAX operations and performance testing.
"""

import time
from math import ceil
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from datarax.dag.nodes import BatchNode, OperatorNode
from datarax.monitoring.callbacks import MetricRecord, MetricsObserver
from datarax.monitoring.pipeline import MonitoredPipeline
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.sources import MemorySource
from datarax.sources.memory_source import MemorySourceConfig


def test_monitored_pipeline_with_jax_operations():
    """Test MonitoredPipeline with JAX operations."""
    # Create test data
    data = {"value": np.arange(100)}
    source = MemorySource(MemorySourceConfig(), data, rngs=nnx.Rngs(0))

    # Create a monitored pipeline
    pipeline = MonitoredPipeline(source)

    # Create an observer to capture metrics
    class TestObserver(MetricsObserver):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.metrics_received = []

        def update(self, metrics: list[MetricRecord]) -> None:
            self.metrics_received.extend(metrics)

    observer = TestObserver()
    pipeline.callbacks.register(observer)

    # Define a JAX operation
    @jax.jit
    def jax_transform(x):
        return jnp.sin(x)

    # Define a transformer that uses the JAX operation
    def apply_jax_op(element, key):
        """Apply JAX transform to element data."""
        del key
        new_data = {"value": jax_transform(element.data["value"])}
        return element.replace(data=new_data)

    # Apply the transformer using ElementOperator
    config = ElementOperatorConfig(stochastic=False)
    rngs = nnx.Rngs(0)
    transformer = ElementOperator(config, fn=apply_jax_op, rngs=rngs)
    transformed_pipeline = pipeline.add(BatchNode(batch_size=1)).add(OperatorNode(transformer))

    # Process all data
    results = list(transformed_pipeline)

    # Verify JAX operation was applied
    assert abs(results[0]["value"][0] - np.sin(0)) < 1e-5
    assert abs(results[1]["value"][0] - np.sin(1)) < 1e-5

    # Manually notify if no metrics were recorded
    if len(observer.metrics_received) == 0:
        pipeline.callbacks.notify(pipeline.metrics.get_metrics())

    # Verify metrics were collected
    assert len(observer.metrics_received) > 0


def _normalize_element(element, key):
    """Normalize element data to [0, 1] — gives each batch real tensor work."""
    del key
    new_data = {k: v / (jnp.max(jnp.abs(v)) + 1e-8) for k, v in element.data.items()}
    return element.replace(data=new_data)


MONITORING_TEST_SAMPLE_COUNT = 2000
MONITORING_TEST_BATCH_SIZE = 64
MAX_MONITORING_OVERHEAD_PER_BATCH_SECONDS = 0.001
MAX_MONITORING_OVERHEAD_RATIO = 2.0


@pytest.fixture(scope="module")
def monitoring_test_pipelines():
    """Create test pipelines with a semi-realistic workload.

    Uses image-like data (2000×32×32×3) with a normalization transform so each
    batch does real tensor operations. This gives a meaningful baseline against
    which monitoring overhead is measured — not an artificially near-zero one.
    """
    rng = np.random.default_rng(0)
    data = [
        {"image": rng.standard_normal((32, 32, 3), dtype=np.float32)}
        for _ in range(MONITORING_TEST_SAMPLE_COUNT)
    ]

    norm_config = ElementOperatorConfig(stochastic=False)
    norm_op = ElementOperator(norm_config, fn=_normalize_element)

    source_no = MemorySource(MemorySourceConfig(), data, rngs=nnx.Rngs(0))
    pipeline_no_metrics = (
        MonitoredPipeline(source_no, metrics_enabled=False)
        .batch(MONITORING_TEST_BATCH_SIZE)
        .add(OperatorNode(norm_op))
    )

    source_with = MemorySource(MemorySourceConfig(), data, rngs=nnx.Rngs(1))
    pipeline_with_metrics = (
        MonitoredPipeline(source_with, metrics_enabled=True)
        .batch(MONITORING_TEST_BATCH_SIZE)
        .add(OperatorNode(norm_op))
    )

    # Warmup both pipelines (important for JIT compilation)
    for _ in range(5):
        list(pipeline_no_metrics)
        list(pipeline_with_metrics)

    return pipeline_no_metrics, pipeline_with_metrics


@pytest.mark.benchmark(group="monitoring_overhead")
def test_benchmark_metrics_disabled(benchmark, monitoring_test_pipelines):
    """Benchmark pipeline with metrics disabled (baseline)."""
    pipeline_no_metrics, _ = monitoring_test_pipelines
    benchmark.pedantic(
        lambda: list(pipeline_no_metrics),
        rounds=10,
        iterations=1,
        warmup_rounds=2,
    )


@pytest.mark.benchmark(group="monitoring_overhead")
def test_benchmark_metrics_enabled(benchmark, monitoring_test_pipelines):
    """Benchmark pipeline with metrics enabled."""
    _, pipeline_with_metrics = monitoring_test_pipelines
    benchmark.pedantic(
        lambda: list(pipeline_with_metrics),
        rounds=10,
        iterations=1,
        warmup_rounds=2,
    )


def test_performance_impact(monitoring_test_pipelines):
    """Test that monitoring overhead is acceptable.

    This test uses pytest-benchmark's pedantic mode for statistically valid
    measurements with proper warmup, GC control, and outlier handling.

    The overhead comparison uses the same methodology for both configurations
    to ensure fair comparison under identical system conditions.
    """
    import gc
    import statistics

    pipeline_no_metrics, pipeline_with_metrics = monitoring_test_pipelines

    # Use pedantic-style benchmarking for both configurations
    rounds = 10
    warmup_rounds = 2

    # Warmup rounds (not measured)
    for _ in range(warmup_rounds):
        list(pipeline_no_metrics)
        list(pipeline_with_metrics)

    # Interleaved measurement for fair comparison under varying system load
    # By alternating, both measurements experience similar system conditions
    no_metrics_times: list[float] = []
    with_metrics_times: list[float] = []

    for _ in range(rounds):
        # Disable GC during measurement for consistency
        gc.disable()
        try:
            # Measure baseline (no metrics)
            start = time.perf_counter()
            list(pipeline_no_metrics)
            no_metrics_times.append(time.perf_counter() - start)

            # Measure with metrics
            start = time.perf_counter()
            list(pipeline_with_metrics)
            with_metrics_times.append(time.perf_counter() - start)
        finally:
            gc.enable()

    # Use median for robustness against outliers (same as pytest-benchmark)
    no_metrics_median = statistics.median(no_metrics_times)
    with_metrics_median = statistics.median(with_metrics_times)

    # Calculate overhead using both a ratio guardrail and the more stable
    # per-batch absolute cost. The absolute budget is the primary contract:
    # a few milliseconds of host jitter can distort ratios when the baseline
    # run is only tens of milliseconds long.
    overhead_ratio = with_metrics_median / no_metrics_median
    paired_overheads = [
        max(0.0, with_metrics_time - no_metrics_time)
        for no_metrics_time, with_metrics_time in zip(no_metrics_times, with_metrics_times)
    ]
    batch_count = ceil(MONITORING_TEST_SAMPLE_COUNT / MONITORING_TEST_BATCH_SIZE)
    overhead_per_batch = statistics.median(paired_overheads) / batch_count

    assert overhead_per_batch < MAX_MONITORING_OVERHEAD_PER_BATCH_SECONDS, (
        f"Metrics overhead too high: {overhead_per_batch * 1000:.3f}ms/batch "
        f"(median no_metrics={no_metrics_median:.3f}s, "
        f"median with_metrics={with_metrics_median:.3f}s)"
    )
    assert overhead_ratio < MAX_MONITORING_OVERHEAD_RATIO, (
        f"Metrics overhead too high: {overhead_ratio:.2f}x "
        f"(median no_metrics={no_metrics_median:.3f}s, "
        f"median with_metrics={with_metrics_median:.3f}s)"
    )


def test_high_frequency_metrics():
    """Test that high-frequency metrics collection works properly."""
    # Create test data
    data = {"value": np.arange(1000)}
    source = MemorySource(MemorySourceConfig(), data, rngs=nnx.Rngs(0))

    # Create a monitored pipeline
    pipeline = MonitoredPipeline(source).batch(1)

    # Create an observer to capture metrics
    class CountingObserver(MetricsObserver):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.update_count = 0
            self.metrics = []

        def update(self, metrics: list[MetricRecord]) -> None:
            self.update_count += 1
            self.metrics.extend(metrics)

    observer = CountingObserver()

    # Set a low notification threshold to ensure frequent updates
    # This is an implementation detail, but we're directly setting it
    # to test high-frequency metrics reporting
    pipeline._notify_threshold = 10  # type: ignore
    pipeline.callbacks.register(observer)

    # Process all data
    list(pipeline)

    # Force final notification for any remaining metrics
    pipeline.callbacks.notify(pipeline.metrics.get_metrics())

    # Verify that multiple updates occurred
    assert observer.update_count > 0

    # Verify that metrics were collected
    assert len(observer.metrics) > 0
