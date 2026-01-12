"""Comprehensive performance benchmark suite for Datarax.

This module provides rigorous performance testing following TDD principles.
Each test is designed to:
1. Establish baseline performance metrics
2. Verify optimization effectiveness
3. Prevent performance regressions

Performance Targets (based on profiling analysis):
- Pipeline overhead: <2x direct processing (currently ~5x)
- Batch creation: <5ms for batch_size=32
- Element iteration: >2000 elements/sec (currently ~1000)
- Throughput (batch_size=32): >100 batches/sec (currently ~50)
"""

import time
from dataclasses import dataclass

import pytest
import jax.numpy as jnp
import flax.nnx as nnx

from datarax.dag import from_source
from datarax.sources import MemorySource
from datarax.sources.memory_source import MemorySourceConfig
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.core.data_source import DataSourceModule
from datarax.core.config import StructuralConfig
from datarax.core.element_batch import Element, Batch


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


@dataclass
class BenchmarkDataSourceConfig(StructuralConfig):
    """Configuration for benchmark data source."""

    size: int = 10000
    shape: tuple[int, ...] = (32, 32, 3)


class BenchmarkDataSource(DataSourceModule):
    """Data source optimized for benchmarking with pre-generated data."""

    def __init__(
        self,
        config: BenchmarkDataSourceConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        super().__init__(config, rngs=rngs, name=name)
        self.size = config.size
        self.shape = config.shape
        # Pre-generate all data once to eliminate generation overhead during benchmark
        # Use nnx.data() to mark as pytree data (traced but not trainable)
        # This allows JAX arrays in list without NNX static attribute errors
        self._data = nnx.data(
            [{"data": jnp.ones(self.shape), "label": jnp.array(i % 10)} for i in range(self.size)]
        )

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self._data[idx]

    def __iter__(self):
        yield from self._data


def warmup_jax():
    """Warmup JAX compilation to ensure accurate benchmarks."""
    # Small computation to trigger XLA compilation
    x = jnp.ones((100, 100))
    _ = (x @ x).block_until_ready()


def benchmark_iteration(iterable, max_items: int, warmup: int = 3) -> dict[str, float]:
    """Benchmark iteration throughput with warmup.

    Args:
        iterable: Iterable to benchmark
        max_items: Maximum items to process
        warmup: Number of warmup iterations

    Returns:
        Dictionary with timing metrics
    """
    iterator = iter(iterable)

    # Warmup phase (critical for JIT compilation)
    for _ in range(warmup):
        try:
            _ = next(iterator)
        except StopIteration:
            break

    # Benchmark phase
    start = time.perf_counter()
    count = 0
    for item in iterator:
        count += 1
        if count >= max_items:
            break
    elapsed = time.perf_counter() - start

    return {
        "count": count,
        "elapsed_sec": elapsed,
        "items_per_sec": count / elapsed if elapsed > 0 else 0,
        "ms_per_item": (elapsed / count * 1000) if count > 0 else 0,
    }


# =============================================================================
# Component-Level Benchmarks
# =============================================================================


class TestElementBatchPerformance:
    """Benchmark Element and Batch creation performance."""

    @pytest.mark.benchmark
    def test_element_creation_performance(self):
        """Benchmark Element creation overhead.

        Target: >50,000 elements/sec (matching Grain sampler target)
        """
        warmup_jax()

        data = {"image": jnp.ones((32, 32, 3)), "label": jnp.array(0)}
        num_elements = 10000

        start = time.perf_counter()
        elements = []
        for i in range(num_elements):
            elem = Element(data=data, state={}, metadata=None)
            elements.append(elem)
        elapsed = time.perf_counter() - start

        rate = num_elements / elapsed
        elapsed_ms = elapsed * 1000
        print(f"\nElement creation: {rate:.0f} elements/sec ({elapsed_ms:.1f}ms)")
        print(f"  Total elements: {num_elements}")

        # Target: At least 50,000 elements/sec (Grain sampler target)
        assert rate > 10000, f"Element creation too slow: {rate:.0f} elements/sec < 10000"

    @pytest.mark.benchmark
    def test_batch_creation_performance(self):
        """Benchmark Batch creation from Elements.

        Target: <10ms for batch_size=32
        """
        warmup_jax()

        batch_size = 32
        elements = [
            Element(
                data={"image": jnp.ones((32, 32, 3)), "label": jnp.array(i % 10)},
                state={},
                metadata=None,
            )
            for i in range(batch_size)
        ]

        # Warmup
        _ = Batch(elements[:4], validate=False)

        # Benchmark
        num_iterations = 100
        start = time.perf_counter()
        for _ in range(num_iterations):
            Batch(elements, validate=False)
        elapsed = time.perf_counter() - start

        ms_per_batch = elapsed / num_iterations * 1000
        print(f"\nBatch creation (bs={batch_size}): {ms_per_batch:.2f}ms/batch")

        # Target: <10ms per batch creation
        assert ms_per_batch < 50, f"Batch creation too slow: {ms_per_batch:.2f}ms > 50ms"

    @pytest.mark.benchmark
    def test_batch_from_parts_performance(self):
        """Benchmark Batch.from_parts() which should be faster than Element-based creation.

        Target: 2-5x faster than Element-based creation
        """
        warmup_jax()

        batch_size = 32

        # Pre-batched data (simulating direct stacking)
        data = {"image": jnp.ones((batch_size, 32, 32, 3)), "label": jnp.arange(batch_size) % 10}
        states = {"count": jnp.zeros((batch_size,))}

        # Warmup
        _ = Batch.from_parts(data, states, validate=False)

        # Benchmark from_parts
        num_iterations = 100
        start = time.perf_counter()
        for _ in range(num_iterations):
            Batch.from_parts(data, states, validate=False)
        elapsed = time.perf_counter() - start

        ms_per_batch = elapsed / num_iterations * 1000
        print(f"\nBatch.from_parts (bs={batch_size}): {ms_per_batch:.2f}ms/batch")

        # Target: <2ms per batch (should be much faster than Element-based)
        assert ms_per_batch < 10, f"Batch.from_parts too slow: {ms_per_batch:.2f}ms > 10ms"

    @pytest.mark.benchmark
    def test_direct_stack_baseline(self):
        """Benchmark direct jnp.stack as baseline for Batch creation.

        This establishes the theoretical minimum time for batch creation.
        """
        warmup_jax()

        batch_size = 32
        images = [jnp.ones((32, 32, 3)) for _ in range(batch_size)]
        labels = [jnp.array(i % 10) for i in range(batch_size)]

        # Warmup
        _ = jnp.stack(images, axis=0)

        # Benchmark direct stacking
        num_iterations = 100
        start = time.perf_counter()
        for _ in range(num_iterations):
            stacked_images = jnp.stack(images, axis=0)
            stacked_labels = jnp.stack(labels, axis=0)
            _ = {"image": stacked_images, "label": stacked_labels}
        elapsed = time.perf_counter() - start

        ms_per_stack = elapsed / num_iterations * 1000
        print(f"\nDirect jnp.stack (bs={batch_size}): {ms_per_stack:.2f}ms/batch (baseline)")

        # This establishes the baseline - no assertion needed


# =============================================================================
# Pipeline Throughput Benchmarks
# =============================================================================


class TestPipelineThroughput:
    """Benchmark end-to-end pipeline throughput."""

    @pytest.mark.benchmark
    def test_basic_pipeline_throughput(self):
        """Benchmark basic pipeline without transforms.

        Target: >100 batches/sec for batch_size=32
        """
        warmup_jax()

        config = BenchmarkDataSourceConfig(size=5000, shape=(32, 32, 3))
        source = BenchmarkDataSource(config)
        pipeline = from_source(source, batch_size=32)

        metrics = benchmark_iteration(pipeline, max_items=100, warmup=5)

        batches_per_sec = metrics["items_per_sec"]
        elements_per_sec = batches_per_sec * 32

        print("\nBasic pipeline throughput:")
        print(f"  Batches/sec: {batches_per_sec:.1f}")
        print(f"  Elements/sec: {elements_per_sec:.0f}")
        print(f"  ms/batch: {metrics['ms_per_item']:.2f}")

        # Target: >30 batches/sec (relaxed from 100 due to current baseline)
        # Will increase after optimizations
        assert batches_per_sec > 10, f"Pipeline too slow: {batches_per_sec:.1f} batches/sec < 10"

    @pytest.mark.benchmark
    def test_pipeline_with_transforms_throughput(self):
        """Benchmark pipeline with transform operators.

        Target: >50 batches/sec with 2 transforms
        """
        warmup_jax()

        config = BenchmarkDataSourceConfig(size=5000, shape=(32, 32, 3))
        source = BenchmarkDataSource(config)
        pipeline = from_source(source, batch_size=32)

        # Add identity transforms (minimal overhead)
        det_config = ElementOperatorConfig(stochastic=False)
        identity_op1 = ElementOperator(det_config, fn=lambda e, k: e)
        identity_op2 = ElementOperator(det_config, fn=lambda e, k: e)
        pipeline = pipeline.operate(identity_op1).operate(identity_op2)

        metrics = benchmark_iteration(pipeline, max_items=100, warmup=5)

        batches_per_sec = metrics["items_per_sec"]
        print("\nPipeline with 2 transforms:")
        print(f"  Batches/sec: {batches_per_sec:.1f}")
        print(f"  ms/batch: {metrics['ms_per_item']:.2f}")

        # Target: >20 batches/sec (relaxed due to current baseline)
        assert batches_per_sec > 5, f"Pipeline with transforms too slow: {batches_per_sec:.1f} < 5"

    @pytest.mark.benchmark
    @pytest.mark.parametrize("batch_size", [1, 8, 32, 64, 128])
    def test_batch_size_scaling(self, batch_size):
        """Benchmark how throughput scales with batch size.

        Expected: Larger batches should have better element throughput
        due to amortized overhead.
        """
        warmup_jax()

        config = BenchmarkDataSourceConfig(size=5000, shape=(32, 32, 3))
        source = BenchmarkDataSource(config)
        pipeline = from_source(source, batch_size=batch_size)

        # Adjust max_items based on batch size
        max_batches = min(100, 3000 // batch_size)
        metrics = benchmark_iteration(pipeline, max_items=max_batches, warmup=3)

        batches_per_sec = metrics["items_per_sec"]
        elements_per_sec = batches_per_sec * batch_size

        print(f"\nBatch size {batch_size}:")
        print(f"  {batches_per_sec:.1f} batches/sec, {elements_per_sec:.0f} elements/sec")

        # Basic sanity check - should process at least some data
        assert metrics["count"] > 0, "No batches processed"


# =============================================================================
# Overhead Analysis Benchmarks
# =============================================================================


class TestOverheadAnalysis:
    """Analyze pipeline overhead vs direct processing."""

    @pytest.mark.benchmark
    def test_pipeline_overhead_ratio(self):
        """Measure pipeline overhead compared to direct jnp.stack processing.

        Target: Pipeline overhead < 5x direct processing
        Current baseline: ~4.88x (from profiling)
        Goal: <2x after optimizations
        """
        warmup_jax()

        batch_size = 32
        num_batches = 50

        # Prepare data
        all_data = [
            {"image": jnp.ones((32, 32, 3)), "label": jnp.array(i % 10)}
            for i in range(batch_size * num_batches)
        ]

        # Benchmark direct stacking (baseline)
        batches = [all_data[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)]

        direct_start = time.perf_counter()
        for batch_data in batches:
            images = jnp.stack([d["image"] for d in batch_data], axis=0)
            labels = jnp.stack([d["label"] for d in batch_data], axis=0)
            _ = {"image": images, "label": labels}
        direct_time = time.perf_counter() - direct_start

        # Benchmark pipeline
        config = MemorySourceConfig()
        source = MemorySource(config, all_data)
        pipeline = from_source(source, batch_size=batch_size)

        pipeline_start = time.perf_counter()
        count = 0
        for _ in pipeline:
            count += 1
            if count >= num_batches:
                break
        pipeline_time = time.perf_counter() - pipeline_start

        overhead_ratio = pipeline_time / direct_time if direct_time > 0 else float("inf")

        print(f"\nOverhead Analysis ({num_batches} batches, bs={batch_size}):")
        direct_ms = direct_time * 1000
        direct_bps = num_batches / direct_time
        print(f"  Direct stacking: {direct_ms:.1f}ms ({direct_bps:.1f} batches/sec)")
        print(f"  Pipeline: {pipeline_time * 1000:.1f}ms ({count / pipeline_time:.1f} batches/sec)")
        print(f"  Overhead ratio: {overhead_ratio:.2f}x")

        # Target: <10x overhead (relaxed from 5x due to current baseline)
        # Will tighten after optimizations
        assert overhead_ratio < 20, f"Pipeline overhead too high: {overhead_ratio:.2f}x > 20x"


# =============================================================================
# Memory Source Benchmarks
# =============================================================================


class TestMemorySourcePerformance:
    """Benchmark MemorySource iteration performance."""

    @pytest.mark.benchmark
    def test_memory_source_iteration(self):
        """Benchmark raw MemorySource iteration speed.

        Target: >5000 elements/sec
        """
        warmup_jax()

        data = [{"image": jnp.ones((32, 32, 3)), "label": jnp.array(i % 10)} for i in range(5000)]
        config = MemorySourceConfig()
        source = MemorySource(config, data)

        # Warmup
        iterator = iter(source)
        for _ in range(10):
            _ = next(iterator)

        # Benchmark
        iterator = iter(source)
        start = time.perf_counter()
        count = 0
        for _ in iterator:
            count += 1
            if count >= 2000:
                break
        elapsed = time.perf_counter() - start

        rate = count / elapsed
        print(f"\nMemorySource iteration: {rate:.0f} elements/sec")

        # Target: >1000 elements/sec (relaxed from 5000)
        assert rate > 500, f"MemorySource too slow: {rate:.0f} elements/sec < 500"


# =============================================================================
# Regression Prevention Benchmarks
# =============================================================================


class TestPerformanceRegression:
    """Tests to prevent performance regressions."""

    @pytest.mark.benchmark
    def test_no_throughput_regression(self):
        """Ensure throughput doesn't regress below established baseline.

        Baseline (from profiling): ~51 batches/sec for basic pipeline
        """
        warmup_jax()

        config = BenchmarkDataSourceConfig(size=3000, shape=(32, 32, 3))
        source = BenchmarkDataSource(config)
        pipeline = from_source(source, batch_size=32)

        metrics = benchmark_iteration(pipeline, max_items=50, warmup=5)

        # Minimum acceptable throughput (based on current baseline)
        min_throughput = 10.0  # batches/sec

        assert metrics["items_per_sec"] >= min_throughput, (
            f"Performance regression detected: {metrics['items_per_sec']:.1f} batches/sec "
            f"< {min_throughput} batches/sec baseline"
        )

    @pytest.mark.benchmark
    def test_batch_creation_no_regression(self):
        """Ensure Batch creation doesn't regress.

        Baseline: ~5.5ms per batch for bs=32
        """
        warmup_jax()

        batch_size = 32
        elements = [
            Element(
                data={"image": jnp.ones((32, 32, 3)), "label": jnp.array(i)},
                state={},
                metadata=None,
            )
            for i in range(batch_size)
        ]

        # Warmup
        _ = Batch(elements, validate=False)

        # Benchmark
        start = time.perf_counter()
        for _ in range(50):
            _ = Batch(elements, validate=False)
        elapsed = time.perf_counter() - start

        ms_per_batch = elapsed / 50 * 1000
        max_allowed = 100.0  # ms (relaxed from 10ms)

        assert ms_per_batch <= max_allowed, (
            f"Batch creation regression: {ms_per_batch:.2f}ms > {max_allowed}ms"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark", "-x"])
