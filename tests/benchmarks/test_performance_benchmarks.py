# File: tests/performance/test_performance_benchmarks.py

import time
from dataclasses import dataclass

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule
from datarax.dag import build_source_pipeline
from datarax.memory.shared_memory_manager import SharedMemoryManager
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.utils.console import emit


@dataclass(frozen=True)
class MockDataSourceConfig(StructuralConfig):
    """Configuration for MockDataSource."""

    size: int = 10000


class MockDataSource(DataSourceModule):
    """Mock data source for testing."""

    def __init__(
        self,
        config: MockDataSourceConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        super().__init__(config, rngs=rngs, name=name)
        self.size = config.size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"data": jax.random.normal(jax.random.key(idx), (224, 224, 3)), "value": idx}

    def __iter__(self):
        for i in range(self.size):
            yield self[i]


class TestPerformanceBenchmarks:
    """Performance benchmarks for Datarax components."""

    @pytest.mark.benchmark
    def test_pipeline_throughput(self):
        """Benchmark Pipeline throughput as replacement for DataLoaderModule."""
        # Create large mock dataset using proper DataSourceModule
        source = MockDataSource(MockDataSourceConfig(size=10000))

        # Test different batch sizes
        batch_sizes = [1, 10, 32, 64]
        results = {}

        for batch_size in batch_sizes:
            # Create pipeline using build_source_pipeline
            dag_pipeline = build_source_pipeline(source, batch_size=batch_size)

            start = time.time()
            count = 0
            for _ in dag_pipeline:
                count += 1
                if count >= 100:
                    break
            elapsed = time.time() - start

            throughput = count / elapsed
            results[batch_size] = throughput

            emit(f"Batch size: {batch_size}, Throughput: {throughput:.2f} batches/sec")

        # Verify that Pipeline is processing data at reasonable speeds
        # CI runners are significantly slower than local hardware
        assert results[32] >= 1.0  # At least 1 batch/sec (relaxed for CI)
        assert results[1] >= 10.0  # At least 10 single-item batches/sec

    @pytest.mark.benchmark
    def test_shared_memory_performance(self):
        """Benchmark shared memory performance."""
        manager = SharedMemoryManager()

        # Test different array sizes
        sizes = [(100, 100), (1000, 1000), (5000, 5000)]

        for size in sizes:
            array = jax.random.normal(jax.random.key(0), size)

            # Time making shared
            start = time.time()
            manager.make_shared(f"test_{size}", array)
            make_time = time.time() - start

            # Time retrieval
            start = time.time()
            retrieved = manager.get_shared(f"test_{size}")
            get_time = time.time() - start

            emit(f"Size: {size}, Make: {make_time:.4f}s, Get: {get_time:.4f}s")

            # Verify correctness (allow small floating point differences)
            assert retrieved is not None
            assert jnp.allclose(retrieved, array, rtol=1e-7, atol=1e-7)

        manager.cleanup()

    @pytest.mark.benchmark
    def test_dataset_iteration_performance(self):
        """Benchmark dataset iteration performance."""

        # Create mock source using proper DataSourceModule
        source = MockDataSource(MockDataSourceConfig(size=10000))

        dataset = build_source_pipeline(source, batch_size=1)

        # Add operators (transforms) - deterministic, Element API: fn(element, key) -> element
        config = ElementOperatorConfig(stochastic=False)
        multiply_op = ElementOperator(
            config,
            fn=lambda e, _k: e.replace(
                data={"data": e.data["data"], "value": jnp.asarray(e.data["value"]) * 2}
            ),
            name="multiply_by_2",
        )
        add_op = ElementOperator(
            config,
            fn=lambda e, _k: e.replace(
                data={"data": e.data["data"], "value": jnp.asarray(e.data["value"]) + 1}
            ),
            name="add_1",
        )
        dataset = dataset.operate(multiply_op).operate(add_op)

        # Time iteration
        start = time.time()
        count = 0
        for batch in dataset:
            count += 1
            if count >= 1000:
                break
        elapsed = time.time() - start

        throughput = count / elapsed
        emit(f"Dataset iteration: {throughput:.2f} elements/sec")

        # CI runners may be significantly slower than local hardware
        # Relaxed threshold to avoid flaky failures on shared CI infrastructure
        assert throughput > 10
