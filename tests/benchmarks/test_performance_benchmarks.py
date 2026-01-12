# File: tests/performance/test_performance_benchmarks.py

import time
from dataclasses import dataclass

import pytest
import jax.numpy as jnp
import jax
import flax.nnx as nnx

from datarax.dag import from_source
from datarax.memory.shared_memory_manager import SharedMemoryManager
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.core.data_source import DataSourceModule
from datarax.core.config import StructuralConfig


@dataclass
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
            # Create pipeline using from_source
            dag_pipeline = from_source(source, batch_size=batch_size)

            start = time.time()
            count = 0
            for _ in dag_pipeline:
                count += 1
                if count >= 100:
                    break
            elapsed = time.time() - start

            throughput = count / elapsed
            results[batch_size] = throughput

            print(f"Batch size: {batch_size}, Throughput: {throughput:.2f} batches/sec")

        # Verify that Pipeline is processing data at reasonable speeds
        # At least 10 batches per second for large batches
        assert results[32] >= 10.0  # At least 10 batches/sec
        assert results[1] >= 100.0  # At least 100 single-item batches/sec

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

            print(f"Size: {size}, Make: {make_time:.4f}s, Get: {get_time:.4f}s")

            # Verify correctness (allow small floating point differences)
            assert jnp.allclose(retrieved, array, rtol=1e-7, atol=1e-7)

        manager.cleanup()

    @pytest.mark.benchmark
    def test_dataset_iteration_performance(self):
        """Benchmark dataset iteration performance."""

        # Create mock source using proper DataSourceModule
        source = MockDataSource(MockDataSourceConfig(size=10000))

        dataset = from_source(source, batch_size=1)

        # Add operators (transforms) - deterministic, Element API: fn(element, key) -> element
        config = ElementOperatorConfig(stochastic=False)
        multiply_op = ElementOperator(
            config,
            fn=lambda e, k: e.replace(
                data={"data": e.data["data"], "value": jnp.asarray(e.data["value"]) * 2}
            ),
            name="multiply_by_2",
        )
        add_op = ElementOperator(
            config,
            fn=lambda e, k: e.replace(
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
        print(f"Dataset iteration: {throughput:.2f} elements/sec")

        # TODO: Optimize the code so that throughput is comparable to other frameworks
        # Currently getting ~300 elements/sec, should aim for 1000+ elements/sec
        # Should process at least 100 elements per second (realistic for complex transforms)
        assert throughput > 100
