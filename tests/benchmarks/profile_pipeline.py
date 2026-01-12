"""Profile script for identifying pipeline performance bottlenecks.

This script provides detailed profiling information about the Datarax pipeline
to identify optimization opportunities.
"""

import time
import cProfile
import pstats
from io import StringIO
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from datarax.dag import DAGExecutor
from datarax.sources import MemorySource
from datarax.sources.memory_source import MemorySourceConfig
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.core.data_source import DataSourceModule
from datarax.core.config import StructuralConfig


@dataclass
class MockDataSourceConfig(StructuralConfig):
    """Configuration for MockDataSource."""

    size: int = 10000


class MockDataSource(DataSourceModule):
    """Mock data source for profiling."""

    def __init__(
        self,
        config: MockDataSourceConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        super().__init__(config, rngs=rngs, name=name)
        self.size = config.size
        # Pre-generate all data for consistent profiling
        self._data = [
            {"data": jax.random.normal(jax.random.key(i), (224, 224, 3)), "value": i}
            for i in range(self.size)
        ]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self._data[idx]

    def __iter__(self):
        for i in range(self.size):
            yield self._data[i]


def profile_basic_pipeline(num_samples: int = 1000, batch_size: int = 32) -> dict[str, float]:
    """Profile basic pipeline iteration without transforms."""
    # Generate test data
    data = [
        {"image": jnp.ones((32, 32, 3)), "label": jnp.array(i % 10)} for i in range(num_samples)
    ]

    config = MemorySourceConfig()
    source = MemorySource(config, data)

    # Create pipeline
    pipeline = DAGExecutor().add(source).batch(batch_size=batch_size)

    # Profile iteration
    timings = {
        "iterator_creation": 0.0,
        "batch_processing": 0.0,
        "total_batches": 0,
        "total_time": 0.0,
    }

    start_total = time.perf_counter()

    start_iter = time.perf_counter()
    iterator = iter(pipeline)
    timings["iterator_creation"] = time.perf_counter() - start_iter

    start_batch = time.perf_counter()
    batch_count = 0
    for batch in iterator:
        batch_count += 1
        if batch_count >= 50:
            break
    timings["batch_processing"] = time.perf_counter() - start_batch
    timings["total_batches"] = batch_count
    timings["total_time"] = time.perf_counter() - start_total

    return timings


def profile_with_transforms(
    num_samples: int = 1000, batch_size: int = 32, num_transforms: int = 2
) -> dict[str, float]:
    """Profile pipeline with transforms."""
    # Generate test data
    data = [
        {"image": jnp.ones((32, 32, 3)), "label": jnp.array(i % 10)} for i in range(num_samples)
    ]

    config = MemorySourceConfig()
    source = MemorySource(config, data)

    # Create pipeline with transforms
    pipeline = DAGExecutor().add(source).batch(batch_size=batch_size)

    # Add identity transforms
    det_config = ElementOperatorConfig(stochastic=False)
    for i in range(num_transforms):
        identity_op = ElementOperator(det_config, fn=lambda e, k: e)
        pipeline = pipeline.operate(identity_op)

    # Profile iteration
    timings = {
        "iterator_creation": 0.0,
        "batch_processing": 0.0,
        "total_batches": 0,
        "total_time": 0.0,
    }

    start_total = time.perf_counter()

    start_iter = time.perf_counter()
    iterator = iter(pipeline)
    timings["iterator_creation"] = time.perf_counter() - start_iter

    start_batch = time.perf_counter()
    batch_count = 0
    for batch in iterator:
        batch_count += 1
        if batch_count >= 50:
            break
    timings["batch_processing"] = time.perf_counter() - start_batch
    timings["total_batches"] = batch_count
    timings["total_time"] = time.perf_counter() - start_total

    return timings


def profile_component_breakdown(num_samples: int = 1000, batch_size: int = 32) -> dict[str, float]:
    """Profile individual components of the pipeline."""
    # Generate test data
    data = [
        {"image": jnp.ones((32, 32, 3)), "label": jnp.array(i % 10)} for i in range(num_samples)
    ]

    timings = {}

    # 1. Profile MemorySource iteration
    config = MemorySourceConfig()
    source = MemorySource(config, data)

    start = time.perf_counter()
    count = 0
    for item in source:
        count += 1
        if count >= 500:
            break
    timings["source_iteration_500"] = time.perf_counter() - start

    # 2. Profile Element creation
    from datarax.core.element_batch import Element

    start = time.perf_counter()
    elements = []
    for item in data[:500]:
        elements.append(Element(data=item, state={}, metadata=None))
    timings["element_creation_500"] = time.perf_counter() - start

    # 3. Profile Batch creation from Elements
    from datarax.core.element_batch import Batch

    start = time.perf_counter()
    Batch(elements[:batch_size], validate=False)
    timings[f"batch_creation_{batch_size}"] = time.perf_counter() - start

    # 4. Profile jnp.stack operations directly
    images = [jnp.ones((32, 32, 3)) for _ in range(batch_size)]
    start = time.perf_counter()
    jnp.stack(images, axis=0)
    timings["jnp_stack_batch"] = time.perf_counter() - start

    # 5. Profile tree.map stacking
    element_data_list = [
        {"image": jnp.ones((32, 32, 3)), "label": jnp.array(i)} for i in range(batch_size)
    ]
    start = time.perf_counter()
    jax.tree.map(lambda *arrays: jnp.stack(arrays, axis=0), *element_data_list)
    timings["tree_map_stack"] = time.perf_counter() - start

    return timings


def profile_with_cprofile(num_samples: int = 500, batch_size: int = 32) -> str:
    """Run cProfile on the pipeline and return formatted stats."""
    # Generate test data
    data = [
        {"image": jnp.ones((32, 32, 3)), "label": jnp.array(i % 10)} for i in range(num_samples)
    ]

    def run_pipeline():
        config = MemorySourceConfig()
        source = MemorySource(config, data)
        pipeline = DAGExecutor().add(source).batch(batch_size=batch_size)

        for batch in pipeline:
            pass  # Just iterate through

    # Profile with cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    run_pipeline()
    profiler.disable()

    # Format output
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(30)  # Top 30 functions

    return stream.getvalue()


def run_all_profiles():
    """Run all profiling scenarios and print results."""
    print("=" * 80)
    print("Datarax Pipeline Performance Profile")
    print("=" * 80)

    # Warm up JAX
    print("\nWarming up JAX...")
    _ = jnp.ones((100, 100)) @ jnp.ones((100, 100))
    print("Done.\n")

    # 1. Basic pipeline
    print("-" * 40)
    print("1. Basic Pipeline (no transforms)")
    print("-" * 40)
    timings = profile_basic_pipeline(num_samples=2000, batch_size=32)
    print(f"  Iterator creation: {timings['iterator_creation'] * 1000:.2f} ms")
    batches = timings["total_batches"]
    batch_proc_ms = timings["batch_processing"] * 1000
    print(f"  Batch processing ({batches} batches): {batch_proc_ms:.2f} ms")
    time_per_batch = timings["batch_processing"] / max(batches, 1) * 1000
    print(f"  Time per batch: {time_per_batch:.2f} ms")
    print(f"  Batches/sec: {timings['total_batches'] / max(timings['total_time'], 0.001):.2f}")

    # 2. With transforms
    print("\n" + "-" * 40)
    print("2. Pipeline with 2 Identity Transforms")
    print("-" * 40)
    timings = profile_with_transforms(num_samples=2000, batch_size=32, num_transforms=2)
    print(f"  Iterator creation: {timings['iterator_creation'] * 1000:.2f} ms")
    batches = timings["total_batches"]
    batch_proc_ms = timings["batch_processing"] * 1000
    print(f"  Batch processing ({batches} batches): {batch_proc_ms:.2f} ms")
    time_per_batch = timings["batch_processing"] / max(batches, 1) * 1000
    print(f"  Time per batch: {time_per_batch:.2f} ms")
    print(f"  Batches/sec: {timings['total_batches'] / max(timings['total_time'], 0.001):.2f}")

    # 3. Component breakdown
    print("\n" + "-" * 40)
    print("3. Component Breakdown")
    print("-" * 40)
    timings = profile_component_breakdown()
    for name, time_val in timings.items():
        print(f"  {name}: {time_val * 1000:.2f} ms")

    # 4. cProfile output
    print("\n" + "-" * 40)
    print("4. cProfile Analysis (Top Functions)")
    print("-" * 40)
    profile_output = profile_with_cprofile(num_samples=1000, batch_size=32)
    print(profile_output)

    # 5. Comparison: Direct vs Pipeline
    print("\n" + "-" * 40)
    print("5. Direct Processing vs Pipeline Overhead")
    print("-" * 40)

    # Direct jnp.stack for comparison
    num_batches = 50
    batch_size = 32
    data_per_batch = [
        [{"image": jnp.ones((32, 32, 3)), "label": jnp.array(i)} for i in range(batch_size)]
        for _ in range(num_batches)
    ]

    start = time.perf_counter()
    for batch_data in data_per_batch:
        # Direct stacking (what an optimal implementation would do)
        jnp.stack([d["image"] for d in batch_data], axis=0)
        jnp.stack([d["label"] for d in batch_data], axis=0)
    direct_time = time.perf_counter() - start

    print(f"  Direct jnp.stack ({num_batches} batches): {direct_time * 1000:.2f} ms")
    print(f"  Direct batches/sec: {num_batches / direct_time:.2f}")

    # Compare with pipeline
    all_data = [item for batch_data in data_per_batch for item in batch_data]
    config = MemorySourceConfig()
    source = MemorySource(config, all_data)
    pipeline = DAGExecutor().add(source).batch(batch_size=batch_size)

    start = time.perf_counter()
    batch_count = 0
    for batch in pipeline:
        batch_count += 1
        if batch_count >= num_batches:
            break
    pipeline_time = time.perf_counter() - start

    print(f"  Pipeline ({batch_count} batches): {pipeline_time * 1000:.2f} ms")
    print(f"  Pipeline batches/sec: {batch_count / pipeline_time:.2f}")
    print(f"  Overhead ratio: {pipeline_time / direct_time:.2f}x")


if __name__ == "__main__":
    run_all_profiles()
