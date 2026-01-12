"""Benchmark tests for Datarax data sources."""

import time
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from datarax import DAGExecutor
from flax import nnx
from datarax.sources import MemorySource
from datarax.sources.memory_source import MemorySourceConfig
from datarax.operators import ElementOperator, ElementOperatorConfig

# Use mark.benchmark to indicate these are performance tests
pytestmark = pytest.mark.benchmark


@pytest.fixture
def large_sample_data() -> list[dict[str, Any]]:
    """Generate larger sample data for benchmarking."""
    return [{"image": jnp.ones((32, 32, 3)), "label": jnp.array(i % 10)} for i in range(10000)]


@pytest.fixture
def memory_source(large_sample_data):
    """Create a memory data source for benchmarking."""

    return MemorySource(MemorySourceConfig(), large_sample_data)


@pytest.fixture
def data_stream(memory_source):
    """Create a data stream for benchmarking."""

    # Create DAGExecutor with proper RNG initialization
    rngs = nnx.Rngs(default=0, augment=1)
    return DAGExecutor(rngs=rngs).add(memory_source)


def measure_iteration_time(data_stream, batch_size, num_batches=100) -> tuple[float, float]:
    """Measure time to iterate through batches."""
    # Set up the batched stream
    batched_stream = data_stream.batch(batch_size=batch_size)

    # Warmup
    for _, _ in zip(range(10), batched_stream):
        pass

    # Reset the pipeline to clear any buffered state
    batched_stream.reset()

    # Create new iterator for timing
    batched_stream_iter = iter(batched_stream)

    start_time = time.time()
    batch_times = []

    for _, batch in zip(range(num_batches), batched_stream_iter):
        batch_start = time.time()
        # Force materialization of arrays if needed
        _ = jax.tree.map(lambda x: x + 0 if isinstance(x, jax.Array) else x, batch)
        batch_end = time.time()
        batch_times.append(batch_end - batch_start)

    end_time = time.time()

    total_time = end_time - start_time
    # Handle case where no batches were produced
    if len(batch_times) == 0:
        return 0.0, 0.0
    avg_batch_time = sum(batch_times) / len(batch_times)

    return total_time, avg_batch_time


def test_memory_source_iteration_benchmark(memory_source, large_sample_data):
    """Benchmark memory source iteration with different batch sizes."""
    batch_sizes = [1, 8, 16, 32, 64, 128, 256]

    results = {}

    for batch_size in batch_sizes:
        # Create fresh pipeline for each batch size test

        # Create fresh source and executor for clean state
        fresh_source = MemorySource(MemorySourceConfig(), large_sample_data)
        rngs = nnx.Rngs(default=0, augment=1)
        fresh_stream = DAGExecutor(rngs=rngs).add(fresh_source)

        total_time, avg_batch_time = measure_iteration_time(
            fresh_stream, batch_size, num_batches=100
        )

        # Skip if no batches were produced (e.g., batch_size > data_size)
        if avg_batch_time == 0:
            print(f"\nBatch size: {batch_size} - Skipped (insufficient data)")
            continue

        # Record results for reporting
        results[batch_size] = {
            "total_time": total_time,
            "avg_batch_time": avg_batch_time,
            "examples_per_second": batch_size / avg_batch_time,
        }

        # Log results
        print("\nBatch size: " + str(batch_size))
        print("Total time for 100 batches: " + str(total_time) + "s")
        print("Average batch processing time: " + str(avg_batch_time) + "s")
        print("Examples per second: " + str(batch_size / avg_batch_time))

    # Verify performance is within acceptable range
    # This is an example threshold - adjust based on actual performance
    assert results[32]["examples_per_second"] > 100, "Performance below threshold"


@pytest.mark.integration
def test_transform_performance(data_stream):
    """Benchmark performance of transformation operations."""
    from datarax.core.element_batch import Element

    # Simple normalize transform - Element API: fn(element, key) -> element
    def normalize(element: Element, key: jax.Array) -> Element:
        new_data = {
            "image": element.data["image"] / 255.0,
            "label": jnp.asarray(element.data["label"]),
        }
        return element.replace(data=new_data)

    config = ElementOperatorConfig(stochastic=False)
    normalizer = ElementOperator(config, fn=normalize)

    # Set up the pipeline with transform - batch first, then operate
    pipeline = data_stream.batch(batch_size=32).operate(normalizer)

    # Warmup
    for _, _ in zip(range(10), pipeline):
        pass

    # Measure performance
    start_time = time.time()
    batches_processed = 0

    for _, _ in zip(range(100), pipeline):
        batches_processed += 1

    end_time = time.time()

    total_time = end_time - start_time
    examples_per_second = batches_processed * 32 / total_time

    print("\nTransform Pipeline:")
    print("Total time for " + str(batches_processed) + " batches: " + str(total_time) + "s")
    print("Examples per second: " + str(examples_per_second))

    # Verify performance is within acceptable range
    assert examples_per_second > 100, "Transform performance below threshold"


@pytest.mark.integration
def test_augment_performance(data_stream, rng_key):
    """Benchmark performance of augmentation operations."""
    from datarax.core.element_batch import Element

    # Simple augmentation (horizontal flip with 50% probability)
    # Element API: fn(element, key) -> element
    def augment_fn(element: Element, key: jax.Array) -> Element:
        flip = jax.random.uniform(key) > 0.5
        # Extract data before jax.lax.cond to avoid tracing issues
        image = element.data["image"]
        label = element.data["label"]
        new_image = jax.lax.cond(flip, lambda x: jnp.flip(x, axis=1), lambda x: x, image)
        new_data = {"image": new_image, "label": label}
        return element.replace(data=new_data)

    # Set up the pipeline with augmentation - batch first, then operate
    config = ElementOperatorConfig(stochastic=True, stream_name="augment")
    rngs = nnx.Rngs(augment=0)
    augmenter = ElementOperator(config, fn=augment_fn, rngs=rngs)
    pipeline = data_stream.batch(batch_size=32).operate(augmenter)

    # Warmup
    for _, _ in zip(range(10), pipeline):
        pass

    # Measure performance
    start_time = time.time()
    batches_processed = 0

    for _, _ in zip(range(100), pipeline):
        batches_processed += 1

    end_time = time.time()

    total_time = end_time - start_time
    examples_per_second = batches_processed * 32 / total_time

    print("\nAugmentation Pipeline:")
    print("Total time for " + str(batches_processed) + " batches: " + str(total_time) + "s")
    print("Examples per second: " + str(examples_per_second))

    # Verify performance is within acceptable range
    assert examples_per_second > 50, "Augmentation performance below threshold"


@pytest.mark.end_to_end
def test_end_to_end_pipeline_benchmark(data_stream, rng_key):
    """Benchmark end-to-end pipeline with operators and batch."""
    from datarax.core.element_batch import Element

    # Define transformations - Element API: fn(element, key) -> element
    def normalize(element: Element, key: jax.Array) -> Element:
        new_data = {"image": element.data["image"] / 255.0, "label": element.data["label"]}
        return element.replace(data=new_data)

    # Define augmentations - Element API: fn(element, key) -> element
    def augment_fn(element: Element, key: jax.Array) -> Element:
        flip = jax.random.uniform(key) > 0.5
        # Extract data before jax.lax.cond to avoid tracing issues
        image = element.data["image"]
        label = element.data["label"]
        new_image = jax.lax.cond(flip, lambda x: jnp.flip(x, axis=1), lambda x: x, image)
        new_data = {"image": new_image, "label": label}
        return element.replace(data=new_data)

    # Create operators
    normalize_config = ElementOperatorConfig(stochastic=False)
    normalizer = ElementOperator(normalize_config, fn=normalize)

    augment_config = ElementOperatorConfig(stochastic=True, stream_name="augment")
    rngs = nnx.Rngs(augment=0)
    augmenter = ElementOperator(augment_config, fn=augment_fn, rngs=rngs)

    # Set up the complete pipeline - batch first, then operators (batch-first principle)
    pipeline = data_stream.batch(batch_size=32).operate(normalizer).operate(augmenter)

    # Warmup
    for _, _ in zip(range(10), pipeline):
        pass

    # Measure performance
    start_time = time.time()
    batches_processed = 0

    for _, _ in zip(range(100), pipeline):
        batches_processed += 1

    end_time = time.time()

    total_time = end_time - start_time
    examples_per_second = batches_processed * 32 / total_time

    print("\nEnd-to-End Pipeline:")
    print("Total time for " + str(batches_processed) + " batches: " + str(total_time) + "s")
    print("Examples per second: " + str(examples_per_second))

    # Verify performance is within acceptable range
    assert examples_per_second > 30, "End-to-end performance below threshold"


if __name__ == "__main__":
    # Allow running this file directly (outside pytest) for quick benchmarking
    pytest.main(["-xvs", __file__])
