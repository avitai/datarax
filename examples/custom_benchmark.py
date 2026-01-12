"""Custom benchmark script for Datarax.

This script demonstrates how to use Datarax's benchmark utilities directly
in Python code, without using the CLI.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax import from_source
from datarax.core.nodes import OperatorNode
from datarax.dag import DAGExecutor
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.sources import MemorySource, MemorySourceConfig
from datarax.benchmarking.pipeline_throughput import (
    BatchSizeBenchmark,
    PipelineBenchmark,
    ProfileReport,
    benchmark_comparison,
)


def generate_sample_image_data(num_samples: int = 1000, image_size: int = 32) -> dict:
    """Generate sample image data for benchmarking.

    Args:
        num_samples: Number of samples to generate.
        image_size: Size of each image (image_size x image_size x 3).

    Returns:
        Dictionary with 'image' and 'label' arrays.
    """
    rng = np.random.RandomState(42)
    return {
        "image": rng.rand(num_samples, image_size, image_size, 3).astype(np.float32),
        "label": rng.randint(0, 10, (num_samples,)).astype(np.int32),
    }


def normalize_transform(element, key=None):
    """Normalize image values to [0, 1] range.

    Args:
        element: Element containing 'image' and 'label' data.
        key: Unused PRNG key (for API compatibility).

    Returns:
        Element with normalized image.
    """
    return element.update_data({"image": element.data["image"] / 255.0})


def random_flip_transform(element, key):
    """Apply random horizontal flip to image.

    Args:
        element: Element containing 'image' and 'label' data.
        key: JAX PRNG key for randomness.

    Returns:
        Element with potentially flipped image.
    """
    image = element.data["image"]
    flip = jax.random.bernoulli(key, 0.5)
    flipped_image = jnp.where(flip, jnp.fliplr(image), image)
    return element.update_data({"image": flipped_image})


def simulated_heavy_transform(element, key):
    """Simulate a compute-intensive operation.

    Args:
        element: Element containing 'image' and 'label' data.
        key: JAX PRNG key (unused in this transform).

    Returns:
        Element with slightly modified image.
    """
    image = element.data["image"]
    # Simulate a compute-intensive operation
    for _ in range(10):
        image = image * 0.99
    return element.update_data({"image": image})


def create_basic_pipeline(batch_size: int = 32) -> DAGExecutor:
    """Create a basic image pipeline with minimal processing.

    Args:
        batch_size: Number of samples per batch.

    Returns:
        DAGExecutor configured with source and normalizer.
    """
    # Generate sample data
    data = generate_sample_image_data()

    # Create data source using config-based API
    source_config = MemorySourceConfig()
    source = MemorySource(source_config, data=data, rngs=nnx.Rngs(0))

    # Create normalizer operator (deterministic)
    normalizer_config = ElementOperatorConfig(stochastic=False)
    normalizer = ElementOperator(normalizer_config, fn=normalize_transform, rngs=nnx.Rngs(0))

    # Build pipeline using DAG-based API
    pipeline = from_source(source, batch_size=batch_size).add(OperatorNode(normalizer))

    return pipeline


def create_advanced_pipeline(batch_size: int = 32) -> DAGExecutor:
    """Create a more complex image pipeline with augmentation.

    Args:
        batch_size: Number of samples per batch.

    Returns:
        DAGExecutor configured with source, normalizer, and augmenters.
    """
    # Generate sample data
    data = generate_sample_image_data()

    # Create data source using config-based API
    source_config = MemorySourceConfig()
    source = MemorySource(source_config, data=data, rngs=nnx.Rngs(0))

    # Create normalizer operator (deterministic)
    normalizer_config = ElementOperatorConfig(stochastic=False)
    normalizer = ElementOperator(normalizer_config, fn=normalize_transform, rngs=nnx.Rngs(0))

    # Create flip augmenter (stochastic)
    flip_config = ElementOperatorConfig(stochastic=True, stream_name="flip")
    flip_augmenter = ElementOperator(flip_config, fn=random_flip_transform, rngs=nnx.Rngs(flip=42))

    # Create heavy transform (stochastic for API consistency)
    heavy_config = ElementOperatorConfig(stochastic=True, stream_name="heavy")
    heavy_transform = ElementOperator(
        heavy_config, fn=simulated_heavy_transform, rngs=nnx.Rngs(heavy=43)
    )

    # Build pipeline using DAG-based API
    pipeline = (
        from_source(source, batch_size=batch_size)
        .add(OperatorNode(normalizer))
        .add(OperatorNode(flip_augmenter))
        .add(OperatorNode(heavy_transform))
    )

    return pipeline


def create_unbatched_pipeline(batch_size: int = 32) -> DAGExecutor:
    """Create a pipeline for batch size benchmarks.

    This factory creates a complete pipeline with the specified batch size.
    BatchSizeBenchmark calls this with different batch sizes to compare performance.

    Args:
        batch_size: The batch size to use for this pipeline instance.

    Returns:
        DAGExecutor configured with the specified batch size.
    """
    # Generate sample data
    data = generate_sample_image_data()

    # Create data source using config-based API
    source_config = MemorySourceConfig()
    source = MemorySource(source_config, data=data, rngs=nnx.Rngs(0))

    # Create normalizer operator (deterministic)
    normalizer_config = ElementOperatorConfig(stochastic=False)
    normalizer = ElementOperator(normalizer_config, fn=normalize_transform, rngs=nnx.Rngs(0))

    # Build complete pipeline WITH batching at the specified size
    # Operators are added AFTER BatchNode so they receive Batch objects
    pipeline = from_source(source, batch_size=batch_size, enforce_batch=True).add(
        OperatorNode(normalizer)
    )

    return pipeline


def run_pipeline_benchmark():
    """Run a basic pipeline benchmark."""
    print("\n=== Running Pipeline Benchmark ===")
    pipeline = create_basic_pipeline(batch_size=32)

    benchmark = PipelineBenchmark(
        pipeline,
        num_batches=50,
        warmup_batches=5,
    )

    print("Running benchmark...")
    results = benchmark.run(pipeline_seed=42)
    benchmark.print_results()

    return results


def run_comparison_benchmark():
    """Run a comparison benchmark between different pipelines."""
    print("\n=== Running Comparison Benchmark ===")

    configurations = {
        "basic": create_basic_pipeline(batch_size=32),
        "advanced": create_advanced_pipeline(batch_size=32),
    }

    print("Comparing pipelines...")
    results = benchmark_comparison(
        configurations,
        num_batches=30,
        warmup_batches=5,
    )

    print("\nComparison Results:")
    print("-" * 80)
    print("Configuration |   Examples/s    |    Batches/s    |  Duration (s)  ")
    print("-" * 80)
    for name, metrics in results.items():
        print(
            f"{name:^13} | {metrics['examples_per_second']:^15.2f} | "
            f"{metrics['batches_per_second']:^15.2f} | "
            f"{metrics['duration_seconds']:^14.4f}"
        )

    return results


def run_profile_report():
    """Run a profile report."""
    print("\n=== Running Profile Report ===")

    pipeline = create_advanced_pipeline(batch_size=32)

    profile = ProfileReport(pipeline)

    print("Running profile...")
    profile.run(num_batches=10, pipeline_seed=42)
    profile.print_report()

    return profile.metrics


def run_batch_size_benchmark():
    """Run a batch size benchmark."""
    print("\n=== Running Batch Size Benchmark ===")

    batch_sizes = [8, 16, 32, 64, 128]

    benchmark = BatchSizeBenchmark(
        data_stream_factory=create_unbatched_pipeline,
        batch_sizes=batch_sizes,
        num_batches=30,
        warmup_batches=5,
    )

    print(f"Running batch size benchmark with sizes {batch_sizes}...")
    results = benchmark.run(pipeline_seed=42)

    print("\nBatch Size Benchmark Results:")
    print("-" * 80)
    print("Batch Size |   Examples/s    |    Batches/s    |  Duration (s)  ")
    print("-" * 80)
    for batch_size, metrics in results.items():
        print(
            f"{batch_size:^10} | {metrics['examples_per_second']:^15.2f} | "
            f"{metrics['batches_per_second']:^15.2f} | "
            f"{metrics['duration_seconds']:^14.4f}"
        )

    return results


def main():
    """Run all benchmarks."""
    start_time = time.time()

    # Run various benchmarks
    pipeline_results = run_pipeline_benchmark()
    comparison_results = run_comparison_benchmark()
    batch_size_results = run_batch_size_benchmark()
    profile_results = run_profile_report()

    # Save all results
    all_results = {
        "pipeline_benchmark": pipeline_results,
        "comparison_benchmark": comparison_results,
        "batch_size_benchmark": batch_size_results,
        "profile_report": profile_results,
    }

    print(f"\nAll benchmarks completed in {time.time() - start_time:.2f} seconds")

    return all_results


if __name__ == "__main__":
    main()
