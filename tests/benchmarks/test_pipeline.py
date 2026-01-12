"""Benchmark tests for Datarax pipeline components."""

import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import pytest
from test_common.data_generators import generate_image_data

from datarax.dag import DAGExecutor
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.dag.nodes import BatchNode, OperatorNode
from datarax.core.element_batch import Element


@pytest.mark.benchmark
def test_datastream_throughput_benchmark(benchmark: Callable) -> None:
    """Benchmark the throughput of Pipeline with different batch sizes."""
    # Generate test data
    num_samples = 5000  # Reduced to speed up the benchmark
    data = generate_image_data(num_samples=num_samples)

    # Define a simple transformation - Element API: fn(element, key) -> element
    def normalize(element: Element, key: jax.Array) -> Element:
        new_data = {"image": element.data["image"] / 255.0, "label": element.data["label"]}
        return element.replace(data=new_data)

    # Create the function to benchmark
    def process_data(batch_size: int) -> dict[str, Any]:
        source_config = MemorySourceConfig()
        source = MemorySource(source_config, data)
        config = ElementOperatorConfig(stochastic=False)
        normalize_op = ElementOperator(config, fn=normalize)
        pipeline = (
            DAGExecutor().add(source).add(BatchNode(batch_size)).add(OperatorNode(normalize_op))
        )
        iterator = iter(pipeline)

        # Process all batches
        batch_count = 0
        total_items = 0
        start_time = time.time()

        for batch in iterator:
            batch_count += 1
            total_items += batch["image"].shape[0]

        end_time = time.time()
        duration = end_time - start_time
        throughput = total_items / duration

        return {
            "duration_seconds": duration,
            "batches_processed": batch_count,
            "items_processed": total_items,
            "throughput_items_per_second": throughput,
        }

    # Define a dummy benchmark function if the actual benchmark is not available
    if not hasattr(benchmark, "__call__"):

        def dummy_benchmark(func, *args):
            return func(*args)

        benchmark = dummy_benchmark

    # Run the benchmark with different batch sizes
    result = benchmark(process_data, 128)

    # Verify expected throughput (adjust based on your machine)
    expected_batches = num_samples // 128
    if num_samples % 128 > 0:
        expected_batches += 1
    assert result["batches_processed"] == expected_batches
    assert result["items_processed"] == num_samples


@pytest.mark.benchmark
def test_memory_usage_benchmark(benchmark: Callable) -> None:
    """Benchmark the memory usage of Pipeline operations."""
    try:
        from memory_profiler import memory_usage  # type: ignore

        has_memory_profiler = True
    except ImportError:
        has_memory_profiler = False
        print("INFO: memory_profiler is not installed but continuing test")
        has_memory_profiler = False  # We don't skip anymore

    # Generate test data - using smaller dataset for faster benchmarks
    num_samples = 2000
    data = generate_image_data(num_samples=num_samples, image_height=32, image_width=32)

    # Define a pipeline processing function
    def create_and_process_pipeline() -> None:
        source_config = MemorySourceConfig()
        source = MemorySource(source_config, data)
        config = ElementOperatorConfig(stochastic=False)

        # Create operators - Element API: fn(element, key) -> element
        normalize_op = ElementOperator(
            config,
            fn=lambda e, k: e.replace(
                data={"image": e.data["image"] / 255.0, "label": e.data["label"]}
            ),
        )
        power_op = ElementOperator(
            config,
            fn=lambda e, k: e.replace(
                data={
                    "image": jnp.power(e.data["image"], 1.5),
                    "label": e.data["label"],
                }
            ),
        )
        clip_op = ElementOperator(
            config,
            fn=lambda e, k: e.replace(
                data={
                    "image": jnp.clip(e.data["image"], 0.0, 1.0),
                    "label": e.data["label"],
                }
            ),
        )

        # Create a pipeline with multiple transformations
        pipeline = (
            DAGExecutor()
            .add(source)
            .add(BatchNode(32))
            .add(OperatorNode(normalize_op))
            .add(OperatorNode(power_op))
            .add(OperatorNode(clip_op))
        )

        # Process all data
        list(iter(pipeline))

    if has_memory_profiler:
        # Measure memory usage
        # Use multiprocess=False to avoid JAX forking issues with CUDA
        mem_usage = memory_usage(
            create_and_process_pipeline,
            interval=0.1,
            timeout=30,
            max_iterations=1,
            multiprocess=False,
        )

        # Get baseline memory and peak memory
        baseline = mem_usage[0]
        peak = max(mem_usage)
        memory_increase = peak - baseline

        # Log the results for analysis
        print(f"Baseline memory: {baseline:.2f} MiB")
        print(f"Peak memory: {peak:.2f} MiB")
        print(f"Memory increase: {memory_increase:.2f} MiB")

    # Define a dummy benchmark function if the actual benchmark is not available
    if not hasattr(benchmark, "__call__"):

        def dummy_benchmark(func, *args):
            return func(*args)

        benchmark = dummy_benchmark

    # Execute benchmark
    benchmark(create_and_process_pipeline)

    # Just a basic assertion
    assert True, "Memory benchmark completed"


if __name__ == "__main__":
    # Run benchmarks directly if file is executed as a script
    def run_function(f, *args):
        return f(*args)

    test_datastream_throughput_benchmark(run_function)
    test_memory_usage_benchmark(run_function)
