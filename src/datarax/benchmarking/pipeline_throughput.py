"""Pipeline throughput benchmarking tools.

This module provides utilities for measuring Datarax pipeline throughput,
including performance measurement and reporting.
"""

import time
from typing import Callable

import jax

from datarax.core.element_batch import Batch
from datarax.dag import DAGExecutor


class Timer:
    """A simple timer for benchmarking operations."""

    def __init__(self, name: str | None = None):
        """Initialize a Timer object.

        Args:
            name: Optional name for the timer.
        """
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        """Start the timer when entering a context."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the timer when exiting a context."""
        self.end_time = time.time()

    @property
    def elapsed(self) -> float:
        """Get the elapsed time in seconds.

        Returns:
            The elapsed time in seconds.
        """
        if self.start_time is None:
            raise ValueError("Timer has not been started.")
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time


class PipelineBenchmark:
    """Benchmark utility for measuring Datarax pipeline performance."""

    def __init__(
        self,
        data_stream: DAGExecutor,
        num_batches: int = 50,
        warmup_batches: int = 5,
    ):
        """Initialize a PipelineBenchmark object.

        Args:
            data_stream: The Datarax Pipeline to benchmark.
            num_batches: Number of batches to process for the benchmark.
            warmup_batches: Number of batches to process before starting timing.
        """
        self.data_stream = data_stream
        self.num_batches = num_batches
        self.warmup_batches = warmup_batches
        self.results: dict[str, float] = {}

    def _count_examples(self, batch: Batch) -> int:
        """Count the number of examples in a batch.

        Args:
            batch: A Batch object from the pipeline.

        Returns:
            The number of examples in the batch.
        """
        data = batch.get_data()
        # Check common keys
        for key in ("image", "text", "data", "features"):
            if key in data:
                return data[key].shape[0]
        # Otherwise use the first leaf node's first dimension
        first_value = jax.tree_util.tree_leaves(data)[0]
        return first_value.shape[0]

    def run(self, pipeline_seed: int = 42) -> dict[str, float]:
        """Run the benchmark and collect metrics.

        Args:
            pipeline_seed: Seed for the pipeline iterator.

        Returns:
            Dictionary of benchmark metrics.
        """
        # Create iterator and do warmup
        iterator = iter(self.data_stream)

        # Warmup phase
        for i, _ in zip(range(self.warmup_batches), iterator):
            pass

        # Reset iterator
        iterator = iter(self.data_stream)

        # Benchmark phase
        with Timer() as timer:
            num_batches = 0
            num_examples = 0

            for batch in iterator:
                num_batches += 1
                num_examples += self._count_examples(batch)

                if num_batches >= self.num_batches:
                    break

        # Calculate metrics
        duration = timer.elapsed
        examples_per_second = num_examples / duration
        batches_per_second = num_batches / duration

        # Store results
        self.results = {
            "duration_seconds": duration,
            "batches_processed": num_batches,
            "examples_processed": num_examples,
            "examples_per_second": examples_per_second,
            "batches_per_second": batches_per_second,
        }

        return self.results

    def print_results(self) -> None:
        """Print benchmark results in a formatted way."""
        if not self.results:
            print("No benchmark results available. Run the benchmark first.")
            return

        print("\nPipeline Benchmark Results:")
        print(f"  Duration: {self.results['duration_seconds']:.4f}s")
        print(f"  Batches Processed: {self.results['batches_processed']}")
        print(f"  Examples Processed: {self.results['examples_processed']}")
        print(f"  Throughput: {self.results['examples_per_second']:.2f} examples/second")
        print(f"  Batch Rate: {self.results['batches_per_second']:.2f} batches/second")


class BatchSizeBenchmark:
    """Benchmark the impact of batch size on pipeline performance."""

    def __init__(
        self,
        data_stream_factory: Callable[[int], DAGExecutor],
        batch_sizes: list[int],
        num_batches: int = 30,
        warmup_batches: int = 5,
    ):
        """Initialize a BatchSizeBenchmark object.

        Args:
            data_stream_factory: Factory function that takes batch_size and returns a Pipeline.
            batch_sizes: List of batch sizes to benchmark.
            num_batches: Number of batches to process for each benchmark.
            warmup_batches: Number of batches to process before starting timing.
        """
        self.data_stream_factory = data_stream_factory
        self.batch_sizes = batch_sizes
        self.num_batches = num_batches
        self.warmup_batches = warmup_batches
        self.results: dict[int, dict[str, float]] = {}

    def run(self, pipeline_seed: int = 42) -> dict[int, dict[str, float]]:
        """Run benchmarks for all batch sizes.

        Args:
            pipeline_seed: Seed for the pipeline iterator.

        Returns:
            Dictionary mapping batch sizes to benchmark metrics.
        """
        for batch_size in self.batch_sizes:
            # Create data stream with the current batch size
            # Factory takes batch_size as parameter to create complete pipeline
            data_stream = self.data_stream_factory(batch_size)

            # Run benchmark
            benchmark = PipelineBenchmark(
                data_stream,
                num_batches=self.num_batches,
                warmup_batches=self.warmup_batches,
            )
            self.results[batch_size] = benchmark.run(pipeline_seed=pipeline_seed)

            # Print progress
            print(f"Completed benchmark for batch_size={batch_size}")

        return self.results

    def print_results(self) -> None:
        """Print benchmark results in a formatted way."""
        if not self.results:
            print("No benchmark results available. Run the benchmark first.")
            return

        print("\nBatch Size Benchmark Results:")
        print("-" * 80)
        print(f"{'Batch Size':^10} | {'Examples/s':^15} | {'Batches/s':^15} | {'Duration (s)':^15}")
        print("-" * 80)

        for batch_size, metrics in sorted(self.results.items()):
            print(
                f"{batch_size:^10} | "
                f"{metrics['examples_per_second']:^15.2f} | "
                f"{metrics['batches_per_second']:^15.2f} | "
                f"{metrics['duration_seconds']:^15.4f}"
            )


class ProfileReport:
    """Generate a performance profile report for a Datarax pipeline."""

    def __init__(self, data_stream: DAGExecutor):
        """Initialize a ProfileReport object.

        Args:
            data_stream: The Datarax Pipeline to profile.
        """
        self.data_stream = data_stream
        self.metrics: dict[str, dict[str, float]] = {}

    def run(self, num_batches: int = 10, pipeline_seed: int = 42) -> None:
        """Run the profiling and collect metrics.

        Args:
            num_batches: Number of batches to process.
            pipeline_seed: Seed for the pipeline iterator.
        """
        # Profile end-to-end pipeline
        benchmark = PipelineBenchmark(self.data_stream, num_batches=num_batches, warmup_batches=2)
        self.metrics["pipeline"] = benchmark.run(pipeline_seed=pipeline_seed)

        # Get pipeline component timing to analyze bottlenecks
        # This is a simplified approach - more complete profiling would
        # instrument individual transforms and operations
        self._profile_pipeline_components(num_batches, pipeline_seed)

    def _profile_pipeline_components(self, num_batches: int, pipeline_seed: int) -> None:
        """Profile individual pipeline components to identify bottlenecks.

        Args:
            num_batches: Number of batches to process.
            pipeline_seed: Seed for the pipeline iterator.
        """
        # Profile data source only (if possible)
        try:
            # TODO: Re-enable when Pipeline abstraction is resolved
            # source_only = Pipeline(self.data_stream.source).batch(batch_size=32)
            # benchmark = PipelineBenchmark(source_only, num_batches=num_batches, warmup_batches=2)
            # self.metrics["source_only"] = benchmark.run(pipeline_seed=pipeline_seed)
            pass
        except Exception:  # nosec B110
            pass

    def print_report(self) -> None:
        """Print the profiling report."""
        if not self.metrics:
            print("No profiling results available. Run the profiling first.")
            return

        print("\n===== Datarax Pipeline Performance Profile =====\n")

        print("Overall Pipeline Performance:")
        print(
            f"  Throughput: {self.metrics['pipeline']['examples_per_second']:.2f} examples/second"
        )
        print(f"  Batch Rate: {self.metrics['pipeline']['batches_per_second']:.2f} batches/second")

        if "source_only" in self.metrics:
            source_throughput = self.metrics["source_only"]["examples_per_second"]
            pipeline_throughput = self.metrics["pipeline"]["examples_per_second"]

            # Calculate overhead due to transforms/augmentations
            if source_throughput > 0:
                overhead_pct = (1 - pipeline_throughput / source_throughput) * 100
                print("\nData Source vs Full Pipeline:")
                print(f"  Source Only: {source_throughput:.2f} examples/second")
                print(f"  Full Pipeline: {pipeline_throughput:.2f} examples/second")
                print(f"  Processing Overhead: {overhead_pct:.1f}%")

                if overhead_pct > 50:
                    print("\nRecommendation: Consider optimizing transforms and augmenters")
                    print("  - Check for expensive operations in transforms")
                    print("  - Consider using prefetch() to parallelize processing")
                    print("  - Profile individual operations to identify bottlenecks")


def benchmark_comparison(
    configurations: dict[str, DAGExecutor],
    num_batches: int = 30,
    warmup_batches: int = 5,
    pipeline_seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Compare performance of multiple pipeline configurations.

    Args:
        configurations: Dictionary mapping configuration names to Pipelines.
        num_batches: Number of batches to process for each benchmark.
        warmup_batches: Number of batches to warm up before benchmarking.
        pipeline_seed: Seed for the pipeline iterator.

    Returns:
        Dictionary mapping configuration names to benchmark metrics.
    """
    results = {}

    for name, data_stream in configurations.items():
        benchmark = PipelineBenchmark(
            data_stream,
            num_batches=num_batches,
            warmup_batches=warmup_batches,
        )
        results[name] = benchmark.run(pipeline_seed=pipeline_seed)

        # Print progress
        print(f"Completed benchmark for configuration: {name}")

    # Print comparison
    print("\nConfiguration Comparison:")
    print("-" * 90)
    print(f"{'Configuration':^20} | {'Examples/s':^15} | {'Batches/s':^15} | {'Duration (s)':^15}")
    print("-" * 90)

    for name, metrics in results.items():
        print(
            f"{name:^20} | "
            f"{metrics['examples_per_second']:^15.2f} | "
            f"{metrics['batches_per_second']:^15.2f} | "
            f"{metrics['duration_seconds']:^15.4f}"
        )

    return results
