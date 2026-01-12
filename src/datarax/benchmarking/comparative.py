"""Comparative benchmarking tools for Datarax.

This module provides utilities for comparing Datarax performance against
other data processing libraries and different pipeline configurations.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import jax

from datarax.benchmarking.profiler import AdvancedProfiler, ProfileResult


@dataclass
class BenchmarkComparison:
    """Results of comparing multiple benchmark configurations.

    Attributes:
        configurations: Dictionary mapping config names to results
        best_config: Name of the best-performing configuration
        worst_config: Name of the worst-performing configuration
        metrics_summary: Summary statistics across configurations
        timestamp: When the comparison was performed
    """

    configurations: dict[str, ProfileResult] = field(default_factory=dict)
    best_config: str | None = None
    worst_config: str | None = None
    metrics_summary: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def add_result(self, config_name: str, result: ProfileResult) -> None:
        """Add a benchmark result for a configuration."""
        self.configurations[config_name] = result
        self._update_summary()

    def _update_summary(self) -> None:
        """Update summary statistics."""
        if not self.configurations:
            return

        # Extract timing metrics for comparison
        timing_data: dict[str, dict[str, float]] = {}
        for config_name, result in self.configurations.items():
            for metric, value in result.timing_metrics.items():
                if metric not in timing_data:
                    timing_data[metric] = {}
                timing_data[metric][config_name] = value

        # Find best and worst for mean iteration time
        if "mean_iteration_time" in timing_data:
            times = timing_data["mean_iteration_time"]
            self.best_config = min(times, key=times.get)
            self.worst_config = max(times, key=times.get)

        # Calculate summary statistics
        self.metrics_summary = {
            "num_configurations": len(self.configurations),
            "timing_metrics": timing_data,
        }

    def get_performance_ratio(self, metric: str = "mean_iteration_time") -> dict[str, float]:
        """Get performance ratios relative to the best configuration."""
        if not self.configurations or self.best_config is None:
            return {}

        ratios: dict[str, float] = {}
        best_value = self.configurations[self.best_config].timing_metrics.get(metric, 0)

        if best_value <= 0:
            return {}

        for config_name, result in self.configurations.items():
            current_value = result.timing_metrics.get(metric, 0)
            if current_value > 0:
                ratios[config_name] = current_value / best_value

        return ratios

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "configurations": {k: v.to_dict() for k, v in self.configurations.items()},
            "best_config": self.best_config,
            "worst_config": self.worst_config,
            "metrics_summary": self.metrics_summary,
            "timestamp": self.timestamp,
        }

    def save(self, filepath: str | Path) -> None:
        """Save comparison to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


@dataclass
class LibraryComparison:
    """Comparison results between different libraries."""

    datarax_result: ProfileResult | None = None
    other_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    comparison_metrics: dict[str, float] = field(default_factory=dict)

    def add_datarax_result(self, result: ProfileResult) -> None:
        """Add Datarax benchmark result."""
        self.datarax_result = result
        self._update_comparison()

    def add_other_library_result(self, library_name: str, metrics: dict[str, Any]) -> None:
        """Add result from another library."""
        self.other_results[library_name] = metrics
        self._update_comparison()

    def _update_comparison(self) -> None:
        """Update comparison metrics."""
        if not self.datarax_result:
            return

        datarax_time = self.datarax_result.timing_metrics.get("mean_iteration_time", 0)

        for lib_name, metrics in self.other_results.items():
            other_time = metrics.get("mean_iteration_time", 0)

            if datarax_time > 0 and other_time > 0:
                speedup = other_time / datarax_time
                self.comparison_metrics[f"{lib_name}_speedup"] = speedup


class ComparativeBenchmark:
    """System for running comparative benchmarks.

    This class provides tools to compare different pipeline configurations,
    batch sizes, and device types to identify optimal settings.

    Examples:
        from datarax.benchmarking import ComparativeBenchmark

        def pipeline_v1():
            # ... pipeline construction ...
            return pipeline

        def pipeline_v2():
            # ... optimization ...
            return pipeline

        benchmark = ComparativeBenchmark()
        results = benchmark.compare_configurations(
            {"v1": pipeline_v1, "v2": pipeline_v2},
            sample_data=data
        )
        print(results.best_config)
    """

    def __init__(self, profiler: AdvancedProfiler | None = None):
        """Initialize comparative benchmark system.

        Args:
            profiler: Advanced profiler to use. If None, creates a new
                ``AdvancedProfiler`` instance with default settings.
        """
        self.profiler = profiler or AdvancedProfiler()

    def compare_configurations(
        self,
        pipeline_factories: dict[str, Callable],
        sample_data: Any,
        num_iterations: int = 10,
        warmup_iterations: int = 2,
    ) -> BenchmarkComparison:
        """Compare different pipeline configurations.

        Args:
            pipeline_factories: Dict mapping config names to pipeline factory functions
            sample_data: Sample data to use for benchmarking
            num_iterations: Number of iterations per configuration
            warmup_iterations: Number of warmup iterations

        Returns:
            BenchmarkComparison with results
        """
        comparison = BenchmarkComparison()

        for config_name, factory in pipeline_factories.items():
            print(f"Benchmarking configuration: {config_name}")

            # Create pipeline function
            pipeline_fn = factory()

            # Profile the configuration
            result = self.profiler.profile_pipeline(
                pipeline_fn,
                sample_data,
                num_iterations=num_iterations,
                warmup_iterations=warmup_iterations,
            )

            comparison.add_result(config_name, result)

        return comparison

    def compare_batch_sizes(
        self,
        pipeline_factory: Callable[[int], Callable],
        batch_sizes: list[int],
        sample_data: Any,
        num_iterations: int = 10,
    ) -> BenchmarkComparison:
        """Compare performance across different batch sizes.

        Args:
            pipeline_factory: Function that takes batch_size and returns pipeline function
            batch_sizes: List of batch sizes to test
            sample_data: Sample data to use
            num_iterations: Number of iterations per batch size

        Returns:
            BenchmarkComparison with results
        """
        comparison = BenchmarkComparison()

        for batch_size in batch_sizes:
            config_name = f"batch_size_{batch_size}"
            print(f"Benchmarking batch size: {batch_size}")

            pipeline_fn = pipeline_factory(batch_size)

            result = self.profiler.profile_pipeline(
                pipeline_fn,
                sample_data,
                num_iterations=num_iterations,
                warmup_iterations=2,
            )

            comparison.add_result(config_name, result)

        return comparison

    def find_optimal_batch_size(
        self,
        pipeline_factory: Callable[[int], Callable],
        sample_data: Any,
        min_batch_size: int = 8,
        max_batch_size: int = 256,
        num_iterations: int = 5,
    ) -> dict[str, Any]:
        """Find the optimal batch size for a pipeline.

        Args:
            pipeline_factory: Function that takes batch_size and returns pipeline function
            sample_data: Sample data to use
            min_batch_size: Minimum batch size to test
            max_batch_size: Maximum batch size to test
            num_iterations: Number of iterations per test

        Returns:
            Dictionary with optimization results
        """
        # Test powers of 2 between min and max
        batch_sizes = []
        size = min_batch_size
        while size <= max_batch_size:
            batch_sizes.append(size)
            size *= 2

        comparison = self.compare_batch_sizes(
            pipeline_factory, batch_sizes, sample_data, num_iterations
        )

        # Find optimal batch size (best throughput)
        best_throughput = 0.0
        optimal_batch_size = batch_sizes[0]

        for config_name, result in comparison.configurations.items():
            throughput = result.timing_metrics.get("iterations_per_second", 0)
            if throughput > best_throughput:
                best_throughput = throughput
                optimal_batch_size = int(config_name.split("_")[-1])

        return {
            "optimal_batch_size": optimal_batch_size,
            "best_throughput": best_throughput,
            "all_results": comparison,
            "throughput_by_batch_size": {
                int(config.split("_")[-1]): result.timing_metrics.get("iterations_per_second", 0)
                for config, result in comparison.configurations.items()
            },
        }

    def compare_device_types(
        self,
        pipeline_factory: Callable[[], Callable],
        sample_data: Any,
        num_iterations: int = 10,
    ) -> BenchmarkComparison:
        """Compare performance across different device types (CPU vs GPU).

        Args:
            pipeline_factory: Function that returns a pipeline function
            sample_data: Sample data to use
            num_iterations: Number of iterations per device

        Returns:
            BenchmarkComparison with results
        """
        comparison = BenchmarkComparison()

        # Get available devices
        cpu_devices = jax.devices("cpu")
        try:
            gpu_devices = jax.devices("gpu")
        except (RuntimeError, ValueError):
            gpu_devices = []

        # Test on CPU
        if cpu_devices:
            with jax.default_device(cpu_devices[0]):
                print("Benchmarking on CPU")
                pipeline_fn = pipeline_factory()
                result = self.profiler.profile_pipeline(
                    pipeline_fn, sample_data, num_iterations=num_iterations
                )
                comparison.add_result("CPU", result)

        # Test on GPU
        if gpu_devices:
            with jax.default_device(gpu_devices[0]):
                print("Benchmarking on GPU")
                pipeline_fn = pipeline_factory()
                result = self.profiler.profile_pipeline(
                    pipeline_fn, sample_data, num_iterations=num_iterations
                )
                comparison.add_result("GPU", result)

        return comparison

    def benchmark_scaling(
        self,
        pipeline_factory: Callable[[int], Callable],
        data_sizes: list[int],
        sample_data_factory: Callable[[int], Any],
        num_iterations: int = 5,
    ) -> BenchmarkComparison:
        """Benchmark how performance scales with data size.

        Args:
            pipeline_factory: Function that takes data_size and returns pipeline function
            data_sizes: List of data sizes to test
            sample_data_factory: Function that generates sample data for given size
            num_iterations: Number of iterations per size

        Returns:
            BenchmarkComparison with scaling results
        """
        comparison = BenchmarkComparison()

        for data_size in data_sizes:
            config_name = f"data_size_{data_size}"
            print(f"Benchmarking data size: {data_size}")

            # Generate sample data for this size
            sample_data = sample_data_factory(data_size)

            # Create pipeline for this size
            pipeline_fn = pipeline_factory(data_size)

            # Benchmark
            result = self.profiler.profile_pipeline(
                pipeline_fn,
                sample_data,
                num_iterations=num_iterations,
                warmup_iterations=1,
            )

            comparison.add_result(config_name, result)

        return comparison
