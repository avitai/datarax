"""Tests for comparative benchmarking tools.

Testing Strategy:
- BenchmarkComparison: Result aggregation, summary stats, performance ratios
- LibraryComparison: Cross-library comparisons and speedup calculations
- ComparativeBenchmark: Core comparison methods and optimization

Coverage Target: 50%+ (180+ lines out of 357 total)
Focus: Core functionality first (dataclass methods, comparison logic)
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock


from datarax.benchmarking.comparative import (
    BenchmarkComparison,
    ComparativeBenchmark,
    LibraryComparison,
)
from datarax.benchmarking.profiler import ProfileResult


class TestBenchmarkComparison:
    """Test BenchmarkComparison dataclass and methods."""

    def test_initialization_defaults(self):
        """Test default initialization creates empty comparison."""
        comparison = BenchmarkComparison()

        assert comparison.configurations == {}
        assert comparison.best_config is None
        assert comparison.worst_config is None
        assert comparison.metrics_summary == {}
        assert isinstance(comparison.timestamp, float)
        assert comparison.timestamp > 0

    def test_add_single_result(self):
        """Test adding a single benchmark result."""
        comparison = BenchmarkComparison()

        # Create mock ProfileResult
        result = ProfileResult(
            timing_metrics={"mean_iteration_time": 0.5, "total_time": 5.0},
            memory_metrics={},
            gpu_metrics={},
        )

        comparison.add_result("config_a", result)

        assert "config_a" in comparison.configurations
        assert comparison.configurations["config_a"] == result
        assert comparison.best_config == "config_a"
        assert comparison.worst_config == "config_a"

    def test_add_multiple_results_identifies_best_worst(self):
        """Test adding multiple results correctly identifies best/worst."""
        comparison = BenchmarkComparison()

        # Fast config
        fast_result = ProfileResult(
            timing_metrics={"mean_iteration_time": 0.1},
            memory_metrics={},
            gpu_metrics={},
        )

        # Medium config
        medium_result = ProfileResult(
            timing_metrics={"mean_iteration_time": 0.5},
            memory_metrics={},
            gpu_metrics={},
        )

        # Slow config
        slow_result = ProfileResult(
            timing_metrics={"mean_iteration_time": 1.0},
            memory_metrics={},
            gpu_metrics={},
        )

        comparison.add_result("fast", fast_result)
        comparison.add_result("medium", medium_result)
        comparison.add_result("slow", slow_result)

        assert comparison.best_config == "fast"
        assert comparison.worst_config == "slow"
        assert len(comparison.configurations) == 3

    def test_update_summary_metrics(self):
        """Test summary metrics are correctly calculated."""
        comparison = BenchmarkComparison()

        result1 = ProfileResult(
            timing_metrics={"mean_iteration_time": 0.5, "total_time": 5.0},
            memory_metrics={},
            gpu_metrics={},
        )

        result2 = ProfileResult(
            timing_metrics={"mean_iteration_time": 1.0, "total_time": 10.0},
            memory_metrics={},
            gpu_metrics={},
        )

        comparison.add_result("config_1", result1)
        comparison.add_result("config_2", result2)

        # Check summary structure
        assert comparison.metrics_summary["num_configurations"] == 2
        assert "timing_metrics" in comparison.metrics_summary

        # Check timing data structure
        timing_data = comparison.metrics_summary["timing_metrics"]
        assert "mean_iteration_time" in timing_data
        assert timing_data["mean_iteration_time"]["config_1"] == 0.5
        assert timing_data["mean_iteration_time"]["config_2"] == 1.0

    def test_get_performance_ratio_basic(self):
        """Test performance ratio calculation for simple case."""
        comparison = BenchmarkComparison()

        # Baseline (fastest)
        fast_result = ProfileResult(
            timing_metrics={"mean_iteration_time": 1.0},
            memory_metrics={},
            gpu_metrics={},
        )

        # 2x slower
        slow_result = ProfileResult(
            timing_metrics={"mean_iteration_time": 2.0},
            memory_metrics={},
            gpu_metrics={},
        )

        comparison.add_result("fast", fast_result)
        comparison.add_result("slow", slow_result)

        ratios = comparison.get_performance_ratio()

        assert ratios["fast"] == 1.0  # Baseline
        assert ratios["slow"] == 2.0  # 2x slower

    def test_get_performance_ratio_empty_comparison(self):
        """Test performance ratio returns empty for empty comparison."""
        comparison = BenchmarkComparison()
        ratios = comparison.get_performance_ratio()

        assert ratios == {}

    def test_get_performance_ratio_zero_best_value(self):
        """Test performance ratio handles zero best value gracefully."""
        comparison = BenchmarkComparison()

        result = ProfileResult(
            timing_metrics={"mean_iteration_time": 0.0},
            memory_metrics={},
            gpu_metrics={},
        )

        comparison.add_result("config", result)
        ratios = comparison.get_performance_ratio()

        assert ratios == {}

    def test_get_performance_ratio_custom_metric(self):
        """Test performance ratio with custom metric."""
        comparison = BenchmarkComparison()

        result1 = ProfileResult(
            timing_metrics={"custom_metric": 10.0},
            memory_metrics={},
            gpu_metrics={},
        )

        result2 = ProfileResult(
            timing_metrics={"custom_metric": 20.0},
            memory_metrics={},
            gpu_metrics={},
        )

        comparison.add_result("config_1", result1)
        comparison.add_result("config_2", result2)

        # Note: best_config determined by mean_iteration_time (not present)
        # So this tests edge case handling
        ratios = comparison.get_performance_ratio(metric="custom_metric")

        # Should handle missing mean_iteration_time gracefully
        assert isinstance(ratios, dict)

    def test_to_dict_conversion(self):
        """Test conversion to dictionary for serialization."""
        comparison = BenchmarkComparison()

        result = ProfileResult(
            timing_metrics={"mean_iteration_time": 0.5},
            memory_metrics={"peak_memory": 100.0},
            gpu_metrics={},
        )

        comparison.add_result("test_config", result)

        result_dict = comparison.to_dict()

        assert "configurations" in result_dict
        assert "best_config" in result_dict
        assert "worst_config" in result_dict
        assert "metrics_summary" in result_dict
        assert "timestamp" in result_dict

        # Check nested structure
        assert "test_config" in result_dict["configurations"]
        assert result_dict["best_config"] == "test_config"

    def test_save_to_file(self):
        """Test saving comparison to JSON file."""
        comparison = BenchmarkComparison()

        result = ProfileResult(
            timing_metrics={"mean_iteration_time": 0.5},
            memory_metrics={},
            gpu_metrics={},
        )

        comparison.add_result("config", result)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "subdir" / "comparison.json"
            comparison.save(filepath)

            # Verify file exists
            assert filepath.exists()

            # Verify content
            with open(filepath) as f:
                data = json.load(f)

            assert "configurations" in data
            assert "config" in data["configurations"]


class TestLibraryComparison:
    """Test LibraryComparison for cross-library benchmarks."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        comparison = LibraryComparison()

        assert comparison.datarax_result is None
        assert comparison.other_results == {}
        assert comparison.comparison_metrics == {}

    def test_add_datarax_result(self):
        """Test adding Datarax benchmark result."""
        comparison = LibraryComparison()

        result = ProfileResult(
            timing_metrics={"mean_iteration_time": 0.5},
            memory_metrics={},
            gpu_metrics={},
        )

        comparison.add_datarax_result(result)

        assert comparison.datarax_result == result

    def test_add_other_library_result(self):
        """Test adding result from another library."""
        comparison = LibraryComparison()

        metrics = {"mean_iteration_time": 1.0, "peak_memory": 200.0}

        comparison.add_other_library_result("tensorflow", metrics)

        assert "tensorflow" in comparison.other_results
        assert comparison.other_results["tensorflow"] == metrics

    def test_speedup_calculation(self):
        """Test speedup calculation between Datarax and other libraries."""
        comparison = LibraryComparison()

        # Datarax is 2x faster
        datarax_result = ProfileResult(
            timing_metrics={"mean_iteration_time": 0.5},
            memory_metrics={},
            gpu_metrics={},
        )

        comparison.add_datarax_result(datarax_result)
        comparison.add_other_library_result("pytorch", {"mean_iteration_time": 1.0})

        # Speedup should be 2.0 (pytorch_time / datarax_time)
        assert "pytorch_speedup" in comparison.comparison_metrics
        assert comparison.comparison_metrics["pytorch_speedup"] == 2.0

    def test_speedup_multiple_libraries(self):
        """Test speedup calculation with multiple libraries."""
        comparison = LibraryComparison()

        datarax_result = ProfileResult(
            timing_metrics={"mean_iteration_time": 1.0},
            memory_metrics={},
            gpu_metrics={},
        )

        comparison.add_datarax_result(datarax_result)
        comparison.add_other_library_result("lib_a", {"mean_iteration_time": 2.0})
        comparison.add_other_library_result("lib_b", {"mean_iteration_time": 3.0})

        assert comparison.comparison_metrics["lib_a_speedup"] == 2.0
        assert comparison.comparison_metrics["lib_b_speedup"] == 3.0

    def test_no_speedup_without_datarax_result(self):
        """Test no speedup calculated without Datarax result."""
        comparison = LibraryComparison()

        comparison.add_other_library_result("pytorch", {"mean_iteration_time": 1.0})

        # Should not crash, metrics should be empty
        assert comparison.comparison_metrics == {}

    def test_speedup_handles_zero_times(self):
        """Test speedup calculation handles zero times gracefully."""
        comparison = LibraryComparison()

        datarax_result = ProfileResult(
            timing_metrics={"mean_iteration_time": 0.0},
            memory_metrics={},
            gpu_metrics={},
        )

        comparison.add_datarax_result(datarax_result)
        comparison.add_other_library_result("pytorch", {"mean_iteration_time": 1.0})

        # Should not calculate speedup with zero Datarax time
        assert "pytorch_speedup" not in comparison.comparison_metrics


class TestComparativeBenchmark:
    """Test ComparativeBenchmark system."""

    def test_initialization_default_profiler(self):
        """Test initialization with default profiler."""
        benchmark = ComparativeBenchmark()

        assert benchmark.profiler is not None

    def test_initialization_custom_profiler(self):
        """Test initialization with custom profiler."""
        mock_profiler = Mock()
        benchmark = ComparativeBenchmark(profiler=mock_profiler)

        assert benchmark.profiler == mock_profiler

    def test_compare_configurations_single_config(self):
        """Test comparing single configuration."""
        mock_profiler = Mock()

        # Mock profile_pipeline to return a ProfileResult
        mock_result = ProfileResult(
            timing_metrics={"mean_iteration_time": 0.5},
            memory_metrics={},
            gpu_metrics={},
        )
        mock_profiler.profile_pipeline.return_value = mock_result

        benchmark = ComparativeBenchmark(profiler=mock_profiler)

        # Create pipeline factory
        def pipeline_factory():
            return lambda x: x

        pipeline_factories = {"config_a": pipeline_factory}
        sample_data = [1, 2, 3]

        comparison = benchmark.compare_configurations(pipeline_factories, sample_data)

        # Verify profiler was called
        assert mock_profiler.profile_pipeline.called
        assert "config_a" in comparison.configurations

    def test_compare_configurations_multiple_configs(self):
        """Test comparing multiple configurations."""
        mock_profiler = Mock()

        # Mock different results for different configs
        result1 = ProfileResult(
            timing_metrics={"mean_iteration_time": 0.5},
            memory_metrics={},
            gpu_metrics={},
        )
        result2 = ProfileResult(
            timing_metrics={"mean_iteration_time": 1.0},
            memory_metrics={},
            gpu_metrics={},
        )

        mock_profiler.profile_pipeline.side_effect = [result1, result2]

        benchmark = ComparativeBenchmark(profiler=mock_profiler)

        pipeline_factories = {
            "fast": lambda: lambda x: x,
            "slow": lambda: lambda x: x,
        }

        comparison = benchmark.compare_configurations(pipeline_factories, [1, 2, 3])

        assert len(comparison.configurations) == 2
        assert "fast" in comparison.configurations
        assert "slow" in comparison.configurations
        assert comparison.best_config == "fast"
        assert comparison.worst_config == "slow"

    def test_compare_batch_sizes_basic(self):
        """Test comparing different batch sizes."""
        mock_profiler = Mock()

        # Mock results for different batch sizes
        result1 = ProfileResult(
            timing_metrics={"mean_iteration_time": 0.5},
            memory_metrics={},
            gpu_metrics={},
        )
        result2 = ProfileResult(
            timing_metrics={"mean_iteration_time": 0.3},
            memory_metrics={},
            gpu_metrics={},
        )

        mock_profiler.profile_pipeline.side_effect = [result1, result2]

        benchmark = ComparativeBenchmark(profiler=mock_profiler)

        def pipeline_factory(batch_size):
            return lambda x: x

        batch_sizes = [32, 64]
        comparison = benchmark.compare_batch_sizes(pipeline_factory, batch_sizes, [1, 2, 3])

        assert len(comparison.configurations) == 2
        assert "batch_size_32" in comparison.configurations
        assert "batch_size_64" in comparison.configurations

    def test_find_optimal_batch_size_basic(self):
        """Test finding optimal batch size."""
        mock_profiler = Mock()

        # Mock results: batch_size 16 is fastest (highest iterations_per_second)
        result1 = ProfileResult(
            timing_metrics={
                "mean_iteration_time": 0.5,
                "iterations_per_second": 100.0,
            },
            memory_metrics={},
            gpu_metrics={},
        )
        result2 = ProfileResult(
            timing_metrics={
                "mean_iteration_time": 0.3,
                "iterations_per_second": 200.0,  # Best throughput
            },
            memory_metrics={},
            gpu_metrics={},
        )
        result3 = ProfileResult(
            timing_metrics={
                "mean_iteration_time": 0.4,
                "iterations_per_second": 150.0,
            },
            memory_metrics={},
            gpu_metrics={},
        )

        mock_profiler.profile_pipeline.side_effect = [result1, result2, result3]

        benchmark = ComparativeBenchmark(profiler=mock_profiler)

        def pipeline_factory(batch_size):
            return lambda x: x

        result = benchmark.find_optimal_batch_size(
            pipeline_factory,
            [1, 2, 3],
            min_batch_size=8,
            max_batch_size=32,
        )

        assert "optimal_batch_size" in result
        assert "best_throughput" in result
        assert "all_results" in result
        assert "throughput_by_batch_size" in result

        # Should select batch_size_16 (middle one with best throughput)
        assert result["best_throughput"] == 200.0
        assert result["optimal_batch_size"] == 16

    def test_find_optimal_batch_size_powers_of_two(self):
        """Test that batch size search uses powers of 2."""
        mock_profiler = Mock()
        mock_profiler.profile_pipeline.return_value = ProfileResult(
            timing_metrics={
                "mean_iteration_time": 0.5,
                "iterations_per_second": 100.0,
            },
            memory_metrics={},
            gpu_metrics={},
        )

        benchmark = ComparativeBenchmark(profiler=mock_profiler)

        def pipeline_factory(batch_size):
            return lambda x: x

        result = benchmark.find_optimal_batch_size(
            pipeline_factory,
            [1, 2, 3],
            min_batch_size=8,
            max_batch_size=64,
            num_iterations=1,
        )

        # Should test: 8, 16, 32, 64
        assert mock_profiler.profile_pipeline.call_count == 4
        assert 8 in result["throughput_by_batch_size"]
        assert 16 in result["throughput_by_batch_size"]
        assert 32 in result["throughput_by_batch_size"]
        assert 64 in result["throughput_by_batch_size"]
