"""Tests for Datarax benchmarking utilities.

Following TDD principles - tests define expected behavior.
Target: 80%+ coverage of utils/benchmark.py.
"""

import time
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest

from datarax.benchmarking.pipeline_throughput import (
    BatchSizeBenchmark,
    PipelineBenchmark,
    ProfileReport,
    Timer,
    benchmark_comparison,
)


class TestTimer:
    """Test the Timer context manager utility.

    Timer is a simple utility for measuring elapsed time.
    """

    def test_timer_measures_elapsed_time(self) -> None:
        """Test that Timer correctly measures elapsed time."""
        with Timer() as timer:
            time.sleep(0.1)  # Sleep for 100ms

        assert timer.elapsed >= 0.1
        assert timer.elapsed < 0.2  # Should complete within 200ms

    def test_timer_with_name(self) -> None:
        """Test Timer with a custom name."""
        timer = Timer(name="test_operation")
        assert timer.name == "test_operation"

    def test_timer_elapsed_before_start_raises_error(self) -> None:
        """Test that accessing elapsed before starting raises ValueError."""
        timer = Timer()

        with pytest.raises(ValueError, match="Timer has not been started"):
            _ = timer.elapsed

    def test_timer_elapsed_during_execution(self) -> None:
        """Test that elapsed can be checked during execution."""
        timer = Timer()
        timer.__enter__()
        time.sleep(0.05)

        # Should return current elapsed time
        elapsed = timer.elapsed
        assert elapsed >= 0.05

        timer.__exit__(None, None, None)

    def test_timer_stores_start_and_end_times(self) -> None:
        """Test that Timer stores start and end times."""
        timer = Timer()

        assert timer.start_time is None
        assert timer.end_time is None

        with timer:
            assert timer.start_time is not None
            assert timer.end_time is None  # Not finished yet

        assert timer.end_time is not None


class TestPipelineBenchmark:
    """Test the PipelineBenchmark class.

    PipelineBenchmark measures pipeline throughput and performance.
    """

    def _create_mock_pipeline(self, num_batches: int = 10, batch_size: int = 32) -> MagicMock:
        """Create a mock pipeline that yields batches."""
        mock_pipeline = MagicMock()

        # Create mock Batch objects
        batch_objects = []
        for _ in range(num_batches):
            mock_batch = MagicMock()
            mock_batch.get_data.return_value = {"image": jnp.ones((batch_size, 28, 28, 3))}
            batch_objects.append(mock_batch)

        mock_pipeline.__iter__.side_effect = lambda: iter(batch_objects)
        return mock_pipeline

    def test_pipeline_benchmark_initialization(self) -> None:
        """Test PipelineBenchmark initialization."""
        mock_pipeline = self._create_mock_pipeline()
        benchmark = PipelineBenchmark(mock_pipeline, num_batches=50, warmup_batches=5)

        assert benchmark.data_stream == mock_pipeline
        assert benchmark.num_batches == 50
        assert benchmark.warmup_batches == 5
        assert benchmark.results == {}

    def test_pipeline_benchmark_run_collects_metrics(self) -> None:
        """Test that run() collects performance metrics."""
        mock_pipeline = self._create_mock_pipeline(num_batches=20, batch_size=16)
        benchmark = PipelineBenchmark(mock_pipeline, num_batches=10, warmup_batches=2)

        results = benchmark.run(pipeline_seed=42)

        # Verify all expected metrics are present
        assert "duration_seconds" in results
        assert "batches_processed" in results
        assert "examples_processed" in results
        assert "examples_per_second" in results
        assert "batches_per_second" in results

        # Verify reasonable values
        assert results["batches_processed"] == 10
        assert results["examples_processed"] == 160  # 10 batches * 16 examples
        assert results["duration_seconds"] > 0
        assert results["examples_per_second"] > 0

    def test_pipeline_benchmark_count_examples_dict_with_image(self) -> None:
        """Test _count_examples with dict containing 'image' key."""
        mock_pipeline = self._create_mock_pipeline()
        benchmark = PipelineBenchmark(mock_pipeline)

        data = {"image": jnp.ones((32, 28, 28, 3)), "label": jnp.ones(32)}
        mock_batch = MagicMock()
        mock_batch.get_data.return_value = data

        count = benchmark._count_examples(mock_batch)

        assert count == 32

    def test_pipeline_benchmark_count_examples_dict_with_text(self) -> None:
        """Test _count_examples with dict containing 'text' key."""
        mock_pipeline = self._create_mock_pipeline()
        benchmark = PipelineBenchmark(mock_pipeline)

        data = {"text": jnp.ones((16, 512)), "mask": jnp.ones((16, 512))}
        mock_batch = MagicMock()
        mock_batch.get_data.return_value = data

        count = benchmark._count_examples(mock_batch)

        assert count == 16

    def test_pipeline_benchmark_count_examples_dict_with_data(self) -> None:
        """Test _count_examples with dict containing 'data' key."""
        mock_pipeline = self._create_mock_pipeline()
        benchmark = PipelineBenchmark(mock_pipeline)

        data = {"data": jnp.ones((24, 100)), "info": "test"}
        mock_batch = MagicMock()
        mock_batch.get_data.return_value = data

        count = benchmark._count_examples(mock_batch)

        assert count == 24

    def test_pipeline_benchmark_count_examples_generic_tree(self) -> None:
        """Test _count_examples with generic pytree structure."""
        mock_pipeline = self._create_mock_pipeline()
        benchmark = PipelineBenchmark(mock_pipeline)

        # Nested structure without standard keys
        data = {"custom_key": jnp.ones((8, 64))}
        mock_batch = MagicMock()
        mock_batch.get_data.return_value = data

        count = benchmark._count_examples(mock_batch)

        assert count == 8

    def test_pipeline_benchmark_print_results_with_data(self, capsys) -> None:
        """Test print_results displays formatted output."""
        mock_pipeline = self._create_mock_pipeline(num_batches=5, batch_size=10)
        benchmark = PipelineBenchmark(mock_pipeline, num_batches=5, warmup_batches=0)

        benchmark.run()
        benchmark.print_results()

        captured = capsys.readouterr()
        assert "Pipeline Benchmark Results:" in captured.out
        assert "Duration:" in captured.out
        assert "Batches Processed: 5" in captured.out
        assert "Examples Processed: 50" in captured.out
        assert "Throughput:" in captured.out

    def test_pipeline_benchmark_print_results_without_run(self, capsys) -> None:
        """Test print_results without running benchmark shows warning."""
        mock_pipeline = self._create_mock_pipeline()
        benchmark = PipelineBenchmark(mock_pipeline)

        benchmark.print_results()

        captured = capsys.readouterr()
        assert "No benchmark results available" in captured.out


class TestBatchSizeBenchmark:
    """Test the BatchSizeBenchmark class.

    BatchSizeBenchmark compares performance across different batch sizes.
    """

    def _create_mock_factory(self, num_batches: int = 10) -> MagicMock:
        """Create a mock factory function that returns pipelines."""

        def factory(batch_size):
            mock_pipeline = MagicMock()
            mock_pipeline.batch.return_value = mock_pipeline

            # Create batches - batch size will be set dynamically
            # Use mock objects for batches so _count_examples works
            batches = []
            for _ in range(num_batches):
                mock_batch = MagicMock()
                mock_batch.get_data.return_value = {"image": jnp.ones((batch_size, 28, 28, 3))}
                batches.append(mock_batch)

            mock_pipeline.__iter__.side_effect = lambda: iter(batches)

            return mock_pipeline

        return factory

    def test_batch_size_benchmark_initialization(self) -> None:
        """Test BatchSizeBenchmark initialization."""
        factory = self._create_mock_factory()
        benchmark = BatchSizeBenchmark(
            factory, batch_sizes=[8, 16, 32], num_batches=30, warmup_batches=5
        )

        assert benchmark.data_stream_factory == factory
        assert benchmark.batch_sizes == [8, 16, 32]
        assert benchmark.num_batches == 30
        assert benchmark.warmup_batches == 5
        assert benchmark.results == {}

    def test_batch_size_benchmark_run_all_sizes(self, capsys) -> None:
        """Test that run() benchmarks all batch sizes."""
        factory = self._create_mock_factory(num_batches=20)
        benchmark = BatchSizeBenchmark(factory, batch_sizes=[4, 8], num_batches=5, warmup_batches=1)

        results = benchmark.run(pipeline_seed=42)

        # Verify results for all batch sizes
        assert 4 in results
        assert 8 in results

        # Verify each has complete metrics
        for batch_size in [4, 8]:
            assert "examples_per_second" in results[batch_size]
            assert "batches_per_second" in results[batch_size]
            assert "duration_seconds" in results[batch_size]

        # Check progress output
        captured = capsys.readouterr()
        assert "Completed benchmark for batch_size=4" in captured.out
        assert "Completed benchmark for batch_size=8" in captured.out

    def test_batch_size_benchmark_print_results_with_data(self, capsys) -> None:
        """Test print_results displays formatted comparison table."""
        factory = self._create_mock_factory(num_batches=10)
        benchmark = BatchSizeBenchmark(
            factory, batch_sizes=[8, 16], num_batches=5, warmup_batches=0
        )

        benchmark.run()
        benchmark.print_results()

        captured = capsys.readouterr()
        assert "Batch Size Benchmark Results:" in captured.out
        assert "Batch Size" in captured.out
        assert "Examples/s" in captured.out
        assert "8" in captured.out
        assert "16" in captured.out

    def test_batch_size_benchmark_print_results_without_run(self, capsys) -> None:
        """Test print_results without running shows warning."""
        factory = self._create_mock_factory()
        benchmark = BatchSizeBenchmark(factory, batch_sizes=[8])

        benchmark.print_results()

        captured = capsys.readouterr()
        assert "No benchmark results available" in captured.out


class TestProfileReport:
    """Test the ProfileReport class.

    ProfileReport generates detailed performance profiles.
    """

    def _create_mock_pipeline(self, num_batches: int = 10, batch_size: int = 32) -> MagicMock:
        """Create a mock pipeline for profiling."""
        mock_pipeline = MagicMock()

        batch_objects = []
        for _ in range(num_batches):
            mock_batch = MagicMock()
            mock_batch.get_data.return_value = {"image": jnp.ones((batch_size, 28, 28, 3))}
            batch_objects.append(mock_batch)

        mock_pipeline.iterator.return_value = iter(batch_objects)

        return mock_pipeline

    def test_profile_report_initialization(self) -> None:
        """Test ProfileReport initialization."""
        mock_pipeline = self._create_mock_pipeline()
        profile = ProfileReport(mock_pipeline)

        assert profile.data_stream == mock_pipeline
        assert profile.metrics == {}

    def test_profile_report_run_collects_metrics(self) -> None:
        """Test that run() collects profiling metrics."""
        mock_pipeline = self._create_mock_pipeline(num_batches=15, batch_size=16)
        profile = ProfileReport(mock_pipeline)

        profile.run(num_batches=10, pipeline_seed=42)

        # Should have pipeline metrics
        assert "pipeline" in profile.metrics
        assert "examples_per_second" in profile.metrics["pipeline"]

    def test_profile_report_print_with_data(self, capsys) -> None:
        """Test print_report displays formatted profile."""
        mock_pipeline = self._create_mock_pipeline(num_batches=10, batch_size=8)
        profile = ProfileReport(mock_pipeline)

        profile.run(num_batches=5)
        profile.print_report()

        captured = capsys.readouterr()
        assert "Datarax Pipeline Performance Profile" in captured.out
        assert "Overall Pipeline Performance:" in captured.out
        assert "Throughput:" in captured.out
        assert "examples/second" in captured.out

    def test_profile_report_print_without_run(self, capsys) -> None:
        """Test print_report without running shows warning."""
        mock_pipeline = self._create_mock_pipeline()
        profile = ProfileReport(mock_pipeline)

        profile.print_report()

        captured = capsys.readouterr()
        assert "No profiling results available" in captured.out


class TestBenchmarkComparison:
    """Test the benchmark_comparison utility function.

    This function compares multiple pipeline configurations.
    """

    def _create_mock_pipeline(self, name: str, batch_size: int = 16) -> MagicMock:
        """Create a mock pipeline with specific characteristics."""
        mock_pipeline = MagicMock()
        batch_objects = []
        for _ in range(15):
            mock_batch = MagicMock()
            mock_batch.get_data.return_value = {"data": jnp.ones((batch_size, 100))}
            batch_objects.append(mock_batch)

        mock_pipeline.__iter__.side_effect = lambda: iter(batch_objects)
        return mock_pipeline

    def test_benchmark_comparison_multiple_configs(self, capsys) -> None:
        """Test comparing multiple pipeline configurations."""
        configurations = {
            "baseline": self._create_mock_pipeline("baseline", batch_size=16),
            "optimized": self._create_mock_pipeline("optimized", batch_size=32),
        }

        results = benchmark_comparison(configurations, num_batches=5, warmup_batches=1)

        # Verify results for all configurations
        assert "baseline" in results
        assert "optimized" in results

        # Verify metrics structure
        for config_name in ["baseline", "optimized"]:
            assert "examples_per_second" in results[config_name]
            assert "duration_seconds" in results[config_name]

        # Verify output formatting
        captured = capsys.readouterr()
        assert "Configuration Comparison:" in captured.out
        assert "baseline" in captured.out
        assert "optimized" in captured.out

    def test_benchmark_comparison_single_config(self) -> None:
        """Test comparison with a single configuration."""
        configurations = {"single": self._create_mock_pipeline("single")}

        results = benchmark_comparison(configurations, num_batches=3, warmup_batches=0)

        assert len(results) == 1
        assert "single" in results
        assert "examples_per_second" in results["single"]

    def test_benchmark_comparison_uses_pipeline_seed(self) -> None:
        """Test that benchmark_comparison passes pipeline_seed correctly."""
        mock_pipeline = self._create_mock_pipeline("test")
        configurations = {"test": mock_pipeline}

        # Run with specific seed
        with patch("datarax.benchmarking.pipeline_throughput.PipelineBenchmark") as MockBenchmark:
            mock_instance = MagicMock()
            mock_instance.run.return_value = {
                "examples_per_second": 100.0,
                "batches_per_second": 10.0,
                "duration_seconds": 1.0,
            }
            MockBenchmark.return_value = mock_instance

            benchmark_comparison(configurations, pipeline_seed=99)

            # Verify run was called with correct seed
            mock_instance.run.assert_called_once_with(pipeline_seed=99)
