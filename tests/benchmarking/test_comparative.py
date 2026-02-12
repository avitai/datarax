"""Tests for comparative benchmarking tools.

Testing Strategy:
- BenchmarkComparison: Result aggregation, summary stats, performance ratios

Uses BenchmarkResult (replaces ProfileResult). LibraryComparison and
ComparativeBenchmark have been removed (replaced by adapter pattern, Section 7).
"""

import json
import tempfile
from pathlib import Path

from datarax.benchmarking.comparative import BenchmarkComparison
from tests.benchmarking.conftest import make_result_for_throughput as _make_result


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

        result = _make_result(wall_clock_sec=1.0, num_elements=1000)
        comparison.add_result("config_a", result)

        assert "config_a" in comparison.configurations
        assert comparison.configurations["config_a"] == result
        assert comparison.best_config == "config_a"
        assert comparison.worst_config == "config_a"

    def test_add_multiple_results_identifies_best_worst(self):
        """Test adding multiple results correctly identifies best/worst by throughput."""
        comparison = BenchmarkComparison()

        # Higher throughput = better: fast has 10000 el/s, slow has 1000 el/s
        fast_result = _make_result(wall_clock_sec=0.1, num_elements=1000)
        medium_result = _make_result(wall_clock_sec=0.5, num_elements=1000)
        slow_result = _make_result(wall_clock_sec=1.0, num_elements=1000)

        comparison.add_result("fast", fast_result)
        comparison.add_result("medium", medium_result)
        comparison.add_result("slow", slow_result)

        assert comparison.best_config == "fast"
        assert comparison.worst_config == "slow"
        assert len(comparison.configurations) == 3

    def test_update_summary_metrics(self):
        """Test summary metrics are correctly calculated."""
        comparison = BenchmarkComparison()

        result1 = _make_result(wall_clock_sec=1.0, num_elements=1000)
        result2 = _make_result(wall_clock_sec=2.0, num_elements=1000)

        comparison.add_result("config_1", result1)
        comparison.add_result("config_2", result2)

        # Check summary structure
        assert comparison.metrics_summary["num_configurations"] == 2
        assert "throughputs" in comparison.metrics_summary

        # Check throughput data
        throughputs = comparison.metrics_summary["throughputs"]
        assert throughputs["config_1"] == 1000.0  # 1000 elements / 1.0s
        assert throughputs["config_2"] == 500.0  # 1000 elements / 2.0s

    def test_get_performance_ratio_basic(self):
        """Test throughput-based performance ratio calculation."""
        comparison = BenchmarkComparison()

        # Fast: 1000 el/s (1000 elements in 1.0s)
        fast_result = _make_result(wall_clock_sec=1.0, num_elements=1000)
        # Slow: 500 el/s (1000 elements in 2.0s)
        slow_result = _make_result(wall_clock_sec=2.0, num_elements=1000)

        comparison.add_result("fast", fast_result)
        comparison.add_result("slow", slow_result)

        ratios = comparison.get_performance_ratio()

        assert ratios["fast"] == 1.0  # Best = baseline
        assert ratios["slow"] == 0.5  # Half the throughput

    def test_get_performance_ratio_empty_comparison(self):
        """Test performance ratio returns empty for empty comparison."""
        comparison = BenchmarkComparison()
        ratios = comparison.get_performance_ratio()

        assert ratios == {}

    def test_get_performance_ratio_zero_throughput(self):
        """Test performance ratio handles zero throughput gracefully."""
        comparison = BenchmarkComparison()

        # Zero elements → zero throughput
        result = _make_result(wall_clock_sec=1.0, num_elements=0)
        comparison.add_result("config", result)

        ratios = comparison.get_performance_ratio()
        assert ratios == {}

    def test_to_dict_conversion(self):
        """Test conversion to dictionary for serialization."""
        comparison = BenchmarkComparison()

        result = _make_result(wall_clock_sec=1.0, num_elements=1000)
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

        result = _make_result(wall_clock_sec=1.0, num_elements=1000)
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
