"""Tests for BenchmarkResult.

TDD tests written first per Section 6.2.3 of the benchmark report.
Verifies: creation, to_dict() serialization, save()/load() round-trip,
throughput_elements_sec(), latency_percentiles(), and edge cases.
"""

import json

import numpy as np
import pytest

from datarax.benchmarking.results import BenchmarkResult
from datarax.benchmarking.resource_monitor import ResourceSummary
from datarax.benchmarking.timing import TimingSample
from tests.benchmarking.conftest import make_result, make_timing


class TestBenchmarkResultCreation:
    """Tests for BenchmarkResult dataclass creation."""

    def test_creation_with_all_fields(self):
        result = make_result()
        assert result.framework == "Datarax"
        assert result.scenario_id == "CV-1"
        assert result.variant == "small"
        assert isinstance(result.timing, TimingSample)
        assert isinstance(result.resources, ResourceSummary)
        assert result.environment["platform"] == "linux"
        assert result.config["batch_size"] == 32
        assert result.timestamp == 1700000000.0
        assert result.extra_metrics == {}

    def test_creation_with_extra_metrics(self):
        result = make_result(extra_metrics={"cache_hit_rate": 0.95})
        assert result.extra_metrics["cache_hit_rate"] == 0.95

    def test_creation_without_resources(self):
        result = make_result(resources=None)
        assert result.resources is None

    def test_timestamp_default_factory(self):
        """Timestamp defaults to current time when not provided."""
        result = BenchmarkResult(
            framework="Grain",
            scenario_id="NLP-1",
            variant="medium",
            timing=make_timing(),
            resources=None,
            environment={},
            config={},
        )
        assert result.timestamp > 0


class TestBenchmarkResultToDict:
    """Tests for to_dict() serialization."""

    def test_to_dict_returns_dict(self):
        result = make_result()
        d = result.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_contains_all_fields(self):
        result = make_result()
        d = result.to_dict()
        assert d["framework"] == "Datarax"
        assert d["scenario_id"] == "CV-1"
        assert d["variant"] == "small"
        assert d["timestamp"] == 1700000000.0
        assert d["extra_metrics"] == {}

    def test_to_dict_nested_timing(self):
        result = make_result()
        d = result.to_dict()
        assert d["timing"]["wall_clock_sec"] == 2.0
        assert d["timing"]["num_batches"] == 5

    def test_to_dict_nested_resources(self):
        result = make_result()
        d = result.to_dict()
        assert d["resources"]["peak_rss_mb"] == 512.0

    def test_to_dict_resources_none(self):
        result = make_result(resources=None)
        d = result.to_dict()
        assert d["resources"] is None

    def test_to_dict_json_serializable(self):
        """to_dict() output should be JSON-serializable."""
        result = make_result()
        d = result.to_dict()
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

    def test_to_dict_roundtrip_through_json(self):
        """Serialize to JSON and back, verify equality."""
        result = make_result()
        json_str = json.dumps(result.to_dict())
        d = json.loads(json_str)
        assert d["framework"] == result.framework
        assert d["timing"]["wall_clock_sec"] == result.timing.wall_clock_sec
        assert d["resources"]["peak_rss_mb"] == result.resources.peak_rss_mb


class TestBenchmarkResultSaveLoad:
    """Tests for save()/load() round-trip."""

    def test_save_creates_file(self, tmp_path):
        result = make_result()
        filepath = tmp_path / "result.json"
        result.save(filepath)
        assert filepath.exists()

    def test_save_creates_parent_dirs(self, tmp_path):
        result = make_result()
        filepath = tmp_path / "nested" / "dir" / "result.json"
        result.save(filepath)
        assert filepath.exists()

    def test_save_writes_valid_json(self, tmp_path):
        result = make_result()
        filepath = tmp_path / "result.json"
        result.save(filepath)
        data = json.loads(filepath.read_text())
        assert data["framework"] == "Datarax"

    def test_load_restores_benchmark_result(self, tmp_path):
        original = make_result()
        filepath = tmp_path / "result.json"
        original.save(filepath)

        loaded = BenchmarkResult.load(filepath)
        assert isinstance(loaded, BenchmarkResult)
        assert loaded.framework == original.framework
        assert loaded.scenario_id == original.scenario_id
        assert loaded.variant == original.variant

    def test_load_restores_timing(self, tmp_path):
        original = make_result()
        filepath = tmp_path / "result.json"
        original.save(filepath)

        loaded = BenchmarkResult.load(filepath)
        assert isinstance(loaded.timing, TimingSample)
        assert loaded.timing.wall_clock_sec == original.timing.wall_clock_sec
        assert loaded.timing.per_batch_times == original.timing.per_batch_times
        assert loaded.timing.num_batches == original.timing.num_batches

    def test_load_restores_resources(self, tmp_path):
        original = make_result()
        filepath = tmp_path / "result.json"
        original.save(filepath)

        loaded = BenchmarkResult.load(filepath)
        assert isinstance(loaded.resources, ResourceSummary)
        assert loaded.resources.peak_rss_mb == original.resources.peak_rss_mb

    def test_load_restores_none_resources(self, tmp_path):
        original = make_result(resources=None)
        filepath = tmp_path / "result.json"
        original.save(filepath)

        loaded = BenchmarkResult.load(filepath)
        assert loaded.resources is None

    def test_save_load_roundtrip_full(self, tmp_path):
        """Full round-trip: all fields preserved exactly."""
        original = make_result(
            extra_metrics={"throughput": 100.0, "latency_p99": 5.2},
        )
        filepath = tmp_path / "result.json"
        original.save(filepath)

        loaded = BenchmarkResult.load(filepath)
        assert loaded.framework == original.framework
        assert loaded.scenario_id == original.scenario_id
        assert loaded.variant == original.variant
        assert loaded.timing.wall_clock_sec == original.timing.wall_clock_sec
        assert loaded.timing.per_batch_times == original.timing.per_batch_times
        assert loaded.timing.first_batch_time == original.timing.first_batch_time
        assert loaded.timing.num_batches == original.timing.num_batches
        assert loaded.timing.num_elements == original.timing.num_elements
        assert loaded.resources.peak_rss_mb == original.resources.peak_rss_mb
        assert loaded.resources.mean_rss_mb == original.resources.mean_rss_mb
        assert loaded.environment == original.environment
        assert loaded.config == original.config
        assert loaded.timestamp == original.timestamp
        assert loaded.extra_metrics == original.extra_metrics


class TestBenchmarkResultThroughput:
    """Tests for throughput_elements_sec()."""

    def test_throughput_normal(self):
        result = make_result(
            timing=make_timing(wall_clock_sec=2.0, num_elements=160),
        )
        assert result.throughput_elements_sec() == pytest.approx(80.0)

    def test_throughput_zero_wall_clock(self):
        result = make_result(
            timing=make_timing(wall_clock_sec=0.0, num_elements=100),
        )
        assert result.throughput_elements_sec() == 0.0

    def test_throughput_zero_elements(self):
        result = make_result(
            timing=make_timing(wall_clock_sec=1.0, num_elements=0),
        )
        assert result.throughput_elements_sec() == 0.0


class TestBenchmarkResultLatencyPercentiles:
    """Tests for latency_percentiles()."""

    def test_returns_p50_p95_p99(self):
        result = make_result()
        percentiles = result.latency_percentiles()
        assert "p50" in percentiles
        assert "p95" in percentiles
        assert "p99" in percentiles

    def test_values_in_milliseconds(self):
        """Percentiles should be in milliseconds (per_batch_times are in seconds)."""
        times = [0.1] * 100  # 100ms per batch
        result = make_result(
            timing=make_timing(per_batch_times=times, num_batches=100),
        )
        percentiles = result.latency_percentiles()
        assert percentiles["p50"] == pytest.approx(100.0)

    def test_percentile_ordering(self):
        """p50 <= p95 <= p99."""
        rng = np.random.default_rng(42)
        times = rng.exponential(0.1, size=100).tolist()
        result = make_result(
            timing=make_timing(per_batch_times=times, num_batches=100),
        )
        percentiles = result.latency_percentiles()
        assert percentiles["p50"] <= percentiles["p95"] <= percentiles["p99"]

    def test_empty_per_batch_times(self):
        """Empty per_batch_times returns zeroed percentiles."""
        result = make_result(
            timing=make_timing(per_batch_times=[], num_batches=0),
        )
        percentiles = result.latency_percentiles()
        assert percentiles == {"p50": 0.0, "p95": 0.0, "p99": 0.0}
