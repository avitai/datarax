"""Tests for the datarax -> calibrax export adapter and FullExporter."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from calibrax.core import MetricDirection, Run
from calibrax.profiling import ResourceSummary, TimingSample
from calibrax.storage import Store

from benchmarks.core.result_model import build_benchmark_result
from benchmarks.export import export_to_calibrax, FullExporter
from benchmarks.runners.full_runner import ComparativeResults


@pytest.fixture
def sample_comparative() -> ComparativeResults:
    """Build a minimal ComparativeResults for testing."""
    timing_fast = TimingSample(
        wall_clock_sec=1.0,
        per_batch_times=(0.01, 0.009, 0.011, 0.01, 0.01),
        first_batch_time=0.05,
        num_batches=5,
        num_elements=5000,
    )
    timing_slow = TimingSample(
        wall_clock_sec=1.2,
        per_batch_times=(0.012, 0.011, 0.013, 0.012, 0.012),
        first_batch_time=0.06,
        num_batches=5,
        num_elements=5000,
    )

    resources = ResourceSummary(
        peak_rss_mb=512.0,
        mean_rss_mb=400.0,
        peak_gpu_mem_mb=2048.0,
        mean_gpu_util=75.0,
        memory_growth_mb=50.0,
        num_samples=10,
        duration_sec=1.0,
    )

    results = {
        "Datarax": [
            build_benchmark_result(
                framework="Datarax",
                scenario_id="CV-1",
                variant="small",
                timing=timing_fast,
                resources=resources,
                environment={"cpu": "AMD Ryzen"},
                config={"batch_size": 32},
            ),
        ],
        "Grain": [
            build_benchmark_result(
                framework="Grain",
                scenario_id="CV-1",
                variant="small",
                timing=timing_slow,
                resources=None,
                environment={"cpu": "AMD Ryzen"},
                config={"batch_size": 32},
            ),
        ],
    }

    return ComparativeResults(
        results=results,
        environment={"cpu": "AMD Ryzen", "os": "Linux"},
        platform="cpu",
        timestamp=time.time(),
    )


class TestExportToCalibraX:
    def test_returns_run(self, sample_comparative: ComparativeResults):
        run = export_to_calibrax(sample_comparative)
        assert isinstance(run, Run)

    def test_correct_point_count(self, sample_comparative: ComparativeResults):
        run = export_to_calibrax(sample_comparative)
        assert len(run.points) == 2

    def test_metric_defs_direction(self, sample_comparative: ComparativeResults):
        run = export_to_calibrax(sample_comparative)
        assert run.metric_defs["throughput"].direction == MetricDirection.HIGHER
        assert run.metric_defs["latency_p50"].direction == MetricDirection.LOWER
        assert run.metric_defs["peak_memory"].direction == MetricDirection.LOWER

    def test_tags_and_environment(self, sample_comparative: ComparativeResults):
        run = export_to_calibrax(sample_comparative)
        frameworks = {p.tags["framework"] for p in run.points}
        assert frameworks == {"Datarax", "Grain"}
        assert run.environment["cpu"] == "AMD Ryzen"
        assert run.metadata["platform"] == "cpu"


class TestFullExporter:
    @pytest.fixture
    def mock_store(self, tmp_path):
        return Store(tmp_path)

    @pytest.fixture
    def mock_wandb_exporter(self):
        exporter = MagicMock()
        exporter.export_run.return_value = "https://wandb.ai/test/run/abc"
        return exporter

    @pytest.fixture
    def sample_run(self, sample_comparative: ComparativeResults) -> Run:
        return export_to_calibrax(sample_comparative)

    def test_full_export_delegates_to_wandb(
        self,
        sample_comparative: ComparativeResults,
        sample_run: Run,
        mock_store: Store,
        mock_wandb_exporter: MagicMock,
    ):
        pytest.importorskip("wandb")
        exporter = FullExporter(mock_store)
        exporter._exporter = mock_wandb_exporter

        url = exporter.export(sample_comparative, sample_run)

        assert url == "https://wandb.ai/test/run/abc"
        mock_wandb_exporter.export_run.assert_called_once_with(sample_run, finish=False)
        mock_wandb_exporter.export_analysis.assert_called_once()
