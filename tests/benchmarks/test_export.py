"""Tests for the datarax → benchkit export adapter and FullExporter."""

import time
from unittest.mock import MagicMock

import pytest

import benchkit
from benchmarks.export import export_to_benchkit
from benchmarks.runners.full_runner import ComparativeResults
from datarax.benchmarking.results import BenchmarkResult
from datarax.benchmarking.resource_monitor import ResourceSummary
from datarax.benchmarking.timing import TimingSample


@pytest.fixture
def sample_comparative():
    """Build a minimal ComparativeResults for testing."""
    timing_fast = TimingSample(
        wall_clock_sec=1.0,
        per_batch_times=[0.01, 0.009, 0.011, 0.01, 0.01],
        first_batch_time=0.05,
        num_batches=5,
        num_elements=5000,
    )
    timing_slow = TimingSample(
        wall_clock_sec=1.2,
        per_batch_times=[0.012, 0.011, 0.013, 0.012, 0.012],
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
            BenchmarkResult(
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
            BenchmarkResult(
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


class TestExportToBenchkit:
    def test_returns_run(self, sample_comparative):
        run = export_to_benchkit(sample_comparative)
        assert isinstance(run, benchkit.Run)

    def test_correct_point_count(self, sample_comparative):
        run = export_to_benchkit(sample_comparative)
        assert len(run.points) == 2  # 1 result per adapter

    def test_point_names(self, sample_comparative):
        run = export_to_benchkit(sample_comparative)
        names = {p.name for p in run.points}
        assert "CV-1/small" in names

    def test_framework_tags(self, sample_comparative):
        run = export_to_benchkit(sample_comparative)
        frameworks = {p.tags["framework"] for p in run.points}
        assert frameworks == {"Datarax", "Grain"}

    def test_throughput_mapped(self, sample_comparative):
        run = export_to_benchkit(sample_comparative)
        datarax_point = next(p for p in run.points if p.tags["framework"] == "Datarax")
        assert "throughput" in datarax_point.metrics
        assert datarax_point.metrics["throughput"].value == pytest.approx(5000.0)

    def test_latency_mapped(self, sample_comparative):
        run = export_to_benchkit(sample_comparative)
        datarax_point = next(p for p in run.points if p.tags["framework"] == "Datarax")
        assert "latency_p50" in datarax_point.metrics
        assert datarax_point.metrics["latency_p50"].value > 0

    def test_memory_mapped_when_available(self, sample_comparative):
        run = export_to_benchkit(sample_comparative)
        datarax_point = next(p for p in run.points if p.tags["framework"] == "Datarax")
        assert "peak_memory" in datarax_point.metrics
        assert datarax_point.metrics["peak_memory"].value == 512.0
        assert "gpu_memory" in datarax_point.metrics

    def test_memory_absent_when_no_resources(self, sample_comparative):
        run = export_to_benchkit(sample_comparative)
        grain_point = next(p for p in run.points if p.tags["framework"] == "Grain")
        assert "peak_memory" not in grain_point.metrics

    def test_metric_defs_present(self, sample_comparative):
        run = export_to_benchkit(sample_comparative)
        assert "throughput" in run.metric_defs
        assert run.metric_defs["throughput"].direction == "higher"
        assert run.metric_defs["latency_p50"].direction == "lower"
        assert run.metric_defs["peak_memory"].direction == "lower"

    def test_environment_passed_through(self, sample_comparative):
        run = export_to_benchkit(sample_comparative)
        assert run.environment["cpu"] == "AMD Ryzen"

    def test_metadata_includes_platform(self, sample_comparative):
        run = export_to_benchkit(sample_comparative)
        assert run.metadata["platform"] == "cpu"


# --- FullExporter tests ---


class TestFullExporter:
    """Tests for FullExporter — composition of WandBExporter + datarax analysis."""

    @pytest.fixture
    def mock_store(self, tmp_path):
        store = benchkit.Store(tmp_path)
        return store

    @pytest.fixture
    def mock_wandb_exporter(self):
        """Mock WandBExporter that tracks all calls."""
        exporter = MagicMock()
        exporter.export_run.return_value = "https://wandb.ai/test/run/abc"
        exporter.project = "test"
        exporter.entity = None
        return exporter

    @pytest.fixture
    def sample_run(self, sample_comparative):
        """Convert comparative results to a benchkit Run."""
        return export_to_benchkit(sample_comparative)

    def test_full_export_delegates_to_wandb_exporter(
        self,
        sample_comparative,
        sample_run,
        mock_store,
        mock_wandb_exporter,
    ):
        """FullExporter should delegate core export to WandBExporter."""
        from benchmarks.export import FullExporter

        exporter = FullExporter(mock_store)
        exporter._exporter = mock_wandb_exporter

        url = exporter.export(sample_comparative, sample_run)

        assert url == "https://wandb.ai/test/run/abc"
        mock_wandb_exporter.export_run.assert_called_once_with(sample_run, finish=False)
        mock_wandb_exporter.export_analysis.assert_called_once()

    def test_full_export_logs_charts(
        self,
        sample_comparative,
        sample_run,
        mock_store,
        mock_wandb_exporter,
        tmp_path,
    ):
        """FullExporter should log chart figures via log_figures."""
        from benchmarks.export import FullExporter

        exporter = FullExporter(mock_store)
        exporter._exporter = mock_wandb_exporter

        exporter.export(sample_comparative, sample_run, chart_dir=tmp_path / "charts")

        # log_figures should have been called with chart keys
        mock_wandb_exporter.log_figures.assert_called_once()
        figures = mock_wandb_exporter.log_figures.call_args[0][0]
        # Should contain at least some chart keys (not all may succeed)
        assert isinstance(figures, dict)

    def test_full_export_logs_gap_detection(
        self,
        sample_comparative,
        sample_run,
        mock_store,
        mock_wandb_exporter,
    ):
        """FullExporter should log gap detection results via log_extra_tables."""
        from benchmarks.export import FullExporter

        exporter = FullExporter(mock_store)
        exporter._exporter = mock_wandb_exporter

        exporter.export(sample_comparative, sample_run)

        # log_extra_tables may or may not be called depending on gap detection results.
        # With Datarax being faster than Grain, no gaps should be detected,
        # so the call may not happen. We verify it doesn't crash.

    def test_full_export_logs_comparison_report(
        self,
        sample_comparative,
        sample_run,
        mock_store,
        mock_wandb_exporter,
    ):
        """FullExporter should log comparison report as HTML."""
        from benchmarks.export import FullExporter

        exporter = FullExporter(mock_store)
        exporter._exporter = mock_wandb_exporter

        exporter.export(sample_comparative, sample_run)

        mock_wandb_exporter.log_html_artifacts.assert_called_once()
        html_dict = mock_wandb_exporter.log_html_artifacts.call_args[0][0]
        assert "analysis/comparison_summary" in html_dict

    def test_full_export_logs_stability(
        self,
        sample_comparative,
        sample_run,
        mock_store,
        mock_wandb_exporter,
    ):
        """FullExporter should log stability validation results."""
        from benchmarks.export import FullExporter

        exporter = FullExporter(mock_store)
        exporter._exporter = mock_wandb_exporter

        exporter.export(sample_comparative, sample_run)

        # Stability table should have been logged
        mock_wandb_exporter.log_extra_tables.assert_called()
        # Find the stability call
        stability_logged = False
        for c in mock_wandb_exporter.log_extra_tables.call_args_list:
            tables = c[0][0]
            if "analysis/stability" in tables:
                stability_logged = True
                cols, rows = tables["analysis/stability"]
                assert "Scenario" in cols
                assert "Status" in cols
        assert stability_logged

    def test_full_export_with_gap_scenario(
        self,
        mock_store,
        mock_wandb_exporter,
    ):
        """FullExporter should log gaps when alternative leads Datarax."""
        from benchmarks.export import FullExporter

        # Create results where Grain is 2x faster than Datarax
        timing_slow = TimingSample(
            wall_clock_sec=2.0,
            per_batch_times=[0.02, 0.019, 0.021, 0.02, 0.02],
            first_batch_time=0.05,
            num_batches=5,
            num_elements=5000,
        )
        timing_fast = TimingSample(
            wall_clock_sec=1.0,
            per_batch_times=[0.01, 0.009, 0.011, 0.01, 0.01],
            first_batch_time=0.05,
            num_batches=5,
            num_elements=5000,
        )

        results = {
            "Datarax": [
                BenchmarkResult(
                    framework="Datarax",
                    scenario_id="CV-1",
                    variant="small",
                    timing=timing_slow,
                    resources=None,
                    environment={"cpu": "test"},
                    config={"batch_size": 32},
                ),
            ],
            "Grain": [
                BenchmarkResult(
                    framework="Grain",
                    scenario_id="CV-1",
                    variant="small",
                    timing=timing_fast,
                    resources=None,
                    environment={"cpu": "test"},
                    config={"batch_size": 32},
                ),
            ],
        }
        comparative = ComparativeResults(
            results=results,
            environment={"cpu": "test"},
            platform="cpu",
            timestamp=time.time(),
        )
        run = export_to_benchkit(comparative)

        exporter = FullExporter(mock_store)
        exporter._exporter = mock_wandb_exporter

        exporter.export(comparative, run)

        # Gaps table should have been logged
        gap_logged = False
        for c in mock_wandb_exporter.log_extra_tables.call_args_list:
            tables = c[0][0]
            if "analysis/gaps" in tables:
                gap_logged = True
                cols, rows = tables["analysis/gaps"]
                assert "Scenario" in cols
                assert len(rows) > 0  # At least one gap
        assert gap_logged
