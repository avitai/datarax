"""Tests for benchkit W&B exporter using WANDB_MODE=offline."""

from unittest.mock import MagicMock

import pytest

from benchkit.models import Metric, MetricDef, MetricPriority, Point, Run

# Skip entire module if wandb is not installed
wandb = pytest.importorskip("wandb")


@pytest.fixture(autouse=True)
def wandb_offline_mode(tmp_path, monkeypatch):
    """Force W&B into offline mode for testing (no network calls)."""
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    yield
    # Ensure any active run is finished
    if wandb.run is not None:
        wandb.finish()


@pytest.fixture
def baseline_run():
    return Run(
        points=[
            Point(
                "CV-1/small",
                scenario="CV-1",
                tags={"framework": "Datarax", "variant": "small"},
                metrics={"throughput": Metric(4500.0), "latency_p50": Metric(15.0)},
            ),
        ],
        metric_defs={
            "throughput": MetricDef("throughput", "elem/s", "higher"),
            "latency_p50": MetricDef("latency_p50", "ms", "lower"),
        },
    )


class TestWandBExporter:
    def test_export_run_returns_url(self, sample_run):
        from benchkit.exporters.wandb import WandBExporter

        exporter = WandBExporter(project="test-benchkit")
        url = exporter.export_run(sample_run)
        # In offline mode, URL will be empty string or contain offline marker
        assert isinstance(url, str)

    def test_export_run_logs_config(self, sample_run):
        from benchkit.exporters.wandb import WandBExporter

        exporter = WandBExporter(project="test-benchkit")
        exporter.export_run(sample_run)

        # wandb.run should have been created and finished
        # In offline mode we can't inspect the run, but we verify no errors

    def test_export_run_with_tags(self, sample_run):
        from benchkit.exporters.wandb import WandBExporter

        exporter = WandBExporter(
            project="test-benchkit",
            tags=["nightly", "v0.1"],
        )
        url = exporter.export_run(sample_run)
        assert isinstance(url, str)

    def test_export_analysis(self, sample_run, baseline_run):
        from benchkit.exporters.wandb import WandBExporter

        exporter = WandBExporter(project="test-benchkit")
        exporter.export_run(sample_run)
        # export_analysis should work within the same context
        exporter.export_analysis(sample_run, baseline_run)

    def test_export_analysis_no_baseline(self, sample_run):
        from benchkit.exporters.wandb import WandBExporter

        exporter = WandBExporter(project="test-benchkit")
        exporter.export_run(sample_run)
        # No baseline → no regressions, but should not crash
        exporter.export_analysis(sample_run, baseline=None)


class TestGracefulDegradation:
    def test_no_wandb_noop(self, monkeypatch, sample_run):
        """When wandb is not available, exporter is a no-op."""
        from benchkit.exporters import wandb as wandb_exporter_module

        # Simulate wandb not available
        monkeypatch.setattr(wandb_exporter_module, "WANDB_AVAILABLE", False)

        exporter = wandb_exporter_module.WandBExporter(project="test")
        url = exporter.export_run(sample_run)
        assert url == ""  # No URL when wandb unavailable

    def test_missing_api_key_returns_empty(self, monkeypatch, sample_run):
        """When no env var, no offline mode, and no stored creds, check_auth fails."""
        from benchkit.exporters import wandb as wandb_module

        # Must mock wandb to avoid picking up real `wandb login` creds
        mock_wandb = MagicMock()
        mock_wandb.api.api_key = None
        monkeypatch.setattr(wandb_module, "wandb", mock_wandb)
        monkeypatch.setattr(wandb_module, "WANDB_AVAILABLE", True)
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.delenv("WANDB_MODE", raising=False)

        exporter = wandb_module.WandBExporter(project="test")
        assert exporter.check_auth() is False

    def test_api_key_set_check_auth_passes(self, monkeypatch, sample_run):
        """When WANDB_API_KEY is set, check_auth returns True."""
        from benchkit.exporters.wandb import WandBExporter

        monkeypatch.setenv("WANDB_API_KEY", "test-key-12345")
        exporter = WandBExporter(project="test")
        assert exporter.check_auth() is True

    def test_offline_mode_check_auth_passes(self, monkeypatch, sample_run):
        """In offline mode, check_auth returns True even without API key."""
        from benchkit.exporters.wandb import WandBExporter

        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.setenv("WANDB_MODE", "offline")
        exporter = WandBExporter(project="test")
        assert exporter.check_auth() is True


class TestExportTrends:
    """Tests for WandBExporter.export_trends() — trend visualization."""

    def test_export_trends_logs_line_chart(self, tmp_path, sample_run):
        """export_trends should log trend data as a W&B line chart."""
        from datetime import datetime

        from benchkit.exporters.wandb import WandBExporter
        from benchkit.store import Store

        store = Store(tmp_path / "benchmark-data")
        for i in range(3):
            run = Run(
                points=[
                    Point(
                        "CV-1/small",
                        scenario="CV-1",
                        tags={"framework": "Datarax"},
                        metrics={"throughput": Metric(5000.0 + i * 100)},
                    ),
                ],
                timestamp=datetime(2026, 2, 1 + i, 12, 0, 0),
                commit=f"commit_{i}",
                branch="main",
            )
            store.save(run)

        exporter = WandBExporter(project="test-benchkit")
        # Should not raise
        exporter.export_trends(
            store,
            metric="throughput",
            point_name="CV-1/small",
            tags={"framework": "Datarax"},
        )

    def test_export_trends_empty_store(self, tmp_path):
        """export_trends on empty store should not raise."""
        from benchkit.exporters.wandb import WandBExporter
        from benchkit.store import Store

        store = Store(tmp_path / "benchmark-data")
        exporter = WandBExporter(project="test-benchkit")
        # Should handle empty gracefully
        exporter.export_trends(
            store,
            metric="throughput",
            point_name="CV-1/small",
            tags={"framework": "Datarax"},
        )

    def test_export_trends_with_n_runs(self, tmp_path):
        """export_trends with n_runs limits the data."""
        from datetime import datetime

        from benchkit.exporters.wandb import WandBExporter
        from benchkit.store import Store

        store = Store(tmp_path / "benchmark-data")
        for i in range(5):
            run = Run(
                points=[
                    Point(
                        "CV-1/small",
                        scenario="CV-1",
                        tags={"framework": "Datarax"},
                        metrics={"throughput": Metric(5000.0 + i * 100)},
                    ),
                ],
                timestamp=datetime(2026, 2, 1 + i, 12, 0, 0),
                branch="main",
            )
            store.save(run)

        exporter = WandBExporter(project="test-benchkit")
        # Limit to 3 most recent runs
        exporter.export_trends(
            store,
            metric="throughput",
            point_name="CV-1/small",
            tags={"framework": "Datarax"},
            n_runs=3,
        )


class TestFromStore:
    def test_from_store_reads_config(self, tmp_path, monkeypatch):
        """WandBExporter.from_store() reads project/entity from config.json."""
        import json

        from benchkit.exporters.wandb import WandBExporter
        from benchkit.store import Store

        config = {
            "wandb_project": "my-project",
            "wandb_entity": "my-team",
        }
        (tmp_path / "config.json").write_text(json.dumps(config))

        store = Store(tmp_path)
        exporter = WandBExporter.from_store(store)

        assert exporter.project == "my-project"
        assert exporter.entity == "my-team"

    def test_from_store_defaults(self, tmp_path):
        """WandBExporter.from_store() uses defaults when config is missing."""
        from benchkit.exporters.wandb import WandBExporter
        from benchkit.store import Store

        store = Store(tmp_path)
        exporter = WandBExporter.from_store(store)

        assert exporter.project == "benchmarks"
        assert exporter.entity is None

    def test_from_store_with_tags(self, tmp_path):
        """WandBExporter.from_store() passes through tags."""
        import json

        from benchkit.exporters.wandb import WandBExporter
        from benchkit.store import Store

        config = {"wandb_project": "test-proj"}
        (tmp_path / "config.json").write_text(json.dumps(config))

        store = Store(tmp_path)
        exporter = WandBExporter.from_store(store, tags=["nightly"])

        assert exporter.project == "test-proj"
        assert exporter.tags == ["nightly"]


# --- New tests for generic artifact logging methods ---


class TestLogFigures:
    """Tests for WandBExporter.log_figures() — matplotlib Figure → wandb.Image."""

    def test_log_figures_calls_wandb_image(self, monkeypatch):
        """Each figure should be logged as wandb.Image via wandb.log."""
        from benchkit.exporters import wandb as wandb_module

        mock_wandb = MagicMock()
        monkeypatch.setattr(wandb_module, "wandb", mock_wandb)
        monkeypatch.setattr(wandb_module, "WANDB_AVAILABLE", True)

        exporter = wandb_module.WandBExporter(project="test")

        fig1 = MagicMock()
        fig2 = MagicMock()
        exporter.log_figures({"charts/throughput": fig1, "charts/latency": fig2})

        assert mock_wandb.log.call_count == 2
        # Verify wandb.Image was called with each figure
        assert mock_wandb.Image.call_count == 2
        mock_wandb.Image.assert_any_call(fig1)
        mock_wandb.Image.assert_any_call(fig2)

    def test_log_figures_empty_dict_noop(self, monkeypatch):
        """Empty figures dict should not call wandb.log."""
        from benchkit.exporters import wandb as wandb_module

        mock_wandb = MagicMock()
        monkeypatch.setattr(wandb_module, "wandb", mock_wandb)
        monkeypatch.setattr(wandb_module, "WANDB_AVAILABLE", True)

        exporter = wandb_module.WandBExporter(project="test")
        exporter.log_figures({})
        mock_wandb.log.assert_not_called()


class TestLogHtmlArtifacts:
    """Tests for WandBExporter.log_html_artifacts() — HTML → wandb.Html."""

    def test_log_html_artifacts_calls_wandb_html(self, monkeypatch):
        """Each HTML string should be logged as wandb.Html."""
        from benchkit.exporters import wandb as wandb_module

        mock_wandb = MagicMock()
        monkeypatch.setattr(wandb_module, "wandb", mock_wandb)
        monkeypatch.setattr(wandb_module, "WANDB_AVAILABLE", True)

        exporter = wandb_module.WandBExporter(project="test")
        exporter.log_html_artifacts({"analysis/report": "<h1>Report</h1>"})

        mock_wandb.Html.assert_called_once_with("<h1>Report</h1>")
        assert mock_wandb.log.call_count == 1

    def test_log_html_empty_dict_noop(self, monkeypatch):
        """Empty HTML dict should not call wandb.log."""
        from benchkit.exporters import wandb as wandb_module

        mock_wandb = MagicMock()
        monkeypatch.setattr(wandb_module, "wandb", mock_wandb)
        monkeypatch.setattr(wandb_module, "WANDB_AVAILABLE", True)

        exporter = wandb_module.WandBExporter(project="test")
        exporter.log_html_artifacts({})
        mock_wandb.log.assert_not_called()


class TestLogExtraTables:
    """Tests for WandBExporter.log_extra_tables() — arbitrary tables."""

    def test_log_extra_tables_calls_wandb_table(self, monkeypatch):
        """Tables should be logged as wandb.Table with correct columns and data."""
        from benchkit.exporters import wandb as wandb_module

        mock_wandb = MagicMock()
        monkeypatch.setattr(wandb_module, "wandb", mock_wandb)
        monkeypatch.setattr(wandb_module, "WANDB_AVAILABLE", True)

        exporter = wandb_module.WandBExporter(project="test")
        exporter.log_extra_tables(
            {
                "analysis/gaps": (
                    ["Priority", "Scenario", "Gap"],
                    [["P0", "CV-1", "2.5x"], ["P1", "PC-1", "1.8x"]],
                ),
            }
        )

        mock_wandb.Table.assert_called_once_with(
            columns=["Priority", "Scenario", "Gap"],
            data=[["P0", "CV-1", "2.5x"], ["P1", "PC-1", "1.8x"]],
        )
        assert mock_wandb.log.call_count == 1

    def test_log_extra_tables_multiple(self, monkeypatch):
        """Multiple tables should each get their own wandb.log call."""
        from benchkit.exporters import wandb as wandb_module

        mock_wandb = MagicMock()
        monkeypatch.setattr(wandb_module, "wandb", mock_wandb)
        monkeypatch.setattr(wandb_module, "WANDB_AVAILABLE", True)

        exporter = wandb_module.WandBExporter(project="test")
        exporter.log_extra_tables(
            {
                "table/a": (["Col"], [["val"]]),
                "table/b": (["Col"], [["val2"]]),
            }
        )
        assert mock_wandb.Table.call_count == 2
        assert mock_wandb.log.call_count == 2


class TestPerScenarioTables:
    """Tests for per-scenario comparison tables in export_run()."""

    def test_per_scenario_tables_logged(self, monkeypatch):
        """export_run should log one table per scenario under scenarios/ prefix."""
        from benchkit.exporters import wandb as wandb_module

        mock_wandb = MagicMock()
        mock_wb_run = MagicMock()
        mock_wb_run.url = "https://wandb.ai/test/run/abc"
        mock_wandb.init.return_value = mock_wb_run
        monkeypatch.setattr(wandb_module, "wandb", mock_wandb)
        monkeypatch.setattr(wandb_module, "WANDB_AVAILABLE", True)

        # Two scenarios, two frameworks
        run = Run(
            points=[
                Point(
                    "CV-1/small",
                    scenario="CV-1",
                    tags={"framework": "Datarax"},
                    metrics={"throughput": Metric(5000.0)},
                ),
                Point(
                    "CV-1/small",
                    scenario="CV-1",
                    tags={"framework": "Grain"},
                    metrics={"throughput": Metric(4800.0)},
                ),
                Point(
                    "NLP-1/small",
                    scenario="NLP-1",
                    tags={"framework": "Datarax"},
                    metrics={"throughput": Metric(3000.0)},
                ),
            ],
            metric_defs={
                "throughput": MetricDef("throughput", "elem/s", "higher"),
            },
        )

        monkeypatch.setenv("WANDB_API_KEY", "test-key")
        exporter = wandb_module.WandBExporter(project="test")
        exporter.export_run(run)

        # Collect all keys logged to wandb
        logged_keys = set()
        for c in mock_wandb.log.call_args_list:
            if c.args:
                logged_keys.update(c.args[0].keys())
            if c.kwargs:
                logged_keys.update(c.kwargs.keys())

        # Should have per-scenario tables
        assert "scenarios/CV-1" in logged_keys
        assert "scenarios/NLP-1" in logged_keys


class TestAggregateScoresAndPareto:
    """Tests for aggregate_score() and pareto_front() in export_analysis()."""

    def test_aggregate_scores_logged(self, monkeypatch):
        """export_analysis should log aggregate scores table."""
        from benchkit.exporters import wandb as wandb_module

        mock_wandb = MagicMock()
        mock_wandb.run = None  # Will trigger wandb.init inside export_analysis
        mock_wandb.init.return_value = MagicMock()
        monkeypatch.setattr(wandb_module, "wandb", mock_wandb)
        monkeypatch.setattr(wandb_module, "WANDB_AVAILABLE", True)
        monkeypatch.setenv("WANDB_API_KEY", "test-key")

        run = Run(
            points=[
                Point(
                    "CV-1/small",
                    scenario="CV-1",
                    tags={"framework": "Datarax"},
                    metrics={"throughput": Metric(5000.0)},
                ),
                Point(
                    "CV-1/small",
                    scenario="CV-1",
                    tags={"framework": "Grain"},
                    metrics={"throughput": Metric(4800.0)},
                ),
            ],
            metric_defs={
                "throughput": MetricDef(
                    "throughput",
                    "elem/s",
                    "higher",
                    priority=MetricPriority.PRIMARY,
                ),
            },
        )

        exporter = wandb_module.WandBExporter(project="test")
        exporter.export_analysis(run, baseline=None)

        # Check aggregate scores table was logged
        logged_keys = set()
        for c in mock_wandb.log.call_args_list:
            if c.args:
                logged_keys.update(c.args[0].keys())
        assert "analysis/aggregate_scores" in logged_keys

    def test_pareto_front_logged(self, monkeypatch):
        """export_analysis should log Pareto front when throughput + latency exist."""
        from benchkit.exporters import wandb as wandb_module

        mock_wandb = MagicMock()
        mock_wandb.run = None
        mock_wandb.init.return_value = MagicMock()
        monkeypatch.setattr(wandb_module, "wandb", mock_wandb)
        monkeypatch.setattr(wandb_module, "WANDB_AVAILABLE", True)
        monkeypatch.setenv("WANDB_API_KEY", "test-key")

        run = Run(
            points=[
                Point(
                    "CV-1/small",
                    scenario="CV-1",
                    tags={"framework": "Datarax"},
                    metrics={
                        "throughput": Metric(5000.0),
                        "latency_p50": Metric(12.0),
                    },
                ),
                Point(
                    "CV-1/small",
                    scenario="CV-1",
                    tags={"framework": "Grain"},
                    metrics={
                        "throughput": Metric(4800.0),
                        "latency_p50": Metric(14.0),
                    },
                ),
            ],
            metric_defs={
                "throughput": MetricDef("throughput", "elem/s", "higher"),
                "latency_p50": MetricDef("latency_p50", "ms", "lower"),
            },
        )

        exporter = wandb_module.WandBExporter(project="test")
        exporter.export_analysis(run, baseline=None)

        logged_keys = set()
        for c in mock_wandb.log.call_args_list:
            if c.args:
                logged_keys.update(c.args[0].keys())
        assert "analysis/pareto_front" in logged_keys


class TestCheckAuthWandBLogin:
    """Tests for check_auth() accepting `wandb login` stored credentials."""

    def test_check_auth_accepts_wandb_login(self, monkeypatch):
        """check_auth should return True when wandb.api.api_key is set."""
        from benchkit.exporters import wandb as wandb_module

        mock_wandb = MagicMock()
        mock_wandb.api.api_key = "stored-key-from-wandb-login"
        monkeypatch.setattr(wandb_module, "wandb", mock_wandb)
        monkeypatch.setattr(wandb_module, "WANDB_AVAILABLE", True)
        # No env var, no offline mode
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.delenv("WANDB_MODE", raising=False)

        exporter = wandb_module.WandBExporter(project="test")
        assert exporter.check_auth() is True

    def test_check_auth_false_when_no_creds_at_all(self, monkeypatch):
        """check_auth should return False when no env var and no stored creds."""
        from benchkit.exporters import wandb as wandb_module

        mock_wandb = MagicMock()
        mock_wandb.api.api_key = None
        monkeypatch.setattr(wandb_module, "wandb", mock_wandb)
        monkeypatch.setattr(wandb_module, "WANDB_AVAILABLE", True)
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.delenv("WANDB_MODE", raising=False)

        exporter = wandb_module.WandBExporter(project="test")
        assert exporter.check_auth() is False

    def test_check_auth_handles_wandb_api_exception(self, monkeypatch):
        """check_auth should return False if wandb.api.api_key raises."""
        from benchkit.exporters import wandb as wandb_module

        class _BrokenApi:
            @property
            def api_key(self):
                raise AttributeError("no api")

        mock_wandb = MagicMock()
        mock_wandb.api = _BrokenApi()
        monkeypatch.setattr(wandb_module, "wandb", mock_wandb)
        monkeypatch.setattr(wandb_module, "WANDB_AVAILABLE", True)
        monkeypatch.delenv("WANDB_API_KEY", raising=False)
        monkeypatch.delenv("WANDB_MODE", raising=False)

        exporter = wandb_module.WandBExporter(project="test")
        assert exporter.check_auth() is False
