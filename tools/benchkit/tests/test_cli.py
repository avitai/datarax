"""Tests for benchkit CLI using click.testing.CliRunner."""

import json

import pytest
from click.testing import CliRunner

from benchkit.cli import main
from benchkit.models import Metric, MetricDef, MetricPriority, Point, Run
from benchkit.store import Store


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def store_with_data(tmp_path):
    """Create a store with two runs and a baseline."""
    store = Store(tmp_path / "benchmark-data")

    baseline = Run(
        points=[
            Point(
                "CV-1/small",
                scenario="CV-1",
                tags={"framework": "Datarax", "variant": "small"},
                metrics={"throughput": Metric(5000.0)},
            ),
        ],
        commit="baseline_abc",
        branch="main",
        metric_defs={
            "throughput": MetricDef(
                "throughput",
                "elem/s",
                "higher",
                group="Throughput",
                priority=MetricPriority.PRIMARY,
            ),
        },
    )
    store.save(baseline)
    store.set_baseline(baseline.id)

    current = Run(
        points=[
            Point(
                "CV-1/small",
                scenario="CV-1",
                tags={"framework": "Datarax", "variant": "small"},
                metrics={"throughput": Metric(5200.0)},
            ),
        ],
        commit="current_def",
        branch="main",
        metric_defs={
            "throughput": MetricDef(
                "throughput",
                "elem/s",
                "higher",
                group="Throughput",
                priority=MetricPriority.PRIMARY,
            ),
        },
    )
    store.save(current)

    return tmp_path / "benchmark-data", baseline, current


class TestIngest:
    def test_ingest_json(self, runner, tmp_path):
        data_dir = tmp_path / "benchmark-data"
        run = Run(
            points=[
                Point(
                    "CV-1/small",
                    scenario="CV-1",
                    tags={"framework": "Datarax"},
                    metrics={"throughput": Metric(5000.0)},
                ),
            ],
        )
        input_file = tmp_path / "input.json"
        input_file.write_text(json.dumps(run.to_dict(), indent=2))

        result = runner.invoke(
            main,
            [
                "ingest",
                "--data",
                str(data_dir),
                "--input",
                str(input_file),
            ],
        )
        assert result.exit_code == 0
        assert "Ingested" in result.output

    def test_ingest_nonexistent_file(self, runner, tmp_path):
        data_dir = tmp_path / "benchmark-data"
        result = runner.invoke(
            main,
            [
                "ingest",
                "--data",
                str(data_dir),
                "--input",
                str(tmp_path / "nope.json"),
            ],
        )
        assert result.exit_code != 0


class TestCheck:
    def test_check_passes_no_regression(self, runner, store_with_data):
        data_dir, _, _ = store_with_data
        result = runner.invoke(
            main,
            [
                "check",
                "--data",
                str(data_dir),
                "--threshold",
                "0.05",
            ],
        )
        assert result.exit_code == 0
        assert "PASS" in result.output or "pass" in result.output.lower()

    def test_check_fails_on_regression(self, runner, tmp_path):
        """Create a scenario where throughput dropped 20%."""
        store = Store(tmp_path / "benchmark-data")
        defs = {"throughput": MetricDef("throughput", "elem/s", "higher")}

        baseline = Run(
            points=[
                Point(
                    "CV-1/small",
                    scenario="CV-1",
                    tags={"framework": "Datarax"},
                    metrics={"throughput": Metric(5000.0)},
                ),
            ],
            metric_defs=defs,
        )
        store.save(baseline)
        store.set_baseline(baseline.id)

        current = Run(
            points=[
                Point(
                    "CV-1/small",
                    scenario="CV-1",
                    tags={"framework": "Datarax"},
                    metrics={"throughput": Metric(4000.0)},  # -20%
                ),
            ],
            metric_defs=defs,
        )
        store.save(current)

        result = runner.invoke(
            main,
            [
                "check",
                "--data",
                str(tmp_path / "benchmark-data"),
                "--threshold",
                "0.05",
            ],
        )
        assert result.exit_code == 1
        assert "FAIL" in result.output or "regression" in result.output.lower()

    def test_check_no_baseline_warns(self, runner, tmp_path):
        """No baseline set → should warn, not crash."""
        store = Store(tmp_path / "benchmark-data")
        run = Run(points=[], metric_defs={})
        store.save(run)

        result = runner.invoke(
            main,
            [
                "check",
                "--data",
                str(tmp_path / "benchmark-data"),
            ],
        )
        # Should exit 0 (can't check without baseline) and print a message
        assert "baseline" in result.output.lower()


class TestBaseline:
    def test_set_baseline_latest(self, runner, store_with_data):
        data_dir, _, current = store_with_data
        result = runner.invoke(
            main,
            [
                "baseline",
                "--data",
                str(data_dir),
                "--run",
                "latest",
            ],
        )
        assert result.exit_code == 0
        assert "Baseline set" in result.output

    def test_set_baseline_by_id(self, runner, store_with_data):
        data_dir, baseline, _ = store_with_data
        result = runner.invoke(
            main,
            [
                "baseline",
                "--data",
                str(data_dir),
                "--run",
                baseline.id,
            ],
        )
        assert result.exit_code == 0


class TestSummary:
    def test_summary_latest(self, runner, store_with_data):
        data_dir, _, _ = store_with_data
        result = runner.invoke(
            main,
            [
                "summary",
                "--data",
                str(data_dir),
                "--run",
                "latest",
            ],
        )
        assert result.exit_code == 0
        assert "throughput" in result.output.lower() or "CV-1" in result.output

    def test_summary_empty_store(self, runner, tmp_path):
        data_dir = tmp_path / "benchmark-data"
        result = runner.invoke(
            main,
            [
                "summary",
                "--data",
                str(data_dir),
                "--run",
                "latest",
            ],
        )
        assert result.exit_code != 0


class TestTrend:
    """Tests for CLI `benchkit trend` command."""

    @pytest.fixture
    def store_with_trend_data(self, tmp_path):
        """Create a store with 5 runs for trend tracking."""
        from datetime import datetime

        store = Store(tmp_path / "benchmark-data")
        runs = []
        for i in range(5):
            run = Run(
                points=[
                    Point(
                        "CV-1/small",
                        scenario="CV-1",
                        tags={"framework": "Datarax"},
                        metrics={"throughput": Metric(5000.0 + i * 100)},
                    ),
                    Point(
                        "CV-1/small",
                        scenario="CV-1",
                        tags={"framework": "Grain"},
                        metrics={"throughput": Metric(4800.0 + i * 50)},
                    ),
                ],
                timestamp=datetime(2026, 2, 1 + i, 12, 0, 0),
                commit=f"commit_{i}",
                branch="main",
            )
            store.save(run)
            runs.append(run)
        return tmp_path / "benchmark-data", runs

    def test_trend_shows_values(self, runner, store_with_trend_data):
        data_dir, _ = store_with_trend_data
        result = runner.invoke(
            main,
            [
                "trend",
                "--data",
                str(data_dir),
                "--metric",
                "throughput",
                "--point",
                "CV-1/small",
                "--framework",
                "Datarax",
            ],
        )
        assert result.exit_code == 0
        assert "5000" in result.output
        assert "5400" in result.output

    def test_trend_with_n_runs(self, runner, store_with_trend_data):
        data_dir, _ = store_with_trend_data
        result = runner.invoke(
            main,
            [
                "trend",
                "--data",
                str(data_dir),
                "--metric",
                "throughput",
                "--point",
                "CV-1/small",
                "--framework",
                "Datarax",
                "--n-runs",
                "3",
            ],
        )
        assert result.exit_code == 0
        # Should only show the 3 most recent: 5200, 5300, 5400
        assert "5200" in result.output
        assert "5400" in result.output

    def test_trend_shows_commits(self, runner, store_with_trend_data):
        data_dir, _ = store_with_trend_data
        result = runner.invoke(
            main,
            [
                "trend",
                "--data",
                str(data_dir),
                "--metric",
                "throughput",
                "--point",
                "CV-1/small",
                "--framework",
                "Datarax",
            ],
        )
        assert result.exit_code == 0
        assert "commit_0" in result.output

    def test_trend_empty_store(self, runner, tmp_path):
        data_dir = tmp_path / "benchmark-data"
        result = runner.invoke(
            main,
            [
                "trend",
                "--data",
                str(data_dir),
                "--metric",
                "throughput",
                "--point",
                "CV-1/small",
                "--framework",
                "Datarax",
            ],
        )
        assert result.exit_code == 0
        assert "no data" in result.output.lower() or "0 points" in result.output.lower()


class TestExport:
    """Tests for CLI `benchkit export` command."""

    def test_export_no_wandb_shows_error(self, runner, store_with_data, monkeypatch):
        """When wandb is not installed, export should fail with helpful message."""
        data_dir, _, _ = store_with_data

        # Force WANDB_AVAILABLE=False by patching the import check in cli
        import benchkit.exporters.wandb as wandb_mod

        original = wandb_mod.WANDB_AVAILABLE
        monkeypatch.setattr(wandb_mod, "WANDB_AVAILABLE", False)

        result = runner.invoke(
            main,
            [
                "export",
                "--data",
                str(data_dir),
                "--run",
                "latest",
            ],
        )
        assert result.exit_code != 0
        assert "wandb" in result.output.lower()

        monkeypatch.setattr(wandb_mod, "WANDB_AVAILABLE", original)

    def test_export_empty_store(self, runner, tmp_path):
        """Export on empty store should fail gracefully."""
        data_dir = tmp_path / "benchmark-data"
        result = runner.invoke(
            main,
            [
                "export",
                "--data",
                str(data_dir),
                "--run",
                "latest",
            ],
        )
        assert result.exit_code != 0

    def test_export_nonexistent_run(self, runner, store_with_data):
        """Export with bad run ID should fail."""
        data_dir, _, _ = store_with_data
        result = runner.invoke(
            main,
            [
                "export",
                "--data",
                str(data_dir),
                "--run",
                "nonexistent_id",
            ],
        )
        assert result.exit_code != 0
