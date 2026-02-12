"""Tests for the datarax-bench CLI."""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from datarax.benchmarking.results import BenchmarkResult
from datarax.benchmarking.timing import TimingSample


@pytest.fixture
def results_dir(tmp_path):
    """Create a fake results directory with a manifest."""
    results_path = tmp_path / "reports" / "latest"
    results_path.mkdir(parents=True)

    timing = TimingSample(
        wall_clock_sec=1.0,
        per_batch_times=[0.01, 0.009, 0.011, 0.01, 0.01],
        first_batch_time=0.05,
        num_batches=5,
        num_elements=5000,
    )
    result = BenchmarkResult(
        framework="Datarax",
        scenario_id="CV-1",
        variant="small",
        timing=timing,
        resources=None,
        environment={"cpu": "test"},
        config={"batch_size": 32},
    )
    result.save(results_path / "Datarax_CV-1_small.json")

    manifest = {
        "platform": "cpu",
        "timestamp": time.time(),
        "environment": {"cpu": "test"},
        "adapters": {
            "Datarax": ["Datarax_CV-1_small.json"],
        },
    }
    (results_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return results_path


class TestCLI:
    """Tests for datarax-bench CLI commands using Click's CliRunner."""

    def test_cli_module_importable(self):
        """The CLI module should be importable."""
        from benchmarks.cli import main

        assert callable(main)

    def test_run_no_wandb(self, tmp_path):
        """datarax-bench run --no-wandb should complete without W&B."""
        from click.testing import CliRunner
        from benchmarks.cli import main

        runner = CliRunner()
        with patch("benchmarks.cli.FullRunner") as mock_runner_cls:
            mock_runner = MagicMock()
            mock_comparative = MagicMock()
            mock_comparative.results = {"Datarax": []}
            mock_comparative.environment = {"cpu": "test"}
            mock_comparative.platform = "cpu"
            mock_comparative.timestamp = time.time()
            mock_comparative.all_scenario_ids = set()
            mock_runner.run_comparative.return_value = mock_comparative
            mock_runner.output_dir = tmp_path / "reports"
            mock_runner_cls.return_value = mock_runner

            result = runner.invoke(
                main,
                [
                    "run",
                    "--no-wandb",
                    "--no-charts",
                    "--repetitions",
                    "1",
                    "--output-dir",
                    str(tmp_path / "reports"),
                ],
            )

            assert result.exit_code == 0, result.output
            mock_runner.run_comparative.assert_called_once()

    def test_export_loads_and_exports(self, results_dir, tmp_path):
        """datarax-bench export should load existing results and export to W&B."""
        from click.testing import CliRunner
        from benchmarks.cli import main

        runner = CliRunner()
        with patch("benchmarks.cli.FullExporter") as mock_exporter_cls:
            mock_exporter = MagicMock()
            mock_exporter.export.return_value = "https://wandb.ai/test/run/abc"
            mock_exporter_cls.return_value = mock_exporter

            result = runner.invoke(
                main,
                [
                    "export",
                    "--results-dir",
                    str(results_dir),
                    "--data",
                    str(tmp_path),
                ],
            )

            assert result.exit_code == 0, result.output

    def test_analyze_generates_output(self, results_dir, tmp_path):
        """datarax-bench analyze should create analysis output files."""
        from click.testing import CliRunner
        from benchmarks.cli import main

        output_dir = tmp_path / "analysis"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "analyze",
                "--results-dir",
                str(results_dir),
                "--output",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0, result.output
        assert output_dir.exists()

    def test_report_generates_markdown(self, results_dir, tmp_path):
        """datarax-bench report should produce a markdown file."""
        from click.testing import CliRunner
        from benchmarks.cli import main

        report_path = tmp_path / "report.md"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "report",
                "--results-dir",
                str(results_dir),
                "--output",
                str(report_path),
            ],
        )

        assert result.exit_code == 0, result.output
        assert report_path.exists()
        content = report_path.read_text()
        assert "# Comparative Benchmark Analysis" in content
