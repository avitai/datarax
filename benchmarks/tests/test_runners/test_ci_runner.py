"""Tests for benchmark runner and CI runner.

RED phase: defines expected behavior for BenchmarkRunner and CI runner
before full integration testing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.adapters.datarax_adapter import DataraxAdapter
from benchmarks.runners.benchmark_runner import BenchmarkRunner
from benchmarks.scenarios.base import run_scenario
from datarax.benchmarking.results import BenchmarkResult


# ---------------------------------------------------------------------------
# BenchmarkRunner tests
# ---------------------------------------------------------------------------


class TestBenchmarkRunner:
    """Tests for the BenchmarkRunner orchestrator."""

    @pytest.fixture
    def runner(self, tmp_path: Path) -> BenchmarkRunner:
        """Runner with ci_cpu profile and temp output dir."""
        return BenchmarkRunner(
            output_dir=tmp_path / "output",
            hardware_profile="ci_cpu",
        )

    @pytest.fixture
    def adapter(self) -> DataraxAdapter:
        return DataraxAdapter()

    def test_runner_creates_output_dir(self, tmp_path: Path):
        """Runner must create output_dir if it doesn't exist."""
        output = tmp_path / "nonexistent" / "output"
        runner = BenchmarkRunner(output_dir=output, hardware_profile="ci_cpu")
        assert runner.output_dir.exists()

    def test_runner_loads_hardware_profile(self, runner: BenchmarkRunner):
        """Runner must load the ci_cpu profile correctly."""
        assert runner.num_batches > 0
        assert runner.warmup_batches > 0

    def test_run_single_scenario(
        self,
        runner: BenchmarkRunner,
        adapter: DataraxAdapter,
    ):
        """Running a single scenario produces a valid BenchmarkResult."""
        from benchmarks.scenarios.vision import cv1_image_classification as cv1

        result = runner.run_scenario(cv1, adapter, variant_name="small")
        assert isinstance(result, BenchmarkResult)
        assert result.scenario_id == "CV-1"
        assert result.variant == "small"
        assert result.timing.num_batches > 0

    def test_run_scenario_uses_tier1_variant_by_default(
        self,
        runner: BenchmarkRunner,
        adapter: DataraxAdapter,
    ):
        """When variant_name is None, use TIER1_VARIANT."""
        from benchmarks.scenarios.vision import cv1_image_classification as cv1

        result = runner.run_scenario(cv1, adapter, variant_name=None)
        assert result.variant == "small"  # CV-1's TIER1_VARIANT

    def test_run_all_returns_results(
        self,
        runner: BenchmarkRunner,
        adapter: DataraxAdapter,
    ):
        """run_all with a filter should return results for matching scenarios."""
        results = runner.run_all(
            adapter,
            scenario_filter={"CV-1"},
            num_repetitions=1,
        )
        assert len(results) >= 1
        assert all(isinstance(r, BenchmarkResult) for r in results)
        assert results[0].scenario_id == "CV-1"

    def test_run_all_saves_results(
        self,
        runner: BenchmarkRunner,
        adapter: DataraxAdapter,
    ):
        """run_all must save result JSONs to output_dir."""
        runner.run_all(
            adapter,
            scenario_filter={"CV-1"},
            num_repetitions=1,
        )
        json_files = list(runner.output_dir.glob("*.json"))
        assert len(json_files) >= 1


# ---------------------------------------------------------------------------
# Baseline generation tests
# ---------------------------------------------------------------------------


class TestBaselineGeneration:
    """Tests for baseline generation via BenchmarkRunner."""

    def test_generate_baselines_creates_files(self, tmp_path: Path):
        """generate_baselines must create baseline JSON files."""
        BenchmarkRunner(
            output_dir=tmp_path / "output",
            hardware_profile="ci_cpu",
        )
        adapter = DataraxAdapter()
        baselines_dir = tmp_path / "baselines"

        # Only generate for CV-1 small to keep test fast
        from benchmarks.core.baselines import BaselineStore
        from benchmarks.scenarios.vision import cv1_image_classification as cv1

        variant = cv1.get_variant("small")
        result = run_scenario(
            adapter,
            variant,
            num_batches=3,
            warmup_batches=1,
            num_repetitions=1,
        )

        store = BaselineStore(baselines_dir)
        path = store.save("CV-1_small", result)
        assert path.exists()

        loaded = store.load("CV-1_small")
        assert loaded is not None
        assert loaded["scenario_id"] == "CV-1"


# ---------------------------------------------------------------------------
# CI runner tests
# ---------------------------------------------------------------------------


class TestCIRunner:
    """Tests for the CI runner Tier 1 gate."""

    def test_ci_runner_imports(self):
        """CI runner module must be importable."""
        from benchmarks.runners import ci_runner

        assert hasattr(ci_runner, "run_tier1_gate")
        assert hasattr(ci_runner, "generate_ci_report")

    def test_generate_ci_report_format(self):
        """generate_ci_report must return a formatted string."""
        from benchmarks.runners.ci_runner import generate_ci_report

        # Create minimal test data
        result = BenchmarkResult(
            framework="Datarax",
            scenario_id="CV-1",
            variant="small",
            timing=_make_timing_sample(),
            resources=None,
            environment={},
            config={"batch_size": 10, "dataset_size": 100},
        )

        report = generate_ci_report([result], [None])
        assert isinstance(report, str)
        assert "CV-1" in report

    def test_generate_ci_report_with_verdict(self):
        """CI report must include verdict status when provided."""
        from benchmarks.runners.ci_runner import generate_ci_report

        result = BenchmarkResult(
            framework="Datarax",
            scenario_id="CV-1",
            variant="small",
            timing=_make_timing_sample(),
            resources=None,
            environment={},
            config={"batch_size": 10, "dataset_size": 100},
        )

        verdict = {
            "status": "pass",
            "throughput_ratio": 1.05,
            "baseline_throughput": 1000.0,
            "current_throughput": 1050.0,
        }

        report = generate_ci_report([result], [verdict])
        assert "pass" in report.lower()

    def test_run_tier1_gate_returns_results(self):
        """run_tier1_gate must return results and verdicts."""
        from benchmarks.runners.ci_runner import run_tier1_gate

        results, verdicts = run_tier1_gate(
            baselines_dir=None,  # No baselines = all verdicts None
            num_repetitions=1,
        )
        assert isinstance(results, list)
        assert isinstance(verdicts, list)
        assert len(results) == len(verdicts)
        # All verdicts should be None (no baselines)
        assert all(v is None for v in verdicts)
        # Should have at least one result
        assert len(results) > 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_timing_sample():
    """Create a minimal TimingSample for testing."""
    from datarax.benchmarking.timing import TimingSample

    return TimingSample(
        wall_clock_sec=1.0,
        per_batch_times=[0.1, 0.1, 0.1],
        first_batch_time=0.15,
        num_batches=3,
        num_elements=30,
    )
