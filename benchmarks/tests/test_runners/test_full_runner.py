"""Tests for the full comparative benchmark runner.

RED phase: defines expected behavior for FullRunner and ComparativeResults.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from datarax.benchmarking.results import BenchmarkResult
from datarax.benchmarking.timing import TimingSample


def _make_timing_sample(wall_clock: float = 1.0) -> TimingSample:
    return TimingSample(
        wall_clock_sec=wall_clock,
        per_batch_times=[0.05] * 20,
        first_batch_time=0.08,
        num_batches=20,
        num_elements=640,
    )


def _make_result(
    framework: str = "Datarax",
    scenario_id: str = "CV-1",
) -> BenchmarkResult:
    return BenchmarkResult(
        framework=framework,
        scenario_id=scenario_id,
        variant="small",
        timing=_make_timing_sample(),
        resources=None,
        environment={"platform": {"backend": "cpu"}},
        config={"batch_size": 32, "dataset_size": 640},
        timestamp=time.time(),
    )


def _patch_full_runner():
    """Context manager that patches all external dependencies for FullRunner."""
    return patch.multiple(
        "benchmarks.runners.full_runner",
        capture_environment=MagicMock(
            return_value={"platform": {"backend": "cpu", "device_count": 1}},
        ),
        can_run_scenario=MagicMock(return_value=True),
    )


def _make_mock_run_scenario(framework: str, scenario_id: str):
    """Return a function that produces a real BenchmarkResult."""

    def _run(adapter, variant, num_batches=50, warmup_batches=5, num_repetitions=5):
        return _make_result(framework, scenario_id)

    return _run


# ---------------------------------------------------------------------------
# ComparativeResults tests
# ---------------------------------------------------------------------------


class TestComparativeResults:
    """Tests for the ComparativeResults data container."""

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        """Verify ComparativeResults serialize to JSON and load back."""
        from benchmarks.runners.full_runner import ComparativeResults

        results = ComparativeResults(
            results={"Datarax": [_make_result()]},
            environment={"platform": {"backend": "cpu"}},
            platform="cpu",
            timestamp=time.time(),
        )
        results.save(tmp_path)

        loaded = ComparativeResults.load(tmp_path)
        assert "Datarax" in loaded.results
        assert len(loaded.results["Datarax"]) == 1
        assert loaded.platform == "cpu"

    def test_get_results_for_scenario(self):
        """get_scenario_results returns results grouped by adapter."""
        from benchmarks.runners.full_runner import ComparativeResults

        results = ComparativeResults(
            results={
                "Datarax": [_make_result("Datarax", "CV-1")],
                "Grain": [_make_result("Grain", "CV-1"), _make_result("Grain", "NLP-1")],
            },
            environment={},
            platform="cpu",
            timestamp=time.time(),
        )

        cv1 = results.get_scenario_results("CV-1")
        assert "Datarax" in cv1
        assert "Grain" in cv1
        assert cv1["Datarax"].scenario_id == "CV-1"

    def test_get_results_for_adapter(self):
        """get_adapter_results returns all results for one adapter."""
        from benchmarks.runners.full_runner import ComparativeResults

        results = ComparativeResults(
            results={
                "Datarax": [_make_result("Datarax", "CV-1"), _make_result("Datarax", "NLP-1")],
            },
            environment={},
            platform="cpu",
            timestamp=time.time(),
        )

        datarax = results.get_adapter_results("Datarax")
        assert len(datarax) == 2
        assert {r.scenario_id for r in datarax} == {"CV-1", "NLP-1"}


# ---------------------------------------------------------------------------
# FullRunner tests
# ---------------------------------------------------------------------------


def _setup_mocks(adapter_names, scenario_ids):
    """Set up patchers for adapter registry and scenario discovery.

    Returns a tuple of (patchers_dict, context_managers).
    """
    from benchmarks.adapters.base import ScenarioConfig
    from benchmarks.scenarios.base import ScenarioVariant

    # Create mock adapters
    available = {}
    for name in adapter_names:
        adapter = MagicMock()
        adapter.name = name
        adapter.supports_scenario.return_value = True
        adapter.supported_scenarios.return_value = set(scenario_ids)
        adapter_cls = MagicMock(return_value=adapter)
        available[name] = adapter_cls

    # Create mock scenarios
    scenarios = []
    for sid in scenario_ids:
        config = ScenarioConfig(
            scenario_id=sid,
            dataset_size=640,
            element_shape=(32, 32, 3),
            batch_size=32,
            transforms=[],
            seed=42,
            extra={"variant_name": "small"},
        )
        variant = ScenarioVariant(
            config=config,
            data_generator=lambda: {"image": __import__("numpy").zeros((640, 32, 32, 3))},
        )
        mod = MagicMock()
        mod.SCENARIO_ID = sid
        mod.VARIANTS = {"small": variant}
        mod.TIER1_VARIANT = "small"
        mod.get_variant.return_value = variant
        scenarios.append(mod)

    return available, scenarios


class TestFullRunner:
    """Tests for the multi-adapter comparative benchmark runner."""

    def _run_with_mocks(
        self,
        tmp_path,
        adapter_names,
        scenario_ids,
        scenario_filter=None,
        adapter_filter=None,
    ):
        """Helper to run FullRunner with full mocking."""
        from benchmarks.runners.full_runner import FullRunner

        runner = FullRunner(output_dir=tmp_path, hardware_profile="ci_cpu")
        available, scenarios = _setup_mocks(adapter_names, scenario_ids)

        # Build a run_scenario side effect that returns proper results
        def mock_run_scenario(
            adapter,
            variant,
            num_batches=50,
            warmup_batches=5,
            num_repetitions=5,
        ):
            return _make_result(adapter.name, variant.config.scenario_id)

        with (
            _patch_full_runner(),
            patch("benchmarks.runners.full_runner.get_available_adapters", return_value=available),
            patch("benchmarks.runners.full_runner.discover_scenarios", return_value=scenarios),
            patch("benchmarks.runners.full_runner.run_scenario", side_effect=mock_run_scenario),
        ):
            comparative = runner.run_comparative(
                scenario_filter=scenario_filter,
                adapter_filter=adapter_filter,
                num_repetitions=1,
            )

        return runner, comparative

    def test_run_comparative_returns_results_per_adapter(self, tmp_path: Path):
        """run_comparative must return ComparativeResults with all adapters."""
        from benchmarks.runners.full_runner import ComparativeResults

        _, comparative = self._run_with_mocks(
            tmp_path,
            ["Datarax"],
            ["CV-1"],
        )
        assert isinstance(comparative, ComparativeResults)
        assert "Datarax" in comparative.results

    def test_filters_by_scenario(self, tmp_path: Path):
        """run_comparative with scenario_filter only runs matching scenarios."""
        _, comparative = self._run_with_mocks(
            tmp_path,
            ["Datarax"],
            ["CV-1", "NLP-1"],
            scenario_filter={"CV-1"},
        )
        for results in comparative.results.values():
            for r in results:
                assert r.scenario_id == "CV-1"

    def test_saves_results_to_release_dir(self, tmp_path: Path):
        """run_comparative must save result JSONs to output_dir."""
        runner, _ = self._run_with_mocks(
            tmp_path,
            ["Datarax"],
            ["CV-1"],
        )
        json_files = list(tmp_path.glob("**/*.json"))
        assert len(json_files) > 0

    def test_generates_summary_json(self, tmp_path: Path):
        """run_comparative must generate a summary.json alongside results."""
        self._run_with_mocks(tmp_path, ["Datarax"], ["CV-1"])

        summary_path = tmp_path / "summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert "adapters" in summary
        assert "scenarios" in summary

    def test_skips_unavailable_adapters(self, tmp_path: Path):
        """run_comparative with adapter_filter excludes non-matching adapters."""
        _, comparative = self._run_with_mocks(
            tmp_path,
            ["Datarax", "Grain"],
            ["CV-1"],
            adapter_filter={"Datarax"},
        )
        assert "Datarax" in comparative.results
        assert "Grain" not in comparative.results

    def test_clears_caches_between_frameworks(self, tmp_path: Path):
        """run_comparative must clear caches between framework runs."""
        from benchmarks.runners.full_runner import FullRunner

        runner = FullRunner(output_dir=tmp_path, hardware_profile="ci_cpu")
        available, scenarios = _setup_mocks(["Datarax", "Grain"], ["CV-1"])

        def mock_run_scenario(
            adapter,
            variant,
            num_batches=50,
            warmup_batches=5,
            num_repetitions=5,
        ):
            return _make_result(adapter.name, variant.config.scenario_id)

        with (
            _patch_full_runner(),
            patch("benchmarks.runners.full_runner.get_available_adapters", return_value=available),
            patch("benchmarks.runners.full_runner.discover_scenarios", return_value=scenarios),
            patch("benchmarks.runners.full_runner.run_scenario", side_effect=mock_run_scenario),
            patch.object(runner, "_clear_framework_caches") as mock_clear,
        ):
            runner.run_comparative(num_repetitions=1)

        # Should be called between/after frameworks (at least once)
        assert mock_clear.call_count >= 1

    def test_cli_platform_flag(self, tmp_path: Path):
        """Verify FullRunner accepts platform parameter."""
        from benchmarks.runners.full_runner import FullRunner

        runner = FullRunner(
            output_dir=tmp_path,
            hardware_profile="ci_cpu",
            platform="cpu",
        )
        assert runner.platform == "cpu"
