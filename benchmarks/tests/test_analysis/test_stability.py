"""Tests for measurement stability validation.

RED phase: defines expected behavior for StabilityValidator.
"""

from __future__ import annotations

from benchmarks.tests.test_analysis.conftest import make_result


class TestStabilityValidator:
    """Tests for the StabilityValidator measurement quality checker."""

    def test_stable_results_pass(self):
        """Results with CV=0.05 should pass stability validation."""
        from benchmarks.analysis.stability import StabilityValidator
        from benchmarks.runners.full_runner import ComparativeResults

        results_dict = {
            "Datarax": [make_result(cv=0.05, scenario_id="CV-1")],
        }
        comparative = ComparativeResults(
            results=results_dict,
            environment={},
            platform="cpu",
            timestamp=0,
        )

        validator = StabilityValidator(cv_threshold=0.10)
        report = validator.validate(comparative)

        assert len(report.stable_scenarios) == 1
        assert len(report.unstable_scenarios) == 0

    def test_unstable_results_flagged(self):
        """Results with CV=0.15 should be flagged as unstable."""
        from benchmarks.analysis.stability import StabilityValidator
        from benchmarks.runners.full_runner import ComparativeResults

        results_dict = {
            "Datarax": [make_result(cv=0.15, scenario_id="CV-1")],
        }
        comparative = ComparativeResults(
            results=results_dict,
            environment={},
            platform="cpu",
            timestamp=0,
        )

        validator = StabilityValidator(cv_threshold=0.10)
        report = validator.validate(comparative)

        assert len(report.unstable_scenarios) >= 1
        # Unstable tuple: (scenario_id, adapter_name, cv)
        assert report.unstable_scenarios[0][0] == "CV-1"
        assert report.unstable_scenarios[0][1] == "Datarax"

    def test_recommended_reruns_for_unstable(self):
        """Unstable scenarios should get rerun recommendations."""
        from benchmarks.analysis.stability import StabilityValidator
        from benchmarks.runners.full_runner import ComparativeResults

        results_dict = {
            "Datarax": [make_result(cv=0.15, scenario_id="CV-1")],
        }
        comparative = ComparativeResults(
            results=results_dict,
            environment={},
            platform="cpu",
            timestamp=0,
        )

        validator = StabilityValidator(cv_threshold=0.10)
        report = validator.validate(comparative)
        reruns = validator.recommend_reruns(report)

        assert isinstance(reruns, dict)
        # Should recommend more repetitions for unstable scenarios
        assert len(reruns) > 0

    def test_generates_stability_report(self):
        """validate() must return a StabilityReport with correct counts."""
        from benchmarks.analysis.stability import StabilityReport, StabilityValidator
        from benchmarks.runners.full_runner import ComparativeResults

        results_dict = {
            "Datarax": [
                make_result(cv=0.05, scenario_id="CV-1"),
                make_result(cv=0.15, scenario_id="NLP-1"),
            ],
        }
        comparative = ComparativeResults(
            results=results_dict,
            environment={},
            platform="cpu",
            timestamp=0,
        )

        validator = StabilityValidator(cv_threshold=0.10)
        report = validator.validate(comparative)

        assert isinstance(report, StabilityReport)
        assert report.total_results == 2
        assert report.stable_count == 1

    def test_handles_single_result(self):
        """Validator must handle a single result without errors."""
        from benchmarks.analysis.stability import StabilityValidator
        from benchmarks.runners.full_runner import ComparativeResults

        results_dict = {
            "Datarax": [make_result(cv=0.02, scenario_id="CV-1")],
        }
        comparative = ComparativeResults(
            results=results_dict,
            environment={},
            platform="cpu",
            timestamp=0,
        )

        validator = StabilityValidator()
        report = validator.validate(comparative)
        assert report.total_results == 1
