"""Measurement stability validation per Section 8.2.

Checks CV < threshold for all metrics across comparative results.
Flags unstable scenarios and recommends re-run parameters.

Design ref: Section 8.2, 11.4 Task 4.2 of the benchmark report.
"""

from __future__ import annotations

from dataclasses import dataclass

from datarax.benchmarking.statistics import StatisticalAnalyzer

from benchmarks.runners.full_runner import ComparativeResults


@dataclass
class StabilityReport:
    """Result of stability validation.

    Attributes:
        stable_scenarios: List of (scenario_id, adapter) pairs that passed.
        unstable_scenarios: List of (scenario_id, adapter, cv) tuples that failed.
        total_results: Total number of results validated.
        stable_count: Number that passed the CV threshold.
    """

    stable_scenarios: list[tuple[str, str]]
    unstable_scenarios: list[tuple[str, str, float]]
    total_results: int
    stable_count: int


class StabilityValidator:
    """Validates measurement stability across comparative results.

    Checks that the coefficient of variation (CV) of per-batch times
    is below a configurable threshold for each (scenario, adapter) pair.

    Reuses StatisticalAnalyzer.summarize() for CV computation — no
    duplicate statistical logic.

    Args:
        cv_threshold: Maximum acceptable CV. Default 0.10 (10%).
    """

    def __init__(self, cv_threshold: float = 0.10):
        self._threshold = cv_threshold
        self._analyzer = StatisticalAnalyzer()

    def validate(self, results: ComparativeResults) -> StabilityReport:
        """Validate all results in a ComparativeResults set.

        Args:
            results: Comparative benchmark results to validate.

        Returns:
            StabilityReport with stable/unstable classification.
        """
        stable: list[tuple[str, str]] = []
        unstable: list[tuple[str, str, float]] = []

        for adapter_name, adapter_results in results.results.items():
            for r in adapter_results:
                if not r.timing.per_batch_times:
                    continue

                stats = self._analyzer.summarize(r.timing.per_batch_times)

                if stats.cv < self._threshold:
                    stable.append((r.scenario_id, adapter_name))
                else:
                    unstable.append((r.scenario_id, adapter_name, stats.cv))

        total = len(stable) + len(unstable)
        return StabilityReport(
            stable_scenarios=stable,
            unstable_scenarios=unstable,
            total_results=total,
            stable_count=len(stable),
        )

    def recommend_reruns(
        self,
        report: StabilityReport,
    ) -> dict[str, int]:
        """Recommend additional repetitions for unstable scenarios.

        For scenarios with high CV, recommends more repetitions to improve
        the median estimate. Sub-second benchmarks need proportionally more.

        Args:
            report: StabilityReport from validate().

        Returns:
            Dict of "scenario_id/adapter" -> recommended repetitions.
        """
        recommendations: dict[str, int] = {}

        for scenario_id, adapter_name, cv in report.unstable_scenarios:
            key = f"{scenario_id}/{adapter_name}"
            # Higher CV needs more repetitions
            if cv > 0.20:
                recommendations[key] = 15
            elif cv > 0.15:
                recommendations[key] = 10
            else:
                recommendations[key] = 7

        return recommendations
