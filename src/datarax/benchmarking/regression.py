"""Performance regression detection system for Datarax.

This module provides tools for detecting performance regressions by comparing
current benchmark results with historical baselines using statistical analysis.

Note: benchkit (tools/benchkit/src/benchkit/analysis.py) has a parallel
detect_regressions() using its own data model (Run/Point/Metric). That
duplication is intentional — benchkit is zero-dependency by design.

Refactored to use StatisticalAnalyzer and BenchmarkResult (fixes P4).
Design ref: Section 9.2 of the benchmark report.
"""

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from datarax.benchmarking.results import BenchmarkResult, save_json
from datarax.benchmarking.statistics import StatisticalAnalyzer


class RegressionSeverity(Enum):
    """Severity levels for performance regressions."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PerformanceRegression:
    """Represents a detected performance regression.

    Attributes:
        metric_name: Name of the metric that regressed
        baseline_value: Historical baseline mean
        current_value: Current measured value
        regression_percent: Percentage regression (positive = worse performance)
        severity: Severity level of the regression
        description: Human-readable description
        p_value: Statistical significance (from Welch's t-test)
        timestamp: When the regression was detected
        metadata: Additional regression metadata
    """

    metric_name: str
    baseline_value: float
    current_value: float
    regression_percent: float
    severity: RegressionSeverity
    description: str
    p_value: float | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metric_name": self.metric_name,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "regression_percent": self.regression_percent,
            "severity": self.severity.value,
            "description": self.description,
            "p_value": self.p_value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class RegressionReport:
    """Full regression analysis report.

    Attributes:
        regressions: List of detected regressions
        improvements: List of detected improvements
        stable_metrics: List of metrics that remained stable
        summary: Summary statistics of the analysis
        timestamp: When the report was generated
    """

    regressions: list[PerformanceRegression] = field(default_factory=list)
    improvements: list[dict[str, Any]] = field(default_factory=list)
    stable_metrics: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def has_regressions(self) -> bool:
        """Check if any regressions were detected."""
        return len(self.regressions) > 0

    def has_critical_regressions(self) -> bool:
        """Check if any critical regressions were detected."""
        return any(r.severity == RegressionSeverity.CRITICAL for r in self.regressions)

    def get_worst_regression(self) -> PerformanceRegression | None:
        """Get the worst (highest percentage) regression."""
        if not self.regressions:
            return None
        return max(self.regressions, key=lambda r: r.regression_percent)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "regressions": [r.to_dict() for r in self.regressions],
            "improvements": self.improvements,
            "stable_metrics": self.stable_metrics,
            "summary": self.summary,
            "timestamp": self.timestamp.isoformat(),
        }

    def save(self, filepath: str | Path) -> None:
        """Save report to JSON file."""
        save_json(self.to_dict(), filepath)


def _extract_metrics(result: BenchmarkResult) -> dict[str, float]:
    """Extract flat metrics dict from a BenchmarkResult for comparison."""
    metrics: dict[str, float] = {
        "throughput_elements_sec": result.throughput_elements_sec(),
        "wall_clock_sec": result.timing.wall_clock_sec,
        "first_batch_time": result.timing.first_batch_time,
        "num_batches": float(result.timing.num_batches),
    }

    # Resource metrics
    if result.resources is not None:
        metrics["peak_rss_mb"] = result.resources.peak_rss_mb
        metrics["mean_rss_mb"] = result.resources.mean_rss_mb
        metrics["memory_growth_mb"] = result.resources.memory_growth_mb
        if result.resources.peak_gpu_mem_mb is not None:
            metrics["peak_gpu_mem_mb"] = result.resources.peak_gpu_mem_mb

    # Latency percentiles
    percentiles = result.latency_percentiles()
    for k, v in percentiles.items():
        metrics[f"latency_{k}_ms"] = v

    # Extra metrics
    metrics.update(result.extra_metrics)

    return metrics


class RegressionDetector:
    """Automated performance regression detection system.

    Uses StatisticalAnalyzer for significance testing per Section 9.2:
    - Welch's t-test for throughput (warning at p<0.05, failure at p<0.01)
    - Percentage thresholds as fallback for small sample sizes
    """

    # Metrics where higher values mean worse performance
    _HIGHER_IS_WORSE = {
        "wall_clock_sec",
        "first_batch_time",
        "peak_rss_mb",
        "mean_rss_mb",
        "memory_growth_mb",
        "peak_gpu_mem_mb",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
    }

    def __init__(
        self,
        baseline_dir: str | Path,
        regression_threshold: float = 0.1,  # 10% regression threshold
        critical_threshold: float = 0.25,  # 25% critical threshold
        min_samples: int = 3,  # Minimum samples for statistical analysis
        analyzer: StatisticalAnalyzer | None = None,
    ):
        """Initialize regression detector.

        Args:
            baseline_dir: Directory to store and load baseline data
            regression_threshold: Minimum percentage change to consider a regression
            critical_threshold: Percentage change to consider critical
            min_samples: Minimum samples required for statistical analysis
            analyzer: StatisticalAnalyzer instance (created if not provided)
        """
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)

        self.regression_threshold = regression_threshold
        self.critical_threshold = critical_threshold
        self.min_samples = min_samples
        self.analyzer = analyzer or StatisticalAnalyzer()

        self._baselines: dict[str, list[dict[str, Any]]] = {}
        self._load_baselines()

    def add_baseline(self, result: BenchmarkResult, benchmark_name: str) -> None:
        """Add a new baseline measurement.

        Args:
            result: Benchmark result to add as baseline
            benchmark_name: Name of the benchmark
        """
        baseline_data = {
            "timestamp": result.timestamp,
            "metrics": _extract_metrics(result),
        }

        if benchmark_name not in self._baselines:
            self._baselines[benchmark_name] = []

        self._baselines[benchmark_name].append(baseline_data)

        # Keep only the most recent N samples
        max_samples = 50
        if len(self._baselines[benchmark_name]) > max_samples:
            self._baselines[benchmark_name] = self._baselines[benchmark_name][-max_samples:]

        self._save_baselines()

    def detect_regressions(
        self,
        current_result: BenchmarkResult,
        benchmark_name: str,
    ) -> RegressionReport:
        """Detect regressions by comparing current result with baselines.

        Uses Welch's t-test when enough samples are available (Section 9.2).

        Args:
            current_result: Current benchmark result to analyze
            benchmark_name: Name of the benchmark

        Returns:
            Regression report with detected issues
        """
        if benchmark_name not in self._baselines:
            warnings.warn(f"No baseline data found for benchmark '{benchmark_name}'")
            return RegressionReport(
                summary={
                    "status": "no_baseline",
                    "message": f"No baseline data for {benchmark_name}",
                }
            )

        baselines = self._baselines[benchmark_name]
        if len(baselines) < self.min_samples:
            warnings.warn(f"Insufficient baseline samples ({len(baselines)} < {self.min_samples})")
            return RegressionReport(
                summary={
                    "status": "insufficient_data",
                    "samples": len(baselines),
                }
            )

        report = RegressionReport()
        current_metrics = _extract_metrics(current_result)
        baseline_metrics_list = [b["metrics"] for b in baselines]

        # Analyze each metric
        all_metric_names: set[str] = set()
        for bm in baseline_metrics_list:
            all_metric_names.update(bm.keys())
        all_metric_names.update(current_metrics.keys())

        for metric_name in sorted(all_metric_names):
            if metric_name not in current_metrics:
                continue

            # Collect baseline values
            baseline_values = [
                bm[metric_name]
                for bm in baseline_metrics_list
                if metric_name in bm and isinstance(bm[metric_name], int | float)
            ]

            if len(baseline_values) < self.min_samples:
                continue

            current_value = current_metrics[metric_name]
            if not isinstance(current_value, int | float):
                continue

            higher_is_worse = metric_name in self._HIGHER_IS_WORSE
            self._analyze_metric(
                metric_name,
                current_value,
                baseline_values,
                higher_is_worse,
                report,
            )

        # Generate summary
        report.summary = {
            "total_regressions": len(report.regressions),
            "critical_regressions": len(
                [r for r in report.regressions if r.severity == RegressionSeverity.CRITICAL]
            ),
            "total_improvements": len(report.improvements),
            "stable_metrics": len(report.stable_metrics),
            "benchmark_name": benchmark_name,
            "baseline_samples": len(baselines),
        }

        return report

    def _analyze_metric(
        self,
        metric_name: str,
        current_value: float,
        baseline_values: list[float],
        higher_is_worse: bool,
        report: RegressionReport,
    ) -> None:
        """Analyze a single metric for regression using statistical testing."""
        baseline_stats = self.analyzer.summarize(baseline_values)
        baseline_mean = baseline_stats.mean

        # Calculate percentage change
        if baseline_mean != 0:
            percent_change = (current_value - baseline_mean) / baseline_mean * 100
        else:
            percent_change = 0

        # Determine direction
        is_regression = (higher_is_worse and percent_change > 0) or (
            not higher_is_worse and percent_change < 0
        )
        is_improvement = (higher_is_worse and percent_change < 0) or (
            not higher_is_worse and percent_change > 0
        )

        abs_percent_change = abs(percent_change)

        # Attempt Welch's t-test if enough baseline samples
        p_value = None
        if len(baseline_values) >= 5:
            # Use current value as a "sample" — compare baseline distribution
            # against a point estimate. For a proper test, we'd need multiple
            # current measurements. Use 1-sample approximation.
            try:
                _, p_value = self.analyzer.welch_t_test(baseline_values, [current_value] * 3)
            except Exception:
                p_value = None

        # Determine if this is a significant regression
        if is_regression and abs_percent_change >= self.regression_threshold * 100:
            severity = self._determine_severity(abs_percent_change, p_value)

            regression = PerformanceRegression(
                metric_name=metric_name,
                baseline_value=baseline_mean,
                current_value=current_value,
                regression_percent=abs_percent_change,
                severity=severity,
                description=self._generate_regression_description(
                    metric_name, baseline_mean, current_value, abs_percent_change
                ),
                p_value=p_value,
                metadata={
                    "baseline_std": baseline_stats.std,
                    "baseline_cv": baseline_stats.cv,
                    "baseline_samples": baseline_stats.n,
                    "baseline_is_stable": baseline_stats.is_stable,
                },
            )
            report.regressions.append(regression)

        elif is_improvement and abs_percent_change >= self.regression_threshold * 100:
            improvement = {
                "metric_name": metric_name,
                "baseline_value": baseline_mean,
                "current_value": current_value,
                "improvement_percent": abs_percent_change,
                "p_value": p_value,
                "description": self._generate_improvement_description(
                    metric_name, baseline_mean, current_value, abs_percent_change
                ),
            }
            report.improvements.append(improvement)

        else:
            report.stable_metrics.append(metric_name)

    def _determine_severity(
        self, percent_change: float, p_value: float | None
    ) -> RegressionSeverity:
        """Determine severity based on percentage change and statistical significance.

        Per Section 9.2: p<0.01 → failure, p<0.05 → warning.
        """
        # If we have statistical significance, use it to escalate
        if p_value is not None and p_value < 0.01:
            if percent_change >= self.critical_threshold * 100:
                return RegressionSeverity.CRITICAL
            return RegressionSeverity.HIGH

        if percent_change >= self.critical_threshold * 100:
            return RegressionSeverity.CRITICAL
        elif percent_change >= 20:
            return RegressionSeverity.HIGH
        elif percent_change >= 15:
            return RegressionSeverity.MEDIUM
        else:
            return RegressionSeverity.LOW

    def _generate_regression_description(
        self, metric_name: str, baseline: float, current: float, percent: float
    ) -> str:
        """Generate human-readable regression description."""
        return f"{metric_name} regressed by {percent:.1f}% (from {baseline:.3f} to {current:.3f})"

    def _generate_improvement_description(
        self, metric_name: str, baseline: float, current: float, percent: float
    ) -> str:
        """Generate human-readable improvement description."""
        return f"{metric_name} improved by {percent:.1f}% (from {baseline:.3f} to {current:.3f})"

    def _load_baselines(self) -> None:
        """Load baseline data from disk."""
        baseline_file = self.baseline_dir / "baselines.json"
        if baseline_file.exists():
            try:
                with baseline_file.open("r") as f:
                    self._baselines = json.load(f)
            except Exception as e:
                warnings.warn(f"Could not load baselines: {e}")
                self._baselines = {}
        else:
            self._baselines = {}

    def _save_baselines(self) -> None:
        """Save baseline data to disk."""
        baseline_file = self.baseline_dir / "baselines.json"
        try:
            with baseline_file.open("w") as f:
                json.dump(self._baselines, f, indent=2, default=str)
        except Exception as e:
            warnings.warn(f"Could not save baselines: {e}")

    def get_baseline_summary(self) -> dict[str, Any]:
        """Get summary of available baseline data."""
        summary: dict[str, Any] = {}
        for benchmark_name, baselines in self._baselines.items():
            if baselines:
                timestamps = [b["timestamp"] for b in baselines]
                summary[benchmark_name] = {
                    "num_samples": len(baselines),
                    "oldest_sample": min(timestamps),
                    "newest_sample": max(timestamps),
                    "available_metrics": list(baselines[-1].get("metrics", {}).keys()),
                }
        return summary

    def clear_baselines(self, benchmark_name: str | None = None) -> None:
        """Clear baseline data.

        Args:
            benchmark_name: Specific benchmark to clear, or None to clear all
        """
        if benchmark_name:
            if benchmark_name in self._baselines:
                del self._baselines[benchmark_name]
        else:
            self._baselines.clear()

        self._save_baselines()
