"""Performance regression detection system for Datarax.

This module provides tools for detecting performance regressions by comparing
current benchmark results with historical baselines and statistical analysis.
"""

import json
import statistics
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from datarax.benchmarking.profiler import ProfileResult


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
        baseline_value: Historical baseline value
        current_value: Current measured value
        regression_percent: Percentage regression (positive = worse performance)
        severity: Severity level of the regression
        description: Human-readable description
        timestamp: When the regression was detected
        metadata: Additional regression metadata
    """

    metric_name: str
    baseline_value: float
    current_value: float
    regression_percent: float
    severity: RegressionSeverity
    description: str
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
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class RegressionReport:
    """Comprehensive regression analysis report.

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
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class RegressionDetector:
    """Automated performance regression detection system."""

    def __init__(
        self,
        baseline_dir: str | Path,
        regression_threshold: float = 0.1,  # 10% regression threshold
        critical_threshold: float = 0.25,  # 25% critical threshold
        min_samples: int = 3,  # Minimum samples for statistical analysis
    ):
        """Initialize regression detector.

        Args:
            baseline_dir: Directory to store and load baseline data
            regression_threshold: Minimum percentage change to consider a regression
            critical_threshold: Percentage change to consider critical
            min_samples: Minimum samples required for statistical analysis
        """
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)

        self.regression_threshold = regression_threshold
        self.critical_threshold = critical_threshold
        self.min_samples = min_samples

        self._baselines: dict[str, list[dict[str, Any]]] = {}
        self._load_baselines()

    def add_baseline(self, profile_result: ProfileResult, benchmark_name: str) -> None:
        """Add a new baseline measurement.

        Args:
            profile_result: Profile result to add as baseline
            benchmark_name: Name of the benchmark
        """
        baseline_data = {
            "timestamp": profile_result.timestamp,
            "timing_metrics": profile_result.timing_metrics,
            "memory_metrics": profile_result.memory_metrics,
            "gpu_metrics": profile_result.gpu_metrics,
            "metadata": profile_result.metadata,
        }

        if benchmark_name not in self._baselines:
            self._baselines[benchmark_name] = []

        self._baselines[benchmark_name].append(baseline_data)

        # Keep only the most recent N samples to avoid unbounded growth
        max_samples = 50
        if len(self._baselines[benchmark_name]) > max_samples:
            self._baselines[benchmark_name] = self._baselines[benchmark_name][-max_samples:]

        self._save_baselines()

    def detect_regressions(
        self,
        current_result: ProfileResult,
        benchmark_name: str,
    ) -> RegressionReport:
        """Detect regressions by comparing current result with baselines.

        Args:
            current_result: Current profile result to analyze
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

        # Analyze timing metrics
        self._analyze_metric_group(
            current_result.timing_metrics,
            [b["timing_metrics"] for b in baselines],
            "timing",
            report,
            higher_is_worse=True,
        )

        # Analyze memory metrics
        self._analyze_metric_group(
            current_result.memory_metrics,
            [b["memory_metrics"] for b in baselines],
            "memory",
            report,
            higher_is_worse=True,
        )

        # Analyze GPU metrics (some might be higher_is_worse=False for utilization)
        self._analyze_metric_group(
            current_result.gpu_metrics,
            [b["gpu_metrics"] for b in baselines],
            "gpu",
            report,
            higher_is_worse=True,
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

    def _analyze_metric_group(
        self,
        current_metrics: dict[str, float],
        baseline_metrics_list: list[dict[str, float]],
        group_name: str,
        report: RegressionReport,
        higher_is_worse: bool = True,
    ) -> None:
        """Analyze a group of metrics for regressions."""
        # Get all metric names that appear in baselines and current
        all_metric_names: set[str] = set()
        for baseline_metrics in baseline_metrics_list:
            all_metric_names.update(baseline_metrics.keys())
        all_metric_names.update(current_metrics.keys())

        for metric_name in all_metric_names:
            full_metric_name = f"{group_name}.{metric_name}"

            # Skip metrics with errors or missing data
            if metric_name == "error" or metric_name not in current_metrics:
                continue

            # Collect baseline values for this metric
            baseline_values: list[float] = []
            for baseline_metrics in baseline_metrics_list:
                if metric_name in baseline_metrics:
                    value = baseline_metrics[metric_name]
                    if isinstance(value, int | float) and not np.isnan(value):
                        baseline_values.append(value)

            if len(baseline_values) < self.min_samples:
                continue

            current_value = current_metrics[metric_name]
            if not isinstance(current_value, int | float) or np.isnan(current_value):
                continue

            # Calculate baseline statistics
            baseline_mean = statistics.mean(baseline_values)
            baseline_std = statistics.stdev(baseline_values) if len(baseline_values) > 1 else 0

            # Calculate percentage change
            if baseline_mean != 0:
                percent_change = (current_value - baseline_mean) / baseline_mean * 100
            else:
                percent_change = 0

            # Determine if this is a regression based on direction
            is_regression = (higher_is_worse and percent_change > 0) or (
                not higher_is_worse and percent_change < 0
            )
            is_improvement = (higher_is_worse and percent_change < 0) or (
                not higher_is_worse and percent_change > 0
            )

            abs_percent_change = abs(percent_change)

            # Check for regressions
            if is_regression and abs_percent_change >= self.regression_threshold * 100:
                severity = self._determine_severity(abs_percent_change)

                regression = PerformanceRegression(
                    metric_name=full_metric_name,
                    baseline_value=baseline_mean,
                    current_value=current_value,
                    regression_percent=abs_percent_change,
                    severity=severity,
                    description=self._generate_regression_description(
                        full_metric_name, baseline_mean, current_value, abs_percent_change
                    ),
                    metadata={
                        "baseline_std": baseline_std,
                        "baseline_samples": len(baseline_values),
                        "baseline_min": min(baseline_values),
                        "baseline_max": max(baseline_values),
                    },
                )
                report.regressions.append(regression)

            # Check for improvements
            elif is_improvement and abs_percent_change >= self.regression_threshold * 100:
                improvement = {
                    "metric_name": full_metric_name,
                    "baseline_value": baseline_mean,
                    "current_value": current_value,
                    "improvement_percent": abs_percent_change,
                    "description": self._generate_improvement_description(
                        full_metric_name, baseline_mean, current_value, abs_percent_change
                    ),
                }
                report.improvements.append(improvement)

            # Stable metrics
            else:
                report.stable_metrics.append(full_metric_name)

    def _determine_severity(self, percent_change: float) -> RegressionSeverity:
        """Determine severity level based on percentage change."""
        if percent_change >= self.critical_threshold * 100:
            return RegressionSeverity.CRITICAL
        elif percent_change >= 20:  # 20%
            return RegressionSeverity.HIGH
        elif percent_change >= 15:  # 15%
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
                with open(baseline_file, "r") as f:
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
            with open(baseline_file, "w") as f:
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
                    "available_metrics": list(baselines[-1].get("timing_metrics", {}).keys()),
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
