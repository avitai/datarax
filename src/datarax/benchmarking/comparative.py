"""Comparative benchmarking tools for Datarax.

Provides BenchmarkComparison for comparing results across frameworks
and configurations.

Refactored to use BenchmarkResult (fixes P2, P7).
Design ref: Section 7 of the benchmark report.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datarax.benchmarking.results import BenchmarkResult, save_json


@dataclass
class BenchmarkComparison:
    """Results of comparing multiple benchmark configurations.

    Attributes:
        configurations: Dictionary mapping config names to results
        best_config: Name of the best-performing configuration
        worst_config: Name of the worst-performing configuration
        metrics_summary: Summary statistics across configurations
        timestamp: When the comparison was performed
    """

    configurations: dict[str, BenchmarkResult] = field(default_factory=dict)
    best_config: str | None = None
    worst_config: str | None = None
    metrics_summary: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def add_result(self, config_name: str, result: BenchmarkResult) -> None:
        """Add a benchmark result for a configuration."""
        self.configurations[config_name] = result
        self._update_summary()

    def _update_summary(self) -> None:
        """Update summary statistics based on throughput."""
        if not self.configurations:
            return

        # Compare by throughput (elements/sec)
        throughputs = {
            name: result.throughput_elements_sec() for name, result in self.configurations.items()
        }

        if throughputs:
            self.best_config = max(throughputs, key=throughputs.get)
            self.worst_config = min(throughputs, key=throughputs.get)

        # Build metrics summary
        latency_data: dict[str, dict[str, float]] = {}
        for config_name, result in self.configurations.items():
            percentiles = result.latency_percentiles()
            for metric, value in percentiles.items():
                if metric not in latency_data:
                    latency_data[metric] = {}
                latency_data[metric][config_name] = value

        self.metrics_summary = {
            "num_configurations": len(self.configurations),
            "throughputs": throughputs,
            "latency_percentiles": latency_data,
        }

    def get_performance_ratio(self) -> dict[str, float]:
        """Get throughput ratios relative to the best configuration."""
        if not self.configurations or self.best_config is None:
            return {}

        best_throughput = self.configurations[self.best_config].throughput_elements_sec()
        if best_throughput <= 0:
            return {}

        return {
            name: result.throughput_elements_sec() / best_throughput
            for name, result in self.configurations.items()
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "configurations": {k: v.to_dict() for k, v in self.configurations.items()},
            "best_config": self.best_config,
            "worst_config": self.worst_config,
            "metrics_summary": self.metrics_summary,
            "timestamp": self.timestamp,
        }

    def save(self, filepath: str | Path) -> None:
        """Save comparison to JSON file."""
        save_json(self.to_dict(), filepath)
