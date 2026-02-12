"""Framework-agnostic benchmark result container.

Replaces ProfileResult from profiler.py with a cleaner, typed design.
Fixes P1 (god-class decomposition) by extracting result storage.

Design ref: Section 6.2.3 of the benchmark report.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from datarax.benchmarking.resource_monitor import ResourceSummary
from datarax.benchmarking.timing import TimingSample


def save_json(data: dict[str, Any], filepath: str | Path) -> None:
    """Save dict to JSON file, creating parent directories as needed.

    Shared utility for BenchmarkResult, RegressionReport, and BenchmarkComparison.
    Uses ``default=str`` for safe serialization of datetime and other types.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(json.dumps(data, indent=2, default=str))


@dataclass
class BenchmarkResult:
    """Framework-agnostic benchmark result.

    Combines timing measurements, resource usage, and metadata into a single
    serializable container suitable for cross-framework comparison.

    Attributes:
        framework: Name of the framework (e.g., "Datarax", "Grain").
        scenario_id: Benchmark scenario identifier (e.g., "CV-1").
        variant: Size variant (e.g., "small", "medium", "large").
        timing: Timing measurements from TimingCollector.
        resources: Resource usage from ResourceMonitor (None if not collected).
        environment: System environment dict from capture_environment().
        config: Scenario configuration as a dict.
        timestamp: Unix timestamp of the benchmark run.
        extra_metrics: Additional framework-specific metrics.
    """

    framework: str
    scenario_id: str
    variant: str
    timing: TimingSample
    resources: ResourceSummary | None
    environment: dict[str, Any]
    config: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    extra_metrics: dict[str, float] = field(default_factory=dict)

    def throughput_elements_sec(self) -> float:
        """Primary comparison metric: elements processed per second."""
        if self.timing.wall_clock_sec > 0:
            return self.timing.num_elements / self.timing.wall_clock_sec
        return 0.0

    def latency_percentiles(self) -> dict[str, float]:
        """Compute p50, p95, p99 inter-batch latency in milliseconds."""
        if not self.timing.per_batch_times:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        arr = np.array(self.timing.per_batch_times) * 1000  # sec -> ms
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return asdict(self)

    def save(self, filepath: str | Path) -> None:
        """Save result to a JSON file."""
        save_json(self.to_dict(), filepath)

    @classmethod
    def load(cls, filepath: Path) -> "BenchmarkResult":
        """Load a BenchmarkResult from a JSON file."""
        data = json.loads(filepath.read_text())
        data["timing"] = TimingSample(**data["timing"])
        if data["resources"] is not None:
            data["resources"] = ResourceSummary(**data["resources"])
        return cls(**data)
