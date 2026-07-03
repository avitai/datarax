"""Baseline storage and comparison for benchmark regression detection.

Manages JSON baselines on disk and uses StatisticalAnalyzer to compare
current results against stored baselines.

Design ref: Sections 9.1, 9.2 of the benchmark report.
Thresholds: Welch's t-test p<0.05 → warning, p<0.01 → failure.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from calibrax.core import BenchmarkResult
from calibrax.statistics import StatisticalAnalyzer, welch_t_test

from benchmarks.core.result_model import throughput_elements_per_sec


_REQUIRED_SCHEMA_KEYS = frozenset({"name", "metrics", "timing"})
_LEGACY_SCHEMA_KEYS = frozenset({"framework", "scenario_id", "variant"})


class BaselineSchemaError(ValueError):
    """A baseline file does not match the calibrax BenchmarkResult schema."""


def _validate_schema(name: str, data: dict[str, Any]) -> None:
    """Reject baseline files that predate the calibrax result schema.

    Args:
        name: Baseline name, used in the error message.
        data: Parsed baseline JSON.

    Raises:
        BaselineSchemaError: If required result keys are missing; the message
            flags legacy pre-calibrax files explicitly.
    """
    missing = _REQUIRED_SCHEMA_KEYS - data.keys()
    if not missing:
        return
    legacy_hint = " (legacy pre-calibrax schema)" if _LEGACY_SCHEMA_KEYS & data.keys() else ""
    raise BaselineSchemaError(
        f"Baseline '{name}' is missing required keys {sorted(missing)}{legacy_hint}; "
        f"regenerate it with --generate-baselines --force."
    )


class BaselineStore:
    """Manages benchmark baselines on disk.

    Args:
        baselines_dir: Directory for storing baseline JSON files.
    """

    def __init__(self, baselines_dir: Path | str):
        """Initialize the baseline store with the given directory."""
        self.baselines_dir = Path(baselines_dir)
        self.baselines_dir.mkdir(parents=True, exist_ok=True)
        self._analyzer = StatisticalAnalyzer()

    def save(self, name: str, result: BenchmarkResult) -> Path:
        """Save a BenchmarkResult as a baseline.

        Args:
            name: Baseline name (e.g., 'CV-1_small').
            result: BenchmarkResult to store.

        Returns:
            Path to the saved JSON file.
        """
        filepath = self.baselines_dir / f"{name}.json"
        result.save(filepath)
        # calibrax writes JSON without a trailing newline; baselines are
        # committed files, so normalize to POSIX text.
        text = filepath.read_text()
        if not text.endswith("\n"):
            filepath.write_text(text + "\n")
        return filepath

    def load(self, name: str) -> dict[str, Any] | None:
        """Load a baseline by name.

        Args:
            name: Baseline name.

        Returns:
            Baseline data as dict, or None if not found.

        Raises:
            BaselineSchemaError: If the file exists but does not match the
                calibrax result schema (e.g., a legacy pre-calibrax baseline).
        """
        filepath = self.baselines_dir / f"{name}.json"
        if not filepath.exists():
            return None
        with filepath.open() as f:
            data = json.load(f)
        _validate_schema(name, data)
        return data

    def compare(
        self,
        name: str,
        current: BenchmarkResult,
        failure_ratio: float = 0.80,
        warning_ratio: float = 0.90,
    ) -> dict[str, Any] | None:
        """Compare current result against a stored baseline.

        Requires both **statistical** and **practical** significance to flag
        a regression. The Welch's t-test determines if the difference is real
        (not noise), while the throughput ratio gates determine if the
        difference is large enough to matter.

        This dual-gate prevents false positives from microbenchmarks where
        sub-millisecond per-batch times make the t-test overly sensitive to
        normal system noise (scheduling jitter, GPU thermals, etc.).

        Thresholds (per Section 9.2):
          - failure: p<0.01 AND throughput_ratio < failure_ratio (default 0.80)
          - warning: p<0.05 AND throughput_ratio < warning_ratio (default 0.90)

        Args:
            name: Baseline name to compare against.
            current: Current BenchmarkResult.
            failure_ratio: Minimum throughput drop to trigger failure (default 0.80 = 20%).
            warning_ratio: Minimum throughput drop to trigger warning (default 0.90 = 10%).

        Returns:
            Verdict dict with 'status' ('pass', 'warning', 'failure'),
            throughput metrics, and statistical details. None if baseline
            doesn't exist.
        """
        baseline_data = self.load(name)
        if baseline_data is None:
            return None

        # Extract throughput values
        baseline_timing = baseline_data["timing"]
        baseline_throughput = (
            baseline_timing["num_elements"] / baseline_timing["wall_clock_sec"]
            if baseline_timing["wall_clock_sec"] > 0
            else 0.0
        )
        current_throughput = throughput_elements_per_sec(current)

        throughput_ratio = (
            current_throughput / baseline_throughput if baseline_throughput > 0 else 0.0
        )

        # Compare per-batch times using Welch's t-test
        baseline_batch_times = baseline_timing.get("per_batch_times", [])
        assert current.timing is not None
        current_batch_times = current.timing.per_batch_times

        status = "pass"
        p_value = None
        t_stat = None

        if len(baseline_batch_times) >= 2 and len(current_batch_times) >= 2:
            t_stat, p_value = welch_t_test(baseline_batch_times, current_batch_times)

            # Only flag if current is slower (higher batch times)
            current_mean = sum(current_batch_times) / len(current_batch_times)
            baseline_mean = sum(baseline_batch_times) / len(baseline_batch_times)

            # Dual gate: require BOTH statistical AND practical significance
            if current_mean > baseline_mean:
                if p_value < 0.01 and throughput_ratio < failure_ratio:
                    status = "failure"
                elif p_value < 0.05 and throughput_ratio < warning_ratio:
                    status = "warning"
        else:
            # Fallback: simple ratio check when not enough samples
            if throughput_ratio < failure_ratio:
                status = "failure"
            elif throughput_ratio < warning_ratio:
                status = "warning"

        return {
            "status": status,
            "throughput_ratio": throughput_ratio,
            "baseline_throughput": baseline_throughput,
            "current_throughput": current_throughput,
            "p_value": p_value,
            "t_statistic": t_stat,
        }

    def archive(self, name: str) -> None:
        """Move a baseline to an archive subdirectory.

        Args:
            name: Baseline name to archive.
        """
        filepath = self.baselines_dir / f"{name}.json"
        if not filepath.exists():
            return

        archive_dir = self.baselines_dir / "archive"
        archive_dir.mkdir(exist_ok=True)

        timestamp = int(time.time())
        archive_path = archive_dir / f"{name}_{timestamp}.json"
        shutil.move(str(filepath), str(archive_path))

    def list_baselines(self) -> list[str]:
        """List all available baseline names.

        Returns:
            List of baseline names (without .json extension).
        """
        return sorted(p.stem for p in self.baselines_dir.glob("*.json") if p.is_file())
