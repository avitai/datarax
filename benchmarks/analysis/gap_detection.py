"""Performance gap detection and optimization target mapping.

Identifies scenarios where other frameworks lead Datarax and maps
each gap to a prioritized optimization target.

Design ref: Section 12.6, 11.4 Task 4.5 of the benchmark report.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from benchmarks.runners.full_runner import ComparativeResults


@dataclass
class PerformanceGap:
    """A detected performance gap between Datarax and another framework.

    Attributes:
        scenario_id: Benchmark scenario ID.
        datarax_throughput: Datarax throughput in elem/s.
        top_alternative: Name of the leading alternative framework.
        alternative_throughput: Alternative framework throughput in elem/s.
        gap_ratio: alternative / datarax ratio (>1 means alternative leads).
        priority: P0-P5 optimization priority.
        severity: "warning" (>1.5x) or "action_required" (>2x).
        mitigation: Suggested optimization approach.
    """

    scenario_id: str
    datarax_throughput: float
    top_alternative: str
    alternative_throughput: float
    gap_ratio: float
    priority: str
    severity: str
    mitigation: str


# Priority mapping from Section 12.6.3
_PRIORITY_MAP: dict[str, str] = {
    "CV-1": "P0",
    "PC-1": "P1",
    "CV-2": "P2",
    "NLP-1": "P3",
    "PR-2": "P4",
    "PR-1": "P5",
}

# Mitigation suggestions per scenario
_MITIGATION_MAP: dict[str, str] = {
    "CV-1": "Thread-based I/O pipeline (vs SPDL pattern)",
    "PC-1": "JIT transform fusion depth optimization",
    "CV-2": "GPU-accelerated augmentation (vs DALI kernels)",
    "NLP-1": "Memory-efficient tokenization pipeline",
    "PR-2": "Deterministic multi-worker data loading",
    "PR-1": "Checkpoint serialization speed optimization",
}


class GapDetector:
    """Identifies performance gaps and maps to optimization targets.

    Per Section 12.6.2: flags gaps > warning_threshold (default 1.5x)
    as "warning" and > action_threshold (default 2.0x) as "action_required".

    Args:
        results: ComparativeResults from a full comparative run.
        warning_threshold: Ratio threshold for warning (default 1.5).
        action_threshold: Ratio threshold for action_required (default 2.0).
        datarax_name: Name of the Datarax adapter.
    """

    def __init__(
        self,
        results: ComparativeResults,
        warning_threshold: float = 1.5,
        action_threshold: float = 2.0,
        datarax_name: str = "Datarax",
    ) -> None:
        self._results = results
        self._warning = warning_threshold
        self._action = action_threshold
        self._datarax_name = datarax_name

    def detect(self) -> list[PerformanceGap]:
        """Scan all scenarios for performance gaps.

        Returns:
            List of PerformanceGap instances, sorted by gap_ratio descending.
        """
        gaps: list[PerformanceGap] = []

        for scenario_id in self._results.all_scenario_ids:
            datarax_tp = self._get_datarax_throughput(scenario_id)
            if datarax_tp is None or datarax_tp == 0:
                continue

            best_name, best_tp = self._top_alternative(scenario_id)
            if best_tp == 0:
                continue  # No other frameworks for this scenario

            ratio = best_tp / datarax_tp
            if ratio < self._warning:
                continue  # Within acceptable range

            severity = "action_required" if ratio >= self._action else "warning"
            priority = _PRIORITY_MAP.get(scenario_id, "P5")
            mitigation = _MITIGATION_MAP.get(
                scenario_id,
                "Profile and optimize hot path",
            )

            gaps.append(
                PerformanceGap(
                    scenario_id=scenario_id,
                    datarax_throughput=datarax_tp,
                    top_alternative=best_name,
                    alternative_throughput=best_tp,
                    gap_ratio=ratio,
                    priority=priority,
                    severity=severity,
                    mitigation=mitigation,
                )
            )

        return sorted(gaps, key=lambda g: g.gap_ratio, reverse=True)

    def generate_backlog(self, output_path: Path) -> None:
        """Generate an optimization backlog markdown file.

        Args:
            output_path: Path to write the markdown file.
        """
        gaps = self.detect()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# Performance Optimization Backlog\n",
            "Auto-generated from benchmark results. Sorted by gap severity.\n",
        ]

        if not gaps:
            lines.append(
                "No significant performance gaps detected. "
                "Datarax is comparable across all tested scenarios."
            )
        else:
            lines.append("| Priority | Scenario | Gap | Framework | Severity | Mitigation |")
            lines.append("|----------|----------|-----|------------|----------|------------|")

            for gap in gaps:
                lines.append(
                    f"| {gap.priority} | {gap.scenario_id} | "
                    f"{gap.gap_ratio:.1f}x | {gap.top_alternative} | "
                    f"{gap.severity} | {gap.mitigation} |"
                )

            lines.append("")
            lines.append("## Details\n")
            for gap in gaps:
                lines.append(f"### {gap.scenario_id} ({gap.priority})\n")
                lines.append(f"- **Datarax:** {gap.datarax_throughput:.0f} elem/s")
                lines.append(
                    f"- **{gap.top_alternative}:** {gap.alternative_throughput:.0f} elem/s"
                )
                lines.append(f"- **Gap:** {gap.gap_ratio:.1f}x")
                lines.append(f"- **Mitigation:** {gap.mitigation}\n")

        output_path.write_text("\n".join(lines))

    def _get_datarax_throughput(self, scenario_id: str) -> float | None:
        for r in self._results.results.get(self._datarax_name, []):
            if r.scenario_id == scenario_id:
                return r.throughput_elements_sec()
        return None

    def _top_alternative(self, scenario_id: str) -> tuple[str, float]:
        """Find the best-performing alternative framework for a scenario."""
        best_name = ""
        best_tp = 0.0
        for adapter_name, adapter_results in self._results.results.items():
            if adapter_name == self._datarax_name:
                continue
            for r in adapter_results:
                if r.scenario_id == scenario_id:
                    tp = r.throughput_elements_sec()
                    if tp > best_tp:
                        best_tp = tp
                        best_name = adapter_name
        return best_name, best_tp
