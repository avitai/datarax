"""Comparative analysis report generator.

Generates a markdown narrative comparing Datarax performance with
other frameworks, organized into strengths, parity, and gaps sections.

Structure mirrors Section 12 of the benchmark report:
- Strengths (Datarax leads)
- Parity (comparable performance)
- Gaps (optimization opportunities)
- Positioning summary

Design ref: Section 12, 11.4 Task 4.4 of the benchmark report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from benchmarks.runners.full_runner import ComparativeResults


class ComparisonReportGenerator:
    """Generates comparative analysis markdown from benchmark results.

    All numbers come from actual benchmark results — no placeholders.

    Args:
        results: ComparativeResults from a full comparative run.
        datarax_name: Name of the Datarax adapter in results.
    """

    def __init__(
        self,
        results: ComparativeResults,
        datarax_name: str = "Datarax",
    ) -> None:
        self._results = results
        self._datarax_name = datarax_name

    def generate(self, chart_dir: Path | str | None = None) -> str:
        """Generate the full comparative analysis report.

        Args:
            chart_dir: Optional path to chart directory for image references.

        Returns:
            Markdown string with the comparative analysis.
        """
        sections = [
            self._header(),
            self._strengths_section(),
            self._parity_section(),
            self._gaps_section(),
            self._positioning_summary(),
        ]

        if chart_dir is not None:
            sections.append(self._chart_references(str(chart_dir)))

        return "\n\n".join(sections)

    def _header(self) -> str:
        scenarios = self._results.all_scenario_ids
        adapters = list(self._results.results.keys())
        return (
            "# Comparative Benchmark Analysis\n\n"
            f"**Platform:** {self._results.platform} | "
            f"**Adapters:** {len(adapters)} | "
            f"**Scenarios:** {len(scenarios)}"
        )

    def _classify_scenario(
        self,
        scenario_id: str,
    ) -> Literal["strength", "parity", "gap"]:
        """Classify based on throughput ratio.

        - strength: Datarax > 1.2x top alternative (or exclusive)
        - parity: within 0.8x-1.2x
        - gap: Datarax < 0.8x top alternative
        """
        datarax_tp = self._datarax_throughput(scenario_id)
        if datarax_tp is None:
            return "parity"

        best_name, best_tp = self._top_alternative(scenario_id)
        if best_tp == 0:
            return "strength"  # No other frameworks

        ratio = datarax_tp / best_tp
        if ratio > 1.2:
            return "strength"
        elif ratio < 0.8:
            return "gap"
        return "parity"

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

    def _datarax_throughput(self, scenario_id: str) -> float | None:
        """Get Datarax throughput for a scenario."""
        for r in self._results.results.get(self._datarax_name, []):
            if r.scenario_id == scenario_id:
                return r.throughput_elements_sec()
        return None

    def _strengths_section(self) -> str:
        lines = ["## Strengths\n", "Scenarios where Datarax leads:\n"]
        found = False

        for sid in sorted(self._results.all_scenario_ids):
            if self._classify_scenario(sid) == "strength":
                found = True
                datarax_tp = self._datarax_throughput(sid)
                best_name, best_tp = self._top_alternative(sid)
                if best_tp > 0 and datarax_tp:
                    ratio = datarax_tp / best_tp
                    lines.append(
                        f"- **{sid}**: {datarax_tp:.0f} elem/s "
                        f"({ratio:.1f}x vs {best_name} at {best_tp:.0f} elem/s)"
                    )
                elif datarax_tp:
                    lines.append(
                        f"- **{sid}**: {datarax_tp:.0f} elem/s "
                        f"(Datarax exclusive — no other framework support)"
                    )

        if not found:
            lines.append("No clear strengths identified in this benchmark run.")

        return "\n".join(lines)

    def _parity_section(self) -> str:
        lines = [
            "## Comparable Performance\n",
            "Scenarios where Datarax is comparable (within 0.8x-1.2x):\n",
        ]
        found = False

        for sid in sorted(self._results.all_scenario_ids):
            if self._classify_scenario(sid) == "parity":
                found = True
                datarax_tp = self._datarax_throughput(sid)
                best_name, best_tp = self._top_alternative(sid)
                if best_tp > 0 and datarax_tp:
                    ratio = datarax_tp / best_tp
                    lines.append(
                        f"- **{sid}**: {datarax_tp:.0f} elem/s "
                        f"({ratio:.2f}x vs {best_name} at {best_tp:.0f} elem/s)"
                    )

        if not found:
            lines.append("No parity scenarios in this benchmark run.")

        return "\n".join(lines)

    def _gaps_section(self) -> str:
        lines = [
            "## Gaps and Optimization Opportunities\n",
            "Scenarios where other frameworks lead by >1.2x:\n",
        ]
        found = False

        for sid in sorted(self._results.all_scenario_ids):
            if self._classify_scenario(sid) == "gap":
                found = True
                datarax_tp = self._datarax_throughput(sid)
                best_name, best_tp = self._top_alternative(sid)
                if best_tp > 0 and datarax_tp:
                    gap_ratio = best_tp / datarax_tp
                    lines.append(
                        f"- **{sid}**: {datarax_tp:.0f} elem/s "
                        f"({gap_ratio:.1f}x below {best_name} "
                        f"at {best_tp:.0f} elem/s)"
                    )

        if not found:
            lines.append("No significant gaps identified.")

        return "\n".join(lines)

    def _positioning_summary(self) -> str:
        strengths = 0
        parity = 0
        gaps = 0

        for sid in self._results.all_scenario_ids:
            cat = self._classify_scenario(sid)
            if cat == "strength":
                strengths += 1
            elif cat == "parity":
                parity += 1
            else:
                gaps += 1

        total = strengths + parity + gaps
        return (
            (
                "## Positioning Summary\n\n"
                f"| Category | Count | Percentage |\n"
                f"|----------|------:|:-----------|\n"
                f"| Strengths | {strengths} | "
                f"{strengths / total * 100:.0f}% |\n"
                f"| Parity | {parity} | "
                f"{parity / total * 100:.0f}% |\n"
                f"| Gaps | {gaps} | "
                f"{gaps / total * 100:.0f}% |\n"
                f"| **Total** | **{total}** | **100%** |"
            )
            if total > 0
            else "## Positioning Summary\n\nNo scenarios to summarize."
        )

    def _chart_references(self, chart_dir: str) -> str:
        return (
            "## Visualization\n\n"
            f"See charts in `{chart_dir}/`:\n\n"
            f"- `throughput_bars.png` — Throughput comparison\n"
            f"- `throughput_radar.png` — Radar comparison\n"
            f"- `latency_cdf.png` — Latency distributions\n"
            f"- `memory_waterfall.png` — Memory usage\n"
            f"- `scaling_curves.png` — Scaling behavior\n"
            f"- `chain_depth.png` — Chain depth degradation\n"
            f"- `feature_heatmap.png` — Feature support matrix"
        )
