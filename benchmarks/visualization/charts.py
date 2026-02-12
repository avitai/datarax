"""Benchmark visualization: 7 standardized chart types.

Generates publication-ready charts from comparative benchmark results.
Each framework gets a consistent color across all charts.

Design ref: Section 10.2 of the benchmark report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from benchmarks.runners.full_runner import ComparativeResults

# Use non-interactive backend for CI/headless environments
matplotlib.use("Agg")

# Consistent framework color palette (Datarax = indigo, others = palette)
_FRAMEWORK_COLORS: dict[str, str] = {
    "Datarax": "#3F51B5",  # Indigo
    "Grain": "#4CAF50",  # Green
    "PyTorch DataLoader": "#F44336",  # Red
    "tf.data": "#FF9800",  # Orange
    "DALI": "#9C27B0",  # Purple
    "SPDL": "#00BCD4",  # Cyan
    "FFCV": "#795548",  # Brown
    "MosaicML": "#607D8B",  # Blue Grey
    "WebDataset": "#E91E63",  # Pink
    "HF Datasets": "#FFEB3B",  # Yellow
    "Ray Data": "#8BC34A",  # Light Green
    "LitData": "#03A9F4",  # Light Blue
    "Energon": "#FF5722",  # Deep Orange
    "Deep Lake": "#673AB7",  # Deep Purple
    "JAX DataLoader": "#009688",  # Teal
}

_DEFAULT_COLOR = "#9E9E9E"  # Grey fallback

# Chart names for generate_all
_CHART_NAMES = [
    "throughput_bars",
    "throughput_radar",
    "latency_cdf",
    "memory_waterfall",
    "scaling_curves",
    "chain_depth",
    "feature_heatmap",
]


def _get_color(framework: str) -> str:
    return _FRAMEWORK_COLORS.get(framework, _DEFAULT_COLOR)


class ChartGenerator:
    """Generates 7 standardized benchmark visualization charts.

    Args:
        results: ComparativeResults from a full comparative run.
        output_dir: Directory to save chart files.
    """

    def __init__(self, results: ComparativeResults, output_dir: Path) -> None:
        self._results = results
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(
        self,
        formats: tuple[str, ...] = ("png", "svg"),
    ) -> list[Path]:
        """Generate all 7 chart types in the specified formats.

        Returns:
            List of paths to generated chart files.
        """
        chart_methods = [
            ("throughput_bars", self.throughput_bars),
            ("throughput_radar", self.throughput_radar),
            ("latency_cdf", self.latency_cdf),
            ("memory_waterfall", self.memory_waterfall),
            ("scaling_curves", self.scaling_curves),
            ("chain_depth", self.chain_depth),
            ("feature_heatmap", self.feature_heatmap),
        ]

        paths: list[Path] = []
        for name, method in chart_methods:
            fig = method()
            for fmt in formats:
                path = self._output_dir / f"{name}.{fmt}"
                fig.savefig(path, dpi=150, bbox_inches="tight")
                paths.append(path)
            plt.close(fig)

        return paths

    def throughput_bars(self) -> Figure:
        """Grouped bar chart: throughput per scenario per framework."""
        scenario_ids = sorted(self._results.all_scenario_ids)
        adapters = sorted(self._results.results.keys())

        n_scenarios = len(scenario_ids)
        n_adapters = len(adapters)

        fig, ax = plt.subplots(figsize=(max(10, n_scenarios * 1.5), 6))

        bar_width = 0.8 / max(n_adapters, 1)
        x = np.arange(n_scenarios)

        for i, adapter in enumerate(adapters):
            throughputs = []
            for sid in scenario_ids:
                scenario_results = self._results.get_scenario_results(sid)
                r = scenario_results.get(adapter)
                throughputs.append(r.throughput_elements_sec() if r else 0)

            offset = (i - n_adapters / 2 + 0.5) * bar_width
            ax.bar(
                x + offset,
                throughputs,
                bar_width,
                label=adapter,
                color=_get_color(adapter),
            )

        ax.set_xlabel("Scenario")
        ax.set_ylabel("Throughput (elem/s)")
        ax.set_title("Benchmark Throughput Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_ids, rotation=45, ha="right")
        ax.legend(loc="upper right", fontsize="small")
        fig.tight_layout()
        return fig

    def throughput_radar(self, top_n: int = 3) -> Figure:
        """Radar/spider chart: Datarax vs top N frameworks."""
        scenario_ids = sorted(self._results.all_scenario_ids)
        if not scenario_ids:
            return self._empty_figure("No scenarios for radar chart")

        # Find top N frameworks by average throughput
        adapter_avg: dict[str, float] = {}
        for adapter_name, adapter_results in self._results.results.items():
            if adapter_name == "Datarax":
                continue
            tps = [r.throughput_elements_sec() for r in adapter_results]
            adapter_avg[adapter_name] = sum(tps) / len(tps) if tps else 0

        top_adapters = sorted(
            adapter_avg,
            key=adapter_avg.get,
            reverse=True,
        )[:top_n]
        adapters = ["Datarax", *top_adapters]

        angles = np.linspace(0, 2 * np.pi, len(scenario_ids), endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])  # Close the polygon

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})

        for adapter in adapters:
            values = []
            for sid in scenario_ids:
                scenario_r = self._results.get_scenario_results(sid)
                r = scenario_r.get(adapter)
                values.append(r.throughput_elements_sec() if r else 0)
            values.append(values[0])  # Close polygon

            ax.plot(
                angles,
                values,
                "o-",
                label=adapter,
                color=_get_color(adapter),
                linewidth=2,
            )
            ax.fill(angles, values, alpha=0.1, color=_get_color(adapter))

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(scenario_ids, fontsize=9)
        ax.set_title("Throughput Radar", y=1.08)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize="small")
        fig.tight_layout()
        return fig

    def latency_cdf(self, scenario_id: str = "CV-1") -> Figure:
        """CDF of per-batch latencies for a specific scenario."""
        fig, ax = plt.subplots(figsize=(10, 6))

        scenario_results = self._results.get_scenario_results(scenario_id)
        for adapter_name, result in sorted(scenario_results.items()):
            if not result.timing.per_batch_times:
                continue
            times_ms = np.array(result.timing.per_batch_times) * 1000
            sorted_times = np.sort(times_ms)
            cdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)

            ax.step(
                sorted_times,
                cdf,
                label=adapter_name,
                color=_get_color(adapter_name),
                linewidth=2,
            )

        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("CDF")
        ax.set_title(f"Per-Batch Latency CDF — {scenario_id}")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def memory_waterfall(self) -> Figure:
        """Stacked bar chart of memory usage by framework."""
        adapters = sorted(self._results.results.keys())
        scenario_ids = sorted(self._results.all_scenario_ids)

        fig, ax = plt.subplots(figsize=(max(10, len(adapters) * 1.5), 6))

        # Use first scenario's data, or show placeholder
        if not scenario_ids:
            return self._empty_figure("No scenarios for memory waterfall")

        x = np.arange(len(adapters))
        rss_values = []

        for adapter_name in adapters:
            adapter_results = self._results.results.get(adapter_name, [])
            if adapter_results and adapter_results[0].resources:
                rss_values.append(adapter_results[0].resources.peak_rss_mb)
            else:
                # Use config to estimate memory if no ResourceMonitor data
                if adapter_results:
                    cfg = adapter_results[0].config
                    ds = cfg.get("dataset_size", 0)
                    shape = cfg.get("element_shape", [1])
                    from math import prod

                    est_mb = ds * prod(shape) * 4 / (1024**2)
                    rss_values.append(est_mb)
                else:
                    rss_values.append(0)

        ax.bar(
            x,
            rss_values,
            color=[_get_color(a) for a in adapters],
        )

        ax.set_xlabel("Framework")
        ax.set_ylabel("Peak RSS (MB)")
        ax.set_title("Memory Usage Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(adapters, rotation=45, ha="right")
        fig.tight_layout()
        return fig

    def scaling_curves(self, dimension: str = "batch_size") -> Figure:
        """Line chart: throughput vs scaling dimension."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for adapter_name, adapter_results in sorted(self._results.results.items()):
            # Group by dimension value
            points: dict[int, float] = {}
            for r in adapter_results:
                dim_val = r.config.get(dimension, r.config.get("batch_size", 0))
                if isinstance(dim_val, (int, float)):
                    points[int(dim_val)] = r.throughput_elements_sec()

            if points:
                xs = sorted(points.keys())
                ys = [points[x] for x in xs]
                ax.plot(
                    xs,
                    ys,
                    "o-",
                    label=adapter_name,
                    color=_get_color(adapter_name),
                    linewidth=2,
                    markersize=6,
                )

        ax.set_xlabel(dimension.replace("_", " ").title())
        ax.set_ylabel("Throughput (elem/s)")
        ax.set_title(f"Scaling: Throughput vs {dimension.replace('_', ' ').title()}")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def chain_depth(self) -> Figure:
        """Line chart: throughput degradation with chain depth."""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Look for PC-1 variants at different chain depths
        for adapter_name, adapter_results in sorted(self._results.results.items()):
            depths: dict[int, float] = {}
            for r in adapter_results:
                if r.scenario_id.startswith("PC"):
                    depth = r.config.get("extra", {}).get("chain_depth", None)
                    if depth is None:
                        depth = int(r.scenario_id.split("-")[-1]) if "-" in r.scenario_id else 1
                    depths[depth] = r.throughput_elements_sec()

            if depths:
                xs = sorted(depths.keys())
                ys = [depths[x] for x in xs]
                ax.plot(
                    xs,
                    ys,
                    "o-",
                    label=adapter_name,
                    color=_get_color(adapter_name),
                    linewidth=2,
                )

        if not ax.get_lines():
            # Fallback: plot all scenarios as "depth" proxy
            for adapter_name, adapter_results in sorted(self._results.results.items()):
                if adapter_results:
                    xs = list(range(1, len(adapter_results) + 1))
                    ys = [r.throughput_elements_sec() for r in adapter_results]
                    ax.plot(
                        xs,
                        ys,
                        "o-",
                        label=adapter_name,
                        color=_get_color(adapter_name),
                        linewidth=2,
                    )

        ax.set_xlabel("Chain Depth / Complexity")
        ax.set_ylabel("Throughput (elem/s)")
        ax.set_title("Pipeline Complexity vs Throughput")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    def feature_heatmap(self) -> Figure:
        """Heatmap: which adapters support which scenarios."""
        scenario_ids = sorted(self._results.all_scenario_ids)
        adapters = sorted(self._results.results.keys())

        if not scenario_ids or not adapters:
            return self._empty_figure("No data for feature heatmap")

        # Build support matrix: 1 if adapter has results for scenario
        matrix = np.zeros((len(adapters), len(scenario_ids)))
        for i, adapter in enumerate(adapters):
            scenario_results = {r.scenario_id for r in self._results.results.get(adapter, [])}
            for j, sid in enumerate(scenario_ids):
                matrix[i, j] = 1.0 if sid in scenario_results else 0.0

        fig, ax = plt.subplots(
            figsize=(max(8, len(scenario_ids) * 0.8), max(4, len(adapters) * 0.5)),
        )

        im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(scenario_ids)))
        ax.set_xticklabels(scenario_ids, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(adapters)))
        ax.set_yticklabels(adapters, fontsize=9)
        ax.set_title("Framework × Scenario Support Matrix")

        # Add text annotations
        for i in range(len(adapters)):
            for j in range(len(scenario_ids)):
                text = "Y" if matrix[i, j] > 0 else ""
                ax.text(j, i, text, ha="center", va="center", fontsize=7)

        fig.colorbar(im, ax=ax, label="Supported", shrink=0.8)
        fig.tight_layout()
        return fig

    def _empty_figure(self, message: str) -> Figure:
        """Create a placeholder figure with a message."""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return fig
