"""Adapter: convert datarax ComparativeResults → benchkit Run, and FullExporter.

This lives in the datarax repo, not in benchkit itself.
benchkit has no knowledge of datarax internals.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

import benchkit
from benchkit.exporters.wandb import WandBExporter
from benchmarks.analysis.comparison_report import ComparisonReportGenerator
from benchmarks.analysis.gap_detection import GapDetector
from benchmarks.analysis.stability import StabilityValidator
from benchmarks.runners.full_runner import ComparativeResults
from benchmarks.visualization.charts import ChartGenerator

logger = logging.getLogger(__name__)


def export_to_benchkit(comparative: ComparativeResults) -> benchkit.Run:
    """Convert datarax ComparativeResults → benchkit Run.

    Maps each (adapter, BenchmarkResult) pair to a benchkit Point,
    extracting throughput, latency, and memory metrics.
    """
    points: list[benchkit.Point] = []

    for adapter_name, results in comparative.results.items():
        for result in results:
            metrics: dict[str, benchkit.Metric] = {
                "throughput": benchkit.Metric(result.throughput_elements_sec()),
            }

            # Add latency percentiles if available
            latencies = result.latency_percentiles()
            if latencies.get("p50", 0) > 0:
                metrics["latency_p50"] = benchkit.Metric(latencies["p50"])
                metrics["latency_p95"] = benchkit.Metric(latencies["p95"])
                metrics["latency_p99"] = benchkit.Metric(latencies["p99"])

            # Add memory if resource monitor ran
            if result.resources is not None:
                metrics["peak_memory"] = benchkit.Metric(result.resources.peak_rss_mb)
                if result.resources.peak_gpu_mem_mb is not None:
                    metrics["gpu_memory"] = benchkit.Metric(
                        result.resources.peak_gpu_mem_mb,
                    )

            points.append(
                benchkit.Point(
                    name=f"{result.scenario_id}/{result.variant}",
                    scenario=result.scenario_id,
                    tags={
                        "framework": adapter_name,
                        "variant": result.variant,
                    },
                    metrics=metrics,
                )
            )

    # Try to get current commit hash
    commit = None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return benchkit.Run(
        points=points,
        commit=commit,
        environment=comparative.environment,
        metadata={"platform": comparative.platform},
        metric_defs={
            "throughput": benchkit.MetricDef(
                "throughput",
                "elem/s",
                "higher",
                group="Throughput",
                priority=benchkit.MetricPriority.PRIMARY,
            ),
            "latency_p50": benchkit.MetricDef(
                "latency_p50",
                "ms",
                "lower",
                group="Latency",
                priority=benchkit.MetricPriority.PRIMARY,
            ),
            "latency_p95": benchkit.MetricDef(
                "latency_p95",
                "ms",
                "lower",
                group="Latency",
            ),
            "latency_p99": benchkit.MetricDef(
                "latency_p99",
                "ms",
                "lower",
                group="Latency",
            ),
            "peak_memory": benchkit.MetricDef(
                "peak_memory",
                "MB",
                "lower",
                group="Memory",
                priority=benchkit.MetricPriority.PRIMARY,
            ),
            "gpu_memory": benchkit.MetricDef(
                "gpu_memory",
                "MB",
                "lower",
                group="Memory",
            ),
        },
    )


class FullExporter:
    """Full W&B export: benchkit metrics + datarax charts + analysis.

    Composes benchkit's WandBExporter (generic metrics/tables/rankings)
    with datarax-specific analysis (ChartGenerator, GapDetector, etc.).
    """

    def __init__(
        self,
        store: benchkit.Store,
        project: str | None = None,
        entity: str | None = None,
    ) -> None:
        self._store = store
        self._exporter = WandBExporter.from_store(store)
        if project:
            self._exporter.project = project
        if entity:
            self._exporter.entity = entity

    def export(
        self,
        comparative: ComparativeResults,
        run: benchkit.Run,
        *,
        baseline: benchkit.Run | None = None,
        chart_dir: Path | None = None,
        results_dir: Path | None = None,
    ) -> str:
        """Export everything to one W&B run. Returns W&B URL."""
        # 1. Core metrics + comparison tables + rankings (benchkit generic)
        #    Keep the W&B run open so we can log additional artifacts.
        url = self._exporter.export_run(run, finish=False)

        # If W&B init failed (no auth / wandb not installed), skip all logging.
        if not url:
            return ""

        # 2. Charts (datarax-specific)
        self._log_charts(comparative, chart_dir)

        # 3. Gap detection (datarax-specific)
        self._log_gaps(comparative)

        # 4. Comparison report (datarax-specific)
        self._log_comparison_report(comparative)

        # 5. Stability (datarax-specific)
        self._log_stability(comparative)

        # 6. Upload raw results as W&B Artifact (enables re-analysis after VM teardown)
        if results_dir is not None:
            self._upload_results_artifact(results_dir, run)

        # 7. Analysis: aggregate scores + Pareto + regression alerts (benchkit generic)
        #    export_analysis will call wandb.finish() when done.
        self._exporter.export_analysis(run, baseline)

        return url

    def _log_charts(
        self,
        comparative: ComparativeResults,
        chart_dir: Path | None,
    ) -> None:
        chart_gen = ChartGenerator(
            comparative,
            chart_dir or Path("benchmark-data/charts"),
        )
        chart_methods = [
            ("charts/throughput_bars", chart_gen.throughput_bars),
            ("charts/throughput_radar", chart_gen.throughput_radar),
            ("charts/latency_cdf", chart_gen.latency_cdf),
            ("charts/memory_waterfall", chart_gen.memory_waterfall),
            ("charts/scaling_curves", chart_gen.scaling_curves),
            ("charts/chain_depth", chart_gen.chain_depth),
            ("charts/feature_heatmap", chart_gen.feature_heatmap),
        ]
        figures: dict[str, Any] = {}
        for key, method in chart_methods:
            try:
                figures[key] = method()
            except Exception:
                pass  # Skip charts that can't be generated
        self._exporter.log_figures(figures)
        # Close figures to free memory
        for fig in figures.values():
            plt.close(fig)

    def _log_gaps(self, comparative: ComparativeResults) -> None:
        detector = GapDetector(comparative)
        gaps = detector.detect()
        if gaps:
            self._exporter.log_extra_tables(
                {
                    "analysis/gaps": (
                        [
                            "Priority",
                            "Scenario",
                            "Gap Ratio",
                            "Framework",
                            "Severity",
                            "Mitigation",
                        ],
                        [
                            [
                                g.priority,
                                g.scenario_id,
                                f"{g.gap_ratio:.1f}x",
                                g.top_alternative,
                                g.severity,
                                g.mitigation,
                            ]
                            for g in gaps
                        ],
                    ),
                }
            )

    def _log_comparison_report(self, comparative: ComparativeResults) -> None:
        report_gen = ComparisonReportGenerator(comparative)
        markdown = report_gen.generate()
        html = f"<pre style='font-family:monospace;white-space:pre-wrap'>{markdown}</pre>"
        self._exporter.log_html_artifacts({"analysis/comparison_summary": html})

    def _upload_results_artifact(
        self,
        results_dir: Path,
        run: benchkit.Run,
    ) -> None:
        """Upload raw results directory as a W&B Artifact.

        Persists JSON manifests, per-adapter results, and summary files
        so they survive VM teardown when running on ephemeral cloud instances.
        """
        try:
            import wandb
        except ImportError:
            logger.warning("wandb not installed — skipping artifact upload")
            return

        if wandb.run is None:
            logger.warning("No active W&B run — skipping artifact upload")
            return

        results_path = Path(results_dir)
        if not results_path.is_dir():
            logger.warning(
                "Results directory %s not found — skipping artifact upload",
                results_path,
            )
            return

        name = f"benchmark-results-{run.commit or 'latest'}"
        artifact = wandb.Artifact(
            name=name,
            type="benchmark-results",
            metadata={
                "platform": run.metadata.get("platform", "unknown"),
                "commit": run.commit,
                "environment": run.environment,
            },
        )
        artifact.add_dir(str(results_path))
        wandb.log_artifact(artifact)
        logger.info("Uploaded results artifact: %s (%s)", name, results_path)

    def _log_stability(self, comparative: ComparativeResults) -> None:
        validator = StabilityValidator()
        report = validator.validate(comparative)
        rows: list[list[Any]] = []
        for scenario_id, adapter in report.stable_scenarios:
            rows.append([scenario_id, adapter, "—", "STABLE"])
        for scenario_id, adapter, cv in report.unstable_scenarios:
            rows.append([scenario_id, adapter, f"{cv:.3f}", "UNSTABLE"])
        if rows:
            self._exporter.log_extra_tables(
                {
                    "analysis/stability": (
                        ["Scenario", "Adapter", "CV", "Status"],
                        rows,
                    ),
                }
            )
