"""Adapter: convert datarax ComparativeResults -> calibrax Run, and FullExporter."""

from __future__ import annotations

import logging

# Controlled local git metadata lookup.
import subprocess  # nosec B404
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from calibrax.core import (
    Metric,
    MetricDef,
    MetricDirection,
    MetricPriority,
    Point,
    Run,
)
from calibrax.exporters.wandb import WandBExporter
from calibrax.storage import Store

from benchmarks.analysis.comparison_report import ComparisonReportGenerator
from benchmarks.analysis.gap_detection import GapDetector
from benchmarks.analysis.stability import StabilityValidator
from benchmarks.core.result_model import (
    latency_percentiles_ms,
    result_scenario_id,
    result_variant,
    throughput_elements_per_sec,
)
from benchmarks.runners.full_runner import ComparativeResults
from benchmarks.visualization.charts import ChartGenerator

logger = logging.getLogger(__name__)


def _metric_defs() -> dict[str, MetricDef]:
    """Canonical metric definitions for datarax comparative runs."""
    return {
        "throughput": MetricDef(
            name="throughput",
            unit="elem/s",
            direction=MetricDirection.HIGHER,
            group="Throughput",
            priority=MetricPriority.PRIMARY,
        ),
        "latency_p50": MetricDef(
            name="latency_p50",
            unit="ms",
            direction=MetricDirection.LOWER,
            group="Latency",
            priority=MetricPriority.PRIMARY,
        ),
        "latency_p95": MetricDef(
            name="latency_p95",
            unit="ms",
            direction=MetricDirection.LOWER,
            group="Latency",
        ),
        "latency_p99": MetricDef(
            name="latency_p99",
            unit="ms",
            direction=MetricDirection.LOWER,
            group="Latency",
        ),
        "peak_memory": MetricDef(
            name="peak_memory",
            unit="MB",
            direction=MetricDirection.LOWER,
            group="Memory",
            priority=MetricPriority.PRIMARY,
        ),
        "gpu_memory": MetricDef(
            name="gpu_memory",
            unit="MB",
            direction=MetricDirection.LOWER,
            group="Memory",
        ),
    }


def export_to_calibrax(comparative: ComparativeResults) -> Run:
    """Convert datarax ComparativeResults -> calibrax Run."""
    points: list[Point] = []

    for adapter_name, results in comparative.results.items():
        for result in results:
            scenario_id = result_scenario_id(result)
            variant = result_variant(result)
            latencies = latency_percentiles_ms(result)
            metrics: dict[str, Metric] = {
                "throughput": Metric(value=throughput_elements_per_sec(result)),
                "latency_p50": Metric(value=latencies["p50"]),
                "latency_p95": Metric(value=latencies["p95"]),
                "latency_p99": Metric(value=latencies["p99"]),
            }

            if result.resources is not None:
                metrics["peak_memory"] = Metric(value=result.resources.peak_rss_mb)
                if result.resources.peak_gpu_mem_mb is not None:
                    metrics["gpu_memory"] = Metric(value=result.resources.peak_gpu_mem_mb)

            points.append(
                Point(
                    name=f"{scenario_id}/{variant}",
                    scenario=scenario_id,
                    tags={
                        "framework": adapter_name,
                        "variant": variant,
                    },
                    metrics=metrics,
                )
            )

    commit = None
    try:
        # Static git command; no user-controlled input.
        commit = subprocess.check_output(  # nosec B603 B607
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return Run(
        points=tuple(points),
        commit=commit,
        environment=comparative.environment,
        metadata={"platform": comparative.platform},
        metric_defs=_metric_defs(),
    )


class FullExporter:
    """Full W&B export: calibrax metrics + datarax charts + analysis."""

    def __init__(
        self,
        store: Store,
        project: str | None = None,
        entity: str | None = None,
    ) -> None:
        self._store = store
        self._project = project or "datarax-benchmarks"
        self._entity = entity
        self._exporter = WandBExporter(project=self._project, entity=self._entity)

    def export(
        self,
        comparative: ComparativeResults,
        run: Run,
        *,
        baseline: Run | None = None,
        chart_dir: Path | None = None,
        results_dir: Path | None = None,
    ) -> str:
        """Export everything to one W&B run. Returns W&B URL."""
        url = self._exporter.export_run(run, finish=False)
        if not url:
            return ""

        self._log_charts(comparative, chart_dir)
        self._log_gaps(comparative)
        self._log_comparison_report(comparative)
        self._log_stability(comparative)

        if results_dir is not None:
            self._upload_results_artifact(results_dir, run)

        self._exporter.export_analysis(run, baseline)
        return url

    def _log_charts(
        self,
        comparative: ComparativeResults,
        chart_dir: Path | None,
    ) -> None:
        chart_gen = ChartGenerator(comparative, chart_dir or Path("benchmark-data/charts"))
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
                pass
        self._exporter.log_figures(figures)
        for fig in figures.values():
            plt.close(fig)

    def _log_gaps(self, comparative: ComparativeResults) -> None:
        detector = GapDetector(comparative)
        gaps = detector.detect()
        if not gaps:
            return
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

    def _upload_results_artifact(self, results_dir: Path, run: Run) -> None:
        """Upload raw results directory as a W&B Artifact."""
        try:
            import wandb
        except ImportError:
            logger.warning("wandb not installed - skipping artifact upload")
            return

        if wandb.run is None:
            logger.warning("No active W&B run - skipping artifact upload")
            return

        results_path = Path(results_dir)
        if not results_path.is_dir():
            logger.warning(
                "Results directory %s not found - skipping artifact upload", results_path
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
            rows.append([scenario_id, adapter, "-", "STABLE"])
        for scenario_id, adapter, cv in report.unstable_scenarios:
            rows.append([scenario_id, adapter, f"{cv:.3f}", "UNSTABLE"])
        if not rows:
            return
        self._exporter.log_extra_tables(
            {
                "analysis/stability": (
                    ["Scenario", "Adapter", "CV", "Status"],
                    rows,
                ),
            }
        )
