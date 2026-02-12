"""W&B exporter: metrics, comparison tables, HTML bold-best, alerts.

Follows patterns from image-reconstruction repo:
- Guard all W&B calls with WANDB_AVAILABLE check
- WANDB_MODE=offline for testing
- Slash notation for metric groups
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from benchkit.analysis import aggregate_score, detect_regressions, pareto_front, rank_table
from benchkit.exporters.base import Exporter
from benchkit.models import Run, is_higher_better

if TYPE_CHECKING:
    from benchkit.store import Store

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore[assignment]
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


class WandBExporter(Exporter):
    """Export benchkit results to W&B.

    Creates ONE W&B run per benchkit Run (not per framework).
    All frameworks appear as rows in comparison tables.

    Authentication:
        W&B credentials are read exclusively from the WANDB_API_KEY
        environment variable. Never store API keys in config files.

        For local development, set the env var in your shell:
            export WANDB_API_KEY="your-key-here"

        For CI, use GitHub Secrets:
            env:
              WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
    """

    def __init__(
        self,
        project: str,
        entity: str | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """Initialize WandBExporter.

        Args:
            project: Weights & Biases project name.
            entity: Optional W&B team or user entity.
            tags: Optional list of tags to attach to each run.
        """
        self.project = project
        self.entity = entity
        self.tags = tags or []

    @classmethod
    def from_store(
        cls,
        store: Store,
        tags: list[str] | None = None,
    ) -> WandBExporter:
        """Create exporter from a Store's config.json settings.

        Reads ``wandb_project`` and ``wandb_entity`` from the store's
        config.json so callers don't need to pass them every time.

        Args:
            store: Store instance with config loaded.
            tags: Optional W&B run tags.
        """
        return cls(
            project=store.wandb_project or "benchmarks",
            entity=store.wandb_entity,
            tags=tags,
        )

    def check_auth(self) -> bool:
        """Check if W&B authentication is available.

        Returns True if:
        - WANDB_API_KEY env var is set, OR
        - WANDB_MODE is "offline" (no auth needed), OR
        - ``wandb login`` stored credentials are present

        Returns False otherwise.
        """
        if not WANDB_AVAILABLE:
            return False

        # Offline mode doesn't need an API key
        if os.environ.get("WANDB_MODE", "").lower() in ("offline", "disabled"):
            return True

        if os.environ.get("WANDB_API_KEY"):
            return True

        # Accept stored credentials from `wandb login`
        try:
            return bool(wandb.api.api_key)
        except Exception:
            return False

    # --- Generic artifact logging methods ---

    def log_figures(self, figures: dict[str, Any]) -> None:
        """Log matplotlib Figure objects as wandb.Image.

        Generic method -- any repo can pass its own charts.

        Args:
            figures: {key: matplotlib.figure.Figure} mapping.
                Keys become W&B panel names (use slash for grouping).
        """
        for key, fig in figures.items():
            wandb.log({key: wandb.Image(fig)})

    def log_html_artifacts(self, html: dict[str, str]) -> None:
        """Log HTML strings as wandb.Html.

        Args:
            html: {key: html_string} mapping.
        """
        for key, html_str in html.items():
            wandb.log({key: wandb.Html(html_str)})

    def log_extra_tables(self, tables: dict[str, tuple[list[str], list[list]]]) -> None:
        """Log arbitrary tables to W&B.

        Args:
            tables: {key: (columns, rows)} where columns is list of strings
                and rows is list of lists.
        """
        for key, (columns, rows) in tables.items():
            wandb.log({key: wandb.Table(columns=columns, data=rows)})

    # --- Export methods ---

    def export_run(self, run: Run, *, finish: bool = True) -> str:
        """Export a Run to W&B. Returns W&B run URL.

        Args:
            run: Benchmark run to export.
            finish: If True (default), call wandb.finish() after logging.
                Set to False when you need to log additional artifacts to the
                same W&B run (e.g., from a composed exporter).

        Returns empty string if wandb is not installed or auth is missing.
        """
        if not WANDB_AVAILABLE:
            return ""

        if not self.check_auth():
            logger.warning(
                "W&B export skipped: WANDB_API_KEY not set. "
                "Set the environment variable or use WANDB_MODE=offline."
            )
            return ""

        wb_run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=f"{run.timestamp.strftime('%Y-%m-%d')}_{run.id}",
            tags=self.tags,
            config=run.environment,
        )

        if wb_run is None:
            return ""

        # Log all metrics with slash-grouped names
        summary_metrics: dict[str, Any] = {}
        for point in run.points:
            fw = point.tags.get("framework", point.name)
            for metric_name, metric in point.metrics.items():
                md = run.metric_defs.get(metric_name)
                if md and md.group:
                    key = f"{md.group}/{metric_name}/{fw}"
                else:
                    key = f"{metric_name}/{fw}"
                summary_metrics[key] = metric.value

        wandb.log(summary_metrics)

        # Log comparison table
        self._log_comparison_table(run)

        # Log styled HTML comparison
        self._log_html_comparison(run)

        # Log per-scenario comparison tables
        self._log_per_scenario_tables(run)

        url = wb_run.url or ""
        if finish:
            wandb.finish()
        return url

    def export_analysis(self, run: Run, baseline: Run | None = None) -> None:
        """Export analysis results to W&B.

        Logs rankings, regression alerts, aggregate scores, and Pareto front.
        """
        if not WANDB_AVAILABLE:
            return

        if not self.check_auth():
            logger.warning("W&B analysis export skipped: WANDB_API_KEY not set.")
            return

        self._init_analysis_run(run)
        self._log_rank_tables(run)
        self._log_regression_alerts(run, baseline)
        self._log_aggregate_scores(run)
        self._log_pareto_front(run)

        if wandb.run is not None:
            wandb.finish()

    def _init_analysis_run(self, run: Run) -> None:
        """Initialize a W&B run for analysis if one isn't active.

        Args:
            run: Benchmark run (used for naming).
        """
        if wandb.run is None:
            wandb.init(
                project=self.project,
                entity=self.entity,
                name=f"analysis_{run.id}",
            )

    def _log_rank_tables(self, run: Run) -> None:
        """Log ranking tables for all metrics.

        Args:
            run: Benchmark run with metric definitions and points.
        """
        for metric_name in run.metric_defs:
            ranks = rank_table(run, metric_name)
            if not ranks:
                continue
            columns = ["Rank", "Framework", "Value", "Best?", "Delta %"]
            table_data = [[r.rank, r.label, r.value, r.is_best, r.delta_from_best] for r in ranks]
            wandb.log(
                {
                    f"rankings/{metric_name}": wandb.Table(
                        columns=columns,
                        data=table_data,
                    ),
                }
            )

    def _log_regression_alerts(self, run: Run, baseline: Run | None) -> None:
        """Log regression alerts comparing current run to baseline.

        Args:
            run: Current benchmark run.
            baseline: Previous baseline run for comparison.
        """
        if baseline is None:
            return
        regressions = detect_regressions(run, baseline)
        for reg in regressions:
            if wandb.run is not None:
                wandb.run.alert(
                    title=f"Regression: {reg.metric} @ {reg.point_name}",
                    text=(
                        f"{reg.metric} degraded {reg.delta_pct:.1f}% "
                        f"(baseline: {reg.baseline_value}, "
                        f"current: {reg.current_value})"
                    ),
                    level=wandb.AlertLevel.WARN,
                )

    def _log_aggregate_scores(self, run: Run) -> None:
        """Log aggregate scores weighting all primary metrics equally.

        Args:
            run: Benchmark run with metric definitions.
        """
        primary_weights = {
            name: 1.0 for name, md in run.metric_defs.items() if md.priority.value == "primary"
        }
        if not primary_weights:
            return
        scores = aggregate_score(run, primary_weights)
        if not scores:
            return
        score_data = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        wandb.log(
            {
                "analysis/aggregate_scores": wandb.Table(
                    columns=["Framework", "Score"],
                    data=[[fw, f"{score:.3f}"] for fw, score in score_data],
                ),
            }
        )

    def _log_pareto_front(self, run: Run) -> None:
        """Log Pareto front for throughput vs latency.

        Args:
            run: Benchmark run with throughput and latency metrics.
        """
        if "throughput" not in run.metric_defs or "latency_p50" not in run.metric_defs:
            return
        front = pareto_front(
            run.points,
            "throughput",
            "latency_p50",
            metric_defs=run.metric_defs,
        )
        if not front:
            return
        wandb.log(
            {
                "analysis/pareto_front": wandb.Table(
                    columns=["Framework", "Throughput", "Latency p50"],
                    data=[
                        [
                            p.tags.get("framework", p.name),
                            f"{p.metrics['throughput'].value:.2f}",
                            f"{p.metrics['latency_p50'].value:.2f}"
                            if "latency_p50" in p.metrics
                            else "—",
                        ]
                        for p in front
                    ],
                ),
            }
        )

    @staticmethod
    def _discover_metric_names(run: Run) -> list[str]:
        """Get sorted metric names from metric_defs, falling back to points."""
        if run.metric_defs:
            return sorted(run.metric_defs.keys())
        return sorted({m for p in run.points for m in p.metrics})

    @staticmethod
    def _find_best_values(run: Run, metric_names: list[str]) -> dict[str, float | None]:
        """Find the best value per metric (direction-aware)."""
        best: dict[str, float | None] = {}
        for mname in metric_names:
            values = [p.metrics[mname].value for p in run.points if mname in p.metrics]
            if not values:
                best[mname] = None
            elif is_higher_better(run.metric_defs.get(mname)):
                best[mname] = max(values)
            else:
                best[mname] = min(values)
        return best

    def _log_comparison_table(self, run: Run) -> None:
        """Log a wandb.Table with framework comparison."""
        metric_names = self._discover_metric_names(run)
        if not metric_names:
            return

        best_values = self._find_best_values(run, metric_names)

        columns = ["Framework", "Scenario", *list(metric_names)]
        data = []
        for point in run.points:
            fw = point.tags.get("framework", point.name)
            row: list[Any] = [fw, point.scenario]
            for mname in metric_names:
                if mname in point.metrics:
                    val = point.metrics[mname].value
                    cell = f"{val:.2f}"
                    if best_values.get(mname) is not None and val == best_values[mname]:
                        cell += " *"
                    row.append(cell)
                else:
                    row.append("—")
            data.append(row)

        wandb.log(
            {
                "comparison": wandb.Table(columns=columns, data=data),
            }
        )

    def _log_html_comparison(self, run: Run) -> None:
        """Log a styled HTML table with bold-best formatting."""
        metric_names = self._discover_metric_names(run)
        if not metric_names:
            return

        best_values = self._find_best_values(run, metric_names)

        html = ['<table style="border-collapse:collapse;font-family:monospace">']
        html.append("<tr><th>Framework</th><th>Scenario</th>")
        for mname in metric_names:
            md = run.metric_defs.get(mname)
            unit = f" ({md.unit})" if md and md.unit else ""
            html.append(f"<th>{mname}{unit}</th>")
        html.append("</tr>")

        for point in run.points:
            fw = point.tags.get("framework", point.name)
            html.append(f"<tr><td>{fw}</td><td>{point.scenario}</td>")
            for mname in metric_names:
                if mname in point.metrics:
                    val = point.metrics[mname].value
                    is_best = best_values.get(mname) is not None and val == best_values[mname]
                    cell = f"{val:.2f}"
                    if is_best:
                        cell = f"<b>{cell}</b>"
                    html.append(f"<td>{cell}</td>")
                else:
                    html.append("<td>—</td>")
            html.append("</tr>")

        html.append("</table>")

        wandb.log({"comparison_styled": wandb.Html("\n".join(html))})

    def _log_per_scenario_tables(self, run: Run) -> None:
        """Log a comparison table per scenario."""
        scenarios: dict[str, list] = {}
        for point in run.points:
            scenarios.setdefault(point.scenario, []).append(point)

        metric_names = self._discover_metric_names(run)
        for scenario_id, points in scenarios.items():
            columns = ["Framework", *list(metric_names)]
            data = []
            for point in points:
                fw = point.tags.get("framework", point.name)
                row: list[Any] = [fw]
                for mname in metric_names:
                    if mname in point.metrics:
                        row.append(f"{point.metrics[mname].value:.2f}")
                    else:
                        row.append("—")
                data.append(row)
            wandb.log(
                {
                    f"scenarios/{scenario_id}": wandb.Table(columns=columns, data=data),
                }
            )

    def export_trends(
        self,
        store: Store,
        metric: str,
        point_name: str,
        tags: dict[str, str],
        *,
        n_runs: int | None = None,
    ) -> None:
        """Export metric trend as a W&B line chart.

        Extracts trend data from the store and logs it as a wandb.Table
        that renders as a line chart in the W&B dashboard.

        Args:
            store: Store to extract trend data from.
            metric: Metric name to track.
            point_name: Point name to filter.
            tags: Tags that must all match.
            n_runs: Limit to N most recent runs.
        """
        if not WANDB_AVAILABLE:
            return

        if not self.check_auth():
            logger.warning("W&B trend export skipped: WANDB_API_KEY not set.")
            return

        series = store.extract_trend(
            metric=metric,
            point_name=point_name,
            tags=tags,
            n_runs=n_runs,
        )

        if not series.points:
            return

        should_finish = False
        if wandb.run is None:
            tag_str = "_".join(f"{k}={v}" for k, v in sorted(tags.items()))
            wandb.init(
                project=self.project,
                entity=self.entity,
                name=f"trend_{metric}_{tag_str}",
            )
            should_finish = True

        # Log as step-based line chart
        for tp in series.points:
            step_data = {
                f"trend/{metric}": tp.value,
                "trend/timestamp": tp.timestamp.isoformat(),
            }
            if tp.commit:
                step_data["trend/commit"] = tp.commit
            if tp.lower is not None:
                step_data[f"trend/{metric}_lower"] = tp.lower
            if tp.upper is not None:
                step_data[f"trend/{metric}_upper"] = tp.upper
            wandb.log(step_data)

        # Also log as a table for custom visualization
        columns = ["Timestamp", "Value", "Commit"]
        table_data = [[tp.timestamp.isoformat(), tp.value, tp.commit or ""] for tp in series.points]
        wandb.log(
            {
                f"trend_table/{metric}": wandb.Table(
                    columns=columns,
                    data=table_data,
                ),
            }
        )

        if should_finish and wandb.run is not None:
            wandb.finish()
