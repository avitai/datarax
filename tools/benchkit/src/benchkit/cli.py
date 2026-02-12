"""benchkit CLI: ingest, export, check, baseline, summary."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from benchkit.analysis import detect_regressions
from benchkit.store import Store


@click.group()
def main() -> None:
    """Run benchkit benchmark analysis and W&B dashboard tool."""


@main.command()
@click.option("--data", required=True, type=click.Path(), help="Store directory path")
@click.option(
    "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="JSON file to ingest",
)
def ingest(data: str, input_file: str) -> None:
    """Ingest results from a JSON file into the store."""
    store = Store(data)
    run = store.ingest(Path(input_file))
    click.echo(f"Ingested run {run.id} ({len(run.points)} points)")


@main.command()
@click.option("--data", required=True, type=click.Path(), help="Store directory path")
@click.option("--run", "run_id", default="latest", help="Run ID or 'latest'")
@click.option("--project", default=None, help="W&B project (overrides config.json)")
@click.option("--entity", default=None, help="W&B entity (overrides config.json)")
def export(data: str, run_id: str, project: str | None, entity: str | None) -> None:
    """Export a run to W&B.

    Reads W&B project/entity from the store's config.json.
    Use --project/--entity to override. Authentication requires
    WANDB_API_KEY environment variable (never stored in config).
    """
    store = Store(data)

    try:
        run = store.latest() if run_id == "latest" else store.load(run_id)
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from None

    try:
        from benchkit.exporters.wandb import WandBExporter, WANDB_AVAILABLE
    except ImportError:
        WANDB_AVAILABLE = False

    if not WANDB_AVAILABLE:
        raise click.ClickException(
            "wandb is not installed. Install with: pip install benchkit[wandb]"
        )

    # Build exporter: CLI flags override config.json settings
    exporter = WandBExporter.from_store(store)
    if project:
        exporter.project = project
    if entity:
        exporter.entity = entity

    if not exporter.check_auth():
        raise click.ClickException(
            "WANDB_API_KEY environment variable not set.\n"
            "  Set it with: export WANDB_API_KEY='your-key'\n"
            "  Or use offline mode: export WANDB_MODE=offline"
        )

    url = exporter.export_run(run)

    baseline = store.get_baseline()
    exporter.export_analysis(run, baseline)

    if url:
        click.echo(f"Exported to W&B: {url}")
    else:
        click.echo("Exported to W&B (offline mode)")


@main.command()
@click.option("--data", required=True, type=click.Path(), help="Store directory path")
@click.option("--threshold", default=0.05, type=float, help="Regression threshold (fraction)")
def check(data: str, threshold: float) -> None:
    """Run regression check against baseline (CI gate)."""
    store = Store(data)

    baseline = store.get_baseline()
    if baseline is None:
        click.echo("No baseline set. Use 'benchkit baseline' to set one.")
        return

    try:
        current = store.latest()
    except FileNotFoundError:
        click.echo("No runs in store.")
        return

    regressions = detect_regressions(current, baseline, threshold=threshold)

    if not regressions:
        click.echo(f"PASS: No regressions detected (threshold={threshold * 100:.0f}%)")
        sys.exit(0)
    else:
        click.echo(f"FAIL: {len(regressions)} regression(s) detected:")
        for reg in regressions:
            arrow = "↓" if reg.direction == "higher" else "↑"
            click.echo(
                f"  {arrow} {reg.point_name} {reg.metric}: "
                f"{reg.baseline_value:.2f} → {reg.current_value:.2f} "
                f"({reg.delta_pct:+.1f}%)"
            )
        sys.exit(1)


@main.command()
@click.option("--data", required=True, type=click.Path(), help="Store directory path")
@click.option("--run", "run_id", default="latest", help="Run ID or 'latest'")
def baseline(data: str, run_id: str) -> None:
    """Set a run as the baseline for regression detection."""
    store = Store(data)

    try:
        if run_id == "latest":
            run = store.latest()
            store.set_baseline(run.id)
        else:
            store.set_baseline(run_id)
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from None

    click.echo(f"Baseline set to run {run_id if run_id != 'latest' else run.id}")


@main.command()
@click.option("--data", required=True, type=click.Path(), help="Store directory path")
@click.option("--metric", required=True, help="Metric name to track")
@click.option("--point", "point_name", required=True, help="Point name to filter")
@click.option("--framework", required=True, help="Framework tag to filter")
@click.option("--n-runs", default=None, type=int, help="Limit to N most recent runs")
def trend(data: str, metric: str, point_name: str, framework: str, n_runs: int | None) -> None:
    """Show metric trend across runs."""
    store = Store(data)
    series = store.extract_trend(
        metric=metric,
        point_name=point_name,
        tags={"framework": framework},
        n_runs=n_runs,
    )

    if not series.points:
        click.echo(f"Trend: {metric} @ {point_name} (framework={framework}) — no data (0 points)")
        return

    click.echo(f"Trend: {metric} @ {point_name} (framework={framework})")
    click.echo(f"  {len(series.points)} points\n")

    for tp in series.points:
        commit_str = f"  [{tp.commit}]" if tp.commit else ""
        ci_str = ""
        if tp.lower is not None and tp.upper is not None:
            ci_str = f"  CI=[{tp.lower:.2f}, {tp.upper:.2f}]"
        click.echo(
            f"  {tp.timestamp.strftime('%Y-%m-%d %H:%M')}  {tp.value:.2f}{ci_str}{commit_str}"
        )


@main.command()
@click.option("--data", required=True, type=click.Path(), help="Store directory path")
@click.option("--run", "run_id", default="latest", help="Run ID or 'latest'")
def summary(data: str, run_id: str) -> None:
    """Show a human-readable run summary."""
    store = Store(data)

    try:
        run = store.latest() if run_id == "latest" else store.load(run_id)
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from None

    click.echo(f"Run: {run.id}")
    click.echo(f"Timestamp: {run.timestamp.isoformat()}")
    if run.commit:
        click.echo(f"Commit: {run.commit}")
    if run.branch:
        click.echo(f"Branch: {run.branch}")
    click.echo(f"Points: {len(run.points)}")
    click.echo()

    # Group by scenario
    scenarios: dict[str, list] = {}
    for point in run.points:
        scenarios.setdefault(point.scenario, []).append(point)

    for scenario, points in scenarios.items():
        click.echo(f"  {scenario}:")
        for point in points:
            fw = point.tags.get("framework", point.name)
            metrics_str = ", ".join(f"{k}={v.value:.2f}" for k, v in point.metrics.items())
            click.echo(f"    {fw}: {metrics_str}")
