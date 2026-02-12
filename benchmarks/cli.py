"""Unified CLI for datarax benchmarks: run → convert → store → analyze → export.

Usage:
    datarax-bench run    — Full pipeline: run benchmarks, analyze, export to W&B
    datarax-bench export — Re-export existing results to W&B
    datarax-bench analyze — Run analysis only (no W&B)
    datarax-bench report — Generate comparison report markdown
"""

from __future__ import annotations

from pathlib import Path

import click

import benchkit
from benchmarks.export import FullExporter, export_to_benchkit
from benchmarks.runners.full_runner import ComparativeResults, FullRunner


@click.group()
def main() -> None:
    """datarax-bench: Benchmark runner, analyzer, and W&B exporter."""


@main.command()
@click.option("--platform", default="cpu", type=click.Choice(["cpu", "gpu", "tpu"]))
@click.option("--repetitions", default=3, type=int)
@click.option("--output-dir", default="benchmark-data/reports/latest", type=click.Path())
@click.option("--scenarios", multiple=True, help="Specific scenario IDs")
@click.option("--adapters", multiple=True, help="Specific adapter names")
@click.option("--profile", default="ci_cpu", help="Hardware profile")
@click.option("--baseline/--no-baseline", default=True)
@click.option("--wandb/--no-wandb", "use_wandb", default=True)
@click.option("--charts/--no-charts", "use_charts", default=True)
@click.option("--data", default="benchmark-data", type=click.Path(), help="benchkit store dir")
@click.option("--project", default=None, help="W&B project override")
@click.option("--entity", default=None, help="W&B entity override")
def run(
    platform: str,
    repetitions: int,
    output_dir: str,
    scenarios: tuple[str, ...],
    adapters: tuple[str, ...],
    profile: str,
    baseline: bool,
    use_wandb: bool,
    use_charts: bool,
    data: str,
    project: str | None,
    entity: str | None,
) -> None:
    """Run benchmarks end-to-end: execute, convert, store, analyze, export."""
    output_path = Path(output_dir)

    # 1. Run benchmarks
    click.echo(f"Running benchmarks on {platform} (profile={profile}, reps={repetitions})")
    runner = FullRunner(
        output_dir=output_path,
        hardware_profile=profile,
        platform=platform,
    )

    scenario_filter = set(scenarios) if scenarios else None
    adapter_filter = set(adapters) if adapters else None

    comparative = runner.run_comparative(
        scenario_filter=scenario_filter,
        adapter_filter=adapter_filter,
        num_repetitions=repetitions,
    )

    n_adapters = len(comparative.results)
    n_scenarios = len(comparative.all_scenario_ids)
    click.echo(f"Completed: {n_adapters} adapters, {n_scenarios} scenarios")

    # 2. Convert to benchkit Run
    benchkit_run = export_to_benchkit(comparative)

    # 3. Save to benchkit store
    store = benchkit.Store(data)
    store.save(benchkit_run)
    click.echo(f"Saved to store: {data}")

    # 4. Load baseline if requested
    baseline_run = store.get_baseline() if baseline else None

    # 5. Export to W&B (with charts + analysis)
    if use_wandb:
        exporter = FullExporter(store, project=project, entity=entity)
        chart_dir = Path(data) / "charts" if use_charts else None
        url = exporter.export(
            comparative,
            benchkit_run,
            baseline=baseline_run,
            chart_dir=chart_dir,
            results_dir=output_path,
        )
        if url:
            click.echo(f"W&B dashboard: {url}")
        else:
            click.echo("W&B export skipped (no auth or wandb not installed)")
    else:
        click.echo("W&B export disabled (--no-wandb)")

    # 6. Set as baseline for future comparisons
    if baseline:
        store.set_baseline(benchkit_run.id)
        click.echo(f"Set baseline: {benchkit_run.id}")

    click.echo("Done.")


@main.command()
@click.option(
    "--results-dir",
    default="benchmark-data/reports/latest",
    type=click.Path(exists=True),
)
@click.option("--data", default="benchmark-data", type=click.Path(), help="benchkit store dir")
@click.option("--project", default=None, help="W&B project override")
@click.option("--entity", default=None, help="W&B entity override")
def export(
    results_dir: str,
    data: str,
    project: str | None,
    entity: str | None,
) -> None:
    """Re-export existing results to W&B."""
    click.echo(f"Loading results from {results_dir}")
    comparative = ComparativeResults.load(Path(results_dir))
    benchkit_run = export_to_benchkit(comparative)

    store = benchkit.Store(data)
    store.save(benchkit_run)

    exporter = FullExporter(store, project=project, entity=entity)
    url = exporter.export(comparative, benchkit_run, results_dir=Path(results_dir))
    if url:
        click.echo(f"W&B dashboard: {url}")
    else:
        click.echo("W&B export failed (check auth)")


@main.command()
@click.option(
    "--results-dir",
    default="benchmark-data/reports/latest",
    type=click.Path(exists=True),
)
@click.option("--output", default="benchmark-data/analysis", type=click.Path())
def analyze(results_dir: str, output: str) -> None:
    """Run analysis only (no W&B export)."""
    click.echo(f"Analyzing results from {results_dir}")
    comparative = ComparativeResults.load(Path(results_dir))

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Gap detection
    from benchmarks.analysis.gap_detection import GapDetector

    detector = GapDetector(comparative)
    detector.generate_backlog(output_path / "optimization_backlog.md")
    click.echo(f"  Gap backlog: {output_path / 'optimization_backlog.md'}")

    # Comparison report
    from benchmarks.analysis.comparison_report import ComparisonReportGenerator

    report_gen = ComparisonReportGenerator(comparative)
    markdown = report_gen.generate(chart_dir=output_path / "charts")
    (output_path / "comparison_report.md").write_text(markdown)
    click.echo(f"  Comparison report: {output_path / 'comparison_report.md'}")

    # Charts
    from benchmarks.visualization.charts import ChartGenerator

    chart_gen = ChartGenerator(comparative, output_path / "charts")
    paths = chart_gen.generate_all()
    click.echo(f"  Charts: {len(paths)} files in {output_path / 'charts'}")

    click.echo("Analysis complete.")


@main.command()
@click.option(
    "--results-dir",
    default="benchmark-data/reports/latest",
    type=click.Path(exists=True),
)
@click.option(
    "--output",
    default="docs/benchmarks/comparison_report.md",
    type=click.Path(),
)
def report(results_dir: str, output: str) -> None:
    """Generate comparison report markdown."""
    click.echo(f"Generating report from {results_dir}")
    comparative = ComparativeResults.load(Path(results_dir))

    from benchmarks.analysis.comparison_report import ComparisonReportGenerator

    report_gen = ComparisonReportGenerator(comparative)
    markdown = report_gen.generate()

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown)
    click.echo(f"Report saved to {output_path}")
