"""CI runner — Tier 1 regression gate for pull requests.

Runs 6 Tier 1 scenarios with small variants, compares against CPU baselines,
and exits non-zero on failure (blocking merge).

Tier 1 scenarios (Section 9.3):
  CV-1 (small), NLP-1 (small), TAB-1 (small),
  PC-1 (depth_1), IO-1 (memory_source), PR-2 (small)

Design ref: Sections 6.4.3, 9.3 of the benchmark report.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from benchmarks.adapters.datarax_adapter import DataraxAdapter
from benchmarks.core.baselines import BaselineStore
from benchmarks.core.platform import can_run_scenario
from benchmarks.runners.benchmark_runner import BenchmarkRunner
from benchmarks.scenarios import discover_scenarios
from benchmarks.scenarios.base import run_scenario
from datarax.benchmarking.results import BenchmarkResult

# Default paths
_DEFAULT_BASELINES = Path("benchmarks/baselines")
_DEFAULT_OUTPUT = Path("benchmark-data/reports")


def run_tier1_gate(
    baselines_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    num_repetitions: int = 5,
) -> tuple[list[BenchmarkResult], list[dict[str, Any] | None]]:
    """Run all Tier 1 scenarios and compare against baselines.

    Args:
        baselines_dir: Path to baselines directory. None skips comparison.
        output_dir: Path for result output. Defaults to benchmark-data/reports.
        num_repetitions: Repetitions per scenario.

    Returns:
        Tuple of (results, verdicts). Each verdict is None if no baseline exists.
    """
    out = Path(output_dir) if output_dir else _DEFAULT_OUTPUT
    runner = BenchmarkRunner(output_dir=out, hardware_profile="ci_cpu")
    adapter = DataraxAdapter()

    # Discover Tier 1 scenarios
    scenarios = discover_scenarios(tier=1)

    results: list[BenchmarkResult] = []
    for mod in scenarios:
        scenario_id = mod.SCENARIO_ID
        if not adapter.supports_scenario(scenario_id):
            continue

        tier1_variant = getattr(mod, "TIER1_VARIANT", None)
        if tier1_variant is None:
            continue

        try:
            variant = mod.get_variant(tier1_variant)
            if not can_run_scenario(variant):
                print(f"SKIP {scenario_id}/{tier1_variant}: exceeds memory", file=sys.stderr)
                continue
            result = run_scenario(
                adapter,
                variant,
                num_batches=runner.num_batches,
                warmup_batches=runner.warmup_batches,
                num_repetitions=num_repetitions,
            )
            results.append(result)
        except Exception as exc:
            print(f"ERROR running {scenario_id}: {exc}", file=sys.stderr)

    # Compare against baselines
    verdicts: list[dict[str, Any] | None] = []
    if baselines_dir is not None:
        store = BaselineStore(baselines_dir)
        for result in results:
            baseline_name = f"{result.scenario_id}_{result.variant}"
            verdict = store.compare(baseline_name, result)
            verdicts.append(verdict)
    else:
        verdicts = [None] * len(results)

    return results, verdicts


def generate_ci_report(
    results: list[BenchmarkResult],
    verdicts: list[dict[str, Any] | None],
) -> str:
    """Generate a human-readable CI report.

    Args:
        results: BenchmarkResults from the gate run.
        verdicts: Corresponding baseline verdicts (None if no baseline).

    Returns:
        Formatted report string.
    """
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("Datarax Benchmark Tier 1 Gate Report")
    lines.append("=" * 60)
    lines.append("")

    for result, verdict in zip(results, verdicts):
        throughput = result.throughput_elements_sec()
        latencies = result.latency_percentiles()

        status_str = "NO BASELINE"
        if verdict is not None:
            status = verdict.get("status", "unknown")
            ratio = verdict.get("throughput_ratio", 0.0)
            status_str = f"{status.upper()} (ratio: {ratio:.2f})"

        lines.append(f"  {result.scenario_id}/{result.variant}:")
        lines.append(f"    Throughput: {throughput:.0f} elem/s")
        lines.append(f"    p50: {latencies['p50']:.2f}ms  p99: {latencies['p99']:.2f}ms")
        lines.append(f"    Status: {status_str}")
        lines.append("")

    # Summary
    failures = sum(1 for v in verdicts if v is not None and v.get("status") == "failure")
    warnings = sum(1 for v in verdicts if v is not None and v.get("status") == "warning")
    passes = sum(1 for v in verdicts if v is not None and v.get("status") == "pass")
    no_baseline = sum(1 for v in verdicts if v is None)

    lines.append("-" * 60)
    lines.append(
        f"Total: {len(results)} scenarios | "
        f"Pass: {passes} | Warn: {warnings} | "
        f"Fail: {failures} | No baseline: {no_baseline}"
    )
    if failures > 0:
        lines.append("RESULT: FAILED (performance regression detected)")
    else:
        lines.append("RESULT: PASSED")
    lines.append("=" * 60)

    return "\n".join(lines)


def main() -> None:
    """CI regression gate entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Datarax CI Benchmark Gate")
    parser.add_argument(
        "--baselines-dir",
        default=str(_DEFAULT_BASELINES),
        help="Baselines directory (skip comparison if doesn't exist)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_DEFAULT_OUTPUT),
        help="Output directory for results (default: benchmark-data/reports)",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Number of repetitions per scenario",
    )

    args = parser.parse_args()

    baselines_path = Path(args.baselines_dir)
    baselines_dir: Path | None = baselines_path if baselines_path.exists() else None

    results, verdicts = run_tier1_gate(
        baselines_dir=baselines_dir,
        output_dir=args.output_dir,
        num_repetitions=args.repetitions,
    )

    report = generate_ci_report(results, verdicts)
    print(report)

    # Save results
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for result in results:
        result.save(out / f"ci_{result.scenario_id}_{result.variant}.json")

    # Exit code
    if any(v is not None and v.get("status") == "failure" for v in verdicts):
        sys.exit(1)


if __name__ == "__main__":
    main()
