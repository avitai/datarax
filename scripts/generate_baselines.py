"""Generate initial CPU baselines for benchmark scenarios.

Uses BenchmarkRunner.generate_baselines() to ensure settings (num_batches,
warmup_batches) are consistent with what CI and other runners use.

Dynamically checks available system/GPU memory and skips scenarios
that would exceed available resources.

Usage:
    uv run python scripts/generate_baselines.py
    uv run python scripts/generate_baselines.py --repetitions 5
    uv run python scripts/generate_baselines.py --tier 1  # Tier 1 only
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.adapters.datarax_adapter import DataraxAdapter
from benchmarks.core.platform import get_available_memory_mb
from benchmarks.runners.benchmark_runner import BenchmarkRunner


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate benchmark baselines")
    parser.add_argument(
        "--baselines-dir",
        default="benchmarks/baselines",
        help="Directory for baseline JSON files",
    )
    parser.add_argument(
        "--profile",
        default="ci_cpu",
        help="Hardware profile name (controls num_batches, warmup_batches)",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of repetitions per scenario",
    )
    parser.add_argument(
        "--tier",
        type=int,
        default=None,
        choices=[1, 2],
        help="Only generate baselines for this tier (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing baselines",
    )
    args = parser.parse_args()

    available = get_available_memory_mb()
    print(f"Available memory: {available:.0f} MB")

    runner = BenchmarkRunner(
        output_dir=Path("benchmark-data/reports"),
        hardware_profile=args.profile,
    )
    print(
        f"Profile: {args.profile} "
        f"(num_batches={runner.num_batches}, "
        f"warmup={runner.warmup_batches})"
    )

    adapter = DataraxAdapter()

    saved = runner.generate_baselines(
        adapter,
        baselines_dir=args.baselines_dir,
        num_repetitions=args.repetitions,
        tier=args.tier,
        force=args.force,
    )
    print(f"\nGenerated {len(saved)} baselines.")


if __name__ == "__main__":
    main()
