"""Benchmark runner — main orchestrator for scenario execution.

Discovers scenarios, runs them through adapters, and stores results.
Supports baseline generation and per-scenario / per-variant filtering.

Design ref: Section 6.4.3 of the benchmark report.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from benchmarks.adapters.base import BenchmarkAdapter
from benchmarks.core.baselines import BaselineStore
from benchmarks.core.config_loader import load_hardware_profile
from benchmarks.core.platform import can_run_scenario, estimate_scenario_memory_mb
from benchmarks.scenarios import discover_scenarios
from benchmarks.scenarios.base import ScenarioVariant, run_scenario
from datarax.benchmarking.results import BenchmarkResult


class BenchmarkRunner:
    """Orchestrates benchmark scenario execution.

    Args:
        output_dir: Directory for result JSON files.
        hardware_profile: Name of the hardware profile TOML (e.g., "ci_cpu").
    """

    def __init__(
        self,
        output_dir: Path | str,
        hardware_profile: str = "ci_cpu",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.profile = load_hardware_profile(hardware_profile)
        self._profile_settings = self.profile.get("profile", {})

    @property
    def num_batches(self) -> int:
        return self._profile_settings.get("num_batches", 20)

    @property
    def warmup_batches(self) -> int:
        return self._profile_settings.get("warmup_batches", 3)

    def run_scenario(
        self,
        scenario_module: ModuleType,
        adapter: BenchmarkAdapter,
        variant_name: str | None = None,
        num_repetitions: int = 5,
    ) -> BenchmarkResult:
        """Run a single scenario variant and return BenchmarkResult.

        Args:
            scenario_module: Scenario module with VARIANTS, get_variant(), etc.
            adapter: BenchmarkAdapter to test.
            variant_name: Variant name (uses TIER1_VARIANT or first variant if None).
            num_repetitions: Number of repetitions (median selected).

        Returns:
            BenchmarkResult from median repetition.
        """
        if variant_name is None:
            variant_name = getattr(scenario_module, "TIER1_VARIANT", None)
            if variant_name is None:
                # Use first available variant
                variants = scenario_module.VARIANTS
                variant_name = next(iter(variants))

        variant: ScenarioVariant = scenario_module.get_variant(variant_name)

        return run_scenario(
            adapter,
            variant,
            num_batches=self.num_batches,
            warmup_batches=self.warmup_batches,
            num_repetitions=num_repetitions,
        )

    def run_all(
        self,
        adapter: BenchmarkAdapter,
        scenario_filter: set[str] | None = None,
        tier: int | None = None,
        num_repetitions: int = 5,
    ) -> list[BenchmarkResult]:
        """Run all matching scenarios.

        Args:
            adapter: BenchmarkAdapter to test.
            scenario_filter: Only run scenarios with these IDs. None = all.
            tier: If 1, only run Tier 1 scenarios.
            num_repetitions: Repetitions per scenario.

        Returns:
            List of BenchmarkResults.
        """
        scenarios = discover_scenarios(tier=tier)
        results: list[BenchmarkResult] = []

        for mod in scenarios:
            scenario_id = mod.SCENARIO_ID
            if scenario_filter and scenario_id not in scenario_filter:
                continue
            if not adapter.supports_scenario(scenario_id):
                continue

            # Determine variant and check memory
            variant_name = self._get_variant_for_scenario(mod)
            variant = mod.get_variant(variant_name)
            if not can_run_scenario(variant):
                mb = estimate_scenario_memory_mb(
                    variant.config.dataset_size,
                    variant.config.element_shape,
                )
                print(
                    f"SKIP {scenario_id}/{variant_name}: {mb:.0f} MB exceeds available memory",
                    file=sys.stderr,
                )
                continue

            try:
                result = self.run_scenario(
                    mod,
                    adapter,
                    variant_name,
                    num_repetitions,
                )
                result.save(self.output_dir / f"{scenario_id}_{result.variant}.json")
                results.append(result)
            except Exception as exc:
                print(f"SKIP {scenario_id}: {exc}", file=sys.stderr)

        return results

    def generate_baselines(
        self,
        adapter: BenchmarkAdapter,
        baselines_dir: Path | str,
        num_repetitions: int = 5,
        tier: int | None = None,
        force: bool = False,
    ) -> list[Path]:
        """Generate baselines for all supported scenarios.

        Args:
            adapter: BenchmarkAdapter to test.
            baselines_dir: Directory for baseline JSON files.
            num_repetitions: Repetitions per scenario.
            tier: If 1, only generate Tier 1 baselines. None = all.
            force: Overwrite existing baselines if True.

        Returns:
            List of paths to saved baseline files.
        """
        store = BaselineStore(baselines_dir)
        scenarios = discover_scenarios(tier=tier)
        saved: list[Path] = []

        for mod in scenarios:
            scenario_id = mod.SCENARIO_ID
            if not adapter.supports_scenario(scenario_id):
                continue

            for variant_name in mod.VARIANTS:
                variant = mod.get_variant(variant_name)
                baseline_name = f"{scenario_id}_{variant_name}"

                if not force and store.load(baseline_name) is not None:
                    print(f"  EXISTS {baseline_name}")
                    continue

                if not can_run_scenario(variant):
                    mb = estimate_scenario_memory_mb(
                        variant.config.dataset_size,
                        variant.config.element_shape,
                    )
                    print(f"  SKIP {baseline_name} ({mb:.0f} MB exceeds memory)")
                    continue

                try:
                    result = run_scenario(
                        adapter,
                        variant,
                        num_batches=self.num_batches,
                        warmup_batches=self.warmup_batches,
                        num_repetitions=num_repetitions,
                    )
                    path = store.save(baseline_name, result)
                    saved.append(path)
                    throughput = result.throughput_elements_sec()
                    print(f"  OK {baseline_name}: {throughput:.0f} elem/s")
                except Exception as exc:
                    print(
                        f"  FAIL {baseline_name}: {exc}",
                        file=sys.stderr,
                    )

        return saved

    def _get_variant_for_scenario(self, mod: ModuleType) -> str | None:
        """Get variant name from hardware profile, or use default."""
        profile_variants = self._profile_settings.get("variants", {})
        scenario_id = mod.SCENARIO_ID
        if scenario_id in profile_variants:
            return profile_variants[scenario_id]
        # Use TIER1_VARIANT or first variant
        tier1 = getattr(mod, "TIER1_VARIANT", None)
        if tier1 is not None:
            return tier1
        return next(iter(mod.VARIANTS))


def main() -> None:
    """CLI entry point for benchmark runner."""
    import argparse

    from benchmarks.adapters.datarax_adapter import DataraxAdapter

    parser = argparse.ArgumentParser(description="Datarax Benchmark Runner")
    parser.add_argument(
        "--profile",
        default="ci_cpu",
        help="Hardware profile name",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark-data/reports",
        help="Directory for result JSON files",
    )
    parser.add_argument(
        "--generate-baselines",
        action="store_true",
        help="Generate baselines for all scenarios",
    )
    parser.add_argument(
        "--baselines-dir",
        default="benchmarks/baselines",
        help="Directory for baseline JSON files",
    )
    parser.add_argument(
        "--tier",
        type=int,
        default=None,
        choices=[1, 2],
        help="Only run scenarios of this tier",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Number of repetitions per scenario",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="Specific scenario IDs to run",
    )

    args = parser.parse_args()

    runner = BenchmarkRunner(
        output_dir=Path(args.output_dir),
        hardware_profile=args.profile,
    )
    adapter = DataraxAdapter()

    if args.generate_baselines:
        print("Generating baselines...")
        saved = runner.generate_baselines(
            adapter,
            args.baselines_dir,
            args.repetitions,
        )
        print(f"Generated {len(saved)} baselines.")
    else:
        scenario_filter = set(args.scenarios) if args.scenarios else None
        results = runner.run_all(
            adapter,
            scenario_filter=scenario_filter,
            tier=args.tier,
            num_repetitions=args.repetitions,
        )
        print(f"Completed {len(results)} scenarios.")
        for r in results:
            throughput = r.throughput_elements_sec()
            print(f"  {r.scenario_id}/{r.variant}: {throughput:.0f} elem/s")


if __name__ == "__main__":
    main()
