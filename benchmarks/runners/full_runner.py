"""Multi-adapter comparative benchmark runner.

Runs all scenarios × all available adapters in a single session,
with cache clearing between frameworks for fair measurement.

Design ref: Sections 6.4.3, 8.3, 11.4 Task 4.1 of the benchmark report.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.adapters import get_available_adapters
from benchmarks.core.config_loader import load_hardware_profile
from benchmarks.core.environment import capture_environment
from benchmarks.core.platform import can_run_scenario
from benchmarks.runners.benchmark_runner import BenchmarkRunner
from benchmarks.scenarios import discover_scenarios
from benchmarks.scenarios.base import run_scenario
from datarax.benchmarking.results import BenchmarkResult


@dataclass
class ComparativeResults:
    """Container for multi-adapter benchmark results.

    Attributes:
        results: Mapping of adapter_name -> list of BenchmarkResults.
        environment: System fingerprint from capture_environment().
        platform: Target platform (cpu/gpu/tpu).
        timestamp: Unix timestamp of the run.
    """

    results: dict[str, list[BenchmarkResult]]
    environment: dict[str, Any]
    platform: str
    timestamp: float

    def save(self, output_dir: Path) -> None:
        """Save all results as individual JSONs plus a manifest."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest: dict[str, Any] = {
            "platform": self.platform,
            "timestamp": self.timestamp,
            "environment": self.environment,
            "adapters": {},
        }

        for adapter_name, adapter_results in self.results.items():
            adapter_files = []
            for result in adapter_results:
                fname = f"{adapter_name}_{result.scenario_id}_{result.variant}.json"
                # Sanitize filename
                fname = fname.replace(" ", "_")
                result.save(output_dir / fname)
                adapter_files.append(fname)
            manifest["adapters"][adapter_name] = adapter_files

        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    @classmethod
    def load(cls, output_dir: Path) -> ComparativeResults:
        """Load ComparativeResults from a directory with manifest."""
        output_dir = Path(output_dir)
        manifest = json.loads((output_dir / "manifest.json").read_text())

        results: dict[str, list[BenchmarkResult]] = {}
        for adapter_name, filenames in manifest["adapters"].items():
            results[adapter_name] = [
                BenchmarkResult.load(output_dir / fname) for fname in filenames
            ]

        return cls(
            results=results,
            environment=manifest.get("environment", {}),
            platform=manifest["platform"],
            timestamp=manifest["timestamp"],
        )

    def get_scenario_results(self, scenario_id: str) -> dict[str, BenchmarkResult]:
        """Get results for a single scenario across all adapters."""
        out: dict[str, BenchmarkResult] = {}
        for adapter_name, adapter_results in self.results.items():
            for r in adapter_results:
                if r.scenario_id == scenario_id:
                    out[adapter_name] = r
                    break
        return out

    def get_adapter_results(self, adapter_name: str) -> list[BenchmarkResult]:
        """Get all results for a single adapter."""
        return list(self.results.get(adapter_name, []))

    @property
    def all_scenario_ids(self) -> set[str]:
        """Set of all scenario IDs present in results."""
        ids: set[str] = set()
        for adapter_results in self.results.values():
            for r in adapter_results:
                ids.add(r.scenario_id)
        return ids


class FullRunner:
    """Multi-adapter comparative benchmark runner.

    Runs all scenarios × all available adapters, with process isolation
    between frameworks per Section 8.3.

    Args:
        output_dir: Directory for result files.
        hardware_profile: Name of the hardware profile TOML.
        platform: Target platform (cpu/gpu/tpu).
    """

    def __init__(
        self,
        output_dir: Path | str,
        hardware_profile: str = "ci_cpu",
        platform: str = "cpu",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.platform = platform
        self.profile = load_hardware_profile(hardware_profile)
        self._runner = BenchmarkRunner(
            output_dir=self.output_dir,
            hardware_profile=hardware_profile,
        )

    def run_comparative(
        self,
        scenario_filter: set[str] | None = None,
        adapter_filter: set[str] | None = None,
        num_repetitions: int = 5,
    ) -> ComparativeResults:
        """Run all scenarios across all available adapters.

        Args:
            scenario_filter: Only run these scenario IDs. None = all.
            adapter_filter: Only run these adapter names. None = all available.
            num_repetitions: Repetitions per scenario (median selected).

        Returns:
            ComparativeResults containing all results.
        """
        available = get_available_adapters()
        scenarios = discover_scenarios()

        # Apply adapter filter
        if adapter_filter is not None:
            available = {name: cls for name, cls in available.items() if name in adapter_filter}

        # Apply scenario filter
        if scenario_filter is not None:
            scenarios = [mod for mod in scenarios if mod.SCENARIO_ID in scenario_filter]

        env = capture_environment()
        all_results: dict[str, list[BenchmarkResult]] = {}

        for adapter_name, adapter_cls in available.items():
            print(f"--- {adapter_name} ---", file=sys.stderr)
            adapter = adapter_cls()
            adapter_results: list[BenchmarkResult] = []

            for mod in scenarios:
                scenario_id = mod.SCENARIO_ID
                if not adapter.supports_scenario(scenario_id):
                    continue

                variant_name = getattr(mod, "TIER1_VARIANT", None)
                if variant_name is None:
                    variant_name = next(iter(mod.VARIANTS))

                variant = mod.get_variant(variant_name)
                if not can_run_scenario(variant):
                    print(
                        f"  SKIP {scenario_id}: exceeds memory",
                        file=sys.stderr,
                    )
                    continue

                try:
                    result = run_scenario(
                        adapter,
                        variant,
                        num_batches=self._runner.num_batches,
                        warmup_batches=self._runner.warmup_batches,
                        num_repetitions=num_repetitions,
                    )
                    adapter_results.append(result)
                    throughput = result.throughput_elements_sec()
                    print(
                        f"  {scenario_id}/{variant_name}: {throughput:.0f} elem/s",
                        file=sys.stderr,
                    )
                except Exception as exc:
                    print(
                        f"  FAIL {scenario_id}: {exc}",
                        file=sys.stderr,
                    )

            all_results[adapter_name] = adapter_results
            self._clear_framework_caches()

        comparative = ComparativeResults(
            results=all_results,
            environment=env,
            platform=self.platform,
            timestamp=time.time(),
        )

        # Save results and summary
        comparative.save(self.output_dir)
        self._write_summary(comparative)

        return comparative

    def _clear_framework_caches(self) -> None:
        """Clear caches between framework runs (Section 8.3).

        Clears JAX caches, Python GC, and CUDA memory if available.
        """
        try:
            import jax

            jax.clear_caches()
        except (ImportError, AttributeError):
            pass

        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _write_summary(self, comparative: ComparativeResults) -> None:
        """Write a human-readable summary.json."""
        summary: dict[str, Any] = {
            "platform": comparative.platform,
            "timestamp": comparative.timestamp,
            "adapters": list(comparative.results.keys()),
            "scenarios": list(comparative.all_scenario_ids),
            "results_summary": {},
        }

        for adapter_name, results in comparative.results.items():
            summary["results_summary"][adapter_name] = {
                r.scenario_id: {
                    "throughput_elem_s": r.throughput_elements_sec(),
                    "variant": r.variant,
                }
                for r in results
            }

        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2),
        )


def main() -> None:
    """CLI entry point for full comparative runner."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Datarax Full Comparative Benchmark Runner",
    )
    parser.add_argument(
        "--platform",
        default="cpu",
        choices=["cpu", "gpu", "tpu"],
        help="Target platform",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark-data/reports/releases/v1.0",
        help="Output directory for results",
    )
    parser.add_argument(
        "--profile",
        default="ci_cpu",
        help="Hardware profile name",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="Specific scenario IDs to run",
    )
    parser.add_argument(
        "--adapters",
        nargs="*",
        default=None,
        help="Specific adapter names to run",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Number of repetitions per scenario",
    )

    args = parser.parse_args()

    runner = FullRunner(
        output_dir=Path(args.output_dir),
        hardware_profile=args.profile,
        platform=args.platform,
    )

    scenario_filter = set(args.scenarios) if args.scenarios else None
    adapter_filter = set(args.adapters) if args.adapters else None

    comparative = runner.run_comparative(
        scenario_filter=scenario_filter,
        adapter_filter=adapter_filter,
        num_repetitions=args.repetitions,
    )

    print(
        f"\nCompleted: {len(comparative.results)} adapters, "
        f"{len(comparative.all_scenario_ids)} scenarios"
    )
    print(f"Results saved to: {runner.output_dir}")


if __name__ == "__main__":
    main()
