"""Base scenario infrastructure: ScenarioVariant and run_scenario.

Each scenario module defines VARIANTS and uses make_get_variant() to create
a standard get_variant() function. The run_scenario() function handles the
full adapter lifecycle:
data generation -> setup -> warmup -> iterate -> teardown -> result assembly.

Design ref: Section 7 of the benchmark report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from collections.abc import Callable

from benchmarks.adapters.base import BenchmarkAdapter, IterationResult, ScenarioConfig
from benchmarks.core.environment import capture_environment
from datarax.benchmarking.results import BenchmarkResult
from datarax.benchmarking.timing import TimingSample

DEFAULT_SEED: int = 42
"""Default RNG seed used across all benchmark scenarios for reproducibility."""


@dataclass
class ScenarioVariant:
    """A specific variant of a scenario (e.g., CV-1 small).

    Attributes:
        config: Scenario configuration for this variant.
        data_generator: Callable that lazily generates the data dict.
    """

    config: ScenarioConfig
    data_generator: Callable[[], dict[str, Any]]


def make_get_variant(
    variants: dict[str, ScenarioVariant],
) -> Callable[[str], ScenarioVariant]:
    """Create a standard get_variant() lookup for a scenario module.

    Every scenario module needs ``get_variant(name) -> ScenarioVariant``.
    This factory eliminates identical boilerplate across all 25 modules.

    Args:
        variants: The module-level VARIANTS dict.

    Returns:
        A ``get_variant(name)`` function that raises ``KeyError`` on miss.
    """

    def get_variant(name: str) -> ScenarioVariant:
        return variants[name]

    return get_variant


def _iteration_to_timing(result: IterationResult) -> TimingSample:
    """Convert an IterationResult to a TimingSample."""
    return TimingSample(
        wall_clock_sec=result.wall_clock_sec,
        per_batch_times=result.per_batch_times,
        first_batch_time=result.first_batch_time,
        num_batches=result.num_batches,
        num_elements=result.num_elements,
    )


def run_scenario(
    adapter: BenchmarkAdapter,
    variant: ScenarioVariant,
    num_batches: int = 50,
    warmup_batches: int = 5,
    num_repetitions: int = 5,
) -> BenchmarkResult:
    """Run a scenario variant through an adapter, returning BenchmarkResult.

    Handles: data generation -> setup -> warmup -> iterate -> teardown -> result.
    Runs num_repetitions times and returns the median result by wall_clock_sec.

    Args:
        adapter: Benchmark adapter to test.
        variant: Scenario variant with config and data generator.
        num_batches: Number of batches per iteration.
        warmup_batches: Warmup batches before timing.
        num_repetitions: Number of repetitions (median selected).

    Returns:
        BenchmarkResult for the median repetition.
    """
    config = variant.config
    data = variant.data_generator()

    results: list[IterationResult] = []
    for _ in range(num_repetitions):
        adapter.setup(config, data)
        adapter.warmup(warmup_batches)
        result = adapter.iterate(num_batches)
        adapter.teardown()
        results.append(result)

    # Select median by wall_clock_sec
    sorted_results = sorted(results, key=lambda r: r.wall_clock_sec)
    median_idx = len(sorted_results) // 2
    median_result = sorted_results[median_idx]

    timing = _iteration_to_timing(median_result)
    env = capture_environment()

    return BenchmarkResult(
        framework=adapter.name,
        scenario_id=config.scenario_id,
        variant=config.extra.get("variant_name", "default"),
        timing=timing,
        resources=None,
        environment=env,
        config={
            "batch_size": config.batch_size,
            "dataset_size": config.dataset_size,
            "element_shape": list(config.element_shape),
            "transforms": config.transforms,
            "seed": config.seed,
        },
    )
