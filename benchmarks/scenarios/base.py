"""Base scenario infrastructure: ScenarioVariant and run_scenario.

Each scenario module defines VARIANTS and uses make_get_variant() to create
a standard get_variant() function. The run_scenario() function handles the
full adapter lifecycle:
data generation -> setup -> warmup -> iterate -> teardown -> result assembly.

Design ref: Section 7 of the benchmark report.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from calibrax.core import BenchmarkResult
from calibrax.profiling import TimingSample
from substrax.testing.compiles import CompileCountError, expect_compiles

from benchmarks.adapters.base import IterationResult, PipelineAdapter, ScenarioConfig
from benchmarks.core.environment import capture_environment
from benchmarks.core.result_model import build_benchmark_result


DEFAULT_SEED: int = 42
"""Default RNG seed used across all benchmark scenarios for reproducibility."""


@dataclass
class ScenarioVariant:
    """A specific variant of a scenario (e.g., CV-1 small).

    Attributes:
        config: Scenario configuration for this variant.
        data_generator: Callable that lazily generates the data dict for a dataset of the
            given number of elements; :meth:`generate_data` passes ``config.dataset_size``.
    """

    config: ScenarioConfig
    data_generator: Callable[[int], dict[str, Any]]

    def generate_data(self) -> dict[str, Any]:
        """Generate the variant's data at its configured ``dataset_size``.

        Returns:
            The data dict, every field holding ``config.dataset_size`` elements.
        """
        return self.data_generator(self.config.dataset_size)

    def with_dataset_size(self, dataset_size: int) -> ScenarioVariant:
        """The same variant over a dataset of ``dataset_size`` elements.

        Batch size, element shape, transforms and extras are unchanged; the data is generated
        at the new size, never generated at full size and sliced.

        Args:
            dataset_size: Number of elements in the dataset.

        Returns:
            A new variant; this one is unchanged.

        Raises:
            ValueError: If ``dataset_size`` is below 1.
        """
        if dataset_size < 1:
            raise ValueError(f"dataset_size must be at least 1, got {dataset_size}")
        return replace(self, config=replace(self.config, dataset_size=dataset_size))


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
        per_batch_times=tuple(result.per_batch_times),
        first_batch_time=result.first_batch_time,
        num_batches=result.num_batches,
        num_elements=result.num_elements,
    )


def _iterate_compiling_nothing(
    adapter: PipelineAdapter, config: ScenarioConfig, num_batches: int
) -> IterationResult:
    """Run the timed ``adapter.iterate(num_batches)``, refusing any compile inside it.

    Raises:
        CompileCountError: If jax built an executable during the timed iteration, with a note
            naming the adapter, scenario and variant.
    """
    try:
        with expect_compiles(0):
            return adapter.iterate(num_batches)
    except CompileCountError as error:
        error.add_note(
            f"adapter {adapter.name!r}, scenario {config.scenario_id!r}, variant "
            f"{config.extra.get('variant_name', 'default')!r}: the timed iteration compiled "
            "after warmup, so its time includes compilation"
        )
        raise


def run_scenario(
    adapter: PipelineAdapter,
    variant: ScenarioVariant,
    num_batches: int = 50,
    warmup_batches: int = 5,
    num_repetitions: int = 5,
) -> BenchmarkResult:
    """Run a scenario variant through an adapter, returning BenchmarkResult.

    Handles: data generation -> setup -> warmup -> iterate -> teardown -> result.
    Runs num_repetitions times and returns the median result by wall_clock_sec.
    Warmup is told the timed batch count and may compile; the timed iteration must not:
    a compile there raises :class:`~substrax.testing.compiles.CompileCountError`.

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
    data = variant.generate_data()

    results: list[IterationResult] = []
    for _ in range(num_repetitions):
        adapter.setup(config, data)
        try:
            adapter.warmup(warmup_batches, timed_batches=num_batches)
            results.append(_iterate_compiling_nothing(adapter, config, num_batches))
        finally:
            adapter.teardown()

    # Select median by wall_clock_sec
    sorted_results = sorted(results, key=lambda r: r.wall_clock_sec)
    median_idx = len(sorted_results) // 2
    median_result = sorted_results[median_idx]

    timing = _iteration_to_timing(median_result)
    env = capture_environment()

    return build_benchmark_result(
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
