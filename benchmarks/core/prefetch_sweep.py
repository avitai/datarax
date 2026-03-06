"""Prefetch sensitivity sweep utility.

Runs a single scenario at multiple prefetch_size values to quantify the
pipeline benefit of prefetching. Reports throughput at each level to
identify the optimal prefetch depth for a given workload.

Protocol ref: Section 8.4 of the performance audit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from calibrax.core import BenchmarkResult

from benchmarks.adapters.base import PipelineAdapter, ScenarioConfig
from benchmarks.core.result_model import throughput_elements_per_sec
from benchmarks.scenarios.base import run_scenario, ScenarioVariant


@dataclass(frozen=True, slots=True)
class PrefetchSweepPoint:
    """A single measurement point in a prefetch sensitivity sweep.

    Attributes:
        prefetch_size: Prefetch buffer depth used.
        throughput_elem_s: Elements per second achieved.
        wall_clock_sec: Total wall-clock time for the iteration.
        result: Full BenchmarkResult for detailed analysis.
    """

    prefetch_size: int
    throughput_elem_s: float
    wall_clock_sec: float
    result: BenchmarkResult


@dataclass(frozen=True, slots=True, kw_only=True)
class PrefetchSweepResult:
    """Complete results of a prefetch sensitivity sweep.

    Attributes:
        scenario_id: Scenario that was tested.
        variant_name: Variant name within the scenario.
        adapter_name: Name of the adapter used.
        points: Ordered list of sweep points (by prefetch_size).
        optimal_prefetch: Prefetch size that achieved highest throughput.
        speedup_vs_disabled: Ratio of best throughput to prefetch=0 throughput.
    """

    scenario_id: str
    variant_name: str
    adapter_name: str
    points: list[PrefetchSweepPoint] = field(default_factory=list)
    optimal_prefetch: int = 0
    speedup_vs_disabled: float = 1.0


def _override_prefetch(config: ScenarioConfig, prefetch_size: int) -> ScenarioConfig:
    """Create a new ScenarioConfig with a different prefetch_size in extra."""
    new_extra = {**config.extra, "prefetch_size": prefetch_size}
    return ScenarioConfig(
        scenario_id=config.scenario_id,
        dataset_size=config.dataset_size,
        element_shape=config.element_shape,
        batch_size=config.batch_size,
        transforms=list(config.transforms),
        num_workers=config.num_workers,
        seed=config.seed,
        extra=new_extra,
    )


def run_prefetch_sweep(
    adapter: PipelineAdapter,
    variant: ScenarioVariant,
    prefetch_sizes: tuple[int, ...] = (0, 2, 4, 8),
    num_batches: int = 50,
    warmup_batches: int = 5,
    num_repetitions: int = 3,
) -> PrefetchSweepResult:
    """Run a scenario at multiple prefetch sizes and return comparative results.

    Args:
        adapter: PipelineAdapter to test.
        variant: ScenarioVariant to run.
        prefetch_sizes: Prefetch buffer depths to test.
        num_batches: Batches per iteration.
        warmup_batches: Warmup batches before timing.
        num_repetitions: Repetitions per prefetch size (median selected).

    Returns:
        PrefetchSweepResult with all measurement points.
    """
    points: list[PrefetchSweepPoint] = []
    config = variant.config
    variant_name = config.extra.get("variant_name", "default")

    for pf_size in prefetch_sizes:
        modified_config = _override_prefetch(config, pf_size)
        modified_variant = ScenarioVariant(
            config=modified_config,
            data_generator=variant.data_generator,
        )
        result = run_scenario(
            adapter,
            modified_variant,
            num_batches=num_batches,
            warmup_batches=warmup_batches,
            num_repetitions=num_repetitions,
        )
        throughput = throughput_elements_per_sec(result)
        wall = result.timing.wall_clock_sec if result.timing else 0.0

        points.append(
            PrefetchSweepPoint(
                prefetch_size=pf_size,
                throughput_elem_s=throughput,
                wall_clock_sec=wall,
                result=result,
            )
        )

    # Find optimal and compute speedup
    best = max(points, key=lambda p: p.throughput_elem_s)
    baseline = next((p for p in points if p.prefetch_size == 0), points[0])
    speedup = (
        best.throughput_elem_s / baseline.throughput_elem_s
        if baseline.throughput_elem_s > 0
        else 1.0
    )

    return PrefetchSweepResult(
        scenario_id=config.scenario_id,
        variant_name=variant_name,
        adapter_name=adapter.name,
        points=sorted(points, key=lambda p: p.prefetch_size),
        optimal_prefetch=best.prefetch_size,
        speedup_vs_disabled=speedup,
    )
