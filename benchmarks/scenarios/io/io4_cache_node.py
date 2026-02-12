"""IO-4: Cache Node Effectiveness.

Measures the impact of a cache node placed between an expensive transform
and a cheap transform. By running 3 epochs, the first epoch pays the
full cost of ExpensiveTransform while subsequent epochs benefit from
the cache.

The standard get_variant() interface works for single-epoch adapter-based
benchmarking. The run_direct() function provides multi-epoch measurement
for cache effectiveness analysis.

Design ref: Section 7.3 of the benchmark report.
"""

from __future__ import annotations

from typing import Any

from benchmarks.adapters.base import BenchmarkAdapter, ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, ScenarioVariant, make_get_variant, run_scenario
from datarax.benchmarking.results import BenchmarkResult

SCENARIO_ID: str = "IO-4"
TIER1_VARIANT: str | None = None

_DATASET_SIZE = 5_000
_ELEMENT_SHAPE = (64, 64, 3)
_BATCH_SIZE = 64
_TRANSFORMS = ["ExpensiveTransform", "Cache", "CheapTransform"]
_NUM_EPOCHS = 3


def _make_cache_data() -> dict:
    """Generate float32 image data for the cache benchmark."""
    gen = SyntheticDataGenerator(seed=DEFAULT_SEED)
    return {"image": gen.images(_DATASET_SIZE, *_ELEMENT_SHAPE, dtype="float32")}


VARIANTS: dict[str, ScenarioVariant] = {
    "default": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=_DATASET_SIZE,
            element_shape=_ELEMENT_SHAPE,
            batch_size=_BATCH_SIZE,
            transforms=_TRANSFORMS,
            seed=DEFAULT_SEED,
            extra={
                "variant_name": "default",
                "num_epochs": _NUM_EPOCHS,
            },
        ),
        data_generator=_make_cache_data,
    ),
}


get_variant = make_get_variant(VARIANTS)


def run_direct(
    adapter: BenchmarkAdapter,
    num_batches: int = 50,
    warmup_batches: int = 5,
) -> dict[str, Any]:
    """Run multi-epoch cache effectiveness benchmark.

    Runs the default variant for ``num_epochs`` epochs (defined in config)
    and returns per-epoch BenchmarkResults to allow cache hit-rate analysis.

    Args:
        adapter: Benchmark adapter to test.
        num_batches: Number of batches per epoch.
        warmup_batches: Warmup batches before timing (first epoch only).

    Returns:
        Dict with keys:
            - "epoch_results": list of BenchmarkResult per epoch
            - "num_epochs": total epochs run
            - "speedup_ratio": epoch_N throughput / epoch_1 throughput
    """
    variant = VARIANTS["default"]
    num_epochs = variant.config.extra["num_epochs"]

    epoch_results: list[BenchmarkResult] = []
    for epoch in range(num_epochs):
        wb = warmup_batches if epoch == 0 else 0
        result = run_scenario(
            adapter,
            variant,
            num_batches=num_batches,
            warmup_batches=wb,
            num_repetitions=1,
        )
        epoch_results.append(result)

    first_throughput = epoch_results[0].throughput_elements_sec()
    last_throughput = epoch_results[-1].throughput_elements_sec()
    speedup = last_throughput / first_throughput if first_throughput > 0 else 0.0

    return {
        "epoch_results": epoch_results,
        "num_epochs": num_epochs,
        "speedup_ratio": speedup,
    }
