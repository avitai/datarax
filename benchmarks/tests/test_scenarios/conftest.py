"""Shared helpers for benchmark scenario tests.

Provides reusable validation and execution helpers used across all 9
scenario test modules, eliminating duplication (DRY principle).
"""

from __future__ import annotations

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.adapters.datarax_adapter import DataraxAdapter
from benchmarks.scenarios.base import ScenarioVariant, run_scenario
from datarax.benchmarking.results import BenchmarkResult


def assert_valid_variant(variant: ScenarioVariant) -> None:
    """Assert that a ScenarioVariant has all required fields populated.

    Checks:
        - config is a ScenarioConfig
        - scenario_id is non-empty
        - dataset_size > 0
        - element_shape is non-empty
        - batch_size > 0
        - transforms is a list
        - extra contains 'variant_name'
    """
    cfg = variant.config
    assert isinstance(cfg, ScenarioConfig)
    assert isinstance(cfg.scenario_id, str) and len(cfg.scenario_id) > 0
    assert cfg.dataset_size > 0
    assert len(cfg.element_shape) > 0
    assert cfg.batch_size > 0
    assert isinstance(cfg.transforms, list)
    assert "variant_name" in cfg.extra


def run_quick_scenario(
    adapter: DataraxAdapter,
    variant: ScenarioVariant,
) -> BenchmarkResult:
    """Run a scenario with minimal batches for speed.

    Uses 3 batches, 1 warmup, 1 repetition — fast enough for unit tests
    while still exercising the full adapter lifecycle.
    """
    return run_scenario(
        adapter,
        variant,
        num_batches=3,
        warmup_batches=1,
        num_repetitions=1,
    )
