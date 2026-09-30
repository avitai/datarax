"""Structural performance gate: every Tier-1 scenario compiles nothing in steady state.

A compile or retrace in the timed region of a benchmark is exactly countable and independent
of the machine, where a throughput number on a shared runner is neither. For each Tier-1
scenario this runs the Datarax adapter at the scenario's own batch size, element shape and
transforms, over a dataset holding exactly the batches iterated, and requires that after warmup
the timed iteration builds no executable.

The scenarios are read from :func:`~benchmarks.scenarios.discover_scenarios`, so a scenario added
to Tier 1 is gated without editing this module.
"""

from __future__ import annotations

from types import ModuleType

import jax
import pytest
from substrax.testing.compiles import expect_compiles

from benchmarks.adapters.datarax_adapter import DataraxAdapter
from benchmarks.scenarios import discover_scenarios
from benchmarks.scenarios.base import ScenarioVariant


WARMUP_BATCHES = 1
"""Batches served before the timed region: one call builds every program the step needs."""

TIMED_BATCHES = 3
"""Batches in the timed region, each of which must reuse the warmed programs."""

TIER1_SCENARIOS: list[ModuleType] = discover_scenarios(tier=1)


def _tier1_variant(module: ModuleType) -> ScenarioVariant:
    return module.get_variant(module.TIER1_VARIANT)


def _shrunk(variant: ScenarioVariant) -> ScenarioVariant:
    """The variant with a dataset of exactly the warmup and timed batches."""
    batches = WARMUP_BATCHES + TIMED_BATCHES
    return variant.with_dataset_size(batches * variant.config.batch_size)


def test_tier1_is_not_empty() -> None:
    assert TIER1_SCENARIOS, "discover_scenarios(tier=1) found no scenario to gate"


@pytest.mark.parametrize(
    "module", TIER1_SCENARIOS, ids=[module.SCENARIO_ID for module in TIER1_SCENARIOS]
)
def test_steady_state_iteration_compiles_nothing(module: ModuleType) -> None:
    adapter = DataraxAdapter()
    assert adapter.supports_scenario(module.SCENARIO_ID)
    variant = _shrunk(_tier1_variant(module))
    config = variant.config
    data = variant.generate_data()
    leaves = jax.tree.leaves(data)
    assert all(leaf.shape[0] == config.dataset_size for leaf in leaves)
    assert any(tuple(leaf.shape[1:]) == tuple(config.element_shape) for leaf in leaves)

    # jax's in-memory caches are process-wide: a program an earlier test compiled at the same
    # shapes would be a cache hit here and hide a recompile. From a cold cache, warmup builds
    # every program the step needs and the timed region must build none.
    jax.clear_caches()
    adapter.setup(config, data)
    try:
        adapter.warmup(WARMUP_BATCHES, timed_batches=TIMED_BATCHES)
        with expect_compiles(0):
            result = adapter.iterate(TIMED_BATCHES)
    finally:
        adapter.teardown()

    assert result.num_batches == TIMED_BATCHES
    assert result.num_elements == TIMED_BATCHES * config.batch_size
