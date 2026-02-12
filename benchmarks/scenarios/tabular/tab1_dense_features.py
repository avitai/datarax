"""TAB-1: Dense Features (Click-Through Rate) scenario.

Dense float tabular data with Normalize transform, simulating
click-through rate prediction workloads.

Design ref: Section 7 of the benchmark report.
"""

from __future__ import annotations

from typing import Any

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, ScenarioVariant, make_get_variant

SCENARIO_ID: str = "TAB-1"
TIER1_VARIANT: str | None = "small"

# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

_NUM_FEATURES = 100

_VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "small": {
        "dataset_size": 10_000,
        "batch_size": 256,
    },
    "medium": {
        "dataset_size": 1_000_000,
        "batch_size": 1024,
    },
}


def _make_data_generator(
    dataset_size: int, num_features: int, seed: int = DEFAULT_SEED
) -> callable:
    """Create a lazy data generator for dense tabular data."""

    def generate() -> dict[str, Any]:
        gen = SyntheticDataGenerator(seed=seed)
        return {"features": gen.tabular(dataset_size, num_features)}

    return generate


def _build_variants() -> dict[str, ScenarioVariant]:
    """Build all TAB-1 variants from specs."""
    variants: dict[str, ScenarioVariant] = {}
    for name, spec in _VARIANT_SPECS.items():
        config = ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=spec["dataset_size"],
            element_shape=(_NUM_FEATURES,),
            batch_size=spec["batch_size"],
            transforms=["Normalize"],
            extra={"variant_name": name},
        )
        variants[name] = ScenarioVariant(
            config=config,
            data_generator=_make_data_generator(spec["dataset_size"], _NUM_FEATURES),
        )
    return variants


VARIANTS: dict[str, ScenarioVariant] = _build_variants()


get_variant = make_get_variant(VARIANTS)
