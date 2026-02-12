"""CV-2: High-Resolution Medical Imaging (3D).

Benchmarks 3D volumetric data loading with large (128x128x128) float32 volumes.
Stresses memory bandwidth and batch assembly for medical-imaging workloads.
Not a Tier-1 scenario due to high memory requirements.

Design ref: Section 7.3.2 of the benchmark report.
"""

from __future__ import annotations

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, ScenarioVariant, make_get_variant

SCENARIO_ID: str = "CV-2"
TIER1_VARIANT: str | None = None

VARIANTS: dict[str, ScenarioVariant] = {
    "default": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=500,
            element_shape=(64, 64, 64),
            batch_size=2,
            transforms=["Normalize"],
            extra={"variant_name": "default"},
        ),
        data_generator=lambda: {
            "volume": SyntheticDataGenerator(seed=DEFAULT_SEED).volumes_3d(500, 64, 64, 64)
        },
    ),
    "large": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=5_000,
            element_shape=(128, 128, 128),
            batch_size=2,
            transforms=["Normalize"],
            extra={"variant_name": "large"},
        ),
        data_generator=lambda: {
            "volume": SyntheticDataGenerator(seed=DEFAULT_SEED).volumes_3d(5_000, 128, 128, 128)
        },
    ),
}


get_variant = make_get_variant(VARIANTS)
