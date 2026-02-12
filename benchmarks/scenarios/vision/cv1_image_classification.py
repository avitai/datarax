"""CV-1: Image Classification Pipeline (Canonical).

The standard image classification benchmark: loads images in NHWC format,
applies Normalize + CastToFloat32 transforms, and batches for training.
This is the Tier-1 canonical vision scenario used in CI regression guards.

Design ref: Section 7.3.1 of the benchmark report.
"""

from __future__ import annotations

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, ScenarioVariant, make_get_variant

SCENARIO_ID: str = "CV-1"
TIER1_VARIANT: str | None = "small"

VARIANTS: dict[str, ScenarioVariant] = {
    "small": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=10_000,
            element_shape=(32, 32, 3),
            batch_size=64,
            transforms=["Normalize", "CastToFloat32"],
            extra={"variant_name": "small"},
        ),
        data_generator=lambda: {
            "image": SyntheticDataGenerator(seed=DEFAULT_SEED).images(
                10_000, 32, 32, 3, dtype="uint8"
            )
        },
    ),
    "medium": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=5_000,
            element_shape=(128, 128, 3),
            batch_size=128,
            transforms=["Normalize", "CastToFloat32"],
            extra={"variant_name": "medium"},
        ),
        data_generator=lambda: {
            "image": SyntheticDataGenerator(seed=DEFAULT_SEED).images(
                5_000, 128, 128, 3, dtype="uint8"
            )
        },
    ),
    "large": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=50_000,
            element_shape=(256, 256, 3),
            batch_size=64,
            transforms=["Normalize", "CastToFloat32"],
            extra={"variant_name": "large"},
        ),
        data_generator=lambda: {
            "image": SyntheticDataGenerator(seed=DEFAULT_SEED).images(
                50_000, 256, 256, 3, dtype="uint8"
            )
        },
    ),
}


get_variant = make_get_variant(VARIANTS)
