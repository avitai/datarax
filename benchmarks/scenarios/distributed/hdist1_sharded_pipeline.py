"""HDIST-1: Multi-Device Sharded Data Pipeline (Heavy).

Tests pipeline scaling across devices with proper SPMD sharding. Global
batch of 4096 across 8 devices = 512 per device. Measures scaling
efficiency as device count increases.

Uses standard ImageNet-scale data with GPU-heavy transforms.

Design ref: Section 11.3 of the performance audit.
"""

from __future__ import annotations

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, make_get_variant, ScenarioVariant


SCENARIO_ID: str = "HDIST-1"
TIER1_VARIANT: str | None = "shard_small"

VARIANTS: dict[str, ScenarioVariant] = {
    "shard_small": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=50_000,
            element_shape=(224, 224, 3),
            batch_size=256,
            transforms=["RandomResizedCrop", "Normalize", "CastToFloat32"],
            extra={"variant_name": "shard_small"},
        ),
        data_generator=lambda: {
            "image": SyntheticDataGenerator(seed=DEFAULT_SEED).images(
                50_000, 224, 224, 3, dtype="uint8"
            )
        },
    ),
    "shard_medium": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=200_000,
            element_shape=(224, 224, 3),
            batch_size=1024,
            transforms=["RandomResizedCrop", "Normalize", "CastToFloat32"],
            extra={"variant_name": "shard_medium"},
        ),
        data_generator=lambda: {
            "image": SyntheticDataGenerator(seed=DEFAULT_SEED).images(
                200_000, 224, 224, 3, dtype="uint8"
            )
        },
    ),
}


get_variant = make_get_variant(VARIANTS)
