"""CV-4: Multi-Resolution Pipeline.

Benchmarks multi-scale resize pipelines where images are resized to
multiple target resolutions during data loading. Tests the framework's
transform throughput on high-resolution inputs.

Design ref: Section 7.3.4 of the benchmark report.
"""

from __future__ import annotations

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, ScenarioVariant, make_get_variant

SCENARIO_ID: str = "CV-4"
TIER1_VARIANT: str | None = None

VARIANTS: dict[str, ScenarioVariant] = {
    "default": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=2_000,
            element_shape=(128, 128, 3),
            batch_size=64,
            transforms=["MultiScaleResize", "Normalize"],
            extra={"variant_name": "default"},
        ),
        data_generator=lambda: {
            "image": SyntheticDataGenerator(seed=DEFAULT_SEED).images(
                2_000, 128, 128, 3, dtype="uint8"
            )
        },
    ),
    "large": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=50_000,
            element_shape=(512, 512, 3),
            batch_size=64,
            transforms=["MultiScaleResize", "Normalize"],
            extra={"variant_name": "large"},
        ),
        data_generator=lambda: {
            "image": SyntheticDataGenerator(seed=DEFAULT_SEED).images(
                50_000, 512, 512, 3, dtype="uint8"
            )
        },
    ),
}


get_variant = make_get_variant(VARIANTS)
