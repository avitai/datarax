"""CV-3: Batch-Level Mixing (MixUp/CutMix).

Benchmarks batch-level augmentation pipelines that apply MixUp or CutMix
after batching. Tests the framework's ability to perform cross-sample
operations within a batch during data loading.

Design ref: Section 7.3.3 of the benchmark report.
"""

from __future__ import annotations

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, ScenarioVariant, make_get_variant

SCENARIO_ID: str = "CV-3"
TIER1_VARIANT: str | None = None

VARIANTS: dict[str, ScenarioVariant] = {
    "default": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=5_000,
            element_shape=(64, 64, 3),
            batch_size=128,
            transforms=["Normalize", "MixUp"],
            extra={"variant_name": "default"},
        ),
        data_generator=lambda: {
            "image": SyntheticDataGenerator(seed=DEFAULT_SEED).images(
                5_000, 64, 64, 3, dtype="uint8"
            )
        },
    ),
    "large": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=50_000,
            element_shape=(224, 224, 3),
            batch_size=128,
            transforms=["Normalize", "MixUp"],
            extra={"variant_name": "large"},
        ),
        data_generator=lambda: {
            "image": SyntheticDataGenerator(seed=DEFAULT_SEED).images(
                50_000, 224, 224, 3, dtype="uint8"
            )
        },
    ),
}


get_variant = make_get_variant(VARIANTS)
