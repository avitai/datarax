"""CV-3: Batch-Level Mixing (MixUp/CutMix).

Benchmarks batch-level augmentation pipelines that apply MixUp or CutMix
after batching. Tests the framework's ability to perform cross-sample
operations within a batch during data loading.

Design ref: Section 7.3.3 of the benchmark report.
"""

from __future__ import annotations

from benchmarks.adapters.base import Capability, ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, make_get_variant, ScenarioVariant
from benchmarks.scenarios.real_data_variants import cifar10_image_data


SCENARIO_ID: str = "CV-3"
TIER1_VARIANT: str | None = None

VARIANTS: dict[str, ScenarioVariant] = {
    "default": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=5_000,
            element_shape=(64, 64, 3),
            batch_size=128,
            transforms=["Normalize"],
            required_capabilities=[Capability.BATCH_MIXING],
            extra={"variant_name": "default", "mix_mode": "mixup", "mix_alpha": 0.4},
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
            transforms=["Normalize"],
            required_capabilities=[Capability.BATCH_MIXING],
            extra={"variant_name": "large", "mix_mode": "mixup", "mix_alpha": 0.4},
        ),
        data_generator=lambda: {
            "image": SyntheticDataGenerator(seed=DEFAULT_SEED).images(
                50_000, 224, 224, 3, dtype="uint8"
            )
        },
    ),
    "real_cifar10": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=5_000,
            element_shape=(64, 64, 3),
            batch_size=128,
            transforms=["Normalize"],
            required_capabilities=[Capability.BATCH_MIXING],
            extra={"variant_name": "real_cifar10", "mix_mode": "mixup", "mix_alpha": 0.4},
        ),
        data_generator=cifar10_image_data(5_000, h=64, w=64),
    ),
}


get_variant = make_get_variant(VARIANTS)
