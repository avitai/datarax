"""HCV-1: ImageNet-Scale Image Classification (Heavy).

Production-realistic CV benchmark with compute-heavy transforms that stress
the GPU pipeline: RandomResizedCrop (bilinear interpolation), ColorJitter
(per-channel ops), and full augmentation chains. This is the standard
benchmark used by DALI, tf.data, and PyTorch DataLoader.

At 224x224x3 with 5 transforms, the arithmetic intensity (~200 FLOPs/pixel)
is high enough that Datarax's fused JIT compilation should outperform
sequential CPU frameworks.

Design ref: Section 11.3 of the performance audit.
"""

from __future__ import annotations

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, make_get_variant, ScenarioVariant


SCENARIO_ID: str = "HCV-1"
TIER1_VARIANT: str | None = "imagenet_small"

VARIANTS: dict[str, ScenarioVariant] = {
    "imagenet_small": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=50_000,
            element_shape=(224, 224, 3),
            batch_size=256,
            transforms=[
                "RandomResizedCrop",
                "RandomHorizontalFlip",
                "ColorJitter",
                "Normalize",
                "CastToFloat32",
            ],
            extra={"variant_name": "imagenet_small"},
        ),
        data_generator=lambda: {
            "image": SyntheticDataGenerator(seed=DEFAULT_SEED).images(
                50_000, 224, 224, 3, dtype="uint8"
            )
        },
    ),
    "imagenet_medium": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=200_000,
            element_shape=(224, 224, 3),
            batch_size=512,
            transforms=[
                "RandomResizedCrop",
                "RandomHorizontalFlip",
                "ColorJitter",
                "Normalize",
                "CastToFloat32",
            ],
            extra={"variant_name": "imagenet_medium"},
        ),
        data_generator=lambda: {
            "image": SyntheticDataGenerator(seed=DEFAULT_SEED).images(
                200_000, 224, 224, 3, dtype="uint8"
            )
        },
    ),
}


get_variant = make_get_variant(VARIANTS)
