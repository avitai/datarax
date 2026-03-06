"""HPC-1: SSL/Contrastive Learning Augmentation Chain (Heavy).

The standard self-supervised learning augmentation pipeline (SimCLR/BYOL/DINO):
8-operator chain of compute-heavy image transforms. This is where Datarax's
fused JIT compilation should massively outperform sequential CPU frameworks --
XLA compiles the entire 8-op chain into a single kernel, eliminating 7
intermediate memory allocations.

Each operator adds ~10-50 FLOPs/pixel, totaling ~200+ FLOPs/pixel for the
full chain. On A100, this is clearly compute-bound.

Design ref: Section 11.3 of the performance audit.
"""

from __future__ import annotations

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, make_get_variant, ScenarioVariant


SCENARIO_ID: str = "HPC-1"
TIER1_VARIANT: str | None = "ssl_small"

_SSL_TRANSFORMS: list[str] = [
    "RandomResizedCrop",
    "RandomHorizontalFlip",
    "ColorJitter",
    "RandomGrayscale",
    "GaussianBlur",
    "RandomSolarize",
    "Normalize",
    "CastToFloat32",
]
"""Standard 8-transform SSL augmentation chain (SimCLR/BYOL/DINO)."""

VARIANTS: dict[str, ScenarioVariant] = {
    "ssl_small": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=10_000,
            element_shape=(224, 224, 3),
            batch_size=64,
            transforms=_SSL_TRANSFORMS,
            extra={"variant_name": "ssl_small", "chain_depth": 8},
        ),
        data_generator=lambda: {
            "image": SyntheticDataGenerator(seed=DEFAULT_SEED).images(
                10_000, 224, 224, 3, dtype="uint8"
            )
        },
    ),
    "ssl_medium": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=50_000,
            element_shape=(224, 224, 3),
            batch_size=128,
            transforms=_SSL_TRANSFORMS,
            extra={"variant_name": "ssl_medium", "chain_depth": 8},
        ),
        data_generator=lambda: {
            "image": SyntheticDataGenerator(seed=DEFAULT_SEED).images(
                50_000, 224, 224, 3, dtype="uint8"
            )
        },
    ),
    "ssl_large": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=200_000,
            element_shape=(224, 224, 3),
            batch_size=256,
            transforms=_SSL_TRANSFORMS,
            extra={"variant_name": "ssl_large", "chain_depth": 8},
        ),
        data_generator=lambda: {
            "image": SyntheticDataGenerator(seed=DEFAULT_SEED).images(
                200_000, 224, 224, 3, dtype="uint8"
            )
        },
    ),
}


get_variant = make_get_variant(VARIANTS)
