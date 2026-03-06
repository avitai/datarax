"""HCV-2: Dense Prediction Pipeline / Segmentation (Heavy).

High-resolution segmentation/detection pipeline with multi-tensor elements
(image + mask). Exercises Datarax's pytree-aware transforms -- mask must be
resized with nearest-neighbor while image uses bilinear interpolation.

Design ref: Section 11.3 of the performance audit.
"""

from __future__ import annotations

import numpy as np

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, make_get_variant, ScenarioVariant


SCENARIO_ID: str = "HCV-2"
TIER1_VARIANT: str | None = "seg_small"


def _seg_data(n: int, h: int, w: int) -> dict[str, np.ndarray]:
    gen = SyntheticDataGenerator(seed=DEFAULT_SEED)
    images = gen.images(n, h, w, 3, dtype="uint8")
    masks = gen.rng.integers(0, 21, (n, h, w, 1), dtype=np.int32)
    return {"image": images, "mask": masks}


VARIANTS: dict[str, ScenarioVariant] = {
    "seg_small": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=10_000,
            element_shape=(256, 256, 3),
            batch_size=16,
            transforms=[
                "RandomResizedCrop",
                "RandomHorizontalFlip",
                "ColorJitter",
                "Normalize",
                "CastToFloat32",
            ],
            extra={"variant_name": "seg_small"},
        ),
        data_generator=lambda: _seg_data(10_000, 256, 256),
    ),
    "seg_medium": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=50_000,
            element_shape=(640, 640, 3),
            batch_size=8,
            transforms=[
                "RandomResizedCrop",
                "RandomHorizontalFlip",
                "ColorJitter",
                "Normalize",
                "CastToFloat32",
            ],
            extra={"variant_name": "seg_medium"},
        ),
        data_generator=lambda: _seg_data(50_000, 640, 640),
    ),
}


get_variant = make_get_variant(VARIANTS)
