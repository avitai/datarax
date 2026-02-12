"""PR-1: Checkpoint Save/Restore Cycle.

Benchmarks the overhead of saving and restoring pipeline state during training.
Three variants test different data scales: small (CIFAR-sized), medium
(ImageNet-sized), and large (3D volumetric). The adapter handles actual
checkpoint I/O; this module defines configurations and data generators.

Design ref: Section 7.3.8 of the benchmark report.
"""

from __future__ import annotations

import numpy as np

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, ScenarioVariant, make_get_variant

SCENARIO_ID: str = "PR-1"
TIER1_VARIANT: str | None = None

VARIANTS: dict[str, ScenarioVariant] = {
    "small": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=10_000,
            element_shape=(32, 32, 3),
            batch_size=64,
            transforms=["Normalize"],
            seed=DEFAULT_SEED,
            extra={"checkpoint": True, "variant_name": "small"},
        ),
        data_generator=lambda: {
            "image": SyntheticDataGenerator(seed=DEFAULT_SEED).images(
                10_000, 32, 32, 3, dtype="float32"
            )
        },
    ),
    "medium": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=5_000,
            element_shape=(64, 64, 3),
            batch_size=64,
            transforms=["Normalize"],
            seed=DEFAULT_SEED,
            extra={"checkpoint": True, "variant_name": "medium"},
        ),
        data_generator=lambda: {
            "image": SyntheticDataGenerator(seed=DEFAULT_SEED).images(
                5_000, 64, 64, 3, dtype="float32"
            )
        },
    ),
    "large": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=500,
            element_shape=(64, 64, 64),
            batch_size=4,
            transforms=["Normalize"],
            seed=DEFAULT_SEED,
            extra={"checkpoint": True, "variant_name": "large"},
        ),
        data_generator=lambda: {
            "volume": np.random.default_rng(DEFAULT_SEED)
            .standard_normal((500, 64, 64, 64))
            .astype(np.float32)
        },
    ),
}


get_variant = make_get_variant(VARIANTS)
