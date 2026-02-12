"""PR-2: Multi-Epoch Determinism Verification.

Verifies that the data pipeline produces identical output across multiple
runs and epochs when given the same seed. The "small" variant (2 runs,
2 epochs) is Tier-1 for CI regression; "full" (3 runs, 5 epochs) is Tier-2.

Design ref: Section 7.3.8 of the benchmark report.
"""

from __future__ import annotations

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, ScenarioVariant, make_get_variant

SCENARIO_ID: str = "PR-2"
TIER1_VARIANT: str | None = "small"

_DATASET_SIZE = 5_000
_ELEMENT_SHAPE = (64, 64, 3)
_BATCH_SIZE = 64


def _make_data_generator(seed: int = DEFAULT_SEED) -> dict:
    gen = SyntheticDataGenerator(seed=seed)
    return {"image": gen.images(_DATASET_SIZE, *_ELEMENT_SHAPE, dtype="uint8")}


VARIANTS: dict[str, ScenarioVariant] = {
    "small": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=_DATASET_SIZE,
            element_shape=_ELEMENT_SHAPE,
            batch_size=_BATCH_SIZE,
            transforms=["RandomCrop", "RandomFlip"],
            seed=DEFAULT_SEED,
            extra={
                "num_runs": 2,
                "num_epochs": 2,
                "variant_name": "small",
            },
        ),
        data_generator=lambda: _make_data_generator(),
    ),
    "full": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=_DATASET_SIZE,
            element_shape=_ELEMENT_SHAPE,
            batch_size=_BATCH_SIZE,
            transforms=["RandomCrop", "RandomFlip"],
            seed=DEFAULT_SEED,
            extra={
                "num_runs": 3,
                "num_epochs": 5,
                "variant_name": "full",
            },
        ),
        data_generator=lambda: _make_data_generator(),
    ),
}


get_variant = make_get_variant(VARIANTS)
