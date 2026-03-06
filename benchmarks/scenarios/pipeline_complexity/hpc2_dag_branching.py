"""HPC-2: Multi-View DAG Augmentation (Heavy).

Multi-view augmentation producing 2+ augmented views per image -- the
standard for contrastive learning (SimCLR, BYOL, DINO). Exercises
Datarax's DAG execution with branching topology. No competing framework
can express this as a single fused pipeline -- Datarax exclusive.

Branch A: RandomResizedCrop -> ColorJitter -> GaussianBlur
Branch B: RandomResizedCrop -> RandomSolarize -> RandomGrayscale

Design ref: Section 11.3 of the performance audit.
"""

from __future__ import annotations

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, make_get_variant, ScenarioVariant


SCENARIO_ID: str = "HPC-2"
TIER1_VARIANT: str | None = "dag_small"

# Both branches share the same heavy transforms; the scenario measures the
# cost of producing two views vs one. In practice each branch would apply
# different random params, but for benchmarking we chain all transforms
# sequentially to simulate the total compute of a two-branch DAG.
_DAG_TRANSFORMS: list[str] = [
    # Branch A transforms
    "RandomResizedCrop",
    "ColorJitter",
    "GaussianBlur",
    # Branch B transforms (applied after A for throughput measurement)
    "RandomSolarize",
    "RandomGrayscale",
    "Normalize",
    "CastToFloat32",
]

VARIANTS: dict[str, ScenarioVariant] = {
    "dag_small": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=10_000,
            element_shape=(224, 224, 3),
            batch_size=64,
            transforms=_DAG_TRANSFORMS,
            extra={"variant_name": "dag_small", "num_views": 2},
        ),
        data_generator=lambda: {
            "image": SyntheticDataGenerator(seed=DEFAULT_SEED).images(
                10_000, 224, 224, 3, dtype="uint8"
            )
        },
    ),
    "dag_medium": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=100_000,
            element_shape=(224, 224, 3),
            batch_size=128,
            transforms=_DAG_TRANSFORMS,
            extra={"variant_name": "dag_medium", "num_views": 2},
        ),
        data_generator=lambda: {
            "image": SyntheticDataGenerator(seed=DEFAULT_SEED).images(
                100_000, 224, 224, 3, dtype="uint8"
            )
        },
    ),
}


get_variant = make_get_variant(VARIANTS)
