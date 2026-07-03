"""IO-3: Mixed-Source Pipeline.

Benchmarks a pipeline that mixes two separate data sources into a single
weighted stream. This scenario exercises the MixDataSourcesNode topology via
the MIXED_SOURCE capability.

Design ref: Section 7.3 of the benchmark report.
"""

from __future__ import annotations

from benchmarks.adapters.base import Capability, ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, make_get_variant, ScenarioVariant


SCENARIO_ID: str = "IO-3"
TIER1_VARIANT: str | None = None

_DATASET_SIZE = 5_000
_ELEMENT_SHAPE = (64, 64, 3)
_BATCH_SIZE = 64
_TRANSFORMS = ["Normalize"]


def _make_mixed_data() -> dict:
    """Generate image + label arrays for the mixed-source scenario."""
    gen = SyntheticDataGenerator(seed=DEFAULT_SEED)
    images = gen.images(_DATASET_SIZE, *_ELEMENT_SHAPE, dtype="float32")
    labels = gen.token_sequences(_DATASET_SIZE, 1)
    return {"image": images, "label": labels}


VARIANTS: dict[str, ScenarioVariant] = {
    "default": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=_DATASET_SIZE,
            element_shape=_ELEMENT_SHAPE,
            batch_size=_BATCH_SIZE,
            transforms=_TRANSFORMS,
            required_capabilities=[Capability.MIXED_SOURCE],
            seed=DEFAULT_SEED,
            extra={
                "variant_name": "default",
                "topology": "mixed_source",
            },
        ),
        data_generator=_make_mixed_data,
    ),
}


get_variant = make_get_variant(VARIANTS)
