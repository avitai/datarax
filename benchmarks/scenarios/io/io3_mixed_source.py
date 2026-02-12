"""IO-3: Mixed-Source Pipeline.

Benchmarks a pipeline that merges two separate data sources (images and
labels) into a single stream. This scenario exercises the MixDataSourcesNode
topology.

BLOCKED: MixDataSourcesNode is not yet implemented. The scenario module
is fully defined so that it can be unblocked without code changes once the
operator lands.

Design ref: Section 7.3 of the benchmark report.
"""

from __future__ import annotations

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, ScenarioVariant, make_get_variant

SCENARIO_ID: str = "IO-3"
TIER1_VARIANT: str | None = None

BLOCKED: bool = True
BLOCKED_REASON: str = "MixDataSourcesNode not yet implemented"

_DATASET_SIZE = 5_000
_ELEMENT_SHAPE = (64, 64, 3)
_BATCH_SIZE = 64
_TRANSFORMS = ["Normalize", "Merge"]


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
