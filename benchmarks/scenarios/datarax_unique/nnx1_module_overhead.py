"""NNX-1: Flax NNX Module Integration Overhead.

Measures the overhead of routing data through a Flax NNX module versus a
plain Python function. Both variants use the same dataset and transforms
so timing differences isolate the NNX integration cost.

Design ref: Section 7.3.9 of the benchmark report.
"""

from __future__ import annotations

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, ScenarioVariant, make_get_variant

SCENARIO_ID: str = "NNX-1"
TIER1_VARIANT: str | None = None

_DATASET_SIZE = 5_000
_ELEMENT_SHAPE = (64, 64, 3)
_BATCH_SIZE = 128


def _make_data_generator(seed: int = DEFAULT_SEED) -> dict:
    gen = SyntheticDataGenerator(seed=seed)
    return {"image": gen.images(_DATASET_SIZE, *_ELEMENT_SHAPE, dtype="float32")}


VARIANTS: dict[str, ScenarioVariant] = {
    "nnx_module": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=_DATASET_SIZE,
            element_shape=_ELEMENT_SHAPE,
            batch_size=_BATCH_SIZE,
            transforms=["Normalize", "RandomCrop"],
            seed=DEFAULT_SEED,
            extra={"mode": "nnx", "variant_name": "nnx_module"},
        ),
        data_generator=lambda: _make_data_generator(),
    ),
    "plain_function": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=_DATASET_SIZE,
            element_shape=_ELEMENT_SHAPE,
            batch_size=_BATCH_SIZE,
            transforms=["Normalize", "RandomCrop"],
            seed=DEFAULT_SEED,
            extra={"mode": "plain", "variant_name": "plain_function"},
        ),
        data_generator=lambda: _make_data_generator(),
    ),
}


get_variant = make_get_variant(VARIANTS)
