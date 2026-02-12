"""DIST-2: Device Mesh Configuration.

Benchmarks data pipeline behavior under different JAX device mesh topologies.
Variants test 1D and 2D mesh configurations. The adapter handles actual mesh
setup; this module defines the scenario configurations and data generators.

Design ref: Section 7.3.7 of the benchmark report.
"""

from __future__ import annotations

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, ScenarioVariant, make_get_variant

SCENARIO_ID: str = "DIST-2"
TIER1_VARIANT: str | None = None

_DATASET_SIZE = 10_000
_ELEMENT_SHAPE = (32, 32, 3)
_BATCH_SIZE = 64


def _make_data_generator(seed: int = DEFAULT_SEED) -> dict:
    gen = SyntheticDataGenerator(seed=seed)
    return {"image": gen.images(_DATASET_SIZE, *_ELEMENT_SHAPE, dtype="float32")}


VARIANTS: dict[str, ScenarioVariant] = {
    "mesh_1d": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=_DATASET_SIZE,
            element_shape=_ELEMENT_SHAPE,
            batch_size=_BATCH_SIZE,
            transforms=["Normalize"],
            seed=DEFAULT_SEED,
            extra={"mesh_topology": "1d", "variant_name": "mesh_1d"},
        ),
        data_generator=lambda: _make_data_generator(),
    ),
    "mesh_2d": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=_DATASET_SIZE,
            element_shape=_ELEMENT_SHAPE,
            batch_size=_BATCH_SIZE,
            transforms=["Normalize"],
            seed=DEFAULT_SEED,
            extra={"mesh_topology": "2d", "variant_name": "mesh_2d"},
        ),
        data_generator=lambda: _make_data_generator(),
    ),
}


get_variant = make_get_variant(VARIANTS)
