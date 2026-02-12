"""DIST-1: Multi-Device Sharding & Prefetch.

Benchmarks data pipeline throughput when sharding across multiple JAX devices.
Each variant increases the device count: 1, 2, 4, 8. The adapter is responsible
for setting XLA_FLAGS and configuring device placement; this module only defines
the scenario configurations and data generators.

Design ref: Section 7.3.7 of the benchmark report.
"""

from __future__ import annotations

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, ScenarioVariant, make_get_variant

SCENARIO_ID: str = "DIST-1"
TIER1_VARIANT: str | None = None

_DEVICE_COUNTS = [1, 2, 4, 8]
_DATASET_SIZE = 10_000
_ELEMENT_SHAPE = (32, 32, 3)
_BATCH_SIZE = 64


def _make_data_generator(seed: int = DEFAULT_SEED) -> dict:
    gen = SyntheticDataGenerator(seed=seed)
    return {"image": gen.images(_DATASET_SIZE, *_ELEMENT_SHAPE, dtype="float32")}


VARIANTS: dict[str, ScenarioVariant] = {}

for _n in _DEVICE_COUNTS:
    _name = f"{_n}_device{'s' if _n > 1 else ''}"
    VARIANTS[_name] = ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=_DATASET_SIZE,
            element_shape=_ELEMENT_SHAPE,
            batch_size=_BATCH_SIZE,
            transforms=["Normalize"],
            seed=DEFAULT_SEED,
            extra={"device_count": _n, "variant_name": _name},
        ),
        data_generator=lambda: _make_data_generator(),
    )

# Clean up loop variables from module namespace
del _n, _name


get_variant = make_get_variant(VARIANTS)
