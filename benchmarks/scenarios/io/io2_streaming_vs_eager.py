"""IO-2: Streaming vs. Eager Loading.

Measures data loading throughput at three dataset sizes (10K, 50K, 200K)
to understand how eager vs. streaming modes scale. With MemorySource
both modes behave identically; the distinction becomes
meaningful when alternative adapters (TFDS, HF) are plugged in.

Design ref: Section 7.3 of the benchmark report.
"""

from __future__ import annotations

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, ScenarioVariant, make_get_variant

SCENARIO_ID: str = "IO-2"
TIER1_VARIANT: str | None = None

_ELEMENT_SHAPE = (64, 64, 3)
_BATCH_SIZE = 64
_TRANSFORMS = ["Normalize"]


# Size tier definitions: name -> dataset_size
_SIZE_TIERS: dict[str, int] = {
    "10k": 10_000,
    "50k": 50_000,
    "200k": 200_000,
}


def _make_data_generator(n: int):
    """Return a lazy data generator for *n* float32 images."""

    def _generate() -> dict:
        gen = SyntheticDataGenerator(seed=DEFAULT_SEED)
        return {"image": gen.images(n, 64, 64, 3, dtype="float32")}

    return _generate


VARIANTS: dict[str, ScenarioVariant] = {
    name: ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=size,
            element_shape=_ELEMENT_SHAPE,
            batch_size=_BATCH_SIZE,
            transforms=_TRANSFORMS,
            seed=DEFAULT_SEED,
            extra={"variant_name": name},
        ),
        data_generator=_make_data_generator(size),
    )
    for name, size in _SIZE_TIERS.items()
}


get_variant = make_get_variant(VARIANTS)
