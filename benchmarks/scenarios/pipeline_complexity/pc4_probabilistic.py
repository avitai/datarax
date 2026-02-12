"""PC-4: Probabilistic & Conditional Pipeline.

Measures overhead of probabilistic augmentation and conditional transform
selection within a data pipeline. Transforms are applied with a given
probability, introducing control-flow branching at the pipeline level.

Single variant: 50K float32 images (64, 64, 3), batch_size=64,
probability=0.5.
Not a Tier-1 scenario.

Design ref: Section 7.4.4 of the benchmark report.
"""

from __future__ import annotations

from typing import Any

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, ScenarioVariant, make_get_variant

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

SCENARIO_ID: str = "PC-4"
"""Unique scenario identifier."""

TIER1_VARIANT: str | None = None
"""PC-4 is not a Tier-1 scenario."""

_DATASET_SIZE: int = 50_000
_ELEMENT_SHAPE: tuple[int, int, int] = (64, 64, 3)
_BATCH_SIZE: int = 64
_PROBABILITY: float = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data_generator(
    dataset_size: int,
    h: int,
    w: int,
    c: int,
    seed: int,
) -> callable:
    """Return a lazy data generator for float32 images."""

    def _generate() -> dict[str, Any]:
        gen = SyntheticDataGenerator(seed=seed)
        return {"image": gen.images(dataset_size, h, w, c, dtype="float32")}

    return _generate


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

VARIANTS: dict[str, ScenarioVariant] = {
    "default": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=_DATASET_SIZE,
            element_shape=_ELEMENT_SHAPE,
            batch_size=_BATCH_SIZE,
            transforms=["ProbabilisticAugment", "ConditionalSelect"],
            seed=DEFAULT_SEED,
            extra={
                "probability": _PROBABILITY,
                "variant_name": "default",
            },
        ),
        data_generator=_make_data_generator(_DATASET_SIZE, *_ELEMENT_SHAPE, DEFAULT_SEED),
    ),
}
"""Single default variant for probabilistic pipeline benchmarking."""


get_variant = make_get_variant(VARIANTS)
