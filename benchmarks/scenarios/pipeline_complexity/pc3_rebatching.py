"""PC-3: Differentiable Rebatching.

Measures overhead of rebatching from one batch size to another within
a pipeline, e.g. loading at batch_size=256 then reshaping to 32 for
model consumption.

Single variant: 50K float32 images (256, 256, 3), batch_size=256,
target_batch_size=32.
Not a Tier-1 scenario.

Design ref: Section 7.4.3 of the benchmark report.
"""

from __future__ import annotations

from typing import Any

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, ScenarioVariant, make_get_variant

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

SCENARIO_ID: str = "PC-3"
"""Unique scenario identifier."""

TIER1_VARIANT: str | None = None
"""PC-3 is not a Tier-1 scenario."""

_DATASET_SIZE: int = 5_000
_ELEMENT_SHAPE: tuple[int, int, int] = (64, 64, 3)
_BATCH_SIZE: int = 256
_TARGET_BATCH_SIZE: int = 32


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
            transforms=["Normalize"],
            seed=DEFAULT_SEED,
            extra={
                "target_batch_size": _TARGET_BATCH_SIZE,
                "variant_name": "default",
            },
        ),
        data_generator=_make_data_generator(_DATASET_SIZE, *_ELEMENT_SHAPE, DEFAULT_SEED),
    ),
}
"""Single default variant for rebatching benchmarking."""


get_variant = make_get_variant(VARIANTS)
