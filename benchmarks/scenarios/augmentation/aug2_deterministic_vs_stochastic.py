"""AUG-2: Deterministic vs Stochastic Comparison.

Direct A/B comparison of the same pipeline structure with deterministic
vs stochastic transforms. Both variants use identical dataset size,
element shape, and batch size. The only difference is the transform list.

This isolates the throughput cost of the stochastic JIT path
(argument-passing with to_tree/from_tree) versus the deterministic
path (closure-capture, zero per-call overhead).

Not a Tier-1 scenario (requires both variants for meaningful results).

Design ref: Phase 2 optimization -- stochastic JIT support.
"""

from __future__ import annotations

from typing import Any

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, ScenarioVariant, make_get_variant

SCENARIO_ID: str = "AUG-2"
"""Unique scenario identifier."""

TIER1_VARIANT: str | None = None
"""Not a Tier-1 scenario (comparison requires both variants)."""

_DATASET_SIZE: int = 10_000
_ELEMENT_SHAPE: tuple[int, int, int] = (64, 64, 3)
_BATCH_SIZE: int = 64


def _make_data_generator(seed: int) -> callable:
    """Return a lazy data generator for uint8 images."""

    def _generate() -> dict[str, Any]:
        gen = SyntheticDataGenerator(seed=seed)
        return {"image": gen.images(_DATASET_SIZE, *_ELEMENT_SHAPE, dtype="uint8")}

    return _generate


VARIANTS: dict[str, ScenarioVariant] = {
    "deterministic": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=_DATASET_SIZE,
            element_shape=_ELEMENT_SHAPE,
            batch_size=_BATCH_SIZE,
            transforms=["Normalize", "Scale", "Clip"],
            extra={"variant_name": "deterministic"},
        ),
        data_generator=_make_data_generator(DEFAULT_SEED),
    ),
    "stochastic": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=_DATASET_SIZE,
            element_shape=_ELEMENT_SHAPE,
            batch_size=_BATCH_SIZE,
            transforms=["Normalize", "GaussianNoise", "RandomScale"],
            extra={"variant_name": "stochastic"},
        ),
        data_generator=_make_data_generator(DEFAULT_SEED),
    ),
}


get_variant = make_get_variant(VARIANTS)
