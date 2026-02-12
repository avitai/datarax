"""AUG-1: Stochastic Augmentation Chain.

Measures throughput of a stochastic transform chain where operators
use RNG for per-element randomness. This isolates the JIT overhead
of the stochastic path (to_tree/from_tree per batch for NNX state
management) versus the deterministic closure-capture path.

Transforms: Normalize (deterministic) + GaussianNoise + RandomBrightness
(both stochastic). The chain triggers the argument-passing JIT strategy.

Tier-1 variant: small (fast CI regression guard).

Design ref: Phase 2 optimization -- stochastic JIT support.
"""

from __future__ import annotations

from typing import Any

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, ScenarioVariant, make_get_variant

SCENARIO_ID: str = "AUG-1"
"""Unique scenario identifier."""

TIER1_VARIANT: str | None = "small"
"""Small variant for fast CI runs."""

_TRANSFORMS: list[str] = ["Normalize", "GaussianNoise", "RandomBrightness"]


def _make_data_generator(dataset_size: int, h: int, w: int, c: int, seed: int) -> callable:
    """Return a lazy data generator for uint8 images."""

    def _generate() -> dict[str, Any]:
        gen = SyntheticDataGenerator(seed=seed)
        return {"image": gen.images(dataset_size, h, w, c, dtype="uint8")}

    return _generate


VARIANTS: dict[str, ScenarioVariant] = {
    "small": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=10_000,
            element_shape=(32, 32, 3),
            batch_size=64,
            transforms=_TRANSFORMS,
            extra={"variant_name": "small"},
        ),
        data_generator=_make_data_generator(10_000, 32, 32, 3, DEFAULT_SEED),
    ),
    "medium": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=5_000,
            element_shape=(128, 128, 3),
            batch_size=128,
            transforms=_TRANSFORMS,
            extra={"variant_name": "medium"},
        ),
        data_generator=_make_data_generator(5_000, 128, 128, 3, DEFAULT_SEED),
    ),
    "large": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=50_000,
            element_shape=(256, 256, 3),
            batch_size=64,
            transforms=_TRANSFORMS,
            extra={"variant_name": "large"},
        ),
        data_generator=_make_data_generator(50_000, 256, 256, 3, DEFAULT_SEED),
    ),
}


get_variant = make_get_variant(VARIANTS)
