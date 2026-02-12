"""AUG-3: Stochastic Chain Depth Scaling.

Measures how stochastic pipeline overhead scales with chain depth.
Mirrors PC-1 (deterministic chain depth) but with stochastic transforms.

Uses a 3-transform stochastic cycle repeated to reach target depth.
All variants prepend Normalize for uint8-to-float32 conversion.

Not a Tier-1 scenario.

Design ref: Phase 2 optimization -- stochastic JIT support.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.scenarios.base import DEFAULT_SEED, ScenarioVariant, make_get_variant

SCENARIO_ID: str = "AUG-3"
"""Unique scenario identifier."""

TIER1_VARIANT: str | None = None
"""Not a Tier-1 scenario."""

_STOCHASTIC_CYCLE: list[str] = [
    "GaussianNoise",
    "RandomBrightness",
    "RandomScale",
]
"""3-transform stochastic cycle repeated to reach target depth."""

_DATASET_SIZE: int = 50_000
_ELEMENT_SHAPE: tuple[int, int] = (64, 64)
_BATCH_SIZE: int = 64
_DEPTHS: list[int] = [1, 3, 6, 12]


def _build_stochastic_chain(depth: int) -> list[str]:
    """Build a stochastic transform chain of given depth.

    Prepends Normalize (deterministic) for uint8 -> float32 conversion,
    then cycles through the stochastic transforms.
    """
    stochastic_ops = [_STOCHASTIC_CYCLE[i % len(_STOCHASTIC_CYCLE)] for i in range(depth)]
    return ["Normalize", *stochastic_ops]


def _make_data_generator(dataset_size: int, shape: tuple[int, int], seed: int) -> callable:
    """Return a lazy data generator for uint8 2D data."""

    def _generate() -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        return {
            "data": rng.integers(
                0,
                256,
                size=(dataset_size, *shape),
                dtype=np.uint8,
            )
        }

    return _generate


def _build_variant(depth: int) -> ScenarioVariant:
    """Build a ScenarioVariant for a given stochastic chain depth."""
    name = f"depth_{depth}"
    return ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=_DATASET_SIZE,
            element_shape=_ELEMENT_SHAPE,
            batch_size=_BATCH_SIZE,
            transforms=_build_stochastic_chain(depth),
            seed=DEFAULT_SEED,
            extra={
                "stochastic_depth": depth,
                "total_depth": depth + 1,
                "variant_name": name,
            },
        ),
        data_generator=_make_data_generator(_DATASET_SIZE, _ELEMENT_SHAPE, DEFAULT_SEED),
    )


VARIANTS: dict[str, ScenarioVariant] = {f"depth_{d}": _build_variant(d) for d in _DEPTHS}
"""All stochastic chain-depth variants."""


get_variant = make_get_variant(VARIANTS)
