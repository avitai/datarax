"""PC-1: Deep Transform Chain Scaling.

Measures pipeline overhead as the number of chained transforms grows.
Each variant applies N transforms from a repeating 5-transform cycle
to a fixed 50K-element float32 dataset of shape (64, 64).

Tier-1 variant: depth_1 (minimal chain for fast CI).
Full suite depths: [1, 2, 5, 10, 20, 50, 100].

Design ref: Section 7.4.1 of the benchmark report.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.scenarios.base import DEFAULT_SEED, ScenarioVariant, make_get_variant

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

SCENARIO_ID: str = "PC-1"
"""Unique scenario identifier."""

TIER1_VARIANT: str | None = "depth_1"
"""Smallest chain depth variant for fast CI runs."""

_FULL_DEPTHS: list[int] = [1, 2, 5, 10, 20, 50, 100]
"""All chain depths in the full benchmark suite."""

_TIER1_DEPTHS: list[int] = [1, 5, 10]
"""Subset of depths included in Tier-1 CI."""

_TRANSFORM_CYCLE: list[str] = ["Normalize", "Scale", "Clip", "Add", "Multiply"]
"""Canonical 5-transform cycle repeated to reach desired chain depth."""

_DATASET_SIZE: int = 50_000
_ELEMENT_SHAPE: tuple[int, int] = (64, 64)
_BATCH_SIZE: int = 64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_transform_chain(depth: int) -> list[str]:
    """Build a transform list of length *depth* by cycling through the canonical set."""
    return [_TRANSFORM_CYCLE[i % len(_TRANSFORM_CYCLE)] for i in range(depth)]


def _make_data_generator(dataset_size: int, shape: tuple[int, int], seed: int) -> callable:
    """Return a lazy data generator for the given shape."""

    def _generate() -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        return {"data": rng.standard_normal((dataset_size, *shape)).astype(np.float32)}

    return _generate


def _build_variant(depth: int) -> ScenarioVariant:
    """Build a ScenarioVariant for a given chain depth."""
    name = f"depth_{depth}"
    return ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=_DATASET_SIZE,
            element_shape=_ELEMENT_SHAPE,
            batch_size=_BATCH_SIZE,
            transforms=_build_transform_chain(depth),
            seed=DEFAULT_SEED,
            extra={"chain_depth": depth, "variant_name": name},
        ),
        data_generator=_make_data_generator(_DATASET_SIZE, _ELEMENT_SHAPE, DEFAULT_SEED),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

VARIANTS: dict[str, ScenarioVariant] = {f"depth_{d}": _build_variant(d) for d in _FULL_DEPTHS}
"""All chain-depth variants keyed by name."""


get_variant = make_get_variant(VARIANTS)
