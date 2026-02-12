"""PC-5: End-to-End Differentiable Pipeline.

Measures overhead of a fully differentiable data pipeline where
transforms are learnable parameters that participate in gradient
computation. The standard get_variant() interface works for
adapter-based forward-only benchmarking.

For gradient measurement, use ``run_direct()`` which exercises the
Datarax API directly for forward+backward timing (bypasses adapter).

Single variant: 10K float32 images (32, 32, 3), batch_size=32.
Not a Tier-1 scenario.

Design ref: Section 7.4.5 of the benchmark report.
"""

from __future__ import annotations

from typing import Any

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, ScenarioVariant, make_get_variant

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

SCENARIO_ID: str = "PC-5"
"""Unique scenario identifier."""

TIER1_VARIANT: str | None = None
"""PC-5 is not a Tier-1 scenario."""

_DATASET_SIZE: int = 10_000
_ELEMENT_SHAPE: tuple[int, int, int] = (32, 32, 3)
_BATCH_SIZE: int = 32


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
            transforms=["LearnableAugment", "Normalize"],
            seed=DEFAULT_SEED,
            extra={
                "differentiable": True,
                "variant_name": "default",
            },
        ),
        data_generator=_make_data_generator(_DATASET_SIZE, *_ELEMENT_SHAPE, DEFAULT_SEED),
    ),
}
"""Single default variant for differentiable pipeline benchmarking."""


get_variant = make_get_variant(VARIANTS)


def run_direct() -> dict[str, Any]:
    """Run the differentiable pipeline directly for gradient measurement.

    This bypasses the standard BenchmarkAdapter lifecycle and uses the
    Datarax API directly to time forward + backward passes through
    learnable augmentation transforms.

    Returns:
        Dictionary with forward_sec, backward_sec, and total_sec timings.

    Note:
        This is a placeholder for the full differentiable pipeline
        implementation. The actual gradient timing requires Datarax's
        differentiable transform API.
    """
    raise NotImplementedError(
        "run_direct() requires Datarax differentiable transform API. "
        "Use get_variant() + run_scenario() for forward-only benchmarking."
    )
