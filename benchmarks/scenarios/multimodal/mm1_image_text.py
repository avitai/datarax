"""MM-1: Image-Text Pairs (CLIP-style) scenario.

CLIP-style image-text pair loading with Normalize transform.
Images are float32, tokens are int32.

Design ref: Section 7 of the benchmark report.
"""

from __future__ import annotations

from typing import Any

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, ScenarioVariant, make_get_variant

SCENARIO_ID: str = "MM-1"
TIER1_VARIANT: str | None = None

# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

_IMG_SHAPE = (64, 64, 3)
_TEXT_LEN = 77

_VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "default": {
        "dataset_size": 5_000,
        "batch_size": 64,
        "img_shape": (64, 64, 3),
    },
    "large": {
        "dataset_size": 50_000,
        "batch_size": 64,
        "img_shape": (224, 224, 3),
    },
}


def _make_data_generator(
    dataset_size: int,
    img_shape: tuple[int, ...],
    text_len: int,
    seed: int = DEFAULT_SEED,
) -> callable:
    """Create a lazy data generator for image-text pairs."""

    def generate() -> dict[str, Any]:
        gen = SyntheticDataGenerator(seed=seed)
        images, tokens = gen.image_text_pairs(dataset_size, img_shape=img_shape, text_len=text_len)
        return {"image": images, "tokens": tokens}

    return generate


def _build_variants() -> dict[str, ScenarioVariant]:
    """Build all MM-1 variants from specs."""
    variants: dict[str, ScenarioVariant] = {}
    for name, spec in _VARIANT_SPECS.items():
        img_shape = spec.get("img_shape", _IMG_SHAPE)
        config = ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=spec["dataset_size"],
            element_shape=img_shape,
            batch_size=spec["batch_size"],
            transforms=["Normalize"],
            extra={"variant_name": name},
        )
        variants[name] = ScenarioVariant(
            config=config,
            data_generator=_make_data_generator(spec["dataset_size"], img_shape, _TEXT_LEN),
        )
    return variants


VARIANTS: dict[str, ScenarioVariant] = _build_variants()


get_variant = make_get_variant(VARIANTS)
