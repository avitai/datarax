"""HMM-1: Vision-Language Contrastive Pipeline / CLIP-Scale (Heavy).

Large-scale multi-modal training with heterogeneous data types: images
(uint8) + text tokens (int32). Exercises Datarax's pytree handling with
large batch sizes (4096 is standard for CLIP training). Image transforms
are GPU-heavy while text padding is CPU-trivial -- tests mixed-intensity
pipelines.

Design ref: Section 11.3 of the performance audit.
"""

from __future__ import annotations

import numpy as np

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, make_get_variant, ScenarioVariant


SCENARIO_ID: str = "HMM-1"
TIER1_VARIANT: str | None = "clip_small"

_TEXT_LEN = 77


def _clip_data(n: int, h: int, w: int) -> dict[str, np.ndarray]:
    gen = SyntheticDataGenerator(seed=DEFAULT_SEED)
    images = gen.images(n, h, w, 3, dtype="uint8")
    tokens = gen.token_sequences(n, _TEXT_LEN, vocab_size=49408)
    return {"image": images, "tokens": tokens}


VARIANTS: dict[str, ScenarioVariant] = {
    "clip_small": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=50_000,
            element_shape=(224, 224, 3),
            batch_size=256,
            transforms=["RandomResizedCrop", "Normalize", "CastToFloat32"],
            extra={"variant_name": "clip_small", "text_len": _TEXT_LEN},
        ),
        data_generator=lambda: _clip_data(50_000, 224, 224),
    ),
    "clip_medium": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=500_000,
            element_shape=(224, 224, 3),
            batch_size=512,
            transforms=["RandomResizedCrop", "Normalize", "CastToFloat32"],
            extra={"variant_name": "clip_medium", "text_len": _TEXT_LEN},
        ),
        data_generator=lambda: _clip_data(500_000, 224, 224),
    ),
}


get_variant = make_get_variant(VARIANTS)
