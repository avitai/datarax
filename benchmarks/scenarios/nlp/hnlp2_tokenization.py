"""HNLP-2: Text Tokenization Pipeline (Heavy).

End-to-end text preprocessing with padding and masking. Tokenization is
CPU-heavy and not JIT-compilable -- tests Datarax's ability to mix CPU
preprocessing with GPU-accelerated post-processing. This is where a
multiprocessing backend (F11) would matter.

Design ref: Section 11.3 of the performance audit.
"""

from __future__ import annotations

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, make_get_variant, ScenarioVariant


SCENARIO_ID: str = "HNLP-2"
TIER1_VARIANT: str | None = "small"

VARIANTS: dict[str, ScenarioVariant] = {
    "small": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=100_000,
            element_shape=(256,),
            batch_size=128,
            transforms=["CreateAttentionMask"],
            extra={"variant_name": "small"},
        ),
        data_generator=lambda: {
            "tokens": SyntheticDataGenerator(seed=DEFAULT_SEED).token_sequences(
                100_000, 256, vocab_size=32000
            )
        },
    ),
    "medium": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=500_000,
            element_shape=(512,),
            batch_size=64,
            transforms=["CreateAttentionMask"],
            extra={"variant_name": "medium"},
        ),
        data_generator=lambda: {
            "tokens": SyntheticDataGenerator(seed=DEFAULT_SEED).token_sequences(
                500_000, 512, vocab_size=32000
            )
        },
    ),
}


get_variant = make_get_variant(VARIANTS)
