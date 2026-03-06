"""HNLP-1: Long-Context LLM Pretraining Data Pipeline (Heavy).

Production-scale LLM pretraining with long-context sequences (8K tokens).
Tests attention mask and causal mask generation as compute-heavy transforms.
Causal mask generation is O(seq_len^2) -- benefits from JIT compilation.

Design ref: Section 11.3 of the performance audit.
"""

from __future__ import annotations

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, make_get_variant, ScenarioVariant


SCENARIO_ID: str = "HNLP-1"
TIER1_VARIANT: str | None = "short_context"

VARIANTS: dict[str, ScenarioVariant] = {
    "short_context": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=100_000,
            element_shape=(2048,),
            batch_size=64,
            transforms=["CreateAttentionMask", "CausalMaskGeneration"],
            extra={"variant_name": "short_context"},
        ),
        data_generator=lambda: {
            "tokens": SyntheticDataGenerator(seed=DEFAULT_SEED).token_sequences(
                100_000, 2048, vocab_size=32000
            )
        },
    ),
    "long_context": ScenarioVariant(
        config=ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=1_000_000,
            element_shape=(8192,),
            batch_size=32,
            transforms=["CreateAttentionMask", "CausalMaskGeneration"],
            extra={"variant_name": "long_context"},
        ),
        data_generator=lambda: {
            "tokens": SyntheticDataGenerator(seed=DEFAULT_SEED).token_sequences(
                1_000_000, 8192, vocab_size=32000
            )
        },
    ),
}


get_variant = make_get_variant(VARIANTS)
