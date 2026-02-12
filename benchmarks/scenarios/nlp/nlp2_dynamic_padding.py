"""NLP-2: Dynamic Padding Pipeline scenario.

Variable-length sequences pre-padded to max_len with attention masks.
Tests pipeline handling of padded sequences with DynamicPad transform.

Design ref: Section 7 of the benchmark report.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, ScenarioVariant, make_get_variant

SCENARIO_ID: str = "NLP-2"
TIER1_VARIANT: str | None = None

# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

_VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "default": {
        "dataset_size": 100_000,
        "max_len": 512,
        "batch_size": 32,
    },
}


def _make_data_generator(dataset_size: int, max_len: int, seed: int = DEFAULT_SEED) -> callable:
    """Create a lazy data generator for padded variable-length sequences."""

    def generate() -> dict[str, Any]:
        gen = SyntheticDataGenerator(seed=seed)
        var_seqs = gen.variable_length_tokens(dataset_size, min_len=10, max_len=max_len)
        padded = np.zeros((dataset_size, max_len), dtype=np.int32)
        masks = np.zeros((dataset_size, max_len), dtype=np.float32)
        for i, seq in enumerate(var_seqs):
            length = len(seq)
            padded[i, :length] = seq
            masks[i, :length] = 1.0
        return {"tokens": padded, "attention_mask": masks}

    return generate


def _build_variants() -> dict[str, ScenarioVariant]:
    """Build all NLP-2 variants from specs."""
    variants: dict[str, ScenarioVariant] = {}
    for name, spec in _VARIANT_SPECS.items():
        config = ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=spec["dataset_size"],
            element_shape=(spec["max_len"],),
            batch_size=spec["batch_size"],
            transforms=["DynamicPad"],
            extra={"variant_name": name},
        )
        variants[name] = ScenarioVariant(
            config=config,
            data_generator=_make_data_generator(spec["dataset_size"], spec["max_len"]),
        )
    return variants


VARIANTS: dict[str, ScenarioVariant] = _build_variants()


get_variant = make_get_variant(VARIANTS)
