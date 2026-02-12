"""NLP-1: LLM Pre-training Pipeline scenario.

Pure iteration throughput for pre-tokenized sequences of fixed length.
Measures raw data loading speed without transform overhead.

Design ref: Section 7 of the benchmark report.
"""

from __future__ import annotations

from typing import Any

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, ScenarioVariant, make_get_variant

SCENARIO_ID: str = "NLP-1"
TIER1_VARIANT: str | None = "small"

# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

_VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "small": {
        "dataset_size": 10_000,
        "seq_len": 128,
        "batch_size": 32,
    },
    "medium": {
        "dataset_size": 100_000,
        "seq_len": 2048,
        "batch_size": 16,
    },
}


def _make_data_generator(dataset_size: int, seq_len: int, seed: int = DEFAULT_SEED) -> callable:
    """Create a lazy data generator for token sequences."""

    def generate() -> dict[str, Any]:
        gen = SyntheticDataGenerator(seed=seed)
        return {"tokens": gen.token_sequences(dataset_size, seq_len)}

    return generate


def _build_variants() -> dict[str, ScenarioVariant]:
    """Build all NLP-1 variants from specs."""
    variants: dict[str, ScenarioVariant] = {}
    for name, spec in _VARIANT_SPECS.items():
        config = ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=spec["dataset_size"],
            element_shape=(spec["seq_len"],),
            batch_size=spec["batch_size"],
            transforms=[],
            extra={"variant_name": name},
        )
        variants[name] = ScenarioVariant(
            config=config,
            data_generator=_make_data_generator(spec["dataset_size"], spec["seq_len"]),
        )
    return variants


VARIANTS: dict[str, ScenarioVariant] = _build_variants()


get_variant = make_get_variant(VARIANTS)
