"""MM-2: Audio-Text Pairs (Speech/Text) scenario.

Audio waveforms paired with token sequences for speech-text tasks.
No transforms applied (pure iteration throughput).

Design ref: Section 7 of the benchmark report.
"""

from __future__ import annotations

from typing import Any

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator
from benchmarks.scenarios.base import DEFAULT_SEED, ScenarioVariant, make_get_variant

SCENARIO_ID: str = "MM-2"
TIER1_VARIANT: str | None = None

# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 16000
_DURATION_SEC = 1.0
_TEXT_LEN = 128

_VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "default": {
        "dataset_size": 5_000,
        "batch_size": 16,
        "duration_sec": 1.0,
    },
    "large": {
        "dataset_size": 50_000,
        "batch_size": 16,
        "duration_sec": 5.0,
    },
}


def _make_data_generator(
    dataset_size: int,
    sample_rate: int,
    duration_sec: float,
    text_len: int,
    seed: int = DEFAULT_SEED,
) -> callable:
    """Create a lazy data generator for audio-text pairs."""

    def generate() -> dict[str, Any]:
        gen = SyntheticDataGenerator(seed=seed)
        waveforms = gen.audio_waveforms(
            dataset_size, sample_rate=sample_rate, duration_sec=duration_sec
        )
        tokens = gen.token_sequences(dataset_size, text_len)
        return {"waveform": waveforms, "tokens": tokens}

    return generate


def _build_variants() -> dict[str, ScenarioVariant]:
    """Build all MM-2 variants from specs."""
    variants: dict[str, ScenarioVariant] = {}
    for name, spec in _VARIANT_SPECS.items():
        duration = spec.get("duration_sec", _DURATION_SEC)
        waveform_len = int(_SAMPLE_RATE * duration)
        config = ScenarioConfig(
            scenario_id=SCENARIO_ID,
            dataset_size=spec["dataset_size"],
            element_shape=(waveform_len,),
            batch_size=spec["batch_size"],
            transforms=[],
            extra={"variant_name": name},
        )
        variants[name] = ScenarioVariant(
            config=config,
            data_generator=_make_data_generator(
                spec["dataset_size"],
                _SAMPLE_RATE,
                duration,
                _TEXT_LEN,
            ),
        )
    return variants


VARIANTS: dict[str, ScenarioVariant] = _build_variants()


get_variant = make_get_variant(VARIANTS)
