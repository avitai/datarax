"""Tests for multimodal benchmark scenarios MM-1 and MM-2.

RED phase: these tests define the expected API and behavior for each
multimodal scenario module before implementation exists.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.adapters.datarax_adapter import DataraxAdapter
from benchmarks.scenarios.base import ScenarioVariant
from benchmarks.tests.test_scenarios.conftest import assert_valid_variant, run_quick_scenario
from datarax.benchmarking.results import BenchmarkResult


# ===========================================================================
# MM-1: Image-Text Pairs (CLIP-style)
# ===========================================================================


class TestMM1Scenario:
    """Tests for mm1_image_text scenario module."""

    @pytest.fixture(autouse=True)
    def _import_module(self):
        from benchmarks.scenarios.multimodal import mm1_image_text as mod

        self.mod = mod

    def test_variant_configs_are_valid(self):
        """All variants must have valid config with Normalize transform."""
        assert self.mod.SCENARIO_ID == "MM-1"
        assert isinstance(self.mod.VARIANTS, dict)
        assert len(self.mod.VARIANTS) == 2  # default, large

        for name, variant in self.mod.VARIANTS.items():
            assert_valid_variant(variant)
            assert variant.config.scenario_id == "MM-1"
            assert variant.config.extra["variant_name"] == name
            assert "Normalize" in variant.config.transforms

    def test_data_generation_shapes(self):
        """Generated data must have image and tokens arrays."""
        variant = self.mod.get_variant("default")
        data = variant.data_generator()

        assert "image" in data
        assert "tokens" in data

        image = data["image"]
        tokens = data["tokens"]

        assert isinstance(image, np.ndarray)
        assert isinstance(tokens, np.ndarray)

        # Images: (N, H, W, C)
        assert image.ndim == 4
        assert image.shape[0] == variant.config.dataset_size
        assert image.dtype == np.float32

        # Tokens: (N, text_len)
        assert tokens.ndim == 2
        assert tokens.shape[0] == variant.config.dataset_size
        assert tokens.dtype == np.int32

    def test_tier1_variant_is_none(self):
        """MM-1 is not a Tier-1 scenario."""
        assert self.mod.TIER1_VARIANT is None

    def test_get_variant_raises_on_unknown(self):
        """get_variant must raise KeyError for unknown names."""
        with pytest.raises(KeyError):
            self.mod.get_variant("nonexistent")

    def test_run_produces_benchmark_result(self, datarax_adapter: DataraxAdapter):
        """Running MM-1 through the adapter produces a valid result."""
        n = 50
        rng = np.random.default_rng(42)
        tiny_variant = ScenarioVariant(
            config=ScenarioConfig(
                scenario_id="MM-1",
                dataset_size=n,
                element_shape=(8, 8, 3),
                batch_size=10,
                transforms=["Normalize"],
                extra={"variant_name": "test_tiny"},
            ),
            data_generator=lambda: {
                "image": rng.standard_normal((n, 8, 8, 3)).astype(np.float32),
                "tokens": rng.integers(0, 32000, (n, 16), dtype=np.int32),
            },
        )
        result = run_quick_scenario(datarax_adapter, tiny_variant)
        assert isinstance(result, BenchmarkResult)
        assert result.scenario_id == "MM-1"
        assert result.timing.num_batches > 0


# ===========================================================================
# MM-2: Audio-Text Pairs (Speech/Text)
# ===========================================================================


class TestMM2Scenario:
    """Tests for mm2_audio_text scenario module."""

    @pytest.fixture(autouse=True)
    def _import_module(self):
        from benchmarks.scenarios.multimodal import mm2_audio_text as mod

        self.mod = mod

    def test_variant_configs_are_valid(self):
        """All variants must have valid config."""
        assert self.mod.SCENARIO_ID == "MM-2"
        assert isinstance(self.mod.VARIANTS, dict)
        assert len(self.mod.VARIANTS) == 2  # default, large

        for name, variant in self.mod.VARIANTS.items():
            assert_valid_variant(variant)
            assert variant.config.scenario_id == "MM-2"
            assert variant.config.extra["variant_name"] == name
            assert variant.config.transforms == []

    def test_data_generation_shapes(self):
        """Generated data must have waveform and tokens arrays."""
        variant = self.mod.get_variant("default")
        data = variant.data_generator()

        assert "waveform" in data
        assert "tokens" in data

        waveform = data["waveform"]
        tokens = data["tokens"]

        assert isinstance(waveform, np.ndarray)
        assert isinstance(tokens, np.ndarray)

        # Waveforms: (N, num_samples)
        assert waveform.ndim == 2
        assert waveform.shape[0] == variant.config.dataset_size
        assert waveform.dtype == np.float32

        # Tokens: (N, text_len)
        assert tokens.ndim == 2
        assert tokens.shape[0] == variant.config.dataset_size
        assert tokens.dtype == np.int32

    def test_tier1_variant_is_none(self):
        """MM-2 is not a Tier-1 scenario."""
        assert self.mod.TIER1_VARIANT is None

    def test_get_variant_raises_on_unknown(self):
        """get_variant must raise KeyError for unknown names."""
        with pytest.raises(KeyError):
            self.mod.get_variant("nonexistent")

    def test_run_produces_benchmark_result(self, datarax_adapter: DataraxAdapter):
        """Running MM-2 through the adapter produces a valid result."""
        n = 50
        rng = np.random.default_rng(42)
        # Use very short waveforms for test speed
        tiny_variant = ScenarioVariant(
            config=ScenarioConfig(
                scenario_id="MM-2",
                dataset_size=n,
                element_shape=(160,),
                batch_size=10,
                transforms=[],
                extra={"variant_name": "test_tiny"},
            ),
            data_generator=lambda: {
                "waveform": rng.standard_normal((n, 160)).astype(np.float32),
                "tokens": rng.integers(0, 32000, (n, 16), dtype=np.int32),
            },
        )
        result = run_quick_scenario(datarax_adapter, tiny_variant)
        assert isinstance(result, BenchmarkResult)
        assert result.scenario_id == "MM-2"
        assert result.timing.num_batches > 0
