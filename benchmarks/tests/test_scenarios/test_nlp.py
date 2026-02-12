"""Tests for NLP benchmark scenarios NLP-1 and NLP-2.

RED phase: these tests define the expected API and behavior for each
NLP scenario module before implementation exists.
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
# NLP-1: LLM Pre-training Pipeline
# ===========================================================================


class TestNLP1Scenario:
    """Tests for nlp1_llm_pretraining scenario module."""

    @pytest.fixture(autouse=True)
    def _import_module(self):
        from benchmarks.scenarios.nlp import nlp1_llm_pretraining as mod

        self.mod = mod

    def test_variant_configs_are_valid(self):
        """Each variant must have required ScenarioConfig fields."""
        assert self.mod.SCENARIO_ID == "NLP-1"
        assert isinstance(self.mod.VARIANTS, dict)
        assert len(self.mod.VARIANTS) == 2  # small, medium

        for name, variant in self.mod.VARIANTS.items():
            assert_valid_variant(variant)
            assert variant.config.scenario_id == "NLP-1"
            assert variant.config.extra["variant_name"] == name
            assert variant.config.transforms == []

    def test_data_generation_shapes(self):
        """Generated token data must have correct (N, seq_len) shape."""
        small = self.mod.get_variant("small")
        data = small.data_generator()
        assert "tokens" in data
        tokens = data["tokens"]
        assert isinstance(tokens, np.ndarray)
        assert tokens.ndim == 2  # (N, seq_len)
        assert tokens.shape[0] == small.config.dataset_size
        seq_len = small.config.element_shape[0]
        assert tokens.shape[1] == seq_len
        assert tokens.dtype == np.int32

    def test_tier1_variant_exists(self):
        """NLP-1 must define TIER1_VARIANT pointing to 'small'."""
        assert self.mod.TIER1_VARIANT == "small"
        tier1 = self.mod.get_variant(self.mod.TIER1_VARIANT)
        assert tier1.config.extra["variant_name"] == "small"

    def test_get_variant_raises_on_unknown(self):
        """get_variant must raise KeyError for unknown names."""
        with pytest.raises(KeyError):
            self.mod.get_variant("nonexistent")

    def test_run_produces_benchmark_result(self, datarax_adapter: DataraxAdapter):
        """Running NLP-1 small through the adapter produces a valid result."""
        tiny_variant = ScenarioVariant(
            config=ScenarioConfig(
                scenario_id="NLP-1",
                dataset_size=100,
                element_shape=(128,),
                batch_size=10,
                transforms=[],
                extra={"variant_name": "test_tiny"},
            ),
            data_generator=lambda: {
                "tokens": np.random.default_rng(42).integers(0, 32000, (100, 128), dtype=np.int32)
            },
        )
        result = run_quick_scenario(datarax_adapter, tiny_variant)
        assert isinstance(result, BenchmarkResult)
        assert result.scenario_id == "NLP-1"
        assert result.timing.num_batches > 0


# ===========================================================================
# NLP-2: Dynamic Padding Pipeline
# ===========================================================================


class TestNLP2Scenario:
    """Tests for nlp2_dynamic_padding scenario module."""

    @pytest.fixture(autouse=True)
    def _import_module(self):
        from benchmarks.scenarios.nlp import nlp2_dynamic_padding as mod

        self.mod = mod

    def test_variant_configs_are_valid(self):
        """Single variant must have valid config with DynamicPad transform."""
        assert self.mod.SCENARIO_ID == "NLP-2"
        assert isinstance(self.mod.VARIANTS, dict)
        assert len(self.mod.VARIANTS) == 1

        for name, variant in self.mod.VARIANTS.items():
            assert_valid_variant(variant)
            assert variant.config.scenario_id == "NLP-2"
            assert variant.config.extra["variant_name"] == name
            assert "DynamicPad" in variant.config.transforms

    def test_data_generation_shapes(self):
        """Generated padded data must have tokens and attention_mask."""
        variant = self.mod.get_variant("default")
        data = variant.data_generator()

        assert "tokens" in data
        assert "attention_mask" in data

        tokens = data["tokens"]
        mask = data["attention_mask"]

        assert isinstance(tokens, np.ndarray)
        assert isinstance(mask, np.ndarray)
        assert tokens.ndim == 2  # (N, max_len)
        assert mask.ndim == 2  # (N, max_len)
        assert tokens.shape == mask.shape
        assert tokens.shape[0] == variant.config.dataset_size
        assert tokens.dtype == np.int32
        assert mask.dtype == np.float32

        # Attention mask should have values 0.0 and 1.0 only
        assert set(np.unique(mask)).issubset({0.0, 1.0})

    def test_tier1_variant_is_none(self):
        """NLP-2 is not a Tier-1 scenario."""
        assert self.mod.TIER1_VARIANT is None

    def test_get_variant_raises_on_unknown(self):
        """get_variant must raise KeyError for unknown names."""
        with pytest.raises(KeyError):
            self.mod.get_variant("nonexistent")

    def test_run_produces_benchmark_result(self, datarax_adapter: DataraxAdapter):
        """Running NLP-2 through the adapter produces a valid result."""
        n = 100
        rng = np.random.default_rng(42)
        padded = np.zeros((n, 64), dtype=np.int32)
        masks = np.zeros((n, 64), dtype=np.float32)
        for i in range(n):
            length = rng.integers(5, 64)
            padded[i, :length] = rng.integers(0, 32000, length, dtype=np.int32)
            masks[i, :length] = 1.0

        tiny_variant = ScenarioVariant(
            config=ScenarioConfig(
                scenario_id="NLP-2",
                dataset_size=n,
                element_shape=(64,),
                batch_size=10,
                transforms=["DynamicPad"],
                extra={"variant_name": "test_tiny"},
            ),
            data_generator=lambda: {"tokens": padded, "attention_mask": masks},
        )
        result = run_quick_scenario(datarax_adapter, tiny_variant)
        assert isinstance(result, BenchmarkResult)
        assert result.scenario_id == "NLP-2"
        assert result.timing.num_batches > 0
