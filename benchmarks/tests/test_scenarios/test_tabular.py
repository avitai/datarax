"""Tests for tabular benchmark scenarios TAB-1 and TAB-2.

RED phase: these tests define the expected API and behavior for each
tabular scenario module before implementation exists.
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
# TAB-1: Dense Features (Click-Through Rate)
# ===========================================================================


class TestTAB1Scenario:
    """Tests for tab1_dense_features scenario module."""

    @pytest.fixture(autouse=True)
    def _import_module(self):
        from benchmarks.scenarios.tabular import tab1_dense_features as mod

        self.mod = mod

    def test_variant_configs_are_valid(self):
        """Each variant must have required ScenarioConfig fields."""
        assert self.mod.SCENARIO_ID == "TAB-1"
        assert isinstance(self.mod.VARIANTS, dict)
        assert len(self.mod.VARIANTS) == 2  # small, medium

        for name, variant in self.mod.VARIANTS.items():
            assert_valid_variant(variant)
            assert variant.config.scenario_id == "TAB-1"
            assert variant.config.extra["variant_name"] == name
            assert "Normalize" in variant.config.transforms

    def test_data_generation_shapes(self):
        """Generated tabular data must have correct (N, features) shape."""
        small = self.mod.get_variant("small")
        data = small.data_generator()
        assert "features" in data
        features = data["features"]
        assert isinstance(features, np.ndarray)
        assert features.ndim == 2  # (N, features)
        assert features.shape[0] == small.config.dataset_size
        num_features = small.config.element_shape[0]
        assert features.shape[1] == num_features
        assert features.dtype == np.float32

    def test_tier1_variant_exists(self):
        """TAB-1 must define TIER1_VARIANT pointing to 'small'."""
        assert self.mod.TIER1_VARIANT == "small"
        tier1 = self.mod.get_variant(self.mod.TIER1_VARIANT)
        assert tier1.config.extra["variant_name"] == "small"

    def test_get_variant_raises_on_unknown(self):
        """get_variant must raise KeyError for unknown names."""
        with pytest.raises(KeyError):
            self.mod.get_variant("nonexistent")

    def test_run_produces_benchmark_result(self, datarax_adapter: DataraxAdapter):
        """Running TAB-1 small through the adapter produces a valid result."""
        tiny_variant = ScenarioVariant(
            config=ScenarioConfig(
                scenario_id="TAB-1",
                dataset_size=100,
                element_shape=(100,),
                batch_size=10,
                transforms=["Normalize"],
                extra={"variant_name": "test_tiny"},
            ),
            data_generator=lambda: {
                "features": np.random.default_rng(42).standard_normal((100, 100)).astype(np.float32)
            },
        )
        result = run_quick_scenario(datarax_adapter, tiny_variant)
        assert isinstance(result, BenchmarkResult)
        assert result.scenario_id == "TAB-1"
        assert result.timing.num_batches > 0


# ===========================================================================
# TAB-2: Sparse Features (DLRM-style)
# ===========================================================================


class TestTAB2Scenario:
    """Tests for tab2_sparse_features scenario module."""

    @pytest.fixture(autouse=True)
    def _import_module(self):
        from benchmarks.scenarios.tabular import tab2_sparse_features as mod

        self.mod = mod

    def test_variant_configs_are_valid(self):
        """Single variant must have valid config."""
        assert self.mod.SCENARIO_ID == "TAB-2"
        assert isinstance(self.mod.VARIANTS, dict)
        assert len(self.mod.VARIANTS) == 1

        for name, variant in self.mod.VARIANTS.items():
            assert_valid_variant(variant)
            assert variant.config.scenario_id == "TAB-2"
            assert variant.config.extra["variant_name"] == name
            assert variant.config.transforms == []

    def test_data_generation_shapes(self):
        """Generated sparse data must have dense and sparse_* keys."""
        variant = self.mod.get_variant("default")
        data = variant.data_generator()

        # Must have a 'dense' key
        assert "dense" in data
        dense = data["dense"]
        assert isinstance(dense, np.ndarray)
        assert dense.ndim == 2  # (N, num_dense)
        assert dense.shape[0] == variant.config.dataset_size
        assert dense.dtype == np.float32

        # Must have sparse_0 through sparse_25 (26 sparse features)
        for i in range(26):
            key = f"sparse_{i}"
            assert key in data, f"Missing key: {key}"
            sparse = data[key]
            assert isinstance(sparse, np.ndarray)
            assert sparse.shape[0] == variant.config.dataset_size

    def test_tier1_variant_is_none(self):
        """TAB-2 is not a Tier-1 scenario."""
        assert self.mod.TIER1_VARIANT is None

    def test_get_variant_raises_on_unknown(self):
        """get_variant must raise KeyError for unknown names."""
        with pytest.raises(KeyError):
            self.mod.get_variant("nonexistent")

    def test_run_produces_benchmark_result(self, datarax_adapter: DataraxAdapter):
        """Running TAB-2 through the adapter produces a valid result."""
        n = 100
        rng = np.random.default_rng(42)
        data_dict: dict[str, np.ndarray] = {
            "dense": rng.standard_normal((n, 13)).astype(np.float32),
        }
        for i in range(26):
            data_dict[f"sparse_{i}"] = rng.integers(0, 1000, n, dtype=np.int64)

        tiny_variant = ScenarioVariant(
            config=ScenarioConfig(
                scenario_id="TAB-2",
                dataset_size=n,
                element_shape=(13,),
                batch_size=10,
                transforms=[],
                extra={"variant_name": "test_tiny"},
            ),
            data_generator=lambda: data_dict,
        )
        result = run_quick_scenario(datarax_adapter, tiny_variant)
        assert isinstance(result, BenchmarkResult)
        assert result.scenario_id == "TAB-2"
        assert result.timing.num_batches > 0
