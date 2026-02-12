"""Tests for I/O benchmark scenarios IO-1 through IO-4.

RED phase: these tests define the expected API and behavior for each
I/O scenario module before implementation exists.
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
# IO-1: Source Backend Comparison
# ===========================================================================


class TestIO1Scenario:
    """Tests for io1_backend_comparison scenario module."""

    @pytest.fixture(autouse=True)
    def _import_module(self):
        from benchmarks.scenarios.io import io1_backend_comparison as mod

        self.mod = mod

    def test_variant_configs_are_valid(self):
        """All IO-1 variants must have valid ScenarioConfig fields."""
        assert self.mod.SCENARIO_ID == "IO-1"
        assert isinstance(self.mod.VARIANTS, dict)
        # Must have at least memory_source; Tier-2 backends are optional
        assert "memory_source" in self.mod.VARIANTS

        for name, variant in self.mod.VARIANTS.items():
            assert_valid_variant(variant)
            assert variant.config.scenario_id == "IO-1"
            assert variant.config.extra["variant_name"] == name
            assert variant.config.transforms == ["Normalize"]

    def test_data_generation_shapes(self):
        """memory_source variant must generate NHWC uint8 image data."""
        variant = self.mod.get_variant("memory_source")
        data = variant.data_generator()
        assert "image" in data
        img = data["image"]
        assert isinstance(img, np.ndarray)
        assert img.ndim == 4  # (N, H, W, C)
        assert img.shape[0] == variant.config.dataset_size
        h, w, c = variant.config.element_shape
        assert img.shape[1:] == (h, w, c)
        assert img.dtype == np.uint8

    def test_tier1_variant_exists(self):
        """IO-1 must define TIER1_VARIANT pointing to 'memory_source'."""
        assert self.mod.TIER1_VARIANT == "memory_source"
        tier1 = self.mod.get_variant(self.mod.TIER1_VARIANT)
        assert tier1.config.extra["variant_name"] == "memory_source"

    def test_get_variant_raises_on_unknown(self):
        """get_variant must raise KeyError for unknown names."""
        with pytest.raises(KeyError):
            self.mod.get_variant("nonexistent")

    def test_run_produces_benchmark_result(self, datarax_adapter: DataraxAdapter):
        """Running IO-1 memory_source through the adapter produces a valid result."""
        tiny_variant = ScenarioVariant(
            config=ScenarioConfig(
                scenario_id="IO-1",
                dataset_size=100,
                element_shape=(32, 32, 3),
                batch_size=10,
                transforms=["Normalize"],
                extra={"variant_name": "test_tiny"},
            ),
            data_generator=lambda: {
                "image": np.random.default_rng(42).integers(
                    0, 256, (100, 32, 32, 3), dtype=np.uint8
                )
            },
        )
        result = run_quick_scenario(datarax_adapter, tiny_variant)
        assert isinstance(result, BenchmarkResult)
        assert result.scenario_id == "IO-1"
        assert result.timing.num_batches > 0


# ===========================================================================
# IO-2: Streaming vs. Eager Loading
# ===========================================================================


class TestIO2Scenario:
    """Tests for io2_streaming_vs_eager scenario module."""

    @pytest.fixture(autouse=True)
    def _import_module(self):
        from benchmarks.scenarios.io import io2_streaming_vs_eager as mod

        self.mod = mod

    def test_variant_configs_are_valid(self):
        """IO-2 must have size-based variants with valid configs."""
        assert self.mod.SCENARIO_ID == "IO-2"
        assert isinstance(self.mod.VARIANTS, dict)
        assert len(self.mod.VARIANTS) == 3  # 10k, 50k, 200k

        expected_sizes = {"10k": 10_000, "50k": 50_000, "200k": 200_000}
        for name, variant in self.mod.VARIANTS.items():
            assert_valid_variant(variant)
            assert variant.config.scenario_id == "IO-2"
            assert variant.config.extra["variant_name"] == name
            assert variant.config.transforms == ["Normalize"]
            assert name in expected_sizes
            assert variant.config.dataset_size == expected_sizes[name]

    def test_data_generation_shapes(self):
        """Generated float32 image data must have correct NHWC shape."""
        variant = self.mod.get_variant("10k")
        data = variant.data_generator()
        assert "image" in data
        img = data["image"]
        assert isinstance(img, np.ndarray)
        assert img.ndim == 4  # (N, H, W, C)
        assert img.shape[0] == variant.config.dataset_size
        h, w, c = variant.config.element_shape
        assert img.shape[1:] == (h, w, c)
        assert img.dtype == np.float32

    def test_tier1_variant_is_none(self):
        """IO-2 is not a Tier-1 scenario."""
        assert self.mod.TIER1_VARIANT is None

    def test_get_variant_raises_on_unknown(self):
        """get_variant must raise KeyError for unknown names."""
        with pytest.raises(KeyError):
            self.mod.get_variant("nonexistent")

    def test_run_produces_benchmark_result(self, datarax_adapter: DataraxAdapter):
        """Running IO-2 10k through the adapter produces a valid result."""
        tiny_variant = ScenarioVariant(
            config=ScenarioConfig(
                scenario_id="IO-2",
                dataset_size=100,
                element_shape=(64, 64, 3),
                batch_size=10,
                transforms=["Normalize"],
                extra={"variant_name": "test_tiny"},
            ),
            data_generator=lambda: {
                "image": np.random.default_rng(42)
                .standard_normal((100, 64, 64, 3))
                .astype(np.float32)
            },
        )
        result = run_quick_scenario(datarax_adapter, tiny_variant)
        assert isinstance(result, BenchmarkResult)
        assert result.scenario_id == "IO-2"
        assert result.timing.num_batches > 0


# ===========================================================================
# IO-3: Mixed-Source Pipeline
# ===========================================================================


class TestIO3Scenario:
    """Tests for io3_mixed_source scenario module."""

    @pytest.fixture(autouse=True)
    def _import_module(self):
        from benchmarks.scenarios.io import io3_mixed_source as mod

        self.mod = mod

    def test_variant_configs_are_valid(self):
        """IO-3 must have a single 'default' variant with valid config."""
        assert self.mod.SCENARIO_ID == "IO-3"
        assert isinstance(self.mod.VARIANTS, dict)
        assert len(self.mod.VARIANTS) == 1
        assert "default" in self.mod.VARIANTS

        variant = self.mod.VARIANTS["default"]
        assert_valid_variant(variant)
        assert variant.config.scenario_id == "IO-3"
        assert variant.config.extra["variant_name"] == "default"
        assert "Normalize" in variant.config.transforms
        assert "Merge" in variant.config.transforms
        assert variant.config.extra["topology"] == "mixed_source"

    def test_data_generation_shapes(self):
        """Generated data must have both image and label arrays."""
        variant = self.mod.get_variant("default")
        data = variant.data_generator()
        assert "image" in data
        assert "label" in data

        img = data["image"]
        lbl = data["label"]
        assert isinstance(img, np.ndarray)
        assert isinstance(lbl, np.ndarray)
        assert img.ndim == 4  # (N, H, W, C)
        assert img.shape[0] == variant.config.dataset_size
        assert img.dtype == np.float32
        assert lbl.shape[0] == variant.config.dataset_size
        assert lbl.ndim == 2  # (N, 1)

    def test_tier1_variant_is_none(self):
        """IO-3 is not a Tier-1 scenario."""
        assert self.mod.TIER1_VARIANT is None

    def test_blocked_flag(self):
        """IO-3 must be marked as BLOCKED until MixDataSourcesNode exists."""
        assert self.mod.BLOCKED is True
        assert isinstance(self.mod.BLOCKED_REASON, str)
        assert len(self.mod.BLOCKED_REASON) > 0

    def test_get_variant_raises_on_unknown(self):
        """get_variant must raise KeyError for unknown names."""
        with pytest.raises(KeyError):
            self.mod.get_variant("nonexistent")


# ===========================================================================
# IO-4: Cache Node Effectiveness
# ===========================================================================


class TestIO4Scenario:
    """Tests for io4_cache_node scenario module."""

    @pytest.fixture(autouse=True)
    def _import_module(self):
        from benchmarks.scenarios.io import io4_cache_node as mod

        self.mod = mod

    def test_variant_configs_are_valid(self):
        """IO-4 must have a single 'default' variant with valid config."""
        assert self.mod.SCENARIO_ID == "IO-4"
        assert isinstance(self.mod.VARIANTS, dict)
        assert len(self.mod.VARIANTS) == 1
        assert "default" in self.mod.VARIANTS

        variant = self.mod.VARIANTS["default"]
        assert_valid_variant(variant)
        assert variant.config.scenario_id == "IO-4"
        assert variant.config.extra["variant_name"] == "default"
        assert variant.config.transforms == [
            "ExpensiveTransform",
            "Cache",
            "CheapTransform",
        ]
        assert variant.config.extra["num_epochs"] == 3

    def test_data_generation_shapes(self):
        """Generated float32 image data must have correct NHWC shape."""
        variant = self.mod.get_variant("default")
        data = variant.data_generator()
        assert "image" in data
        img = data["image"]
        assert isinstance(img, np.ndarray)
        assert img.ndim == 4  # (N, H, W, C)
        assert img.shape[0] == variant.config.dataset_size
        h, w, c = variant.config.element_shape
        assert img.shape[1:] == (h, w, c)
        assert img.dtype == np.float32

    def test_tier1_variant_is_none(self):
        """IO-4 is not a Tier-1 scenario."""
        assert self.mod.TIER1_VARIANT is None

    def test_get_variant_raises_on_unknown(self):
        """get_variant must raise KeyError for unknown names."""
        with pytest.raises(KeyError):
            self.mod.get_variant("nonexistent")

    def test_run_produces_benchmark_result(self, datarax_adapter: DataraxAdapter):
        """Running IO-4 through the adapter produces a valid result."""
        tiny_variant = ScenarioVariant(
            config=ScenarioConfig(
                scenario_id="IO-4",
                dataset_size=100,
                element_shape=(32, 32, 3),
                batch_size=10,
                transforms=["ExpensiveTransform", "Cache", "CheapTransform"],
                extra={"variant_name": "test_tiny", "num_epochs": 3},
            ),
            data_generator=lambda: {
                "image": np.random.default_rng(42)
                .standard_normal((100, 32, 32, 3))
                .astype(np.float32)
            },
        )
        result = run_quick_scenario(datarax_adapter, tiny_variant)
        assert isinstance(result, BenchmarkResult)
        assert result.scenario_id == "IO-4"
        assert result.timing.num_batches > 0
