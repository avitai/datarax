"""Tests for production readiness benchmark scenarios PR-1 and PR-2.

RED phase: these tests define the expected API and behavior for each
production scenario module before implementation exists.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.tests.test_scenarios.conftest import assert_valid_variant


# ===========================================================================
# PR-1: Checkpoint Save/Restore Cycle
# ===========================================================================


class TestPR1Scenario:
    """Tests for pr1_checkpoint scenario module."""

    @pytest.fixture(autouse=True)
    def _import_module(self):
        from benchmarks.scenarios.production import pr1_checkpoint as mod

        self.mod = mod

    def test_module_exports(self):
        """PR-1 must export SCENARIO_ID, VARIANTS, TIER1_VARIANT, get_variant."""
        assert self.mod.SCENARIO_ID == "PR-1"
        assert self.mod.TIER1_VARIANT is None
        assert isinstance(self.mod.VARIANTS, dict)
        assert callable(self.mod.get_variant)

    def test_variant_configs_are_valid(self):
        """Each variant must have required ScenarioConfig fields."""
        expected_variants = {"small", "medium", "large"}
        assert set(self.mod.VARIANTS.keys()) == expected_variants

        for name, variant in self.mod.VARIANTS.items():
            assert_valid_variant(variant)
            assert variant.config.scenario_id == "PR-1"
            assert variant.config.extra["variant_name"] == name
            assert variant.config.extra["checkpoint"] is True

    def test_small_data_generation(self):
        """Small variant generates (10K, 32, 32, 3) float32 images."""
        variant = self.mod.get_variant("small")
        assert variant.config.dataset_size == 10_000
        assert variant.config.element_shape == (32, 32, 3)
        data = variant.data_generator()
        assert "image" in data
        img = data["image"]
        assert isinstance(img, np.ndarray)
        assert img.shape == (10_000, 32, 32, 3)
        assert img.dtype == np.float32

    def test_large_variant_generates_volumes(self):
        """Large variant generates 3D volume data, not image data."""
        variant = self.mod.get_variant("large")
        assert variant.config.dataset_size == 500
        assert variant.config.element_shape == (64, 64, 64)
        data = variant.data_generator()
        assert "volume" in data
        vol = data["volume"]
        assert isinstance(vol, np.ndarray)
        assert vol.shape == (500, 64, 64, 64)
        assert vol.dtype == np.float32

    def test_get_variant_raises_on_unknown(self):
        """get_variant must raise KeyError for unknown names."""
        with pytest.raises(KeyError):
            self.mod.get_variant("nonexistent")


# ===========================================================================
# PR-2: Multi-Epoch Determinism Verification
# ===========================================================================


class TestPR2Scenario:
    """Tests for pr2_determinism scenario module."""

    @pytest.fixture(autouse=True)
    def _import_module(self):
        from benchmarks.scenarios.production import pr2_determinism as mod

        self.mod = mod

    def test_module_exports(self):
        """PR-2 must export SCENARIO_ID, VARIANTS, TIER1_VARIANT, get_variant."""
        assert self.mod.SCENARIO_ID == "PR-2"
        assert self.mod.TIER1_VARIANT == "small"
        assert isinstance(self.mod.VARIANTS, dict)
        assert callable(self.mod.get_variant)

    def test_variant_configs_are_valid(self):
        """Each variant must have required ScenarioConfig fields."""
        expected_variants = {"small", "full"}
        assert set(self.mod.VARIANTS.keys()) == expected_variants

        for name, variant in self.mod.VARIANTS.items():
            assert_valid_variant(variant)
            assert variant.config.scenario_id == "PR-2"
            assert variant.config.extra["variant_name"] == name
            assert "num_runs" in variant.config.extra
            assert "num_epochs" in variant.config.extra

    def test_small_variant_is_tier1(self):
        """Small variant has 2 runs and 2 epochs (Tier 1)."""
        variant = self.mod.get_variant("small")
        assert variant.config.extra["num_runs"] == 2
        assert variant.config.extra["num_epochs"] == 2

    def test_full_variant_is_tier2(self):
        """Full variant has 3 runs and 5 epochs (Tier 2)."""
        variant = self.mod.get_variant("full")
        assert variant.config.extra["num_runs"] == 3
        assert variant.config.extra["num_epochs"] == 5

    def test_data_generation_shapes(self):
        """Generated image data must have correct NHWC uint8 shape."""
        variant = self.mod.get_variant("small")
        data = variant.data_generator()
        assert "image" in data
        img = data["image"]
        assert isinstance(img, np.ndarray)
        assert img.ndim == 4
        assert img.shape[0] == variant.config.dataset_size
        h, w, c = variant.config.element_shape
        assert img.shape[1:] == (h, w, c)
        assert img.dtype == np.uint8
