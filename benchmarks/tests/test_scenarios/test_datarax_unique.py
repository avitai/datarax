"""Tests for datarax-unique benchmark scenarios NNX-1 and XFMR-1.

RED phase: these tests define the expected API and behavior for each
datarax-unique scenario module before implementation exists.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.tests.test_scenarios.conftest import assert_valid_variant


# ===========================================================================
# NNX-1: Flax NNX Module Integration Overhead
# ===========================================================================


class TestNNX1Scenario:
    """Tests for nnx1_module_overhead scenario module."""

    @pytest.fixture(autouse=True)
    def _import_module(self):
        from benchmarks.scenarios.datarax_unique import nnx1_module_overhead as mod

        self.mod = mod

    def test_module_exports(self):
        """NNX-1 must export SCENARIO_ID, VARIANTS, TIER1_VARIANT, get_variant."""
        assert self.mod.SCENARIO_ID == "NNX-1"
        assert self.mod.TIER1_VARIANT is None
        assert isinstance(self.mod.VARIANTS, dict)
        assert callable(self.mod.get_variant)

    def test_variant_configs_are_valid(self):
        """Each variant must have required ScenarioConfig fields."""
        expected_variants = {"nnx_module", "plain_function"}
        assert set(self.mod.VARIANTS.keys()) == expected_variants

        for name, variant in self.mod.VARIANTS.items():
            assert_valid_variant(variant)
            assert variant.config.scenario_id == "NNX-1"
            assert variant.config.extra["variant_name"] == name
            assert "mode" in variant.config.extra

    def test_mode_values(self):
        """NNX variant uses 'nnx' mode, plain uses 'plain' mode."""
        nnx = self.mod.get_variant("nnx_module")
        assert nnx.config.extra["mode"] == "nnx"
        plain = self.mod.get_variant("plain_function")
        assert plain.config.extra["mode"] == "plain"

    def test_data_generation_shapes(self):
        """Both variants produce (50K, 224, 224, 3) float32 images."""
        for name in ("nnx_module", "plain_function"):
            variant = self.mod.get_variant(name)
            data = variant.data_generator()
            assert "image" in data
            img = data["image"]
            assert isinstance(img, np.ndarray)
            assert img.ndim == 4
            assert img.shape[0] == variant.config.dataset_size
            h, w, c = variant.config.element_shape
            assert img.shape[1:] == (h, w, c)
            assert img.dtype == np.float32

    def test_get_variant_raises_on_unknown(self):
        """get_variant must raise KeyError for unknown names."""
        with pytest.raises(KeyError):
            self.mod.get_variant("nonexistent")


# ===========================================================================
# XFMR-1: JIT + vmap Transform Acceleration
# ===========================================================================


class TestXFMR1Scenario:
    """Tests for xfmr1_jit_vmap scenario module."""

    @pytest.fixture(autouse=True)
    def _import_module(self):
        from benchmarks.scenarios.datarax_unique import xfmr1_jit_vmap as mod

        self.mod = mod

    def test_module_exports(self):
        """XFMR-1 must export SCENARIO_ID, VARIANTS, TIER1_VARIANT, get_variant."""
        assert self.mod.SCENARIO_ID == "XFMR-1"
        assert self.mod.TIER1_VARIANT is None
        assert isinstance(self.mod.VARIANTS, dict)
        assert callable(self.mod.get_variant)

    def test_variant_configs_are_valid(self):
        """Each resolution variant must have valid config."""
        expected_variants = {"32x32", "128x128", "512x512"}
        assert set(self.mod.VARIANTS.keys()) == expected_variants

        for name, variant in self.mod.VARIANTS.items():
            assert_valid_variant(variant)
            assert variant.config.scenario_id == "XFMR-1"
            assert variant.config.extra["variant_name"] == name
            assert "modes" in variant.config.extra

    def test_resolution_specific_configs(self):
        """Each resolution has appropriate dataset_size and batch_size."""
        v32 = self.mod.get_variant("32x32")
        assert v32.config.dataset_size == 50_000
        assert v32.config.element_shape == (32, 32, 3)
        assert v32.config.batch_size == 128

        v128 = self.mod.get_variant("128x128")
        assert v128.config.dataset_size == 50_000
        assert v128.config.element_shape == (128, 128, 3)
        assert v128.config.batch_size == 64

        v512 = self.mod.get_variant("512x512")
        assert v512.config.dataset_size == 10_000
        assert v512.config.element_shape == (512, 512, 3)
        assert v512.config.batch_size == 16

    def test_modes_include_all_compilation_variants(self):
        """All variants must include the four compilation modes."""
        expected_modes = ["eager", "vmap", "jit_vmap", "jit_vmap_fused"]
        for name, variant in self.mod.VARIANTS.items():
            assert variant.config.extra["modes"] == expected_modes

    def test_data_generation_shapes(self):
        """Generated image data must have correct NHWC float32 shape."""
        variant = self.mod.get_variant("32x32")
        data = variant.data_generator()
        assert "image" in data
        img = data["image"]
        assert isinstance(img, np.ndarray)
        assert img.ndim == 4
        assert img.shape[0] == variant.config.dataset_size
        h, w, c = variant.config.element_shape
        assert img.shape[1:] == (h, w, c)
        assert img.dtype == np.float32
