"""Tests for distributed benchmark scenarios DIST-1 and DIST-2.

RED phase: these tests define the expected API and behavior for each
distributed scenario module before implementation exists.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.tests.test_scenarios.conftest import assert_valid_variant


# ===========================================================================
# DIST-1: Multi-Device Sharding & Prefetch
# ===========================================================================


class TestDIST1Scenario:
    """Tests for dist1_multi_device scenario module."""

    @pytest.fixture(autouse=True)
    def _import_module(self):
        from benchmarks.scenarios.distributed import dist1_multi_device as mod

        self.mod = mod

    def test_module_exports(self):
        """DIST-1 must export SCENARIO_ID, VARIANTS, TIER1_VARIANT, get_variant."""
        assert self.mod.SCENARIO_ID == "DIST-1"
        assert self.mod.TIER1_VARIANT is None
        assert isinstance(self.mod.VARIANTS, dict)
        assert callable(self.mod.get_variant)

    def test_variant_configs_are_valid(self):
        """Each variant must have required ScenarioConfig fields."""
        expected_variants = {"1_device", "2_devices", "4_devices", "8_devices"}
        assert set(self.mod.VARIANTS.keys()) == expected_variants

        for name, variant in self.mod.VARIANTS.items():
            assert_valid_variant(variant)
            assert variant.config.scenario_id == "DIST-1"
            assert variant.config.extra["variant_name"] == name
            assert "device_count" in variant.config.extra

    def test_device_counts_match_variant_names(self):
        """Each variant's device_count must match its name."""
        expected = {"1_device": 1, "2_devices": 2, "4_devices": 4, "8_devices": 8}
        for name, count in expected.items():
            variant = self.mod.get_variant(name)
            assert variant.config.extra["device_count"] == count

    def test_data_generation_shapes(self):
        """Generated image data must have correct NHWC float32 shape."""
        variant = self.mod.get_variant("1_device")
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
# DIST-2: Device Mesh Configuration
# ===========================================================================


class TestDIST2Scenario:
    """Tests for dist2_device_mesh scenario module."""

    @pytest.fixture(autouse=True)
    def _import_module(self):
        from benchmarks.scenarios.distributed import dist2_device_mesh as mod

        self.mod = mod

    def test_module_exports(self):
        """DIST-2 must export SCENARIO_ID, VARIANTS, TIER1_VARIANT, get_variant."""
        assert self.mod.SCENARIO_ID == "DIST-2"
        assert self.mod.TIER1_VARIANT is None
        assert isinstance(self.mod.VARIANTS, dict)
        assert callable(self.mod.get_variant)

    def test_variant_configs_are_valid(self):
        """Each variant must have required ScenarioConfig fields."""
        expected_variants = {"mesh_1d", "mesh_2d"}
        assert set(self.mod.VARIANTS.keys()) == expected_variants

        for name, variant in self.mod.VARIANTS.items():
            assert_valid_variant(variant)
            assert variant.config.scenario_id == "DIST-2"
            assert variant.config.extra["variant_name"] == name
            assert "mesh_topology" in variant.config.extra

    def test_mesh_topology_values(self):
        """Each variant's mesh_topology must match its name."""
        assert self.mod.get_variant("mesh_1d").config.extra["mesh_topology"] == "1d"
        assert self.mod.get_variant("mesh_2d").config.extra["mesh_topology"] == "2d"

    def test_data_generation_shapes(self):
        """Generated image data must have correct NHWC float32 shape."""
        variant = self.mod.get_variant("mesh_1d")
        data = variant.data_generator()
        assert "image" in data
        img = data["image"]
        assert isinstance(img, np.ndarray)
        assert img.ndim == 4
        assert img.shape[0] == variant.config.dataset_size
        assert img.dtype == np.float32

    def test_get_variant_raises_on_unknown(self):
        """get_variant must raise KeyError for unknown names."""
        with pytest.raises(KeyError):
            self.mod.get_variant("nonexistent")
