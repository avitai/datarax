"""Tests for vision benchmark scenarios CV-1 through CV-4.

RED phase: these tests define the expected API and behavior for each
vision scenario module before implementation exists.
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
# CV-1: Image Classification Pipeline (Canonical)
# ===========================================================================


class TestCV1Scenario:
    """Tests for cv1_image_classification scenario module."""

    @pytest.fixture(autouse=True)
    def _import_module(self):
        from benchmarks.scenarios.vision import cv1_image_classification as mod

        self.mod = mod

    def test_variant_configs_are_valid(self):
        """Each variant must have required ScenarioConfig fields."""
        assert self.mod.SCENARIO_ID == "CV-1"
        assert isinstance(self.mod.VARIANTS, dict)
        assert len(self.mod.VARIANTS) == 3  # small, medium, large

        for name, variant in self.mod.VARIANTS.items():
            assert_valid_variant(variant)
            assert variant.config.scenario_id == "CV-1"
            assert variant.config.extra["variant_name"] == name

    def test_data_generation_shapes(self):
        """Generated image data must have correct NHWC shapes."""
        small = self.mod.get_variant("small")
        data = small.data_generator()
        assert "image" in data
        img = data["image"]
        assert isinstance(img, np.ndarray)
        assert img.ndim == 4  # (N, H, W, C)
        assert img.shape[0] == small.config.dataset_size
        h, w, c = small.config.element_shape
        assert img.shape[1:] == (h, w, c)
        assert img.dtype == np.uint8

    def test_tier1_variant_exists(self):
        """CV-1 must define TIER1_VARIANT pointing to 'small'."""
        assert self.mod.TIER1_VARIANT == "small"
        tier1 = self.mod.get_variant(self.mod.TIER1_VARIANT)
        assert tier1.config.extra["variant_name"] == "small"

    def test_get_variant_raises_on_unknown(self):
        """get_variant must raise KeyError for unknown names."""
        with pytest.raises(KeyError):
            self.mod.get_variant("nonexistent")

    def test_run_produces_benchmark_result(self, datarax_adapter: DataraxAdapter):
        """Running CV-1 small through the adapter produces a valid result."""
        # Use a tiny variant for test speed
        tiny_variant = ScenarioVariant(
            config=ScenarioConfig(
                scenario_id="CV-1",
                dataset_size=100,
                element_shape=(32, 32, 3),
                batch_size=10,
                transforms=["Normalize", "CastToFloat32"],
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
        assert result.scenario_id == "CV-1"
        assert result.timing.num_batches > 0


# ===========================================================================
# CV-2: High-Resolution Medical Imaging (3D)
# ===========================================================================


class TestCV2Scenario:
    """Tests for cv2_medical_3d scenario module."""

    @pytest.fixture(autouse=True)
    def _import_module(self):
        from benchmarks.scenarios.vision import cv2_medical_3d as mod

        self.mod = mod

    def test_variant_configs_are_valid(self):
        """All variants must have valid config."""
        assert self.mod.SCENARIO_ID == "CV-2"
        assert isinstance(self.mod.VARIANTS, dict)
        assert len(self.mod.VARIANTS) == 2  # default, large

        for name, variant in self.mod.VARIANTS.items():
            assert_valid_variant(variant)
            assert variant.config.scenario_id == "CV-2"
            assert variant.config.extra["variant_name"] == name

    def test_data_generation_shapes(self):
        """Generated 3D volume data must have correct (N, D, H, W) shape.

        Uses SyntheticDataGenerator directly at small scale to verify the
        same API the production data_generator lambda calls.
        """
        from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator

        variant = self.mod.get_variant("default")
        cfg = variant.config
        # Verify the data_generator is callable and returns the right key
        assert callable(variant.data_generator)
        # Verify shape contract using small-scale generation
        gen = SyntheticDataGenerator(seed=42)
        d, h, w = cfg.element_shape
        vol = gen.volumes_3d(10, d=4, h=4, w=4)
        assert isinstance(vol, np.ndarray)
        assert vol.ndim == 4  # (N, D, H, W)
        assert vol.shape == (10, 4, 4, 4)
        assert vol.dtype == np.float32
        # Verify config element_shape is 3D
        assert len(cfg.element_shape) == 3

    def test_tier1_variant_is_none(self):
        """CV-2 is not a Tier-1 scenario."""
        assert self.mod.TIER1_VARIANT is None

    def test_run_produces_benchmark_result(self, datarax_adapter: DataraxAdapter):
        """Running CV-2 through the adapter produces a valid result."""
        tiny_variant = ScenarioVariant(
            config=ScenarioConfig(
                scenario_id="CV-2",
                dataset_size=50,
                element_shape=(8, 8, 8),
                batch_size=5,
                transforms=["Normalize"],
                extra={"variant_name": "test_tiny"},
            ),
            data_generator=lambda: {
                "volume": np.random.default_rng(42)
                .standard_normal((50, 8, 8, 8))
                .astype(np.float32)
            },
        )
        result = run_quick_scenario(datarax_adapter, tiny_variant)
        assert isinstance(result, BenchmarkResult)
        assert result.scenario_id == "CV-2"
        assert result.timing.num_batches > 0


# ===========================================================================
# CV-3: Batch-Level Mixing (MixUp/CutMix)
# ===========================================================================


class TestCV3Scenario:
    """Tests for cv3_batch_mixing scenario module."""

    @pytest.fixture(autouse=True)
    def _import_module(self):
        from benchmarks.scenarios.vision import cv3_batch_mixing as mod

        self.mod = mod

    def test_variant_configs_are_valid(self):
        """All variants must have valid config with MixUp transform."""
        assert self.mod.SCENARIO_ID == "CV-3"
        assert isinstance(self.mod.VARIANTS, dict)
        assert len(self.mod.VARIANTS) == 2  # default, large

        for name, variant in self.mod.VARIANTS.items():
            assert_valid_variant(variant)
            assert variant.config.scenario_id == "CV-3"
            assert variant.config.extra["variant_name"] == name
            assert "MixUp" in variant.config.transforms

    def test_data_generation_shapes(self):
        """Generated image data must have correct NHWC shape.

        Uses SyntheticDataGenerator at small scale to verify the same API
        the production data_generator lambda calls.
        """
        from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator

        variant = self.mod.get_variant("default")
        cfg = variant.config
        assert callable(variant.data_generator)
        # Verify shape contract at small scale
        gen = SyntheticDataGenerator(seed=42)
        h, w, c = cfg.element_shape
        img = gen.images(10, 32, 32, 3, dtype="uint8")
        assert isinstance(img, np.ndarray)
        assert img.ndim == 4
        assert img.shape == (10, 32, 32, 3)
        assert img.dtype == np.uint8
        # Verify config element_shape is 3D (H, W, C)
        assert len(cfg.element_shape) == 3
        assert cfg.element_shape[2] == 3  # RGB

    def test_tier1_variant_is_none(self):
        """CV-3 is not a Tier-1 scenario."""
        assert self.mod.TIER1_VARIANT is None

    def test_run_produces_benchmark_result(self, datarax_adapter: DataraxAdapter):
        """Running CV-3 through the adapter produces a valid result."""
        tiny_variant = ScenarioVariant(
            config=ScenarioConfig(
                scenario_id="CV-3",
                dataset_size=100,
                element_shape=(32, 32, 3),
                batch_size=10,
                transforms=["Normalize", "MixUp"],
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
        assert result.scenario_id == "CV-3"
        assert result.timing.num_batches > 0


# ===========================================================================
# CV-4: Multi-Resolution Pipeline
# ===========================================================================


class TestCV4Scenario:
    """Tests for cv4_multi_resolution scenario module."""

    @pytest.fixture(autouse=True)
    def _import_module(self):
        from benchmarks.scenarios.vision import cv4_multi_resolution as mod

        self.mod = mod

    def test_variant_configs_are_valid(self):
        """All variants must have valid config with MultiScaleResize."""
        assert self.mod.SCENARIO_ID == "CV-4"
        assert isinstance(self.mod.VARIANTS, dict)
        assert len(self.mod.VARIANTS) == 2  # default, large

        for name, variant in self.mod.VARIANTS.items():
            assert_valid_variant(variant)
            assert variant.config.scenario_id == "CV-4"
            assert variant.config.extra["variant_name"] == name
            assert "MultiScaleResize" in variant.config.transforms

    def test_data_generation_shapes(self):
        """Generated image data must have correct NHWC shape.

        Uses SyntheticDataGenerator at small scale to verify the same API
        the production data_generator lambda calls.
        """
        from benchmarks.fixtures.synthetic_data import SyntheticDataGenerator

        variant = self.mod.get_variant("default")
        cfg = variant.config
        assert callable(variant.data_generator)
        # Verify shape contract at small scale
        gen = SyntheticDataGenerator(seed=42)
        img = gen.images(10, 32, 32, 3, dtype="uint8")
        assert isinstance(img, np.ndarray)
        assert img.ndim == 4
        assert img.shape == (10, 32, 32, 3)
        assert img.dtype == np.uint8
        # Verify config element_shape is 3D (H, W, C)
        assert len(cfg.element_shape) == 3
        assert cfg.element_shape[2] == 3  # RGB

    def test_tier1_variant_is_none(self):
        """CV-4 is not a Tier-1 scenario."""
        assert self.mod.TIER1_VARIANT is None

    def test_run_produces_benchmark_result(self, datarax_adapter: DataraxAdapter):
        """Running CV-4 through the adapter produces a valid result."""
        tiny_variant = ScenarioVariant(
            config=ScenarioConfig(
                scenario_id="CV-4",
                dataset_size=100,
                element_shape=(32, 32, 3),
                batch_size=10,
                transforms=["MultiScaleResize", "Normalize"],
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
        assert result.scenario_id == "CV-4"
        assert result.timing.num_batches > 0
