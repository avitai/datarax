"""Tests for platform resource detection and scenario memory gating.

TDD RED phase: defines expected behavior for memory-aware scenario filtering
across CPU (system RAM) and GPU (device VRAM) backends.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.core.platform import (
    estimate_scenario_memory_mb,
    get_available_memory_mb,
    can_run_scenario,
)
from benchmarks.scenarios.base import ScenarioVariant


# ---------------------------------------------------------------------------
# get_available_memory_mb
# ---------------------------------------------------------------------------


class TestGetAvailableMemoryMB:
    """Tests for system/device memory detection."""

    def test_returns_positive_float(self):
        """Must return a positive number representing available memory in MB."""
        mb = get_available_memory_mb()
        assert isinstance(mb, float)
        assert mb > 0

    def test_returns_reasonable_range(self):
        """Available memory should be between 100 MB and 10 TB."""
        mb = get_available_memory_mb()
        assert 100 < mb < 10_000_000

    def test_cpu_backend_returns_system_ram(self):
        """On CPU backend, should return system RAM."""
        mb = get_available_memory_mb(backend="cpu")
        assert isinstance(mb, float)
        assert mb > 0

    def test_gpu_backend_with_no_gpu_returns_system_ram(self):
        """On GPU backend without actual GPU, should fall back to system RAM."""
        mb = get_available_memory_mb(backend="gpu")
        assert isinstance(mb, float)
        assert mb > 0


# ---------------------------------------------------------------------------
# estimate_scenario_memory_mb
# ---------------------------------------------------------------------------


class TestEstimateScenarioMemoryMB:
    """Tests for dataset memory estimation."""

    def test_small_image_dataset(self):
        """CV-1 small: 10K x (32,32,3) float32 = ~117 MB."""
        mb = estimate_scenario_memory_mb(
            dataset_size=10_000,
            element_shape=(32, 32, 3),
        )
        assert abs(mb - 117.19) < 1.0

    def test_tiny_tabular_dataset(self):
        """TAB-1 small: 10K x (100,) float32 = ~3.8 MB."""
        mb = estimate_scenario_memory_mb(
            dataset_size=10_000,
            element_shape=(100,),
        )
        assert abs(mb - 3.81) < 0.5

    def test_large_image_dataset(self):
        """CV-1 large: 200K x (512,512,3) float32 = ~600K MB."""
        mb = estimate_scenario_memory_mb(
            dataset_size=200_000,
            element_shape=(512, 512, 3),
        )
        assert mb > 500_000

    def test_custom_dtype_bytes(self):
        """int8 data should use 1 byte per element instead of 4."""
        mb_f32 = estimate_scenario_memory_mb(
            dataset_size=1000,
            element_shape=(100,),
            dtype_bytes=4,
        )
        mb_i8 = estimate_scenario_memory_mb(
            dataset_size=1000,
            element_shape=(100,),
            dtype_bytes=1,
        )
        assert mb_f32 == pytest.approx(4 * mb_i8)

    def test_from_config(self):
        """Should work with ScenarioConfig fields directly."""
        config = ScenarioConfig(
            scenario_id="TEST",
            dataset_size=10_000,
            element_shape=(32, 32, 3),
            batch_size=32,
            transforms=[],
        )
        mb = estimate_scenario_memory_mb(
            dataset_size=config.dataset_size,
            element_shape=config.element_shape,
        )
        assert mb > 0


# ---------------------------------------------------------------------------
# can_run_scenario
# ---------------------------------------------------------------------------


class TestCanRunScenario:
    """Tests for the memory-gated scenario check."""

    def _make_variant(
        self,
        dataset_size: int,
        element_shape: tuple[int, ...],
    ) -> ScenarioVariant:
        return ScenarioVariant(
            config=ScenarioConfig(
                scenario_id="TEST",
                dataset_size=dataset_size,
                element_shape=element_shape,
                batch_size=32,
                transforms=[],
            ),
            data_generator=lambda: {},
        )

    def test_small_variant_can_run(self):
        """A tiny variant should always be runnable."""
        variant = self._make_variant(100, (10,))
        assert can_run_scenario(variant) is True

    def test_huge_variant_cannot_run(self):
        """A 600 GB variant should never be runnable on any real machine."""
        variant = self._make_variant(200_000, (512, 512, 3))
        assert can_run_scenario(variant) is False

    @patch("benchmarks.core.platform.get_available_memory_mb", return_value=3000.0)
    def test_safety_factor_applies(self, _mock_mem):
        """Higher safety factor should reject more scenarios."""
        # 3000 MB available, variant uses ~1500 MB
        # ds * 100 * 4 / 1024^2 = 1500 → ds = 1500 * 1024^2 / 400 = 3932160
        variant = self._make_variant(3_932_160, (100,))

        # safety_factor=2.5 → needs 3750 MB → should fail (> 3000)
        assert can_run_scenario(variant, safety_factor=2.5) is False
        # safety_factor=1.5 → needs 2250 MB → should pass (< 3000)
        assert can_run_scenario(variant, safety_factor=1.5) is True

    @patch("benchmarks.core.platform.get_available_memory_mb", return_value=1000.0)
    def test_with_mocked_memory(self, _mock_mem):
        """With 1000 MB available and safety_factor=2.5: max = 400 MB."""
        small = self._make_variant(1000, (100,))  # ~0.38 MB
        big = self._make_variant(100_000_000, (100,))  # ~38K MB

        assert can_run_scenario(small, safety_factor=2.5) is True
        assert can_run_scenario(big, safety_factor=2.5) is False

    @patch("benchmarks.core.platform.get_available_memory_mb", return_value=1000.0)
    def test_boundary_at_exact_limit(self, _mock_mem):
        """Variant exactly at the limit should be rejected (strict <)."""
        # 1000 / 2.5 = 400 MB limit
        # 400 MB = ds * 1 * 4 / 1024^2 → ds = 104857600
        variant = self._make_variant(104_857_600, (1,))
        assert can_run_scenario(variant, safety_factor=2.5) is False

    @patch("benchmarks.core.platform.get_available_memory_mb", return_value=16000.0)
    def test_gpu_memory_respected(self, _mock_mem):
        """When backend=gpu, the GPU VRAM limit should be used."""
        # 16 GB VRAM, safety_factor=2.5 → 6400 MB usable
        # Create a variant using 7000 MB → should fail
        shape = (100,)
        ds = int(7000 * 1024**2 / (100 * 4))
        variant = self._make_variant(ds, shape)
        assert can_run_scenario(variant, safety_factor=2.5) is False

        # Create a variant using 5000 MB → should pass
        ds_small = int(5000 * 1024**2 / (100 * 4))
        variant_small = self._make_variant(ds_small, shape)
        assert can_run_scenario(variant_small, safety_factor=2.5) is True
