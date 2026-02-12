"""Tests for AdaptiveOperation hardware detection and optimization."""

import unittest
from unittest.mock import MagicMock, patch

import jax.numpy as jnp

from datarax.benchmarking.profiler import AdaptiveOperation


class TestAdaptiveOperation(unittest.TestCase):
    @patch("jax.devices")
    def test_adaptive_operation_gpu(self, mock_devices):
        """Test hardware detection for GPU."""
        mock_device = MagicMock()
        mock_device.device_kind = "NVIDIA A100-SXM4-40GB"
        mock_devices.return_value = [mock_device]

        with patch("jax.default_backend", return_value="gpu"):
            op = AdaptiveOperation()
            config = op.hw_config

            self.assertEqual(config["platform"], "gpu_modern")
            self.assertEqual(config["precision"], jnp.bfloat16)
            self.assertEqual(config["critical_batch_size"], 298)

    def test_grain_optimization_hook(self):
        """Test Grain optimization hook (mocking grain)."""
        op = AdaptiveOperation()

        mock_grain = MagicMock()
        mock_grain.experimental.pick_performance_config.return_value.read_options = (
            "mock_read_options"
        )
        mock_grain.experimental.pick_performance_config.return_value.multiprocessing_options = (
            "mock_mp_options"
        )

        mock_ds = MagicMock()
        mock_ds.to_iter_dataset.return_value = mock_ds
        mock_ds.mp_prefetch.return_value = mock_ds

        with patch.dict("sys.modules", {"grain.python": mock_grain}):
            with patch(
                "builtins.__import__",
                side_effect=lambda name, *args, **kwargs: mock_grain
                if name == "grain.python"
                else __import__(name, *args, **kwargs),
            ):
                pass

            res = op.optimize_grain_dataset(mock_ds)
            self.assertEqual(res, mock_ds)

    def test_optimize_shapes(self):
        """Test shape optimization for hardware alignment."""
        op = AdaptiveOperation()
        tile_size = op.hw_config["tile_size"]

        # Shape already aligned
        aligned = (128, 128)
        result = op.optimize_shapes(aligned)
        # Should remain same or be padded to tile_size multiples
        assert len(result) == 1

        # Shape not aligned
        unaligned = (100, 100)
        result = op.optimize_shapes(unaligned)
        assert len(result) == 1
        # Each dimension should be padded to next tile_size multiple
        for dim in result[0][-2:]:
            assert dim % tile_size == 0

    def test_cpu_default_config(self):
        """Test that CPU config has sensible defaults."""
        with patch("jax.default_backend", return_value="cpu"):
            op = AdaptiveOperation()
            config = op.hw_config
            assert config["platform"] == "cpu"
            assert config["tile_size"] == 64
            assert config["critical_batch_size"] == 32
