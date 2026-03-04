"""Tests for AdaptiveOperation hardware detection and shape optimization."""

from unittest.mock import MagicMock, patch

from calibrax.profiling import AdaptiveOperation


class TestAdaptiveOperation:
    @patch("jax.devices")
    @patch("jax.default_backend", return_value="gpu")
    def test_detects_modern_gpu_config(self, _mock_backend, mock_devices):
        """Modern GPUs should map to the gpu_modern profile."""
        mock_device = MagicMock()
        mock_device.device_kind = "NVIDIA A100-SXM4-40GB"
        mock_devices.return_value = [mock_device]

        op = AdaptiveOperation()
        config = op.config

        assert config.platform == "gpu_modern"
        assert config.precision == "bfloat16"
        assert config.tile_size == 16
        assert config.critical_batch_size == 298

    @patch("jax.devices")
    @patch("jax.default_backend", return_value="gpu")
    def test_detects_legacy_gpu_config(self, _mock_backend, mock_devices):
        """Older GPUs should map to the gpu_legacy profile."""
        mock_device = MagicMock()
        mock_device.device_kind = "Tesla V100-SXM2-32GB"
        mock_devices.return_value = [mock_device]

        op = AdaptiveOperation()
        config = op.config

        assert config.platform == "gpu_legacy"
        assert config.precision == "float32"
        assert config.tile_size == 32
        assert config.critical_batch_size == 128

    @patch("jax.default_backend", return_value="cpu")
    def test_cpu_default_config(self, _mock_backend):
        """CPU backend should use the default CPU tuning profile."""
        op = AdaptiveOperation()
        config = op.config

        assert config.platform == "cpu"
        assert config.precision == "float32"
        assert config.tile_size == 64
        assert config.critical_batch_size == 32

    @patch("jax.default_backend", return_value="cpu")
    def test_optimize_shapes(self, _mock_backend):
        """Shape optimization should pad trailing dims to tile-size multiples."""
        op = AdaptiveOperation()

        aligned = op.optimize_shapes((128, 128))
        assert aligned == [(128, 128)]

        unaligned = op.optimize_shapes((100, 100), (65, 63))
        assert unaligned == [(128, 128), (128, 64)]
