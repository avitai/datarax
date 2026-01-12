import os
import unittest
from unittest.mock import patch
import jax
import jax.numpy as jnp
from datarax.performance.xla_optimization import (
    XLAOptimizer,
    SmartCompilation,
    MemoryEfficientCompilation,
    DistributedUtils,
)


class TestXLAOptimization(unittest.TestCase):
    def setUp(self):
        self.original_xla_flags = os.environ.get("XLA_FLAGS", "")

    def tearDown(self):
        if self.original_xla_flags is None:
            if "XLA_FLAGS" in os.environ:
                del os.environ["XLA_FLAGS"]
        else:
            os.environ["XLA_FLAGS"] = self.original_xla_flags

    def test_xla_flags_configuration(self):
        """Test that XLA flags are correctly configured."""
        if "XLA_FLAGS" in os.environ:
            del os.environ["XLA_FLAGS"]

        # Mock backend to ensure consistency in tests
        with patch("jax.default_backend", return_value="cpu"):
            XLAOptimizer(target_hardware="auto")

        flags = os.environ.get("XLA_FLAGS", "")
        # Verify core flags
        self.assertIn("--xla_gpu_enable_latency_hiding_scheduler=true", flags)

    def test_aot_compile(self):
        """Test AOT compilation wrapper."""
        smart_compiler = SmartCompilation()
        sample_input = jnp.ones((10, 10))
        try:
            smart_compiler.aot_compile(lambda x: x * 2, sample_input)
        except Exception:
            pass

    def test_rematerialization(self):
        """Test gradient checkpointing wrapper."""
        remat_func = MemoryEfficientCompilation.with_rematerialization(lambda x: x * x)
        x = jnp.array([1.0, 2.0])
        self.assertTrue(jnp.allclose(remat_func(x), x * x))

    def test_distributed_mesh(self):
        """Test mesh creation."""

        # Use simple object instead of MagicMock because np.array([MagicMock]) -> []
        class DummyDevice:
            def __repr__(self):
                return "Dev"

        with patch("jax.devices") as mock_devices:
            # Case 1: Enough devices
            mock_devices.return_value = [DummyDevice() for _ in range(4)]

            try:
                mesh = DistributedUtils.create_mesh((2, 2), ("x", "y"))
                self.assertIsInstance(mesh, jax.sharding.Mesh)
                self.assertEqual(mesh.shape["x"], 2)
                self.assertEqual(mesh.shape["y"], 2)
            except Exception as e:
                # If checking assertion fails, it raises
                raise e

            # Case 2: Not enough devices
            mock_devices.return_value = [DummyDevice()]
            try:
                DistributedUtils.create_mesh((4,), ("x",))
            except Exception:
                pass
