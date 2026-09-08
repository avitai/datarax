import os
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp

from datarax.performance.xla_optimization import (
    MemoryEfficientCompilation,
    SmartCompilation,
    XLAOptimizer,
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

    def test_xla_flags_cpu_backend(self):
        """Test that CPU-specific XLA flags are set for CPU backend."""
        if "XLA_FLAGS" in os.environ:
            del os.environ["XLA_FLAGS"]

        with patch("jax.default_backend", return_value="cpu"):
            XLAOptimizer(target_hardware="auto")

        flags = os.environ.get("XLA_FLAGS", "")
        self.assertIn("--xla_cpu_enable_fast_math=true", flags)
        self.assertNotIn("--xla_gpu_", flags)

    def test_xla_flags_gpu_backend(self):
        """Test that GPU-specific XLA flags are set for GPU backend."""
        if "XLA_FLAGS" in os.environ:
            del os.environ["XLA_FLAGS"]

        with patch("jax.default_backend", return_value="gpu"):
            XLAOptimizer(target_hardware="auto")

        flags = os.environ.get("XLA_FLAGS", "")
        self.assertIn("--xla_gpu_enable_latency_hiding_scheduler=true", flags)
        self.assertIn("--xla_gpu_strict_conv_algorithm_picker=false", flags)
        self.assertNotIn("--xla_gpu_enable_async_all_gather", flags)
        self.assertNotIn("--xla_gpu_enable_memory_space_assignment", flags)
        self.assertNotIn("--xla_cpu_", flags)

    def test_aot_compile(self):
        """Test AOT compilation wrapper."""
        smart_compiler = SmartCompilation()
        sample_input = jnp.ones((10, 10))
        try:
            smart_compiler.aot_compile(lambda x: x * 2, sample_input)
        except (NotImplementedError, RuntimeError, TypeError, ValueError):
            self.skipTest("AOT compilation is not available on this backend")

    def test_rematerialization(self):
        """Test gradient checkpointing wrapper."""
        remat_func = MemoryEfficientCompilation.with_rematerialization(lambda x: x * x)
        x = jnp.array([1.0, 2.0])
        self.assertTrue(jnp.allclose(remat_func(x), x * x))

    def test_shard_map_jit_uses_top_level_jax_api(self):
        """SmartCompilation should call top-level jax.shard_map."""
        compiler = SmartCompilation()
        mesh = jax.make_mesh((1,), ("data",))
        calls = []

        def fake_shard_map(func, *, mesh, in_specs, out_specs):
            calls.append((func, mesh, in_specs, out_specs))
            return "wrapped"

        with patch("jax.shard_map", fake_shard_map):
            decorator = compiler.shard_map_jit(mesh, in_specs=None, out_specs=None)
            result = decorator(lambda x: x)

        self.assertEqual(result, "wrapped")
        self.assertEqual(len(calls), 1)
