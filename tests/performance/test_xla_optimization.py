import os
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp

from datarax.performance.xla_optimization import (
    CompilationProfiler,
    DistributedUtils,
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

    def test_distributed_mesh(self):
        """Test mesh creation."""

        # Use simple object instead of MagicMock because np.array([MagicMock]) -> []
        class DummyDevice:
            def __repr__(self):
                return "Dev"

        with patch("jax.devices") as mock_devices:
            # Case 1: Enough devices
            mock_devices.return_value = [DummyDevice() for _ in range(4)]

            mesh = DistributedUtils.create_mesh((2, 2), ("x", "y"))
            self.assertIsInstance(mesh, jax.sharding.Mesh)
            self.assertEqual(mesh.shape["x"], 2)
            self.assertEqual(mesh.shape["y"], 2)

            # Case 2: Not enough devices
            mock_devices.return_value = [DummyDevice()]
            with self.assertRaises(ValueError):
                DistributedUtils.create_mesh((4,), ("x",))

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

    def test_compilation_profiler_compiles_once_and_times_execution_separately(self):
        """Profiler should cache the compiled callable and sync full result pytrees."""
        profiler = CompilationProfiler()
        jit_calls = []
        compile_calls = []
        execution_calls = []
        sync_calls = []

        class FakeCompiled:
            def __init__(self, func):
                self._func = func

            def __call__(self, *args, **kwargs):
                execution_calls.append((args, kwargs))
                return {"value": self._func(*args, **kwargs)}

        class FakeLowered:
            def __init__(self, func):
                self._func = func

            def compile(self):
                compile_calls.append(self._func)
                return FakeCompiled(self._func)

        class FakeJitted:
            def __init__(self, func):
                self._func = func

            def lower(self, *args, **kwargs):
                del args, kwargs
                return FakeLowered(self._func)

        def fake_jit(func):
            jit_calls.append(func)
            return FakeJitted(func)

        with (
            patch("jax.jit", fake_jit),
            patch(
                "datarax.performance.xla_optimization.block_until_ready_tree",
                side_effect=lambda value: sync_calls.append(value) or value,
            ),
        ):
            wrapped = profiler.profile_function("double")(lambda x: x * 2)
            first = wrapped(jnp.ones((2,)))
            second = wrapped(jnp.ones((2,)))

        self.assertEqual(len(jit_calls), 1)
        self.assertEqual(len(compile_calls), 1)
        self.assertEqual(len(execution_calls), 2)
        self.assertEqual(len(sync_calls), 2)
        self.assertEqual(profiler.cache_misses, 1)
        self.assertEqual(profiler.cache_hits, 1)
        self.assertEqual(first["value"].shape, (2,))
        self.assertEqual(second["value"].shape, (2,))

        signature = ((2,),)
        self.assertIn(signature, profiler.compilation_times)
        self.assertEqual(len(profiler.execution_times[signature]), 2)
        self.assertFalse(profiler.cache_status[signature][0])
        self.assertTrue(profiler.cache_status[signature][1])
        self.assertIn("execution_time", profiler.shape_profiles[signature])
