"""Tests the effectiveness of JAX's persistent compilation cache.

Uses direct time.perf_counter() for measurement (replaces AdvancedProfiler).
"""

import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest


@pytest.mark.benchmark
class TestPersistentCache:
    """Tests the effectiveness of JAX's persistent compilation cache."""

    @pytest.fixture(autouse=True)
    def check_gpu(self):
        try:
            jax.devices("gpu")
        except RuntimeError:
            pytest.skip("No GPU devices available for persistent cache testing.")

    def test_cache_hit_vs_miss(self, tmp_path):
        """Measures compilation time with and without cache hit."""

        def heavy_compute(x):
            for _ in range(50):
                x = jnp.sin(x) @ x.T
                x = jnp.tanh(x)
            return jnp.sum(x)

        heavy_jit = jax.jit(heavy_compute)
        dummy_input = jnp.ones((128, 128))

        # Cold compile time
        start_time = time.perf_counter()
        _ = heavy_jit(dummy_input).block_until_ready()
        cold_compile_time = time.perf_counter() - start_time

        print(f"\nCold Compile Time: {cold_compile_time:.4f} s")

        # Hot runtime (no re-compile)
        start_time = time.perf_counter()
        _ = heavy_jit(dummy_input).block_until_ready()
        hot_run_time = time.perf_counter() - start_time
        print(f"Hot Run Time: {hot_run_time:.4f} s")

        assert hot_run_time < cold_compile_time * 0.1, "Hot run should be significantly faster"

        default_idx_cache = os.environ.get("JAX_COMPILATION_CACHE_DIR")
        if default_idx_cache:
            p = Path(default_idx_cache)
            if p.exists():
                cache_files = list(p.glob("*"))
                print(f"Cache Directory ({default_idx_cache}) contains {len(cache_files)} entries.")
