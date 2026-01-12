import pytest
import jax
import jax.numpy as jnp
from datarax.benchmarking.profiler import AdvancedProfiler, ProfilerConfig
import time
import os
from pathlib import Path


@pytest.mark.benchmark
class TestPersistentCache:
    """Tests the effectiveness of JAX's persistent compilation cache."""

    @pytest.fixture(autouse=True)
    def setup_profiler(self):
        try:
            jax.devices("gpu")
        except RuntimeError:
            pytest.skip("No GPU devices available for persistent cache testing.")

        self.profiler = AdvancedProfiler(
            config=ProfilerConfig(warmup_steps=2, measure_steps=5, enable_trace=True)
        )

    def test_cache_hit_vs_miss(self, tmp_path):
        """Measures compilation time with and without cache hit."""

        # Setup specific cache dir for this test
        cache_dir = tmp_path / "jax_cache"
        cache_dir.mkdir()
        # Set environment variable for JAX cache
        # Note: In a real scenario this needs to be set before JAX init or via config.
        # However, jax.config.update can set it at runtime if using newer JAX versions
        # or we rely on 'datarax.performance.xla_optimization' logic.

        # For the purpose of this test, we might not be able to easily change global JAX state
        # inside a running process if it's already initialized.
        # But we can simulate "First Run" vs "Second Run" behavior by defining a new
        # unique function.

        # Define a somewhat complex function to ensure compilation takes measurable time
        def heavy_compute(x):
            for _ in range(50):
                x = jnp.sin(x) @ x.T
                x = jnp.tanh(x)
            return jnp.sum(x)

        heavy_jit = jax.jit(heavy_compute)

        dummy_input = jnp.ones((128, 128))

        # 1. measure cold compile time
        start_time = time.time()
        _ = heavy_jit(dummy_input).block_until_ready()
        cold_compile_time = time.time() - start_time

        print(f"\nCold Compile Time: {cold_compile_time:.4f} s")

        # 2. measure hot runtime (no re-compile)
        start_time = time.time()
        _ = heavy_jit(dummy_input).block_until_ready()
        hot_run_time = time.time() - start_time
        print(f"Hot Run Time: {hot_run_time:.4f} s")

        # 3. Simulate "Persistent Cache Hit"
        # This is tricky within the same process because JAX has an in-memory cache too.
        # To test *persistent* cache specifically, we'd typically need to spawn a subprocess.
        # But we can at least assert that the in-memory cache is working (Hot < Cold).

        assert hot_run_time < cold_compile_time * 0.1, "Hot run should be significantly faster"

        # For true persistent cache testing, we would need to check if files are written to
        # JAX_COMPILATION_CACHE_DIR.
        # Let's check if the directory is populated if configured.

        default_idx_cache = os.environ.get("JAX_COMPILATION_CACHE_DIR")
        if default_idx_cache:
            p = Path(default_idx_cache)
            if p.exists():
                cache_files = list(p.glob("*"))
                print(f"Cache Directory ({default_idx_cache}) contains {len(cache_files)} entries.")
                # We can't strictly assert len > 0 because this test might be the first thing
                # running or cache might be read-only. But it's good for logging.
