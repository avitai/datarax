import pytest
import jax
import jax.numpy as jnp
from datarax.benchmarking.profiler import AdvancedProfiler, ProfilerConfig


@pytest.mark.benchmark
@pytest.mark.gpu
class TestBatchAlignment:
    """Tests the performance impact of hardware-aligned batch sizes (e.g. multiples of 128
    on TPU)."""

    def setup_method(self):
        try:
            jax.devices("gpu")
        except RuntimeError:
            pytest.skip("No GPU devices available for batch alignment testing.")

        self.profiler = AdvancedProfiler(
            config=ProfilerConfig(warmup_steps=10, measure_steps=100, enable_trace=True)
        )

    @pytest.mark.parametrize("batch_size", [127, 128, 255, 256])
    def test_matmul_alignment(self, batch_size):
        """Benchmarks a simple matmul operation with different batch sizes."""

        # Matrix size
        feature_dim = 1024

        # Create random keys
        key1, key2 = jax.random.split(jax.random.PRNGKey(0))

        # Inputs - use random to avoid constant folding (though JIT handles args)
        # We pre-generate inputs to isolate compute time
        A = jax.random.normal(key1, (batch_size, feature_dim))
        B = jax.random.normal(key2, (feature_dim, feature_dim))

        @jax.jit
        def matmul_op(a, b):
            return jnp.matmul(a, b)

        # Compile first
        _ = matmul_op(A, B).block_until_ready()

        def workload():
            _ = matmul_op(A, B).block_until_ready()

        print(f"\nBenchmarking MatMul Batch Size: {batch_size}")
        result = self.profiler.profile(workload, f"matmul_bs_{batch_size}")

        steps_per_sec = result.timing_metrics.get("iterations_per_second", 0.0)
        latency = result.timing_metrics.get("mean_iteration_time", 0.0) * 1000
        print(
            f"Batch Size {batch_size}: {steps_per_sec:.1f} steps/s, Avg Latency: {latency:.4f} ms"
        )

        # We don't assert strictly because CPU might not show difference,
        # but on TPU/GPU 128 should ideally be faster per-element than 127.
        return result

    @pytest.mark.parametrize("batch_size", [127, 128])
    def test_data_transfer_alignment(self, batch_size):
        """Benchmarks host-to-device transfer speed for aligned vs unaligned batches."""

        feature_dim = 1024 * 1024  # 1M floats ~ 4MB per row

        # Host data setup
        jnp.zeros((batch_size, feature_dim), dtype=jnp.float32)
        # Ensure it's on host? jnp.zeros puts it on default device usually.
        # Use numpy for true host data
        import numpy as np

        host_array = np.zeros((batch_size, feature_dim), dtype=np.float32)

        def transfer_workload():
            # Measure time to move to device
            device_array = jax.device_put(host_array)
            device_array.block_until_ready()

        print(f"\nBenchmarking Transfer Batch Size: {batch_size}")
        result = self.profiler.profile(transfer_workload, f"transfer_bs_{batch_size}")

        dataset_size_mb = (batch_size * feature_dim * 4) / 1e6
        steps_per_sec = result.timing_metrics.get("iterations_per_second", 0.0)
        bandwidth = dataset_size_mb * steps_per_sec
        print(f"Batch Size {batch_size}: {bandwidth:.2f} MB/s")
