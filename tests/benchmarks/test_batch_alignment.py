"""Tests the performance impact of hardware-aligned batch sizes.

Uses TimingCollector for measurement (replaces AdvancedProfiler).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from datarax.benchmarking.timing import TimingCollector


@pytest.mark.benchmark
@pytest.mark.gpu
class TestBatchAlignment:
    """Tests the performance impact of hardware-aligned batch sizes."""

    def setup_method(self):
        try:
            jax.devices("gpu")
        except RuntimeError:
            pytest.skip("No GPU devices available for batch alignment testing.")

    def _measure_workload(self, workload_fn, warmup=10, iterations=100):
        """Measure a zero-arg workload function."""
        for _ in range(warmup):
            workload_fn()

        def workload_iter():
            for _ in range(iterations):
                yield workload_fn()

        sync_fn = lambda: jnp.array(0.0).block_until_ready()
        collector = TimingCollector(sync_fn=sync_fn)
        return collector.measure_iteration(workload_iter(), num_batches=iterations)

    @pytest.mark.parametrize("batch_size", [127, 128, 255, 256])
    def test_matmul_alignment(self, batch_size):
        """Benchmarks matmul with different batch sizes."""
        feature_dim = 1024
        key1, key2 = jax.random.split(jax.random.PRNGKey(0))
        A = jax.random.normal(key1, (batch_size, feature_dim))
        B = jax.random.normal(key2, (feature_dim, feature_dim))

        @jax.jit
        def matmul_op(a, b):
            return jnp.matmul(a, b)

        # Compile first
        _ = matmul_op(A, B).block_until_ready()

        def workload():
            return matmul_op(A, B).block_until_ready()

        print(f"\nBenchmarking MatMul Batch Size: {batch_size}")
        sample = self._measure_workload(workload)

        steps_per_sec = (
            sample.num_batches / sample.wall_clock_sec if sample.wall_clock_sec > 0 else 0
        )
        latency = (
            (sample.wall_clock_sec / sample.num_batches * 1000) if sample.num_batches > 0 else 0
        )
        print(
            f"Batch Size {batch_size}: {steps_per_sec:.1f} steps/s, Avg Latency: {latency:.4f} ms"
        )

    @pytest.mark.parametrize("batch_size", [127, 128])
    def test_data_transfer_alignment(self, batch_size):
        """Benchmarks host-to-device transfer for aligned vs unaligned batches."""
        feature_dim = 1024 * 1024

        host_array = np.zeros((batch_size, feature_dim), dtype=np.float32)

        def transfer_workload():
            device_array = jax.device_put(host_array)
            device_array.block_until_ready()

        print(f"\nBenchmarking Transfer Batch Size: {batch_size}")
        sample = self._measure_workload(transfer_workload)

        dataset_size_mb = (batch_size * feature_dim * 4) / 1e6
        steps_per_sec = (
            sample.num_batches / sample.wall_clock_sec if sample.wall_clock_sec > 0 else 0
        )
        bandwidth = dataset_size_mb * steps_per_sec
        print(f"Batch Size {batch_size}: {bandwidth:.2f} MB/s")
