"""DAG complexity scaling benchmark.

Uses TimingCollector for measurement (replaces AdvancedProfiler).
"""

import jax.numpy as jnp
import pytest

from datarax.benchmarking.timing import TimingCollector
from datarax.core.element_batch import Batch
from datarax.dag.dag_executor import DAGExecutor
from tests.benchmarks.complex_dag_builder import ComplexDAGBuilder


@pytest.mark.benchmark
class TestDAGScaling:
    """Benchmarks specific to DAG complexity scaling."""

    def _create_batch(self, batch_size):
        data = {"x": jnp.ones((batch_size, 64))}
        return Batch.from_parts(data=data, states={})

    def _measure_workload(self, workload_fn, warmup=5, iterations=20):
        """Measure a zero-arg workload function."""
        # Warmup
        for _ in range(warmup):
            workload_fn()

        # Measure: create an iterator that yields workload results
        def workload_iter():
            for _ in range(iterations):
                yield workload_fn()

        sync_fn = lambda: jnp.array(0.0).block_until_ready()
        collector = TimingCollector(sync_fn=sync_fn)
        return collector.measure_iteration(workload_iter(), num_batches=iterations)

    @pytest.mark.parametrize("depth", [1, 5, 10, 20])
    def test_dag_depth_scaling(self, depth):
        """Benchmarks execution overhead as DAG depth increases."""
        graph = ComplexDAGBuilder.build_linear_chain(length=depth, compute_intensity=10)
        executor = DAGExecutor(graph=graph, jit_compile=True)
        batch = self._create_batch(128)

        sample = self._measure_workload(lambda: executor(batch))
        steps_per_sec = (
            sample.num_batches / sample.wall_clock_sec if sample.wall_clock_sec > 0 else 0
        )
        print(f"Depth {depth}: {steps_per_sec:.1f} steps/s")

    @pytest.mark.parametrize("width", [1, 4, 8, 16])
    def test_dag_width_scaling(self, width):
        """Benchmarks execution overhead as DAG width increases."""
        graph = ComplexDAGBuilder.build_width_fanout(width=width, compute_intensity=10)
        executor = DAGExecutor(graph=graph, jit_compile=True)
        batch = self._create_batch(128)

        sample = self._measure_workload(lambda: executor(batch))
        steps_per_sec = (
            sample.num_batches / sample.wall_clock_sec if sample.wall_clock_sec > 0 else 0
        )
        print(f"Width {width}: {steps_per_sec:.1f} steps/s")

    def test_dag_mixed_topology(self):
        """Benchmarks a complex mixed topology."""
        depth, width = 5, 4
        graph = ComplexDAGBuilder.build_mixed_topology(depth=depth, width=width)
        executor = DAGExecutor(graph=graph, jit_compile=True)
        batch = self._create_batch(128)

        sample = self._measure_workload(lambda: executor(batch))
        steps_per_sec = (
            sample.num_batches / sample.wall_clock_sec if sample.wall_clock_sec > 0 else 0
        )
        print(f"Mixed (D={depth}, W={width}): {steps_per_sec:.1f} steps/s")
