import pytest
import jax
import jax.numpy as jnp
from datarax.benchmarking.profiler import AdvancedProfiler, ProfilerConfig
from datarax.dag.dag_executor import DAGExecutor
from datarax.core.element_batch import Batch
from tests.benchmarks.complex_dag_builder import ComplexDAGBuilder


@pytest.mark.benchmark
class TestDAGScaling:
    """Benchmarks specific to DAG complexity scaling."""

    @pytest.fixture(autouse=True)
    def setup_profiler(self):
        try:
            jax.devices("gpu")
            enable_gpu = True
            # Explicitly enable trace for detailed metrics if needed, but disable for speed
        except RuntimeError:
            enable_gpu = False

        self.profiler = AdvancedProfiler(
            config=ProfilerConfig(
                warmup_steps=5,
                measure_steps=20,
                enable_trace=True,
                # Enable trace to get timing_metrics populated properly by AdvancedProfiler
                # defaults?
                # Actually default profile_pipeline populates timing_metrics.
                enable_gpu_profiling=enable_gpu,
            )
        )

    def _create_batch(self, batch_size):
        data = {"x": jnp.ones((batch_size, 64))}
        # Batch.from_data does not exist. Use from_parts.
        return Batch.from_parts(data=data, states={})

    @pytest.mark.parametrize("depth", [1, 5, 10, 20])
    def test_dag_depth_scaling(self, depth):
        """Benchmarks execution overhead as DAG depth increases."""

        # Build DAG
        graph = ComplexDAGBuilder.build_linear_chain(length=depth, compute_intensity=10)
        executor = DAGExecutor(graph=graph, jit_compile=True)

        # Input: batch
        batch = self._create_batch(128)

        def run_dag():
            return executor(batch)

        # Profile
        result = self.profiler.profile(run_dag, f"dag_depth_{depth}")

        # metrics access
        steps_per_sec = result.timing_metrics.get("iterations_per_second", 0.0)
        latency = result.timing_metrics.get("mean_iteration_time", 0.0) * 1000
        print(f"Depth {depth}: {steps_per_sec:.1f} steps/s, Latency: {latency:.2f} ms")

    @pytest.mark.parametrize("width", [1, 4, 8, 16])
    def test_dag_width_scaling(self, width):
        """Benchmarks execution overhead as DAG width (parallel branches) increases."""

        graph = ComplexDAGBuilder.build_width_fanout(width=width, compute_intensity=10)
        executor = DAGExecutor(graph=graph, jit_compile=True)

        batch = self._create_batch(128)

        def run_dag():
            return executor(batch)

        result = self.profiler.profile(run_dag, f"dag_width_{width}")

        steps_per_sec = result.timing_metrics.get("iterations_per_second", 0.0)
        latency = result.timing_metrics.get("mean_iteration_time", 0.0) * 1000
        print(f"Width {width}: {steps_per_sec:.1f} steps/s, Latency: {latency:.2f} ms")

    def test_dag_mixed_topology(self):
        """Benchmarks a complex mixed topology (Depth x Width)."""
        depth = 5
        width = 4
        graph = ComplexDAGBuilder.build_mixed_topology(depth=depth, width=width)
        executor = DAGExecutor(graph=graph, jit_compile=True)

        batch = self._create_batch(128)

        def run_dag():
            return executor(batch)

        result = self.profiler.profile(run_dag, "dag_mixed_topology")

        steps_per_sec = result.timing_metrics.get("iterations_per_second", 0.0)
        latency = result.timing_metrics.get("mean_iteration_time", 0.0) * 1000
        print(
            f"Mixed (D={depth}, W={width}): {steps_per_sec:.1f} steps/s, Latency: {latency:.2f} ms"
        )
