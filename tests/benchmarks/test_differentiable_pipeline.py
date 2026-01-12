import pytest
import jax
import jax.numpy as jnp
from datarax.benchmarking.profiler import AdvancedProfiler, ProfilerConfig
from datarax.dag.dag_executor import DAGExecutor
from datarax.core.element_batch import Batch
import flax.nnx as nnx
from tests.benchmarks.complex_dag_builder import ComplexDAGBuilder


@pytest.mark.benchmark
class TestDifferentiablePipeline:
    """Benchmarks for end-to-end differentiable pipelines (training loop simulation)."""

    @pytest.fixture(autouse=True)
    def setup_profiler(self):
        try:
            jax.devices("gpu")
            enable_gpu = True
        except RuntimeError:
            enable_gpu = False

        self.profiler = AdvancedProfiler(
            config=ProfilerConfig(
                warmup_steps=5, measure_steps=20, enable_trace=True, enable_gpu_profiling=enable_gpu
            )
        )

    def test_gradient_flow_and_overhead(self):
        """Verify gradient flow and benchmark training step overhead."""

        # 1. Build a simple differentiable DAG
        # Use a linear chain of compute operators
        input_dim = 64
        output_dim = 64
        depth = 5

        # We can reuse ComplexDAGBuilder or build manually
        graph = ComplexDAGBuilder.build_linear_chain(length=depth, compute_intensity=5)

        # 2. Create Executor (Model)
        executor = DAGExecutor(graph=graph, jit_compile=True)

        # 3. Create Batch
        batch_size = 128
        data = {"x": jnp.ones((batch_size, input_dim))}
        # Target for loss (dummy)
        target = jnp.zeros((batch_size, output_dim))

        batch = Batch.from_parts(data=data, states={})

        # 4. Define Loss Function
        def loss_fn(model, batch, target):
            # Forward pass
            output_batch = model(batch)

            # Extract output (assuming DummyComputeOperator modifies 'x' in place or we take 'x')
            # The DummyComputeOperator in complex_dag_builder maps 'x' -> 'x' (new_data = tree_map)
            pred = output_batch.data["x"]

            # Simple MSE loss
            loss = jnp.mean((pred - target) ** 2)
            return loss

        # 5. Define Training Step (Value and Grad)
        # We use nnx.value_and_grad
        grad_fn = nnx.value_and_grad(loss_fn)

        @nnx.jit
        def train_step(model, batch, target):
            loss, grads = grad_fn(model, batch, target)
            # Simulate optimizer update (simple SGD)
            # model.update(grads) # NNX update pattern might differ, usually we apply updates
            # to state
            # For benchmarking, calculating grads is enough to measure overhead.
            # But let's do a simple update to be realistic.
            lr = 0.01
            nnx.optimizer.sgd(model, grads, lr)  # does nnx have optimizer?
            # Or manually:
            # jax.tree_map(lambda p, g: p - lr * g, model.params, grads)
            # NNX Variables are mutable. We can update them in place?
            # Actually nnx.value_and_grad returns grads relative to parameters.
            # We need to apply them.
            # Since strict NNX separation:
            # We can use optax or simple loop.
            return loss

        # Manual update for now to avoid optax dependency issues if not installed/configured
        @nnx.jit
        def train_step_manual(model, batch, target):
            loss, grads = grad_fn(model, batch, target)

            # Apply gradients to parameters
            # model.parameters() returns a State of params?
            # Actually we can iterate over params.
            # Simplest: just value_and_grad is enough to prove differentiability.
            return loss, grads

        # 6. Verify Gradients (One-off)
        loss, grads = train_step_manual(executor, batch, target)
        assert loss > 0
        # Check if we have gradients
        # grads is a State/PyTree. Check leaves.
        grad_leaves = jax.tree.leaves(grads)
        assert len(grad_leaves) > 0
        # Check norms
        assert any(jnp.sum(jnp.abs(g)) > 0 for g in grad_leaves)

        print(f"Initial Loss: {loss}")

        # 7. Benchmark the Step
        def run_step():
            return train_step_manual(executor, batch, target)

        result = self.profiler.profile(run_step, "differentiable_pipeline_step")

        steps_per_sec = result.timing_metrics.get("iterations_per_second", 0.0)
        mean_time_ms = result.timing_metrics.get("mean_iteration_time", 0.0) * 1000

        print(f"Training Step: {steps_per_sec:.1f} steps/s, Time: {mean_time_ms:.2f} ms")
