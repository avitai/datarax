"""Benchmarks for end-to-end differentiable pipelines.

Uses TimingCollector for measurement (replaces AdvancedProfiler).
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from datarax.benchmarking.timing import TimingCollector
from datarax.core.element_batch import Batch
from datarax.dag.dag_executor import DAGExecutor
from tests.benchmarks.complex_dag_builder import ComplexDAGBuilder


@pytest.mark.benchmark
class TestDifferentiablePipeline:
    """Benchmarks for end-to-end differentiable pipelines."""

    def test_gradient_flow_and_overhead(self):
        """Verify gradient flow and benchmark training step overhead."""
        input_dim = 64
        output_dim = 64
        depth = 5

        graph = ComplexDAGBuilder.build_linear_chain(length=depth, compute_intensity=5)
        executor = DAGExecutor(graph=graph, jit_compile=True)

        batch_size = 128
        data = {"x": jnp.ones((batch_size, input_dim))}
        target = jnp.zeros((batch_size, output_dim))
        batch = Batch.from_parts(data=data, states={})

        def loss_fn(model, batch, target):
            output_batch = model(batch)
            pred = output_batch.data["x"]
            loss = jnp.mean((pred - target) ** 2)
            return loss

        grad_fn = nnx.value_and_grad(loss_fn)

        @nnx.jit
        def train_step_manual(model, batch, target):
            loss, grads = grad_fn(model, batch, target)
            return loss, grads

        # Verify gradients
        loss, grads = train_step_manual(executor, batch, target)
        assert loss > 0
        grad_leaves = jax.tree.leaves(grads)
        assert len(grad_leaves) > 0
        assert any(jnp.sum(jnp.abs(g)) > 0 for g in grad_leaves)

        print(f"Initial Loss: {loss}")

        # Warmup
        for _ in range(5):
            train_step_manual(executor, batch, target)

        # Measure
        def workload_iter():
            for _ in range(20):
                yield train_step_manual(executor, batch, target)

        sync_fn = lambda: jnp.array(0.0).block_until_ready()
        collector = TimingCollector(sync_fn=sync_fn)
        sample = collector.measure_iteration(workload_iter(), num_batches=20)

        steps_per_sec = (
            sample.num_batches / sample.wall_clock_sec if sample.wall_clock_sec > 0 else 0
        )
        print(f"Training Step: {steps_per_sec:.1f} steps/s")
