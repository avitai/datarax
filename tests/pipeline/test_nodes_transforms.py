"""JAX / Flax NNX transform-compatibility tests for the pipeline DAG nodes.

The nodes run inside the jitted ``Pipeline.step`` and under ``nnx.scan``, so they
must compose with ``jax.jit``, ``jax.vmap``, ``jax.grad``, and — through a real
Pipeline — ``nnx.jit`` (step) and ``nnx.scan``.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from datarax import Pipeline
from datarax.pipeline.nodes import RebatchNode, SplitField
from datarax.sources import MemorySource, MemorySourceConfig


def _linear_pipeline(stage: nnx.Module, data: dict) -> Pipeline:
    source = MemorySource(
        config=MemorySourceConfig(shuffle=False), data=dict(data), rngs=nnx.Rngs(0)
    )
    return Pipeline(source=source, stages=[stage], batch_size=4, rngs=nnx.Rngs(0))


def _sum_step(batch: dict) -> jax.Array:
    return jax.tree.leaves(batch)[0].astype(jnp.float32).sum()


class TestRebatchNodeTransforms:
    """RebatchNode composes with every relevant JAX/NNX transform."""

    def test_jit(self):
        out = jax.jit(RebatchNode(2))({"x": jnp.ones((4, 5))})
        assert out["x"].shape == (2, 2, 5)

    def test_vmap(self):
        out = jax.vmap(RebatchNode(2))({"x": jnp.ones((3, 4, 5))})
        assert out["x"].shape == (3, 2, 2, 5)

    def test_grad(self):
        node = RebatchNode(2)

        def loss(x):
            return node({"x": x})["x"].sum()

        grad = jax.grad(loss)(jnp.ones((4, 3)))
        assert grad.shape == (4, 3)

    def test_in_pipeline_step_is_jitted(self):
        pipe = _linear_pipeline(RebatchNode(2), {"image": jnp.ones((16, 8))})
        batch = pipe.step()  # type: ignore[reportCallIssue]  # nnx.jit wrapper
        assert jax.tree.leaves(batch)[0].shape == (2, 2, 8)

    def test_in_pipeline_scan(self):
        pipe = _linear_pipeline(RebatchNode(2), {"image": jnp.ones((16, 8))})
        out = pipe.scan(_sum_step, length=2)
        assert out.shape == (2,)


class TestSplitFieldTransforms:
    """SplitField composes with jit, grad, and the Pipeline transforms."""

    def test_jit(self):
        out = jax.jit(SplitField(["a"]))({"a": jnp.ones((2, 3)), "b": jnp.zeros((2, 3))})
        assert set(out) == {"a"}

    def test_grad_passthrough(self):
        node = SplitField(["a"])

        def loss(x):
            return node({"a": x, "b": x})["a"].sum()

        grad = jax.grad(loss)(jnp.ones((2, 3)))
        assert bool((grad == 1).all())

    def test_in_pipeline_step_and_scan(self):
        data = {"image": jnp.ones((16, 8)), "label": jnp.zeros((16, 1))}
        step_pipe = _linear_pipeline(SplitField(["image"]), data)
        assert sorted(step_pipe.step().keys()) == ["image"]  # type: ignore[reportCallIssue]
        scan_pipe = _linear_pipeline(SplitField(["image"]), data)
        assert scan_pipe.scan(_sum_step, length=2).shape == (2,)
