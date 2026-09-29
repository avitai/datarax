"""JAX / Flax NNX transform-compatibility tests for the pipeline DAG nodes.

The nodes run inside the jitted ``Pipeline.step`` and under ``nnx.scan``, so they
must compose with ``jax.jit``, ``jax.vmap``, ``jax.grad``, and — through a real
Pipeline — ``nnx.jit`` (step) and ``nnx.scan``.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from datarax import Pipeline
from datarax.core import batch_ops
from datarax.core.element_batch import Batch
from datarax.pipeline.nodes import SplitField
from datarax.sources import MemorySource, MemorySourceConfig


def _linear_pipeline(stage: nnx.Module, data: dict) -> Pipeline:
    source = MemorySource(
        config=MemorySourceConfig(shuffle=False), data=dict(data), rngs=nnx.Rngs(0)
    )
    return Pipeline(source=source, stages=[stage], batch_size=4, rngs=nnx.Rngs(0))


def _sum_step(batch: Batch) -> jax.Array:
    return batch["image"].astype(jnp.float32).sum()


def _two_fields() -> Batch:
    return batch_ops.from_arrays({"a": jnp.ones((2, 3)), "b": jnp.zeros((2, 3))})


class TestSplitFieldTransforms:
    """SplitField composes with jit, vmap, grad, and the Pipeline transforms."""

    def test_jit(self):
        out = jax.jit(SplitField(["a"]))(_two_fields())
        assert set(out.data) == {"a"}

    def test_vmap_over_a_batch_prefix(self):
        batch = _two_fields()
        axes = jax.tree.map(lambda _: 0, batch.replace(batch_state=None))
        out = jax.vmap(SplitField(["a"]), in_axes=(axes,))(batch)
        assert set(out.data) == {"a"} and out["a"].shape == (2, 3)

    def test_grad_passthrough(self):
        node = SplitField(["a"])

        def loss(x):
            return node(batch_ops.from_arrays({"a": x, "b": x}))["a"].sum()

        grad = jax.grad(loss)(jnp.ones((2, 3)))
        assert bool((grad == 1).all())

    def test_in_pipeline_step_and_scan(self):
        data = {"image": jnp.ones((16, 8)), "label": jnp.zeros((16, 1))}
        step_pipe = _linear_pipeline(SplitField(["image"]), data)
        assert sorted(step_pipe.step().data) == ["image"]
        scan_pipe = _linear_pipeline(SplitField(["image"]), data)
        assert scan_pipe.scan(_sum_step, length=2).shape == (2,)
