"""Tests for the differentiable RebatchNode."""

import jax
import jax.numpy as jnp
import pytest

from datarax.pipeline.nodes import RebatchNode


class TestRebatchNode:
    """RebatchNode differentiably regroups the batch axis."""

    def test_regroups_leading_axis(self):
        out = RebatchNode(group_size=4)({"x": jnp.ones((8, 3))})
        assert out["x"].shape == (2, 4, 3)

    def test_rejects_non_divisible_batch(self):
        with pytest.raises(ValueError, match="not divisible"):
            RebatchNode(group_size=3)({"x": jnp.ones((8, 3))})

    def test_rejects_invalid_group_size(self):
        with pytest.raises(ValueError, match="group_size"):
            RebatchNode(group_size=0)

    def test_gradient_flows(self):
        node = RebatchNode(group_size=2)

        def loss(x):
            return jnp.sum(node({"x": x})["x"] ** 2)

        grad = jax.grad(loss)(jnp.ones((4, 3)))
        assert grad.shape == (4, 3)
        assert bool(jnp.all(grad != 0))

    def test_jit_safe(self):
        node = RebatchNode(group_size=2)
        out = jax.jit(node)({"x": jnp.ones((4, 3))})
        assert out["x"].shape == (2, 2, 3)

    def test_regroup_is_reshape(self):
        x = jnp.arange(4 * 3).reshape(4, 3).astype(jnp.float32)
        out = RebatchNode(group_size=2)({"x": x})
        assert bool(jnp.array_equal(out["x"].reshape(4, 3), x))
