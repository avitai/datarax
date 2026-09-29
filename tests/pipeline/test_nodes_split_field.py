"""Tests for the SplitField node."""

import jax.numpy as jnp

from datarax.core import batch_ops
from datarax.pipeline.nodes import SplitField


class TestSplitField:
    """SplitField keeps a named subset of a batch's data fields; identities pass through."""

    def test_selects_subset(self):
        batch = batch_ops.from_arrays({"image": jnp.ones((2, 4)), "label": jnp.zeros((2,))})
        out = SplitField(["image"])(batch)
        assert set(out.data) == {"image"}
        assert out.indices is batch.indices and out.epochs is batch.epochs

    def test_skips_absent_fields(self):
        out = SplitField(["image", "missing"])(batch_ops.from_arrays({"image": jnp.ones((2, 4))}))
        assert set(out.data) == {"image"}

    def test_empty_when_none_present(self):
        out = SplitField(["missing"])(batch_ops.from_arrays({"image": jnp.ones((2, 4))}))
        assert out.data == {}
        assert out.batch_size == 2

    def test_preserves_field_values(self):
        x = jnp.arange(6).reshape(2, 3)
        out = SplitField(["a"])(batch_ops.from_arrays({"a": x, "b": jnp.zeros((2, 3))}))
        assert bool(jnp.array_equal(out["a"], x))
