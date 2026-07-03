"""Tests for the SplitField node."""

import jax.numpy as jnp

from datarax.pipeline.nodes import SplitField


class TestSplitField:
    """SplitField keeps a named subset of a batch dict's fields."""

    def test_selects_subset(self):
        out = SplitField(["image"])({"image": jnp.ones((2, 4)), "label": jnp.zeros((2,))})
        assert set(out) == {"image"}

    def test_skips_absent_fields(self):
        out = SplitField(["image", "missing"])({"image": jnp.ones((2, 4))})
        assert set(out) == {"image"}

    def test_empty_when_none_present(self):
        assert SplitField(["missing"])({"image": jnp.ones((2, 4))}) == {}

    def test_preserves_field_values(self):
        x = jnp.arange(6).reshape(2, 3)
        out = SplitField(["a"])({"a": x, "b": jnp.zeros((2, 3))})
        assert bool(jnp.array_equal(out["a"], x))
