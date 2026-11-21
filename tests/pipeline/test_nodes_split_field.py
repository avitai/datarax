"""Tests for the SplitField node."""

import jax.numpy as jnp
import pytest

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

    def test_a_spelling_selects_a_nested_field_by_resolution(self):
        """A name resolves against the record (N3), so ``meta/id`` reaches the nested field."""
        batch = batch_ops.from_arrays(
            {"meta": {"id": jnp.arange(2), "score": jnp.ones(2)}, "image": jnp.ones((2, 4))}
        )
        out = SplitField(["meta/id"])(batch)
        assert set(out.data) == {"meta"} and set(out.data["meta"]) == {"id"}

    def test_a_key_tuple_selects_its_exact_path(self):
        batch = batch_ops.from_arrays({"meta": {"id": jnp.arange(2)}, "meta/id": jnp.ones(2)})
        out = SplitField([("meta", "id")])(batch)
        assert set(out.data) == {"meta"}
        assert bool(jnp.array_equal(out.data["meta"]["id"], jnp.arange(2)))

    def test_a_literal_key_holding_a_slash_stays_one_field(self):
        batch = batch_ops.from_arrays({"image": jnp.ones((2, 4)), "image/id": jnp.arange(2)})
        assert set(SplitField(["image/id"])(batch).data) == {"image/id"}

    def test_a_bare_string_is_refused(self):
        """A bare string would otherwise select its characters as names."""
        with pytest.raises(TypeError, match="SplitField's fields names several fields"):
            SplitField("image")
