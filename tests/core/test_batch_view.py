"""Tests for BatchView lightweight container.

BatchView is a plain Python object (not nnx.Module) that provides the same
dict-like interface as Batch but without NNX Variable overhead.
Used in the fused operator hot path for maximum iteration speed.

Test categories:
1. BatchView creation and data access
2. BatchView.to_batch() conversion
3. BatchView in pipeline iteration
4. Dict-like interface (__getitem__, __contains__, __iter__)
"""

import jax.numpy as jnp
from flax import nnx

from datarax.core.element_batch import Batch, BatchView


# ========================================================================
# Tests: Basic BatchView
# ========================================================================


class TestBatchViewBasic:
    """BatchView creation and data access."""

    def test_creation(self):
        """BatchView can be created from data dict."""
        data = {"image": jnp.ones((4, 8, 8, 3))}
        view = BatchView(data=data, states={}, batch_size=4)
        assert view.batch_size == 4

    def test_get_data(self):
        """get_data() returns the data dict."""
        data = {"image": jnp.ones((4, 8, 8, 3)), "label": jnp.zeros((4,))}
        view = BatchView(data=data, states={}, batch_size=4)
        result = view.get_data()
        assert set(result.keys()) == {"image", "label"}
        assert jnp.allclose(result["image"], jnp.ones((4, 8, 8, 3)))

    def test_getitem(self):
        """Dict-like __getitem__ access."""
        data = {"image": jnp.ones((4, 8, 8, 3))}
        view = BatchView(data=data, states={}, batch_size=4)
        assert jnp.allclose(view["image"], jnp.ones((4, 8, 8, 3)))

    def test_contains(self):
        """Dict-like __contains__ check."""
        data = {"image": jnp.ones((4, 8, 8, 3))}
        view = BatchView(data=data, states={}, batch_size=4)
        assert "image" in view
        assert "missing_key" not in view

    def test_iter(self):
        """Dict-like iteration over keys."""
        data = {"image": jnp.ones((4, 8, 8, 3)), "label": jnp.zeros((4,))}
        view = BatchView(data=data, states={}, batch_size=4)
        keys = list(view)
        assert set(keys) == {"image", "label"}


# ========================================================================
# Tests: to_batch() conversion
# ========================================================================


class TestBatchViewToBatch:
    """BatchView.to_batch() creates valid NNX Batch."""

    def test_to_batch_type(self):
        """to_batch() returns a Batch instance."""
        data = {"image": jnp.ones((4, 8, 8, 3))}
        view = BatchView(data=data, states={}, batch_size=4)
        batch = view.to_batch()
        assert isinstance(batch, Batch)

    def test_to_batch_preserves_data(self):
        """to_batch() preserves data values."""
        data = {"image": jnp.ones((4, 8, 8, 3)) * 2.0}
        view = BatchView(data=data, states={}, batch_size=4)
        batch = view.to_batch()
        assert jnp.allclose(batch.data.get_value()["image"], data["image"])

    def test_to_batch_preserves_batch_size(self):
        """to_batch() preserves batch_size."""
        data = {"image": jnp.ones((8, 4, 4, 3))}
        view = BatchView(data=data, states={}, batch_size=8)
        batch = view.to_batch()
        assert batch.batch_size == 8

    def test_to_batch_with_states(self):
        """to_batch() handles states."""
        data = {"image": jnp.ones((4, 8, 8, 3))}
        states = {"count": jnp.zeros((4,))}
        view = BatchView(data=data, states=states, batch_size=4)
        batch = view.to_batch()
        assert jnp.allclose(batch.states.get_value()["count"], jnp.zeros((4,)))


# ========================================================================
# Tests: Not an NNX Module
# ========================================================================


class TestBatchViewLightweight:
    """BatchView is NOT an NNX Module — it's a plain Python object."""

    def test_not_nnx_module(self):
        """BatchView is not an nnx.Module."""
        data = {"image": jnp.ones((4, 8, 8, 3))}
        view = BatchView(data=data, states={}, batch_size=4)
        assert not isinstance(view, nnx.Module)

    def test_has_slots(self):
        """BatchView uses __slots__ for minimal memory."""
        data = {"image": jnp.ones((4, 8, 8, 3))}
        view = BatchView(data=data, states={}, batch_size=4)
        assert hasattr(BatchView, "__slots__")
        # __slots__ objects don't have __dict__ (unless inherited)
        assert not hasattr(view, "__dict__")
