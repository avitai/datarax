"""Tests for Batch.valid_mask — the per-position validity flag.

Batchers that pad partial last batches mark padded positions as invalid via this
mask, so mask-weighted loss (``sum(loss * mask) / max(sum(mask), 1)``) skips them
without forcing variable batch shapes that would trigger JIT recompilation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from datarax.core.element_batch import Batch, Element


def _make_element(value: float) -> Element:
    return Element(data={"x": jnp.asarray(value, dtype=jnp.float32)})


def test_batch_default_valid_mask_all_true_for_full_batch() -> None:
    """A non-empty batch built from real elements has all-True valid_mask.

    No padding means every position is valid; the mask is the identity for
    loss-weighting purposes.
    """
    batch = Batch([_make_element(1.0), _make_element(2.0), _make_element(3.0)])

    mask = batch.valid_mask[...]
    assert mask.shape == (3,)
    assert mask.dtype == jnp.bool_
    np.testing.assert_array_equal(np.asarray(mask), np.array([True, True, True]))


def test_batch_empty_has_zero_length_valid_mask() -> None:
    """An empty Batch has a zero-length mask (still a valid bool array)."""
    batch = Batch([])

    mask = batch.valid_mask[...]
    assert mask.shape == (0,)
    assert mask.dtype == jnp.bool_


def test_batch_from_parts_accepts_explicit_valid_mask() -> None:
    """``from_parts`` lets callers supply an explicit mask (e.g., padded last batch).

    A batcher building a partial last batch with two real elements and one
    pad position constructs a mask ``[True, True, False]`` so the loss
    aggregator can ignore the padded slot.
    """
    data = {"x": jnp.asarray([1.0, 2.0, 0.0], dtype=jnp.float32)}
    states = {}
    explicit_mask = jnp.asarray([True, True, False], dtype=jnp.bool_)

    batch = Batch.from_parts(data, states, valid_mask=explicit_mask)

    np.testing.assert_array_equal(np.asarray(batch.valid_mask[...]), np.array([True, True, False]))


def test_batch_from_parts_defaults_to_all_true_when_no_mask_supplied() -> None:
    """Without an explicit mask, ``from_parts`` defaults to all-True (no padding)."""
    data = {"x": jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)}
    states = {}

    batch = Batch.from_parts(data, states)

    np.testing.assert_array_equal(np.asarray(batch.valid_mask[...]), np.array([True, True, True]))


def test_batch_from_parts_rejects_mismatched_valid_mask_length() -> None:
    """A mask whose length does not match the batch size is a contract violation."""
    data = {"x": jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)}
    states = {}
    bad_mask = jnp.asarray([True, False], dtype=jnp.bool_)  # length 2 != 3

    with pytest.raises(ValueError, match="valid_mask"):
        Batch.from_parts(data, states, valid_mask=bad_mask)


def test_valid_mask_propagates_through_operator_apply_batch() -> None:
    """A Batch's ``valid_mask`` must survive an operator pass unchanged.

    Without this guarantee, a partial last batch with mask ``[T,T,F]`` would
    silently reset to all-True after the first operator, and downstream
    mask-weighted loss would treat the padded position as valid (a real bug).
    """
    from dataclasses import dataclass  # noqa: PLC0415

    from datarax.core.config import OperatorConfig  # noqa: PLC0415
    from datarax.core.operator import OperatorModule  # noqa: PLC0415

    @dataclass(frozen=True)
    class _NoOpConfig(OperatorConfig):
        pass

    class _NoOpOperator(OperatorModule):
        def apply(self, data, state, metadata, random_params=None, stats=None):  # type: ignore[override]
            return data, state, metadata

    op = _NoOpOperator(_NoOpConfig(stochastic=False))

    data = {"x": jnp.asarray([1.0, 2.0, 0.0], dtype=jnp.float32)}
    states = {}
    explicit_mask = jnp.asarray([True, True, False], dtype=jnp.bool_)
    batch_in = Batch.from_parts(data, states, valid_mask=explicit_mask)

    batch_out = op.apply_batch(batch_in)

    np.testing.assert_array_equal(
        np.asarray(batch_out.valid_mask[...]),
        np.array([True, True, False]),
    )


def test_valid_mask_slices_with_batch() -> None:
    """``Batch.slice`` must slice ``valid_mask`` consistently with data/states.

    Without this, a partial last batch sliced for distribution would silently
    revert to all-True for the slice, masking padded positions.
    """
    data = {"x": jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)}
    explicit_mask = jnp.asarray([True, True, True, False], dtype=jnp.bool_)
    batch = Batch.from_parts(data, {}, valid_mask=explicit_mask)

    sub = batch.slice(2, 4)

    np.testing.assert_array_equal(np.asarray(sub.valid_mask[...]), np.array([True, False]))


def test_valid_mask_concatenates_across_batches() -> None:
    """``BatchOps.concatenate_batch_sequence`` must concatenate ``valid_mask`` per input."""
    from datarax.core.element_batch import BatchOps  # noqa: PLC0415

    a = Batch.from_parts(
        {"x": jnp.asarray([1.0, 2.0], dtype=jnp.float32)},
        {},
        valid_mask=jnp.asarray([True, False], dtype=jnp.bool_),
    )
    b = Batch.from_parts(
        {"x": jnp.asarray([3.0, 4.0], dtype=jnp.float32)},
        {},
        valid_mask=jnp.asarray([True, True], dtype=jnp.bool_),
    )

    merged = BatchOps.concatenate_batch_sequence([a, b])

    np.testing.assert_array_equal(
        np.asarray(merged.valid_mask[...]),
        np.array([True, False, True, True]),
    )


def test_batch_valid_mask_traces_through_nnx_split() -> None:
    """``valid_mask`` participates in NNX state splitting (PyTree-compatible).

    ``nnx.split(batch)`` must return a state PyTree that contains the mask, so
    JIT-compiled scan bodies that thread Batch state through their carry can
    propagate the mask correctly.
    """
    from flax import nnx  # noqa: PLC0415 — avoid top-level import cost in tests

    batch = Batch([_make_element(1.0), _make_element(2.0)])
    _, state = nnx.split(batch)

    # Locate any leaf whose shape matches the mask shape and is bool dtype.
    leaves = jax.tree.leaves(state)
    bool_leaves = [
        leaf
        for leaf in leaves
        if isinstance(leaf, jax.Array) and leaf.dtype == jnp.bool_ and leaf.shape == (2,)
    ]
    assert len(bool_leaves) >= 1, (
        "Expected at least one bool array of shape (2,) in nnx.split(batch) state, "
        "but found none — valid_mask is not flowing through NNX state extraction."
    )


def test_batch_with_valid_mask_compiles_under_nnx_jit() -> None:
    """A function that consumes ``Batch.valid_mask`` must compile and cache under ``nnx.jit``.

    Regression guard for the Phase 1 / 4 contract: the new ``valid_mask`` field
    must not introduce a re-tracing trap in the JIT cache nor leak as a static
    argument that would force per-call recompilation.
    """
    from flax import nnx  # noqa: PLC0415

    @nnx.jit
    def mask_weighted_sum(batch: Batch) -> jax.Array:
        x = batch.data["x"]
        mask = batch.valid_mask[...].astype(x.dtype)
        return jnp.sum(x * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    data = {"x": jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)}
    mask = jnp.asarray([True, True, True, False], dtype=jnp.bool_)
    batch = Batch.from_parts(data, {}, valid_mask=mask)

    result = mask_weighted_sum(batch)
    # 3 valid positions [1, 2, 3] → mean 2.0
    assert jnp.isclose(result, 2.0)


def test_grad_flows_through_mask_weighted_loss() -> None:
    """``nnx.value_and_grad`` through a mask-weighted loss returns finite grads.

    This is the DADA pattern (``sum(loss * mask) / max(sum(mask), 1)``) that
    the differentiable-pipeline vision depends on. Padded positions must
    contribute exactly zero to the gradient — otherwise mask-weighted loss
    silently leaks gradient signal from padding into the learnable params.
    """
    from flax import nnx  # noqa: PLC0415

    class Scaler(nnx.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nnx.Param(jnp.asarray(2.0, dtype=jnp.float32))

        def __call__(self, batch: Batch) -> jax.Array:
            x = batch.data["x"]
            mask = batch.valid_mask[...].astype(x.dtype)
            scaled = self.w[...] * x
            target = jnp.asarray([2.0, 4.0, 6.0, 0.0], dtype=jnp.float32)
            per_example = (scaled - target) ** 2
            return jnp.sum(per_example * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    scaler = Scaler()
    data = {"x": jnp.asarray([1.0, 2.0, 3.0, 99.0], dtype=jnp.float32)}
    mask_with_padding = jnp.asarray([True, True, True, False], dtype=jnp.bool_)
    batch_padded = Batch.from_parts(data, {}, valid_mask=mask_with_padding)

    loss_fn = nnx.value_and_grad(lambda model: model(batch_padded))
    loss_padded, grads_padded = loss_fn(scaler)

    # With w=2 and target [2,4,6,_], the padded position (99) would dominate
    # the loss if mask-weighted aggregation were broken — the result would be
    # huge instead of zero.
    assert jnp.isclose(loss_padded, 0.0, atol=1e-6)

    # Gradients must be finite (no NaN/Inf from masked positions).
    grad_w = grads_padded.w[...]
    assert jnp.isfinite(grad_w)
    # And nearly zero at this synthetic optimum (positions 0..2 are exact fit).
    assert jnp.abs(grad_w) < 1e-5
