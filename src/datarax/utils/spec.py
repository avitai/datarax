"""Helpers for constructing and propagating ``jax.ShapeDtypeStruct`` PyTrees.

Datarax modules (sources, operators, batchers, samplers) declare their output
PyTree shape/dtype as a ``PyTree[jax.ShapeDtypeStruct]`` so downstream consumers
can pre-allocate buffers, auto-size learnable layers, and statically validate
operator chains. The helpers in this module centralize the leaf-level
manipulations so each tier doesn't reinvent them.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


_VALID_MASK_KEY = "valid_mask"


def add_leading_dim(spec_leaf: jax.ShapeDtypeStruct, size: int) -> jax.ShapeDtypeStruct:
    """Return a ``ShapeDtypeStruct`` with ``size`` prepended to ``spec_leaf.shape``.

    Used by Batchers to lift per-element specs into batch-level specs.
    """
    return jax.ShapeDtypeStruct(shape=(size, *spec_leaf.shape), dtype=spec_leaf.dtype)


def batched_spec(element_spec, batch_size: int) -> dict[str, object]:
    """Lift a per-element spec PyTree into a batch-level spec dict with ``valid_mask``.

    The returned dict has the same structure as ``element_spec`` (a leading
    ``batch_size`` dimension prepended to every ``ShapeDtypeStruct`` leaf) plus
    a top-level ``valid_mask`` leaf of shape ``(batch_size,)`` and dtype bool.

    Args:
        element_spec: PyTree of ``jax.ShapeDtypeStruct`` describing per-element
            output (typically a dict from a DataSourceModule's ``element_spec()``).
        batch_size: Number of elements per emitted batch.

    Returns:
        A dict with the batched element spec under the original keys plus a
        ``"valid_mask"`` key. If ``element_spec`` is itself a dict, its keys are
        merged in; otherwise it is placed under ``"data"``.

    Raises:
        ValueError: If ``batch_size`` is not positive.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    batched = jax.tree.map(
        lambda leaf: add_leading_dim(leaf, batch_size),
        element_spec,
        is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct),
    )

    valid_mask_leaf = jax.ShapeDtypeStruct(shape=(batch_size,), dtype=jnp.bool_)

    if isinstance(batched, dict):
        if _VALID_MASK_KEY in batched:
            raise ValueError(
                f"element_spec already contains key '{_VALID_MASK_KEY}'; "
                "this key is reserved for the batcher-injected validity mask."
            )
        return {**batched, _VALID_MASK_KEY: valid_mask_leaf}
    return {"data": batched, _VALID_MASK_KEY: valid_mask_leaf}


def scalar_index_spec(dtype=jnp.int32) -> jax.ShapeDtypeStruct:
    """Return the default ``ShapeDtypeStruct`` for a scalar sampler index.

    Most samplers emit a single integer index per call; specialized samplers
    (windowed, vectorized) override their tier's ``index_spec`` to return a
    different shape.
    """
    return jax.ShapeDtypeStruct(shape=(), dtype=dtype)


def array_to_spec(value) -> jax.ShapeDtypeStruct:
    """Return a ``ShapeDtypeStruct`` matching the shape and JAX dtype of ``value``.

    Used by sources to derive element specs from sample data. Canonicalizes
    NumPy dtypes to JAX dtypes via ``jnp.asarray`` so the resulting spec
    compares correctly against JAX-traced shapes downstream.
    """
    arr = jnp.asarray(value)
    return jax.ShapeDtypeStruct(shape=arr.shape, dtype=arr.dtype)


def array_to_spec_strip_leading(value) -> jax.ShapeDtypeStruct:
    """Return a ``ShapeDtypeStruct`` for a single element of a leading-batched array.

    Used by sources whose stored data has a leading dataset-size dimension
    (e.g., a dict of arrays where each array is shape ``(N, *element_shape)``).
    The returned spec describes one element by stripping the leading axis.

    Raises:
        ValueError: If ``value`` is a scalar (no leading dimension to strip).
    """
    arr = jnp.asarray(value)
    if arr.ndim == 0:
        raise ValueError(
            "array_to_spec_strip_leading requires an array with at least one "
            "axis (got a 0-d array)."
        )
    return jax.ShapeDtypeStruct(shape=arr.shape[1:], dtype=arr.dtype)


__all__ = [
    "add_leading_dim",
    "array_to_spec",
    "array_to_spec_strip_leading",
    "batched_spec",
    "scalar_index_spec",
]
