"""PRNG handling utilities for Datarax.

This module provides utilities for managing random number generation in Datarax,
focusing on compatibility with Flax NNX and JAX's functional paradigm.

Key Utilities:
    create_rngs: Creates `flax.nnx.Rngs` objects with multiple named streams
                 derived from a single master seed.

Usage Note:
    Internal Datarax code should exclusively use `flax.nnx.Rngs` for randomness.
    When interfacing with external libraries that require raw JAX PRNG keys, call
    the specific stream (e.g., `rngs.params()`) to generate a unique key.
"""

import logging

import flax.nnx as nnx
import jax
import jax.numpy as jnp


logger = logging.getLogger(__name__)


def per_record_keys(base_key: jax.Array, global_indices: jax.Array) -> jax.Array:
    """Derive one stateless PRNG key per record from its stable global index.

    This is the JAX-native analogue of Grain's per-record ``Philox(seed + index)``
    contract: each record's key is ``jax.random.fold_in(base_key, index)``, so a
    record's randomness depends only on ``(base_key, global_index)`` — never on
    batch position, batch composition, shuffle order, host count, or resume
    point. Feeding these keys to stochastic operators makes augmentation
    reproducible across all of those, which is the determinism guarantee Datarax
    advertises.

    Args:
        base_key: A stable per-operator PRNG key (drawn once, not per batch).
        global_indices: Integer array ``(batch_size,)`` of monotonic global
            record indices (e.g. the batch start position plus the in-batch
            offset).

    Returns:
        A key array of shape ``(batch_size, ...)`` — one key per record, aligned
        with ``global_indices``.
    """
    indices = jnp.asarray(global_indices, dtype=jnp.uint32)
    return jax.vmap(lambda index: jax.random.fold_in(base_key, index))(indices)


# Standard stream names for consistency
DEFAULT_RNG_STREAMS = ["augment", "dropout", "params", "shuffling", "default"]


def create_rngs(seed: int | None = None, streams: list[str] | None = None) -> nnx.Rngs:
    """Create an Rngs object with the specified streams.

    This is a convenience function that creates multiple RNG streams from a single seed.
    For simple cases, you can use nnx.Rngs directly:
        rngs = nnx.Rngs(42)  # Single default stream
        rngs = nnx.Rngs(params=0, dropout=1)  # Multiple streams

    Args:
        seed: Optional seed for PRNG. Defaults to 0.
        streams: List of stream names. Defaults to DEFAULT_RNG_STREAMS.

    Returns:
        An nnx.Rngs object with keys for each stream.
    """
    if streams is None:
        streams = DEFAULT_RNG_STREAMS

    seed = seed if seed is not None else 0
    key = jax.random.key(seed)
    keys = jax.random.split(key, len(streams))
    return nnx.Rngs(**{name: keys[i] for i, name in enumerate(streams)})
