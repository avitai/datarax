"""Foundational per-record PRNG helpers for Datarax.

Lives in ``datarax.core`` because the operator execution model (the lowest
architectural layer) derives per-record keys here. The named ``nnx.Rngs``
streams a component draws from are :data:`DEFAULT_RNG_STREAMS`; an ``nnx.Rngs``
over them comes from :func:`substrax.rng.rngs_from_seed`, which derives each
stream's key from the seed and the stream's name.
"""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


DEFAULT_RNG_STREAMS: tuple[str, ...] = ("augment", "dropout", "params", "shuffling", "default")
"""The streams a component built from a seeded configuration receives."""


def record_key(
    base_key: jax.Array,
    index: ArrayLike,
    epoch: ArrayLike,
    draw: ArrayLike,
) -> jax.Array:
    """Derive one record's stateless PRNG key from its epoch, draw and 64-bit index.

    The key is ``fold_in(fold_in(fold_in(fold_in(base_key, epoch), draw), hi), lo)``. It depends
    only on ``(base_key, epoch, draw, index)``: not on batch size, batch position, shuffle order,
    padding, how records are split across workers or processes, where a run resumed, or which
    wrapper the operator sits in. The same record draws afresh in every epoch (the epoch), and
    twice in one epoch when a sampler serves it twice (the draw). The index is folded as two
    uint32 words, since ``fold_in`` takes 32-bit data, so indices past ``2^32`` keep distinct
    keys. Every key hangs off one parent, the wide key tree jax recommends.

    Args:
        base_key: A stable per-operator PRNG key (drawn once, not per batch).
        index: uint32 ``(2,)``, the record's index as ``(hi, lo)``.
        epoch: int32 scalar, the record's epoch.
        draw: int32 scalar, the record's draw within its epoch.

    Returns:
        The record's key.
    """
    words = jnp.asarray(index, jnp.uint32)
    key = jax.random.fold_in(jax.random.fold_in(base_key, epoch), draw)
    return jax.random.fold_in(jax.random.fold_in(key, words[0]), words[1])


def per_record_keys(
    base_key: jax.Array,
    indices: ArrayLike,
    epochs: ArrayLike,
    draws: ArrayLike,
) -> jax.Array:
    """Derive every record's key, ``record_key`` mapped over a batch's identities.

    Args:
        base_key: A stable per-operator PRNG key (drawn once, not per batch).
        indices: uint32 ``(B, 2)``, each record's index as ``(hi, lo)``.
        epochs: int32 ``(B,)``, each record's epoch.
        draws: int32 ``(B,)``, each record's draw within its epoch.

    Returns:
        A key array of shape ``(B,)``, one key per record, aligned with ``indices``.
    """
    return jax.vmap(record_key, in_axes=(None, 0, 0, 0))(
        base_key, jnp.asarray(indices, jnp.uint32), jnp.asarray(epochs), jnp.asarray(draws)
    )
