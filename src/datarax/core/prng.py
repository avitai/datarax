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


def per_record_keys(
    base_key: jax.Array,
    indices: ArrayLike,
    epochs: ArrayLike,
    draws: ArrayLike,
) -> jax.Array:
    """Derive one stateless PRNG key per record from its epoch, draw and 64-bit index.

    Record ``r``'s key is ``fold_in(fold_in(fold_in(fold_in(base_key, epoch), draw), hi), lo)``.
    It depends only on ``(base_key, epoch, draw, index)``: not on batch size, batch position,
    shuffle order, padding, how records are split across workers or processes, or where a run
    resumed. The same record draws afresh in every epoch (the epoch), and twice in one epoch
    when a sampler serves it twice (the draw). The index is folded as two uint32 words, since
    ``fold_in`` takes 32-bit data, so indices past ``2^32`` keep distinct keys. Every key hangs
    off one parent, the wide key tree jax recommends.

    Args:
        base_key: A stable per-operator PRNG key (drawn once, not per batch).
        indices: uint32 ``(B, 2)``, each record's index as ``(hi, lo)``.
        epochs: int32 ``(B,)``, each record's epoch.
        draws: int32 ``(B,)``, each record's draw within its epoch.

    Returns:
        A key array of shape ``(B,)``, one key per record, aligned with ``indices``.
    """

    def record_key(index: jax.Array, epoch: jax.Array, draw: jax.Array) -> jax.Array:
        key = jax.random.fold_in(jax.random.fold_in(base_key, epoch), draw)
        return jax.random.fold_in(jax.random.fold_in(key, index[0]), index[1])

    return jax.vmap(record_key)(
        jnp.asarray(indices, jnp.uint32), jnp.asarray(epochs), jnp.asarray(draws)
    )
