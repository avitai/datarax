"""Foundational per-record PRNG helpers for Datarax.

Lives in ``datarax.core`` because the operator execution model (the lowest
architectural layer) derives per-record keys here; higher layers reach these
helpers through :mod:`datarax.utils.prng`, which re-exports them alongside the
``nnx.Rngs`` conveniences.
"""

import jax
import jax.numpy as jnp


def per_record_keys(
    base_key: jax.Array,
    record_indices: jax.Array,
    epoch: jax.Array | int | None = None,
) -> jax.Array:
    """Derive one stateless PRNG key per record from the epoch and the record's stable index.

    Each record's key is ``fold_in(fold_in(base_key, epoch), index)``. Within an epoch a
    record's randomness depends only on ``(base_key, epoch, index)``, so it does not change
    with batch size, batch position, shuffle order, how records are split across workers, or
    where a run resumed; every epoch draws a fresh key for each record. Grain likewise
    changes a record's randomness every epoch, seeding it from a draw index that keeps
    counting across epochs. Folding the epoch and the index in separately keeps each within
    ``fold_in``'s 32-bit data, and every key hangs directly off one parent, the wide key tree
    jax recommends.

    Args:
        base_key: A stable per-operator PRNG key (drawn once, not per batch).
        record_indices: Integer array ``(batch_size,)`` of stable record indices, as
            ``DataSourceModule.record_indices_at`` names them.
        epoch: The epoch counter, or ``None`` for keys that depend on the index alone.

    Returns:
        A key array of shape ``(batch_size, ...)`` — one key per record, aligned
        with ``record_indices``.
    """
    if epoch is not None:
        base_key = jax.random.fold_in(base_key, epoch)
    indices = jnp.asarray(record_indices, dtype=jnp.uint32)
    return jax.vmap(lambda index: jax.random.fold_in(base_key, index))(indices)
