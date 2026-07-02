"""Foundational per-record PRNG helpers for Datarax.

Lives in ``datarax.core`` because the operator execution model (the lowest
architectural layer) derives per-record keys here; higher layers reach these
helpers through :mod:`datarax.utils.prng`, which re-exports them alongside the
``nnx.Rngs`` conveniences.
"""

import jax
import jax.numpy as jnp


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
