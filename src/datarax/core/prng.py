"""Foundational per-record PRNG helpers for Datarax.

Lives in ``datarax.core`` because the operator execution model (the lowest
architectural layer) derives per-record keys here. The named ``nnx.Rngs``
streams a component draws from are :data:`DEFAULT_RNG_STREAMS`; an ``nnx.Rngs``
over them comes from :func:`substrax.rng.rngs_from_seed`, which derives each
stream's key from the seed and the stream's name.

The host stage orders records and seeds stream passes on the host. It reads a key's data once
as uint32 words (:func:`key_words`) and folds them on the CPU device (:func:`host_device`,
:func:`fold_on_host`), so no fold reads back from an accelerator and none needs an implicit
transfer. Every key that names records is a :data:`NAMING_PRNG_IMPL` key, whatever the caller's
``jax_default_prng_impl``.
"""

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import SingleDeviceSharding
from jax.typing import ArrayLike


DEFAULT_RNG_STREAMS: tuple[str, ...] = ("augment", "dropout", "params", "shuffling", "default")
"""The streams a component built from a seeded configuration receives."""

NAMING_PRNG_IMPL = "threefry2x32"
"""The implementation of every key that orders or names records: an epoch's order, a stream's
pass seed. Pinned, so record identity never depends on the caller's ``jax_default_prng_impl``
(which is not part of a jitted program's cache key either, so a cached naming program would
otherwise keep the impl it was first traced under), and exact under ``jax.vmap`` over keys,
which a block of batches named in one call needs. A training step's own randomness follows the
caller's setting."""


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


def host_device() -> SingleDeviceSharding:
    """The CPU device, where the host stage names records and folds keys, as a placement.

    Returns:
        A sharding placing an array whole on the process's first CPU device.

    Raises:
        RuntimeError: If JAX was started without its CPU platform (``JAX_PLATFORMS=cuda``), which
            the host stage computes on; the message names the setting that includes it.
    """
    try:
        return SingleDeviceSharding(jax.devices("cpu")[0])
    except RuntimeError as error:
        raise RuntimeError(
            "datarax names records and folds keys on the CPU device, which this process has "
            "not started: include the CPU platform, e.g. JAX_PLATFORMS=cuda,cpu"
        ) from error


def key_words(key: ArrayLike) -> np.ndarray:
    """A key's data as uint32 words on the host: typed, raw or already host words.

    A device key is read back with one explicit ``jax.device_get``, which a
    ``jax.transfer_guard("disallow")`` allows; host words are returned as they are.

    Args:
        key: A typed key, its raw ``key_data``, or its uint32 words as a NumPy array.

    Returns:
        The key's data, uint32.
    """
    if isinstance(key, np.ndarray):
        return np.asarray(key, np.uint32)
    if isinstance(key, jax.Array) and jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
        key = jax.random.key_data(key)
    return np.asarray(jax.device_get(key), np.uint32)


def naming_key_data(key: jax.Array) -> jax.Array:
    """The data of a :data:`NAMING_PRNG_IMPL` key drawn from ``key``, a typed key of any impl.

    A key of that impl gives its own data; any other seeds one from 64 of its bits, so a caller
    whose default impl is another still orders records with a threefry key, derived from theirs.

    Args:
        key: A typed key.

    Returns:
        The naming key's raw data, uint32 ``(2,)``.
    """
    if str(jax.random.key_impl(key)) == NAMING_PRNG_IMPL:
        return jax.random.key_data(key)
    return jax.random.bits(key, (2,), jnp.uint32)


@functools.cache
def _fold() -> Callable[[jax.Array, jax.Array], jax.Array]:
    """``fold_in`` of raw key words by uint32 data, returning the folded key's words."""

    def fold(words: jax.Array, data: jax.Array) -> jax.Array:
        key = jax.random.wrap_key_data(words, impl=NAMING_PRNG_IMPL)
        return jax.random.key_data(jax.random.fold_in(key, data))

    return jax.jit(fold)


def fold_on_host(words: np.ndarray, data: int) -> np.ndarray:
    """``fold_in(key, data)`` of the :data:`NAMING_PRNG_IMPL` key whose words are ``words``.

    Computed on the CPU device: both operands are placed there explicitly and the result is read
    back explicitly, so the fold runs under a ``jax.transfer_guard("disallow")`` and reads
    nothing back from an accelerator.

    Args:
        words: The key's uint32 words (:func:`key_words`).
        data: The datum folded in, taken modulo ``2**32`` as ``fold_in`` takes it.

    Returns:
        The folded key's uint32 words.
    """
    cpu = host_device()
    folded = _fold()(
        jax.device_put(np.asarray(words, np.uint32), cpu),
        jax.device_put(np.uint32(data % (1 << 32)), cpu),
    )
    return np.asarray(jax.device_get(folded), np.uint32)
