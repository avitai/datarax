"""Keyed index shuffles: the record at each position of a shuffled order, without the order.

A shuffled order is a bijection of ``[0, length)`` keyed by a PRNG key, computed per position,
so serving a batch costs O(batch) at every dataset size and nothing is stored between batches.

The bijection is CCCL's ``cuda::__feistel_bijection``
(``libcudacxx/include/cuda/__random/feistel_bijection.h``, also ``thrust::shuffle``'s
``random_bijection``): the VariablePhilox cipher of Mitchell, Stokes, Frank and Holmes,
"Bandwidth-optimal random shuffling for GPUs", ACM Transactions on Parallel Computing 9(1), 2022
-- an unbalanced Feistel network over ``max(8, bit_width(length - 1))`` bits whose 24 rounds use
the Philox multiplier as round function -- with cycle-walking into ``[0, length)`` (Black and
Rogaway, "Ciphers with Arbitrary Finite Domains", CT-RSA 2002). Adapted to JAX: 32-bit arithmetic
with the 64-bit product built from 16-bit limbs (positions are int32, so the left half has at most
15 bits), and round keys drawn from a JAX key.

Two forms share the cipher. :func:`shuffle_positions` is traceable and takes a JAX key;
:func:`shuffle_positions_host` and its scalar :func:`index_shuffle` run in NumPy for host-side
samplers and iterators and take an integer seed and epoch, whose key is
``fold_in(key(seed), epoch)`` -- so the host order for a seed and epoch is the device order for
that key.

Grain's and TensorFlow's ``index_shuffle`` are not used: they extend Simon's rotation constants
to word sizes the cipher does not define, which at 14-bit words makes every round linear in
parity, size their block one bit short when ``length - 1`` is a power of two, and seed an epoch
with ``seed + epoch``, so one seed's second epoch is the next seed's first.
"""

import functools
import logging

import jax
import jax.numpy as jnp
import numpy as np


logger = logging.getLogger(__name__)

_ROUNDS = 24
_MIN_BITS = 8
_PHILOX_MULTIPLIER = 0xD2B74407B1CE6E93
_MAX_LENGTH = int(np.iinfo(np.int32).max)
# Positions the scalar host form computes at once and caches, so a per-element caller pays a
# vectorized cipher call per block instead of per element.
_HOST_BLOCK = 4096

# uint32 arrays of either framework: the cipher uses only operators both define.
type _Uint32Array = jax.Array | np.ndarray


def _block_bits(length: int) -> int:
    """Bits of the cipher's domain: enough to hold ``length - 1``, at least 8."""
    return max(_MIN_BITS, (length - 1).bit_length())


def _encrypt(values: _Uint32Array, bits: int, round_keys: _Uint32Array) -> _Uint32Array:
    """CCCL's ``__feistel_bijection::operator()`` over ``bits``-bit uint32 values.

    Written with operators NumPy and JAX uint32 arrays share (both wrap modulo 2**32), so the
    host and device forms run the same code.
    """
    left_bits = bits // 2
    right_bits = bits - left_bits
    left_mask = np.uint32((1 << left_bits) - 1)
    right_mask = np.uint32((1 << right_bits) - 1)
    multiplier_high = np.uint32(_PHILOX_MULTIPLIER >> 32)
    multiplier_mid = np.uint32((_PHILOX_MULTIPLIER >> 16) & 0xFFFF)
    multiplier_low = np.uint32(_PHILOX_MULTIPLIER & 0xFFFF)
    left, right = values >> right_bits, values & right_mask
    for round_key in round_keys:
        # product = multiplier * left mod 2**64, as 32-bit halves. left < 2**15, so each
        # 16-bit limb product fits in 32 bits; the low half carries into the high half.
        middle = left * multiplier_mid
        shifted = middle << 16
        product_low = shifted + left * multiplier_low
        carry = (product_low < shifted).astype(np.uint32)
        product_high = (middle >> 16) + carry + left * multiplier_high
        new_left = (product_high ^ round_key) ^ right
        new_right = (product_low << (right_bits - left_bits)) | (right >> left_bits)
        left, right = new_left & left_mask, new_right & right_mask
    return (left << right_bits) | right


def _check_length(length: int) -> None:
    if not 1 <= length <= _MAX_LENGTH:
        raise ValueError(f"length must be in [1, 2**31 - 1], got {length}")


def shuffle_positions(positions: jax.Array, length: int, key: jax.Array) -> jax.Array:  # noqa: DOC502
    """The record at each position of the order ``key`` shuffles ``[0, length)`` into.

    The same ``key`` always gives the same order, and a position maps to the same record
    however positions are batched. JAX-traceable and composable with ``jit``, ``vmap`` and
    ``scan``: ``positions`` and ``key`` may be traced, ``length`` is static.

    Args:
        positions: Int32 positions in ``[0, length)``, any shape.
        length: Number of records the order covers, in ``[1, 2**31 - 1]``.
        key: PRNG key selecting the order.

    Returns:
        Int32 array of record indices, shaped like ``positions``.

    Raises:
        ValueError: If ``length`` is out of range.
    """
    _check_length(length)
    bits = _block_bits(length)
    round_keys = jax.random.bits(key, (_ROUNDS,), jnp.uint32)
    top = jnp.uint32(length - 1)
    # Cycle-walking: the cipher permutes its 2**bits domain, so re-encrypting a value until it
    # lands in range walks its cycle back into the range, and distinct positions reach
    # distinct records.
    records = jax.lax.while_loop(
        lambda values: jnp.any(values > top),
        lambda values: jnp.where(values > top, _encrypt(values, bits, round_keys), values),
        _encrypt(jnp.asarray(positions).astype(jnp.uint32), bits, round_keys),
    )
    return jnp.asarray(records).astype(jnp.int32)


@functools.lru_cache(maxsize=64)
def _host_round_keys(seed: int, epoch: int) -> np.ndarray:
    """Round keys of the order for ``seed`` at ``epoch``: those of ``fold_in(key(seed), epoch)``."""
    key = jax.random.fold_in(jax.random.key(seed), epoch)
    round_keys = np.asarray(jax.random.bits(key, (_ROUNDS,), jnp.uint32))
    round_keys.flags.writeable = False
    return round_keys


def shuffle_positions_host(  # noqa: DOC502
    positions: np.ndarray, length: int, seed: int, epoch: int = 0
) -> np.ndarray:
    """NumPy form of :func:`shuffle_positions` for the order of ``seed`` at ``epoch``.

    Equal to ``shuffle_positions(positions, length, fold_in(key(seed), epoch))``.

    Args:
        positions: Integer positions in ``[0, length)``, any shape.
        length: Number of records the order covers, in ``[1, 2**31 - 1]``.
        seed: Integer seed in ``[0, 2**32)``.
        epoch: Epoch whose order is served.

    Returns:
        Int64 array of record indices, shaped like ``positions``.

    Raises:
        ValueError: If ``length`` is out of range.
    """
    _check_length(length)
    bits = _block_bits(length)
    round_keys = _host_round_keys(seed, epoch)
    top = np.uint32(length - 1)
    values = _encrypt(np.atleast_1d(np.asarray(positions)).astype(np.uint32), bits, round_keys)
    while (outside := values > top).any():
        values = np.where(outside, _encrypt(values, bits, round_keys), values)
    return np.asarray(values).astype(np.int64).reshape(np.shape(positions))


@functools.lru_cache(maxsize=64)
def _host_block(length: int, seed: int, epoch: int, block: int) -> np.ndarray:
    """Record indices of positions ``[block * _HOST_BLOCK, ...)`` of one order, cached."""
    start = block * _HOST_BLOCK
    indices = shuffle_positions_host(
        np.arange(start, min(start + _HOST_BLOCK, length)), length, seed, epoch
    )
    indices.flags.writeable = False
    return indices


def index_shuffle(index: int, seed: int, num_elements: int, epoch: int = 0) -> int:
    """The record at position ``index`` of the order of ``seed`` at ``epoch``.

    Scalar form of :func:`shuffle_positions_host` for callers that walk an order one position
    at a time: positions are computed and cached a block at a time.

    Args:
        index: Position in ``[0, num_elements)``.
        seed: Integer seed in ``[0, 2**32)``.
        num_elements: Number of records the order covers.
        epoch: Epoch whose order is served.

    Returns:
        Record index in ``[0, num_elements)``.

    Raises:
        IndexError: If ``index`` is outside ``[0, num_elements)``.
    """
    if index < 0 or index >= num_elements:
        raise IndexError(f"Index {index} out of range for {num_elements} elements")
    block, offset = divmod(index, _HOST_BLOCK)
    return int(_host_block(num_elements, seed, epoch, block)[offset])
