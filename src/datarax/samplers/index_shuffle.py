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
import math

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
# Chance that values remain out of range after the cycle-walk's fixed passes, sending the walk
# into its loop (one host round trip per iteration on a GPU): the pass count is set from it.
_FALLBACK_CHANCE = 1e-3
# The fixed passes are capped at those a domain within twice the length needs for 2**22 values,
# log2(2**22 / _FALLBACK_CHANCE); only a source shorter than half the smallest domain (2**7
# records) needs more, and its walk continues in the loop.
_MAX_FIXED_PASSES = 32


def _block_bits(length: int) -> int:
    """Bits of the cipher's domain: enough to hold ``length - 1``, at least 8."""
    return max(_MIN_BITS, (length - 1).bit_length())


def _encrypt[A: (jax.Array, np.ndarray)](values: A, bits: int, round_keys: A) -> A:
    """CCCL's ``__feistel_bijection::operator()`` over ``bits``-bit uint32 values.

    The round is written with operators NumPy and JAX uint32 arrays share (both wrap modulo
    2**32), so the host and device forms run the same code. NumPy iterates it in Python; JAX
    iterates it with ``lax.scan(unroll=True)``, which traces the round once and hands the compiler
    the same unrolled rounds -- a Python loop would trace all 24, and every transform applied to
    the shuffle (``vmap``, the platform choice) would walk each of them again.
    """
    left_bits = bits // 2
    right_bits = bits - left_bits
    left_mask = np.uint32((1 << left_bits) - 1)
    right_mask = np.uint32((1 << right_bits) - 1)
    multiplier_high = np.uint32(_PHILOX_MULTIPLIER >> 32)
    multiplier_mid = np.uint32((_PHILOX_MULTIPLIER >> 16) & 0xFFFF)
    multiplier_low = np.uint32(_PHILOX_MULTIPLIER & 0xFFFF)

    def one_round(halves: tuple[A, A], round_key: A) -> tuple[A, A]:
        left, right = halves
        # product = multiplier * left mod 2**64, as 32-bit halves. left < 2**15, so each
        # 16-bit limb product fits in 32 bits; the low half carries into the high half.
        middle = left * multiplier_mid
        shifted = middle << 16
        product_low = shifted + left * multiplier_low
        carry = (product_low < shifted).astype(np.uint32)
        product_high = (middle >> 16) + carry + left * multiplier_high
        new_left = (product_high ^ round_key) ^ right
        new_right = (product_low << (right_bits - left_bits)) | (right >> left_bits)
        return new_left & left_mask, new_right & right_mask

    halves = (values >> right_bits, values & right_mask)
    if isinstance(values, np.ndarray):
        for round_key in round_keys:
            halves = one_round(halves, round_key)
    else:
        halves, _ = jax.lax.scan(
            lambda state, round_key: (one_round(state, round_key), None),
            halves,
            round_keys,
            unroll=True,
        )
    left, right = halves
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
    positions = jnp.asarray(positions)
    passes = _fixed_passes(length, positions.size)
    return jax.lax.platform_dependent(
        positions,
        key,
        cpu=lambda p, k: _cycle_walk(p, length, k, 0),
        default=lambda p, k: _cycle_walk(p, length, k, passes),
    )


def _fixed_passes(length: int, count: int) -> int:
    """The fewest cycle-walk passes leaving ``count`` values in range but for a small chance.

    A value out of range stays out after a pass with probability ``1 - length / 2**bits``, below
    1/2 by the choice of ``bits``, so ``count`` values all land within ``k`` passes except with
    probability at most ``count * (1 - length / 2**bits) ** k``; the returned ``k`` is the least
    making that at most :data:`_FALLBACK_CHANCE`, which grows with ``log(count)``, capped at
    :data:`_MAX_FIXED_PASSES`.

    Args:
        length: Records the order covers.
        count: Values walked together.

    Returns:
        The pass count, at least 1.
    """
    out_of_range = 1.0 - length / (1 << _block_bits(length))
    if out_of_range == 0.0:
        return 1
    passes = math.ceil(math.log(_FALLBACK_CHANCE / max(count, 1)) / math.log(out_of_range))
    return min(max(1, passes), _MAX_FIXED_PASSES)


def _cycle_walk(positions: jax.Array, length: int, key: jax.Array, fixed: int) -> jax.Array:
    """Encrypt ``positions`` and cycle-walk every value into ``[0, length)``.

    The cipher permutes its ``2**bits`` domain, so re-encrypting a value until it lands in range
    walks its cycle back into the range, and distinct positions reach distinct records. The first
    ``fixed`` passes run in a loop of known trip count and the rest in a ``while_loop`` until
    every value is in range: a pass re-encrypts only values still out of range, so any split
    gives the same records. On a GPU XLA reads a while loop's predicate back to the host every
    iteration, and a known trip count needs no such read.

    Args:
        positions: Int32 positions in ``[0, length)``.
        length: Records the order covers.
        key: PRNG key selecting the order.
        fixed: Passes to run before the loop, 0 for the loop alone.

    Returns:
        Int32 record indices, shaped like ``positions``.
    """
    bits = _block_bits(length)
    round_keys = jax.random.bits(key, (_ROUNDS,), jnp.uint32)
    top = jnp.uint32(length - 1)

    def walk(values: jax.Array) -> jax.Array:
        return jnp.where(values > top, _encrypt(values, bits, round_keys), values)

    records = positions.astype(jnp.uint32)
    if fixed == 0:
        records = _encrypt(records, bits, round_keys)
    else:
        # Pass 0 encrypts every position and later passes only values still out of range, so the
        # cipher appears once in the loop body rather than once more before it.
        records = jax.lax.fori_loop(
            0,
            fixed,
            lambda index, values: jnp.where(
                (index == 0) | (values > top), _encrypt(values, bits, round_keys), values
            ),
            records,
        )
    records = jax.lax.while_loop(lambda values: jnp.any(values > top), walk, records)
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
