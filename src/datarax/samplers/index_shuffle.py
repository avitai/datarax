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

Grain's and TensorFlow's ``index_shuffle`` are not used: they extend Simon's rotation constants
to word sizes the cipher does not define, which at 14-bit words makes every round linear in
parity, and size their block one bit short when ``length - 1`` is a power of two.

``index_shuffle`` is the host-side scalar form used by the samplers.
"""

import logging

import jax
import jax.numpy as jnp
from grain.experimental import index_shuffle as grain_index_shuffle


logger = logging.getLogger(__name__)

_ROUNDS = 24
_MIN_BITS = 8
_PHILOX_MULTIPLIER = 0xD2B74407B1CE6E93


def _block_bits(length: int) -> int:
    """Bits of the cipher's domain: enough to hold ``length - 1``, at least 8."""
    return max(_MIN_BITS, (length - 1).bit_length())


def _encrypt(values: jax.Array, bits: int, round_keys: jax.Array) -> jax.Array:
    """CCCL's ``__feistel_bijection::operator()`` over ``bits``-bit uint32 values."""
    left_bits = bits // 2
    right_bits = bits - left_bits
    left_mask = jnp.uint32((1 << left_bits) - 1)
    right_mask = jnp.uint32((1 << right_bits) - 1)
    multiplier_high = jnp.uint32(_PHILOX_MULTIPLIER >> 32)
    multiplier_mid = jnp.uint32((_PHILOX_MULTIPLIER >> 16) & 0xFFFF)
    multiplier_low = jnp.uint32(_PHILOX_MULTIPLIER & 0xFFFF)
    left, right = values >> right_bits, values & right_mask
    for round_key in round_keys:
        # product = multiplier * left mod 2**64, as 32-bit halves. left < 2**15, so each
        # 16-bit limb product fits in 32 bits; the low half carries into the high half.
        middle = left * multiplier_mid
        shifted = middle << 16
        product_low = shifted + left * multiplier_low
        carry = (product_low < shifted).astype(jnp.uint32)
        product_high = (middle >> 16) + carry + left * multiplier_high
        new_left = (product_high ^ round_key) ^ right
        new_right = (product_low << (right_bits - left_bits)) | (right >> left_bits)
        left, right = new_left & left_mask, new_right & right_mask
    return (left << right_bits) | right


def shuffle_positions(positions: jax.Array, length: int, key: jax.Array) -> jax.Array:
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
    if not 1 <= length <= jnp.iinfo(jnp.int32).max:
        raise ValueError(f"length must be in [1, 2**31 - 1], got {length}")
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
    return records.astype(jnp.int32)


def index_shuffle(index: int, seed: int, num_elements: int) -> int:
    """Compute Grain's shuffled position without materializing a permutation.

    Args:
        index: Original element index in [0, num_elements).
        seed: Seed for the permutation (same seed = same permutation).
        num_elements: Total number of elements N.

    Returns:
        Shuffled index in [0, num_elements).

    Raises:
        IndexError: If ``index`` is outside ``[0, num_elements)``.
    """
    if index < 0 or index >= num_elements:
        raise IndexError(f"Index {index} out of range for {num_elements} elements")
    if num_elements <= 1:
        return 0
    return grain_index_shuffle(
        index=index,
        max_index=num_elements - 1,
        seed=seed,
        rounds=4,
    )
