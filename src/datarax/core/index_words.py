"""64-bit record indices and positions as two uint32 words, ``(hi, lo)``.

A record's index is 64-bit, and JAX runs with 64-bit types off by default: its integers are
32-bit and ``fold_in`` takes 32-bit data. datarax therefore carries a 64-bit integer as a uint32
array whose last axis holds ``(hi, lo)``, the layout of ``Batch.indices``. The arithmetic below
works on the two words with the operators NumPy and JAX uint32 arrays share, both wrapping modulo
``2**32``, so the host and the device run the same code and neither needs x64. The high word of a
32-bit product is built from 16-bit limbs, whose products fit in 32 bits.

The all-ones index, :data:`~datarax.core.element_batch.PADDING_INDEX`, marks a row that is not a
record, so a source holds at most :data:`MAX_RECORDS` records, indices ``[0, 2**64 - 1)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast, overload

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike


WORD_BITS = 32
"""Bits per word."""

MAX_RECORDS = (1 << 64) - 1
"""Records a source can hold: every 64-bit index but the all-ones one, which marks padding."""

_LOW_MASK = (1 << WORD_BITS) - 1
_LIMB_MASK = np.uint32(0xFFFF)

type HostIntegers = int | np.integer | Sequence[int] | np.ndarray
"""Integers on the host: Python ints of any size, or NumPy integers and arrays."""


@overload
def to_words(values: jax.Array) -> jax.Array: ...


@overload
def to_words(values: HostIntegers) -> np.ndarray: ...


def to_words(values: jax.Array | HostIntegers) -> jax.Array | np.ndarray:
    """Nonnegative integers as uint32 ``(..., 2)`` words ``(hi, lo)``.

    A JAX array stays a JAX array (and may be traced): its integers are at most 32-bit with x64
    off, so the high word is 0, and with x64 on a 64-bit array is split. Anything else (Python
    ints of any size, NumPy arrays) is split exactly on the host into a NumPy array.

    Args:
        values: Nonnegative integers below ``2**64``, any shape.

    Returns:
        uint32 words with a trailing axis of 2.

    Raises:
        ValueError: If a host value is negative or not below ``2**64``, or ``values`` is not
            integral.
    """
    if isinstance(values, jax.Array):
        if not jnp.issubdtype(values.dtype, jnp.integer):
            raise ValueError(f"record indices are integers; got {values.dtype}")
        low = values.astype(jnp.uint32)
        if values.dtype.itemsize > 4:  # noqa: PLR2004 - a 64-bit integer under x64
            high = (values >> WORD_BITS).astype(jnp.uint32)
        else:
            high = jnp.zeros_like(low)
        return jnp.stack([high, low], axis=-1)
    host = np.asarray(values)
    if host.dtype == object or not np.issubdtype(host.dtype, np.integer):
        host = np.asarray(values, dtype=np.uint64)  # Python ints past int64; refuses others
    elif np.issubdtype(host.dtype, np.signedinteger):
        if (host < 0).any():
            raise ValueError("record indices are nonnegative")
        host = host.astype(np.uint64)
    else:
        host = host.astype(np.uint64)
    return np.stack(
        [(host >> np.uint64(WORD_BITS)).astype(np.uint32), host.astype(np.uint32)], axis=-1
    )


def from_words(words: ArrayLike) -> np.ndarray:
    """uint32 ``(..., 2)`` words ``(hi, lo)`` as uint64 integers, on the host.

    Args:
        words: The words, as :func:`to_words` gives them.

    Returns:
        A uint64 NumPy array shaped like ``words`` without its trailing axis.
    """
    host = np.asarray(words, dtype=np.uint32).astype(np.uint64)
    return (host[..., 0] << np.uint64(WORD_BITS)) | host[..., 1]


def split_constant(value: int) -> tuple[np.uint32, np.uint32]:
    """A Python integer below ``2**64`` as its two words, ``(hi, lo)``."""
    return np.uint32(value >> WORD_BITS), np.uint32(value & _LOW_MASK)


def multiply_high[A: (jax.Array, np.ndarray)](a: A, b: A | np.uint32) -> A:
    """The high word of the 64-bit product ``a * b`` of two uint32 values.

    With ``a = a1 * 2**16 + a0`` and ``b = b1 * 2**16 + b0`` in 16-bit limbs, every limb product
    is below ``2**32``, and the partial sums below never exceed ``2**32 - 1``: ``a1 * b0`` plus a
    16-bit carry, ``a0 * b1`` plus a 16-bit remainder, and the high word itself.

    Args:
        a: uint32 values.
        b: uint32 values or a uint32 constant.

    Returns:
        ``(a * b) >> 32``, uint32.
    """
    a0, a1 = a & _LIMB_MASK, a >> 16
    b0, b1 = b & _LIMB_MASK, b >> 16
    low = a0 * b0
    middle = a1 * b0 + (low >> 16)
    cross = a0 * b1 + (middle & _LIMB_MASK)
    return a1 * b1 + (middle >> 16) + (cross >> 16)


def greater[A: (jax.Array, np.ndarray)](
    a: tuple[A, A], b: tuple[A, A] | tuple[np.uint32, np.uint32]
) -> A:
    """Whether the two-word value ``a`` exceeds ``b``, elementwise: a bool array like ``a``."""
    return cast(A, (a[0] > b[0]) | ((a[0] == b[0]) & (a[1] > b[1])))


def add[A: (jax.Array, np.ndarray)](
    a: tuple[A, A], b: tuple[A, A] | tuple[np.uint32, np.uint32]
) -> tuple[A, A]:
    """``a + b`` modulo ``2**64``, in words."""
    low = a[1] + b[1]
    carry = (low < a[1]).astype(np.uint32)
    return a[0] + b[0] + carry, low


def subtract[A: (jax.Array, np.ndarray)](
    a: tuple[A, A], b: tuple[A, A] | tuple[np.uint32, np.uint32]
) -> tuple[A, A]:
    """``a - b`` modulo ``2**64``, in words."""
    borrow = (a[1] < b[1]).astype(np.uint32)
    return a[0] - b[0] - borrow, a[1] - b[1]


def multiply_word[A: (jax.Array, np.ndarray)](a: tuple[A, A], factor: A | np.uint32) -> tuple[A, A]:
    """``a * factor`` modulo ``2**64`` for a uint32 ``factor``, one or one per element, in words."""
    return a[0] * factor + multiply_high(a[1], factor), a[1] * factor


def divmod_word[A: (jax.Array, np.ndarray)](a: tuple[A, A], divisor: int) -> tuple[tuple[A, A], A]:
    """``divmod(a, divisor)`` of the two-word value ``a`` by a word, in uint32 arithmetic.

    The high word divides directly; its remainder and the low word form a 64-bit value below
    ``divisor * 2**32``, whose quotient is one word. That division is Knuth's algorithm D for two
    16-bit quotient digits as Hacker's Delight writes it for 32-bit machines (Warren, *Hacker's
    Delight*, 2nd ed., section 9-4, ``divlu``): the divisor is shifted until its top bit is set,
    each digit is estimated from the divisor's top 16 bits and corrected at most twice, and the
    remainder is shifted back. The corrections are selected arithmetically rather than by
    branching, so NumPy arrays and traced arrays run the same code. ``divisor`` is a Python int,
    so its normalisation is computed once on the host and a jitted caller compiles once per
    divisor.

    Args:
        a: The dividend's words ``(hi, lo)``, uint32 arrays.
        divisor: A Python int in ``[1, 2**32 - 1]``.

    Returns:
        ``((quotient_hi, quotient_lo), remainder)``, uint32.

    Raises:
        ValueError: If ``divisor`` is not in ``[1, 2**32 - 1]``.
    """
    if not 1 <= divisor <= _LOW_MASK:
        raise ValueError(f"a word divisor is in [1, 2**32 - 1]; got {divisor}")
    word = np.uint32(divisor)
    high_quotient = a[0] // word
    carry = a[0] - high_quotient * word
    low_quotient, remainder = _divide_below_word(carry, a[1], divisor)
    return (high_quotient, low_quotient), remainder


_DIGIT_BASE = np.uint32(1 << 16)


def _divide_below_word[A: (jax.Array, np.ndarray)](high: A, low: A, divisor: int) -> tuple[A, A]:
    """``divmod(high * 2**32 + low, divisor)`` for ``high < divisor``: a one-word quotient.

    Hacker's Delight's ``divlu`` (section 9-4) in 32-bit words with 16-bit digits; see
    :func:`divmod_word`. Every intermediate the algorithm forms modulo ``2**32`` is exact where it
    is read: a product that can wrap is read only when the digit estimate is below ``2**16``.
    """
    shift = WORD_BITS - divisor.bit_length()
    normalised = divisor << shift
    divisor_word = np.uint32(normalised)
    top, bottom = np.uint32(normalised >> 16), np.uint32(normalised & 0xFFFF)
    if shift:
        numerator_high = (high << np.uint32(shift)) | (low >> np.uint32(WORD_BITS - shift))
        numerator_low = low << np.uint32(shift)
    else:
        numerator_high, numerator_low = high, low
    digits = (numerator_low >> np.uint32(16), numerator_low & _LIMB_MASK)

    quotient_digits = []
    partial = numerator_high
    for digit in digits:
        estimate = partial // top
        remainder_estimate = partial - estimate * top

        def too_large(estimate: A, remainder_estimate: A, digit: A = digit) -> A:
            """Whether the digit estimate exceeds the digit (``divlu``'s correction test)."""
            return cast(
                A,
                (estimate >= _DIGIT_BASE)
                | (estimate * bottom > _DIGIT_BASE * remainder_estimate + digit),
            )

        # The estimate exceeds the digit by at most two; the second test applies only after a
        # first correction that left the remainder estimate below the digit base.
        first = too_large(estimate, remainder_estimate)
        estimate = estimate - first.astype(np.uint32)
        remainder_estimate = remainder_estimate + first.astype(np.uint32) * top
        second = (
            first & (remainder_estimate < _DIGIT_BASE) & too_large(estimate, remainder_estimate)
        )
        estimate = estimate - second.astype(np.uint32)
        quotient_digits.append(estimate)
        partial = partial * _DIGIT_BASE + digit - estimate * divisor_word
    quotient = quotient_digits[0] * _DIGIT_BASE + quotient_digits[1]
    return quotient, partial >> np.uint32(shift)


def is_word_start(start: int | ArrayLike) -> bool:
    """Whether a start position is given as its two uint32 words ``(hi, lo)``, shape ``(2,)``.

    The host stage names positions past ``2**31`` in this form; an int32 scalar is the compiled
    session's, a Python int any caller's.
    """
    return not isinstance(start, int | np.integer) and np.shape(start) == (2,)


def wrapped_positions(start: int | ArrayLike, size: int, length: int | None) -> jax.Array:
    """Positions ``start + arange(size)`` wrapped at ``length``, as uint32 ``(size, 2)`` words.

    ``start`` comes in three forms. A Python integer may be any nonnegative size and is reduced on
    the host exactly. Two uint32 words ``(hi, lo)``, NumPy or traced (:func:`is_word_start`), are a
    64-bit position of the order, in ``[0, length)``. A traced integer scalar is an int32 position:
    below ``2**31`` and so below any length past that, and wrapped in int32 for a shorter length
    (negative positions wrap from the end). Each offset ``i mod length`` is a host constant, so a
    position is ``first + offset`` less ``length`` when it reaches it, which
    ``first >= length - offset`` decides without overflowing 64 bits.

    Args:
        start: The first position: a Python int, two uint32 words, or a traced int32 scalar.
        size: Positions (static).
        length: The length positions wrap at, or ``None`` for positions that never wrap.

    Returns:
        The positions, uint32 ``(size, 2)``.
    """
    offsets = np.arange(size, dtype=np.uint64)
    if length is not None:
        offsets %= np.uint64(length)
    if isinstance(start, int | np.integer):
        first_high, first_low = split_constant(int(start) % (length or MAX_RECORDS + 1))
        first = (jnp.asarray(first_high), jnp.asarray(first_low))
    elif is_word_start(start):
        words = jnp.asarray(start, jnp.uint32)
        first = (words[0], words[1])
    else:
        position = jnp.asarray(start)
        if length is not None and length <= np.iinfo(np.int32).max:
            position = position.astype(jnp.int32) % jnp.int32(length)
        first = (jnp.zeros((), jnp.uint32), position.astype(jnp.uint32))
    offset_words = to_words(offsets)
    offset = (jnp.asarray(offset_words[:, 0]), jnp.asarray(offset_words[:, 1]))
    high, low = add(first, offset)
    if length is not None:
        remainder_words = to_words(np.uint64(length) - offsets)
        remainder = (jnp.asarray(remainder_words[:, 0]), jnp.asarray(remainder_words[:, 1]))
        wraps = ~greater(remainder, first)
        wrapped_high, wrapped_low = subtract(first, remainder)
        high, low = jnp.where(wraps, wrapped_high, high), jnp.where(wraps, wrapped_low, low)
    return jnp.stack([high, low], axis=-1)


@overload
def low_words(indices: np.ndarray, length: int) -> np.ndarray: ...


@overload
def low_words(indices: jax.Array, length: int) -> jax.Array: ...


def low_words(indices: jax.Array | np.ndarray, length: int) -> jax.Array | np.ndarray:
    """The low words of ``indices`` into a source of ``length`` records: the rows a gather reads.

    A gather addresses rows with one uint32 word, which reaches every row of a source of at most
    ``2**32`` records, whose indices all have a high word of 0. Host (NumPy) indices give host
    rows, for the host read; any other indices give a JAX array, for a traced gather.

    Args:
        indices: uint32 ``(n, 2)`` record indices.
        length: The source's record count.

    Returns:
        uint32 ``(n,)`` row numbers, NumPy for NumPy indices.

    Raises:
        ValueError: If the source holds more than ``2**32`` records, which one word cannot
            address, or ``indices`` is not ``(n, 2)``.
    """
    if length > 1 << WORD_BITS:
        raise ValueError(
            f"a gather addresses rows with one uint32 word, at most 2**32 rows; this "
            f"source holds {length}"
        )
    if not isinstance(indices, np.ndarray):
        indices = jnp.asarray(indices)
    if indices.ndim != 2 or indices.shape[-1] != 2:  # noqa: PLR2004 - (hi, lo)
        raise ValueError(f"record indices are uint32 (n, 2) words (hi, lo); got {indices.shape}")
    return indices[:, 1]


__all__ = [
    "HostIntegers",
    "MAX_RECORDS",
    "WORD_BITS",
    "add",
    "divmod_word",
    "from_words",
    "greater",
    "is_word_start",
    "low_words",
    "multiply_high",
    "multiply_word",
    "split_constant",
    "subtract",
    "to_words",
    "wrapped_positions",
]
