"""64-bit record indices as two uint32 words, computed alike on the host and the device.

The arithmetic is checked against Python integers at the word boundaries and over random draws,
in NumPy and in jitted JAX with x64 off. Indices on either side of ``2**32`` keep distinct words
and distinct per-record keys, which a truncation to 32 bits would merge.
"""

from __future__ import annotations

import random
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from substrax.testing.compiles import expect_compiles

from datarax.core.element_batch import PADDING_INDEX
from datarax.core.index_words import (
    add,
    divmod_word,
    from_words,
    greater,
    low_words,
    MAX_RECORDS,
    multiply_high,
    multiply_word,
    split_constant,
    subtract,
    to_words,
)
from datarax.core.prng import per_record_keys
from tests.test_common.compiles import expect_first_call_compiles
from tests.test_common.step_jaxpr import host_callbacks


_EDGES = [
    0,
    1,
    (1 << 31) - 1,
    1 << 31,
    (1 << 32) - 1,
    1 << 32,
    (1 << 32) + 1,
    1 << 63,
    MAX_RECORDS - 1,
]


def _draws(count: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return _EDGES + [rng.getrandbits(64) for _ in range(count)]


class TestConversion:
    def test_python_integers_round_trip_through_their_words(self) -> None:
        values = _draws(256, 0)
        words = to_words(values)
        assert words.dtype == np.uint32
        assert words.shape == (len(values), 2)
        assert [int(value) for value in from_words(words)] == values

    def test_an_index_past_two_to_the_32_has_a_high_word(self) -> None:
        np.testing.assert_array_equal(to_words([(1 << 32) - 1, 1 << 32]), [[0, 2**32 - 1], [1, 0]])

    def test_a_jax_array_stays_on_the_device_and_traces(self) -> None:
        words = jax.jit(to_words)(jnp.arange(4, dtype=jnp.int32))
        assert isinstance(words, jax.Array)
        assert words.dtype == jnp.uint32
        np.testing.assert_array_equal(words, [[0, 0], [0, 1], [0, 2], [0, 3]])

    @pytest.mark.parametrize("values", [[-1], np.array([-1, 2]), [1 << 64]])
    def test_a_value_outside_the_64_bit_indices_is_refused(
        self, values: list[int] | np.ndarray
    ) -> None:
        with pytest.raises((ValueError, OverflowError)):
            to_words(values)

    def test_the_padding_index_is_the_one_index_past_the_records(self) -> None:
        assert int(from_words(PADDING_INDEX)) == MAX_RECORDS
        np.testing.assert_array_equal(to_words(MAX_RECORDS), PADDING_INDEX)

    def test_split_constant_gives_the_words_of_a_python_integer(self) -> None:
        assert split_constant((5 << 32) + 7) == (5, 7)


def _pair(words: np.ndarray, *, device: bool) -> tuple[Any, Any]:
    """The words as a ``(hi, lo)`` pair: NumPy on the host, JAX on the device."""
    array = jnp.asarray(words) if device else words
    return array[..., 0], array[..., 1]


def _joined(pair: tuple[Any, Any]) -> np.ndarray:
    return from_words(np.stack([np.asarray(pair[0]), np.asarray(pair[1])], axis=-1))


@pytest.mark.parametrize("device", [False, True], ids=["host", "device"])
class TestArithmetic:
    """Each operation equals Python's integer arithmetic modulo ``2**64``."""

    def test_add_and_subtract_wrap_modulo_two_to_the_64(self, device: bool) -> None:
        a, b = _draws(512, 1), _draws(512, 2)
        x, y = _pair(to_words(a), device=device), _pair(to_words(b), device=device)
        sums = jax.jit(add)(x, y) if device else add(x, y)
        differences = jax.jit(subtract)(x, y) if device else subtract(x, y)
        mask = (1 << 64) - 1
        assert [int(v) for v in _joined(sums)] == [
            (p + q) & mask for p, q in zip(a, b, strict=True)
        ]
        assert [int(v) for v in _joined(differences)] == [
            (p - q) & mask for p, q in zip(a, b, strict=True)
        ]

    def test_greater_orders_by_the_whole_value(self, device: bool) -> None:
        a, b = _draws(512, 3), _draws(512, 4)[::-1]
        b[:3] = a[:3]  # equal values are not greater
        x, y = _pair(to_words(a), device=device), _pair(to_words(b), device=device)
        result = jax.jit(greater)(x, y) if device else greater(x, y)
        assert np.asarray(result).tolist() == [p > q for p, q in zip(a, b, strict=True)]

    def test_multiply_high_is_the_product_s_high_word(self, device: bool) -> None:
        rng = random.Random(5)
        a = [0, 1, 0xFFFF, 0x10000, 0xFFFFFFFF] + [rng.getrandbits(32) for _ in range(512)]
        b = [0xFFFFFFFF, 0xFFFFFFFF, 0x10001, 0xFFFF, 0xFFFFFFFF] + [
            rng.getrandbits(32) for _ in range(512)
        ]
        if device:
            high = jax.jit(multiply_high)(jnp.asarray(a, jnp.uint32), jnp.asarray(b, jnp.uint32))
        else:
            high = multiply_high(np.asarray(a, np.uint32), np.asarray(b, np.uint32))
        assert [int(v) for v in np.asarray(high)] == [
            (p * q) >> 32 for p, q in zip(a, b, strict=True)
        ]

    @pytest.mark.parametrize("factor", [1, 3, 0xFFFF, 0xFFFFFFFF])
    def test_multiply_word_wraps_modulo_two_to_the_64(self, device: bool, factor: int) -> None:
        a = _draws(256, 6)
        x = _pair(to_words(a), device=device)

        def product(value: tuple[Any, Any]) -> tuple[Any, Any]:
            return multiply_word(value, np.uint32(factor))

        result = jax.jit(product)(x) if device else product(x)
        assert [int(v) for v in _joined(result)] == [(p * factor) & ((1 << 64) - 1) for p in a]


_DIVISORS = [1, 3, 333, 0xFFFF, 0x10000, 0x10001, (1 << 31) - 1, 1 << 31, (1 << 32) - 1]


def _quotient_and_remainder(pair: tuple[tuple[Any, Any], Any]) -> list[tuple[int, int]]:
    quotient, remainder = pair
    return [
        (int(q), int(r))
        for q, r in zip(_joined(quotient), np.asarray(remainder).tolist(), strict=True)
    ]


@pytest.mark.parametrize("device", [False, True], ids=["host", "device"])
class TestDivmodWord:
    """``divmod_word`` equals Python's ``divmod`` of a 64-bit value by a word, in uint32 only."""

    @pytest.mark.parametrize("divisor", _DIVISORS)
    def test_equals_python_divmod(self, device: bool, divisor: int) -> None:
        values = _draws(512, divisor)
        x = _pair(to_words(values), device=device)
        result = (
            jax.jit(divmod_word, static_argnums=1)(x, divisor)
            if device
            else divmod_word(x, divisor)
        )
        quotient, remainder = result
        for part in (*quotient, remainder):
            assert part.dtype == np.uint32
            assert isinstance(part, jax.Array) is device
        assert _quotient_and_remainder(result) == [divmod(v, divisor) for v in values]


class TestDivmodWordTraced:
    """The traced form: transforms, compiles, the program it builds, and refused divisors."""

    def test_vmap_and_scan_equal_the_whole_array(self) -> None:
        values = _draws(64, 9)
        high, low = _pair(to_words(values), device=True)
        expected = [divmod(v, 333) for v in values]

        mapped = jax.jit(jax.vmap(lambda h, lo: divmod_word((h, lo), 333)))(high, low)
        assert _quotient_and_remainder(mapped) == expected

        def body(carry: None, word: tuple[jax.Array, jax.Array]) -> tuple[None, Any]:
            return carry, divmod_word(word, 333)

        scanned = jax.jit(lambda h, lo: jax.lax.scan(body, None, (h, lo))[1])(high, low)
        assert _quotient_and_remainder(scanned) == expected

    def test_one_compile_per_divisor(self) -> None:
        divide = jax.jit(divmod_word, static_argnums=1)
        first, second = (_pair(to_words(_draws(8, seed)), device=True) for seed in (1, 2))
        with expect_first_call_compiles("jit(divmod_word)"):
            jax.block_until_ready(divide(first, 333))
        with expect_compiles(0):
            jax.block_until_ready(divide(second, 333))
        with expect_compiles(1):
            jax.block_until_ready(divide(second, 7))

    def test_the_program_holds_no_host_callback_and_no_64_bit_value(self) -> None:
        x = _pair(to_words(_draws(8, 3)), device=True)
        jaxpr = jax.make_jaxpr(lambda value: divmod_word(value, (1 << 31) - 1))(x)
        assert host_callbacks(jaxpr) == []
        dtypes = {
            str(getattr(var.aval, "dtype", None)) for eqn in jaxpr.jaxpr.eqns for var in eqn.outvars
        }
        assert dtypes <= {"uint32", "bool"}, dtypes

    @pytest.mark.parametrize("divisor", [0, -1, 1 << 32])
    def test_a_divisor_outside_one_word_is_refused(self, divisor: int) -> None:
        x = _pair(to_words([7]), device=False)
        with pytest.raises(ValueError, match="divisor"):
            divmod_word(x, divisor)


class TestRecordKeys:
    def test_indices_either_side_of_two_to_the_32_keep_distinct_keys(self) -> None:
        """``fold_in`` takes 32-bit data: one word would merge records ``2**32 - 1`` and ``2**32``
        with records ``2**32 - 1`` and ``0``; two words keep all three apart."""
        indices = jnp.asarray(to_words([(1 << 32) - 1, 1 << 32, 0]))
        assert len({tuple(row) for row in np.asarray(indices).tolist()}) == 3
        keys = jax.random.key_data(
            per_record_keys(
                jax.random.key(0), indices, jnp.zeros(3, jnp.int32), jnp.zeros(3, jnp.int32)
            )
        )
        assert len({tuple(row) for row in np.asarray(keys).tolist()}) == 3


class TestLowWords:
    def test_a_source_of_at_most_two_to_the_32_records_gathers_by_the_low_word(self) -> None:
        np.testing.assert_array_equal(low_words(jnp.asarray(to_words([3, 9])), 1 << 32), [3, 9])

    def test_a_source_past_two_to_the_32_records_is_refused(self) -> None:
        with pytest.raises(ValueError, match="2\\*\\*32"):
            low_words(jnp.asarray(to_words([3])), (1 << 32) + 1)

    def test_indices_other_than_words_are_refused(self) -> None:
        with pytest.raises(ValueError, match=r"\(n, 2\)"):
            low_words(jnp.arange(3), 10)
