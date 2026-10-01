"""The shuffled order: a keyed bijection of the records, computed per batch in O(batch).

``shuffle_positions`` ports CCCL's ``__feistel_bijection`` (VariablePhilox, Mitchell et al. 2022)
with cycle-walking, over domains up to 64 bits held as two uint32 words. The port is checked value
for value against the reference operator transcribed with Python integers at every block width,
against orders recorded before the domain grew past 31 bits, to be a bijection at every block-size
boundary (exhaustively where small, by sampled inverses past ``2**31``), and for quality with
statistics that fail on a known-structured map. No permutation is materialized or stored, so a
batch costs the same at every dataset size and a step writes no order state.
"""

from __future__ import annotations

import random
from math import comb
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.core import ShapedArray
from substrax.testing.compiles import expect_compiles

from datarax.core.index_shuffle import (
    _block_bits,
    _cycle_walk,
    _encrypt,
    _FALLBACK_CHANCE,
    _fixed_passes,
    _MAX_FIXED_PASSES,
    _ROUNDS,
    index_shuffle,
    shuffle_positions,
    shuffle_positions_host,
)
from datarax.core.index_words import from_words, HostIntegers, MAX_RECORDS, to_words
from datarax.pipeline import Pipeline
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from tests.test_common.step_jaxpr import sub_jaxprs, traced_step


_MULTIPLIER = 0xD2B74407B1CE6E93
_MULTIPLIER_INVERSE = pow(_MULTIPLIER, -1, 1 << 64)


def _reference_encrypt(value: int, bits: int, keys: list[int]) -> int:
    """``cuda::__feistel_bijection::operator()`` transcribed with Python integers.

    From ``libcudacxx/include/cuda/__random/feistel_bijection.h`` (CCCL ``5e77be1``), whose domain
    is ``max(8, bit_width(n - 1))`` bits for ``n`` up to ``2**64``. At 64 bits the left half is 32
    bits wide and CCCL's ``__R >> __L_bits_`` shifts a 32-bit word by 32, which C++ leaves
    undefined; Python's shift is the arithmetic one, 0, which is what the port computes.
    """
    left_bits = bits // 2
    right_bits = bits - left_bits
    left, right = value >> right_bits, value & ((1 << right_bits) - 1)
    for key in keys:
        product = (_MULTIPLIER * left) & (2**64 - 1)
        new_left = ((product >> 32) ^ key) ^ right
        new_right = (((product & 0xFFFFFFFF) << (right_bits - left_bits)) & 0xFFFFFFFF) | (
            right >> left_bits
        )
        left, right = new_left & ((1 << left_bits) - 1), new_right & ((1 << right_bits) - 1)
    return (left << right_bits) | right


def _reference_decrypt(value: int, bits: int, keys: list[int]) -> int:
    """The inverse of :func:`_reference_encrypt`, a round at a time in reverse.

    A round's new right half holds the product's low ``left_bits`` bits above the old right half's
    top ``right_bits - left_bits`` bits; the multiplier is odd, so the product's low bits give the
    old left half back, and with it the round function that masked the old right half's low bits.
    """
    left_bits = bits // 2
    right_bits = bits - left_bits
    spill = right_bits - left_bits
    left_mask = (1 << left_bits) - 1
    left, right = value >> right_bits, value & ((1 << right_bits) - 1)
    for key in reversed(keys):
        old_left = ((right >> spill) * _MULTIPLIER_INVERSE) & left_mask
        product = (_MULTIPLIER * old_left) & (2**64 - 1)
        low = (left ^ (product >> 32) ^ key) & left_mask
        right = ((right & ((1 << spill) - 1)) << left_bits) | low
        left = old_left
    return (left << right_bits) | right


def _round_keys(key: jax.Array) -> list[int]:
    return [int(word) for word in np.asarray(jax.random.bits(key, (_ROUNDS,), jnp.uint32))]


def _reference_inverse(record: int, length: int, keys: list[int]) -> int:
    """The position serving ``record``: decrypt along its cycle back into ``[0, length)``."""
    bits = _block_bits(length)
    value = _reference_decrypt(record, bits, keys)
    while value >= length:
        value = _reference_decrypt(value, bits, keys)
    return value


def _words(values: HostIntegers) -> jax.Array:
    return jnp.asarray(to_words(values))


# Sizes at the block boundaries: the smallest block, lengths one past a power of two (where
# the last index needs one more bit), and a prime.
_SIZES = (1, 2, 7, 255, 256, 257, 65536, 65537, 262145, 1_000_003)
# Lengths past int32 positions, each across a word or block boundary.
_WIDE = ((1 << 31) + 1, (1 << 32) - 1, 1 << 32, (1 << 32) + 1, 1 << 40)


def _order(length: int, key: jax.Array) -> np.ndarray:
    return from_words(shuffle_positions(_words(np.arange(length)), length, key))


def _orders(length: int, seed: int, count: int) -> np.ndarray:
    """``count`` orders of ``length`` records under independent keys derived from ``seed``."""
    keys = jax.random.split(jax.random.key(seed), count)
    positions = _words(np.arange(length))
    return from_words(
        jax.jit(jax.vmap(lambda key: shuffle_positions(positions, length, key)))(keys)
    )


def _affine_orders(length: int, seed: int, count: int, served: int | None = None) -> np.ndarray:
    """The statistics' control: a structured bijection, ``(a * i + b) mod length``, odd ``a``.

    Only the first ``served`` positions are computed (all of them by default): ``count`` orders of
    a large ``length`` in full would be ``count * length`` values, 16.8 GB for 2,000 orders of
    2**20 records.
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(0, length // 2, count) * 2 + 1
    b = rng.integers(0, length, count)
    positions = np.arange(length if served is None else served)
    return (a[:, None] * positions[None, :] + b[:, None]) % length


def _chi_square_p(observed: np.ndarray, expected: np.ndarray) -> float:
    """p-value of Pearson's chi-square test over the cells expected to hold at least five."""
    cells = expected >= 5
    statistic = float(((observed[cells] - expected[cells]) ** 2 / expected[cells]).sum())
    return float(jax.scipy.stats.chi2.sf(statistic, int(cells.sum()) - 1))


def _avalanche_p(outputs: np.ndarray, bits: int) -> float:
    """p-value that consecutive outputs differ in Binomial(bits, 1/2) bits (excluding zero)."""
    outputs = outputs.astype(np.uint64)
    flipped = (outputs[:, 1:] ^ outputs[:, :-1]).ravel()
    weights = np.bitwise_count(flipped).astype(np.int64)
    expected = np.array([comb(bits, weight) for weight in range(bits + 1)], dtype=float)
    expected[0] = 0.0
    expected *= weights.size / expected.sum()
    return _chi_square_p(np.bincount(weights, minlength=bits + 1).astype(float), expected)


def _pair_p(orders: np.ndarray) -> float:
    """p-value that the first two records served are a uniform ordered pair of distinct records."""
    orders = orders.astype(np.int64)
    length = orders.shape[1]
    counts = np.bincount(orders[:, 0] * length + orders[:, 1], minlength=length * length)
    cells = [i * length + j for i in range(length) for j in range(length) if i != j]
    observed = counts[cells].astype(float)
    return _chi_square_p(observed, np.full(observed.shape, observed.sum() / observed.size))


def _encrypt_values(values: list[int], bits: int, keys: list[int], *, host: bool) -> np.ndarray:
    """The port's cipher over ``values``, on the host (NumPy) or the device (JAX)."""
    words = to_words(values)
    if host:
        high, low = _encrypt(words[..., 0], words[..., 1], bits, np.asarray(keys, np.uint32))
    else:
        device = jnp.asarray(words)
        high, low = _encrypt(device[..., 0], device[..., 1], bits, jnp.asarray(keys, jnp.uint32))
    return from_words(np.stack([np.asarray(high), np.asarray(low)], axis=-1))


class TestPort:
    """The port computes exactly what the reference operator computes, at every block width."""

    @pytest.mark.parametrize("host", [False, True], ids=["device", "host"])
    @pytest.mark.parametrize("bits", range(8, 65))
    def test_every_width_matches_the_reference(self, bits: int, host: bool) -> None:
        rng = random.Random(bits)
        keys = [rng.getrandbits(32) for _ in range(_ROUNDS)]
        values = [rng.getrandbits(bits) for _ in range(64)] + [0, (1 << bits) - 1]
        expected = [_reference_encrypt(value, bits, keys) for value in values]
        np.testing.assert_array_equal(
            _encrypt_values(values, bits, keys, host=host), np.asarray(expected, np.uint64)
        )

    @pytest.mark.parametrize("bits", [8, 9, 31, 32, 33, 47, 63, 64])
    def test_the_reference_inverse_undoes_the_reference(self, bits: int) -> None:
        """The decryption the sampled bijection checks rely on is the cipher's inverse."""
        rng = random.Random(bits)
        keys = [rng.getrandbits(32) for _ in range(_ROUNDS)]
        for _ in range(64):
            value = rng.getrandbits(bits)
            assert _reference_decrypt(_reference_encrypt(value, bits, keys), bits, keys) == value


_ORACLE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "core" / "shuffle_orders_below_2_31.npz"
)


def _oracle_cases() -> list[tuple[int, int, int]]:
    with np.load(_ORACLE) as oracle:
        return [
            (int(length), int(seed), int(epoch))
            for length in oracle["lengths"]
            for seed, epoch in oracle["seeds_epochs"]
        ]


class TestRegressionOracle:
    """Orders below ``2**31`` records are those served before the domain grew to 64 bits.

    The fixture holds, for 12 lengths and 3 seeds and epochs, the records at sampled positions
    as ``shuffle_positions`` gave them on datarax ``e9abda7``; its provenance is in the file.
    """

    @pytest.mark.parametrize(("length", "seed", "epoch"), _oracle_cases())
    def test_every_form_serves_the_recorded_order(self, length: int, seed: int, epoch: int) -> None:
        with np.load(_ORACLE) as oracle:
            positions = oracle[f"positions_{length}"]
            recorded = oracle[f"order_{length}_{seed}_{epoch}"]
        key = jax.random.fold_in(jax.random.key(seed), epoch)
        traced = jax.jit(shuffle_positions, static_argnums=1)(_words(positions), length, key)
        np.testing.assert_array_equal(from_words(traced), recorded)
        np.testing.assert_array_equal(
            shuffle_positions_host(positions, length, seed, epoch), recorded
        )
        scalar = [index_shuffle(int(p), seed, length, epoch) for p in positions[:64]]
        np.testing.assert_array_equal(np.asarray(scalar, np.uint64), recorded[:64])


class TestBijection:
    @pytest.mark.parametrize("length", _SIZES)
    def test_every_record_is_served_exactly_once(self, length: int) -> None:
        order = _order(length, jax.random.key(0))
        np.testing.assert_array_equal(np.sort(order), np.arange(length, dtype=np.uint64))

    def test_the_same_key_gives_the_same_order(self) -> None:
        np.testing.assert_array_equal(
            _order(1000, jax.random.key(3)), _order(1000, jax.random.key(3))
        )

    def test_different_keys_give_different_orders(self) -> None:
        assert not np.array_equal(_order(1000, jax.random.key(3)), _order(1000, jax.random.key(4)))

    def test_a_position_maps_to_the_same_record_however_it_is_batched(self) -> None:
        key = jax.random.key(5)
        part = shuffle_positions(_words(np.arange(300, 340)), 1000, key)
        np.testing.assert_array_equal(from_words(part), _order(1000, key)[300:340])

    @pytest.mark.parametrize("length", _WIDE)
    def test_sampled_positions_past_int32_map_one_to_one(self, length: int) -> None:
        """No two sampled positions share a record, and each record's inverse is its position.

        Positions only: the order is never materialized, and no data exists. The inverse walks
        the reference cipher's decryption back along the record's cycle.
        """
        rng = np.random.default_rng(length % (1 << 32))
        edges = [0, 1, length - 2, length - 1, (1 << 31) - 1, 1 << 31, (1 << 32) - 1, 1 << 32]
        sampled = [int(p) for p in rng.integers(0, length, 2048, dtype=np.uint64)]
        positions = sorted({p for p in edges + sampled if p < length})
        key = jax.random.key(17)
        records = from_words(shuffle_positions(_words(positions), length, key))

        assert (records < length).all()
        assert np.unique(records).size == len(positions)
        keys = _round_keys(key)
        inverses = [_reference_inverse(int(record), length, keys) for record in records]
        assert inverses == positions

    @pytest.mark.parametrize("length", [0, MAX_RECORDS + 1])
    def test_a_length_outside_the_64_bit_indices_is_refused(self, length: int) -> None:
        with pytest.raises(ValueError, match="length"):
            shuffle_positions(_words([0]), length, jax.random.key(0))
        with pytest.raises(ValueError, match="length"):
            shuffle_positions_host(np.zeros(1, np.uint64), length, 0)

    def test_the_largest_order_never_names_the_padding_index(self) -> None:
        """At ``2**64 - 1`` records the indices are ``[0, 2**64 - 1)``: all-ones marks padding."""
        length = MAX_RECORDS
        positions = [0, 1, 1 << 32, (1 << 63) + 5, length - 2]
        records = from_words(shuffle_positions(_words(positions), length, jax.random.key(1)))
        assert (records < np.uint64(MAX_RECORDS)).all()
        assert np.unique(records).size == len(positions)

    @pytest.mark.parametrize(
        "positions",
        [
            jnp.arange(4, dtype=jnp.int32),
            jnp.zeros((4, 2), jnp.int32),
            jnp.zeros((4, 3), jnp.uint32),
        ],
        ids=["int32", "int32-words", "three-words"],
    )
    def test_positions_other_than_uint32_words_are_refused(self, positions: jax.Array) -> None:
        with pytest.raises(ValueError, match=r"uint32 \(\.\.\., 2\)"):
            shuffle_positions(positions, 10, jax.random.key(0))


class TestQuality:
    """Over many keys the order shows no structure; a structured bijection is the control."""

    @pytest.mark.parametrize("bits", [*range(8, 33), 33, 40, 48, 56, 63, 64])
    def test_consecutive_positions_serve_unrelated_records(self, bits: int) -> None:
        # The raw cipher over its whole domain: no walking, so the statistic sees each round.
        keys = jax.random.split(jax.random.key(11), 2000)
        positions = jnp.arange(256, dtype=jnp.uint32)

        def encrypt(key: jax.Array) -> jax.Array:
            high, low = _encrypt(
                jnp.zeros_like(positions),
                positions,
                bits,
                jax.random.bits(key, (_ROUNDS,), jnp.uint32),
            )
            return jnp.stack([high, low], axis=-1)

        assert _avalanche_p(from_words(jax.vmap(encrypt)(keys)), bits) > 1e-4

    def test_the_avalanche_check_detects_a_structured_bijection(self) -> None:
        assert _avalanche_p(_affine_orders(1 << 20, 11, 2000, served=256), 20) < 1e-4

    @pytest.mark.parametrize("seed", [11, 12, 13])
    def test_the_first_two_records_served_are_uniform_pairs(self, seed: int) -> None:
        assert _pair_p(_orders(16, seed, 20000)) > 1e-3

    def test_the_pair_check_detects_a_structured_bijection(self) -> None:
        assert _pair_p(_affine_orders(16, 11, 20000)) < 1e-3

    @pytest.mark.parametrize("seed", [11, 12, 13])
    def test_every_record_is_equally_likely_at_each_position(self, seed: int) -> None:
        positions_of_record_zero = np.argmax(_orders(64, seed, 20000) == 0, axis=1)
        observed = np.bincount(positions_of_record_zero, minlength=64).astype(float)
        assert _chi_square_p(observed, np.full(64, observed.sum() / 64)) > 1e-3


class TestHost:
    """The host form serves the device form's order for the same seed and epoch."""

    @pytest.mark.parametrize("length", [1, 7, 257, 65537])
    def test_the_host_order_is_the_device_order(self, length: int) -> None:
        host = shuffle_positions_host(np.arange(length), length, seed=9, epoch=2)
        key = jax.random.fold_in(jax.random.key(9), 2)
        assert host.dtype == np.uint64
        np.testing.assert_array_equal(host, _order(length, key))

    @pytest.mark.parametrize("length", [*_WIDE, MAX_RECORDS])
    def test_host_traced_and_scalar_forms_agree_past_int32(self, length: int) -> None:
        positions = [0, 4095, 4096, (1 << 31) - 1, 1 << 31, (1 << 32) - 1, 1 << 32, length - 1]
        positions = [p for p in positions if p < length]
        key = jax.random.fold_in(jax.random.key(4), 6)
        traced = from_words(
            jax.jit(shuffle_positions, static_argnums=1)(_words(positions), length, key)
        )
        host = shuffle_positions_host(positions, length, seed=4, epoch=6)
        scalar = [index_shuffle(p, 4, length, epoch=6) for p in positions]

        np.testing.assert_array_equal(host, traced)
        assert scalar == [int(record) for record in host]
        assert all(isinstance(record, int) for record in scalar)

    @pytest.mark.parametrize("length", _SIZES)
    def test_every_record_is_served_exactly_once(self, length: int) -> None:
        order = shuffle_positions_host(np.arange(length), length, seed=3)
        np.testing.assert_array_equal(np.sort(order), np.arange(length, dtype=np.uint64))

    def test_one_index_at_a_time_matches_the_whole_order_across_blocks(self) -> None:
        length = 10_000
        whole = shuffle_positions_host(np.arange(length), length, seed=4, epoch=1)
        for position in (0, 4095, 4096, 8191, 8192, length - 1):
            assert index_shuffle(position, 4, length, epoch=1) == whole[position]

    def test_epochs_give_different_orders(self) -> None:
        first = shuffle_positions_host(np.arange(1000), 1000, seed=5, epoch=0)
        second = shuffle_positions_host(np.arange(1000), 1000, seed=5, epoch=1)
        assert not np.array_equal(first, second)

    def test_a_seed_and_epoch_never_repeat_the_next_seeds_order(self) -> None:
        # Grain seeds an epoch with seed + epoch, so seed 5 at epoch 1 was seed 6 at epoch 0.
        later_epoch = shuffle_positions_host(np.arange(1000), 1000, seed=5, epoch=1)
        next_seed = shuffle_positions_host(np.arange(1000), 1000, seed=6, epoch=0)
        assert not np.array_equal(later_epoch, next_seed)

    @pytest.mark.parametrize(("index", "length"), [(-1, 10), (10, 10), (1 << 40, 1 << 40)])
    def test_an_index_outside_the_records_is_refused(self, index: int, length: int) -> None:
        with pytest.raises(IndexError, match="out of range"):
            index_shuffle(index, 0, length)

    def test_a_negative_host_position_is_refused(self) -> None:
        with pytest.raises(ValueError, match="nonnegative"):
            shuffle_positions_host(np.array([-1]), 10, seed=0)


class TestTransforms:
    def test_under_vmap_over_keys_matches_each_key(self) -> None:
        keys = jax.random.split(jax.random.key(2), 3)
        positions = _words(np.arange(50))
        batched = jax.vmap(lambda key: shuffle_positions(positions, 50, key))(keys)
        for row, key in zip(from_words(batched), keys, strict=True):
            np.testing.assert_array_equal(row, _order(50, key))

    def test_under_jit_with_traced_positions(self) -> None:
        key = jax.random.key(2)
        jitted = jax.jit(lambda positions: shuffle_positions(positions, 50, key))
        np.testing.assert_array_equal(
            from_words(jitted(_words(np.arange(10, 18)))), _order(50, key)[10:18]
        )

    @pytest.mark.parametrize("length", [1000, (1 << 32) + 1, 1 << 40])
    def test_one_compile_per_length_under_jit(self, length: int) -> None:
        shuffle = jax.jit(shuffle_positions, static_argnums=1)
        first, second = _words(np.arange(256)), _words(np.arange(256, 512))
        first_key, second_key = jax.random.split(jax.random.key(3), 2)
        with expect_compiles(1):
            jax.block_until_ready(shuffle(first, length, first_key))
        with expect_compiles(0):
            jax.block_until_ready(shuffle(second, length, second_key))

    @pytest.mark.parametrize("length", [1000, 1 << 40])
    def test_one_compile_per_length_under_vmap(self, length: int) -> None:
        positions = _words(np.arange(64))
        shuffle = jax.jit(jax.vmap(lambda key: shuffle_positions(positions, length, key)))
        first, second = (
            jax.random.split(jax.random.key(5), 4),
            jax.random.split(jax.random.key(6), 4),
        )
        with expect_compiles(1):
            jax.block_until_ready(shuffle(first))
        with expect_compiles(0):
            jax.block_until_ready(shuffle(second))

    @pytest.mark.parametrize("length", [1000, 1 << 40])
    def test_one_compile_per_length_under_scan(self, length: int) -> None:
        """A scan over batches of positions serves each batch the order's records."""
        key = jax.random.key(7)
        batches = _words(np.arange(4 * 32)).reshape(4, 32, 2)

        @jax.jit
        def scanned(batches: jax.Array) -> jax.Array:
            def body(carry: None, batch: jax.Array) -> tuple[None, jax.Array]:
                return carry, shuffle_positions(batch, length, key)

            return jax.lax.scan(body, None, batches)[1]

        with expect_compiles(1):
            served = jax.block_until_ready(scanned(batches))
        with expect_compiles(0):
            jax.block_until_ready(scanned(batches))
        expected = from_words(shuffle_positions(batches.reshape(-1, 2), length, key))
        np.testing.assert_array_equal(from_words(served).ravel(), expected)


class TestFixedPasses:
    """GPU while loops read their predicate back to the host, so the walk starts with fixed passes.

    A pass applies the cipher only to values still out of range, exactly as the loop does, so any
    number of fixed passes gives the loop's order; the pass count keeps the loop to one check but
    for a small chance. Both paths are tested on every backend: ``shuffle_positions`` picks one
    per platform, and a CPU-only run would otherwise never execute the other.
    """

    @pytest.mark.parametrize("length", [*_SIZES, *_WIDE])
    def test_any_number_of_fixed_passes_gives_the_loop_s_order(self, length: int) -> None:
        positions = _words(np.arange(min(length, 512)))
        walk = jax.jit(_cycle_walk, static_argnums=(1, 3))
        shuffle = jax.jit(shuffle_positions, static_argnums=1)
        for seed in range(4):
            key = jax.random.key(seed)
            loop = np.asarray(walk(positions, length, key, 0))
            for fixed in (1, 2, 7, _fixed_passes(length, positions.shape[0])):
                np.testing.assert_array_equal(np.asarray(walk(positions, length, key, fixed)), loop)
            np.testing.assert_array_equal(np.asarray(shuffle(positions, length, key)), loop)

    @pytest.mark.parametrize("length", [*_SIZES, *_WIDE, (1 << 63) + 1, MAX_RECORDS])
    @pytest.mark.parametrize("count", [1, 256, 4096])
    def test_the_pass_count_is_the_fewest_meeting_the_fallback_bound(
        self, length: int, count: int
    ) -> None:
        domain = 1 << _block_bits(length)
        out_of_range = (domain - length) / domain  # exact numerator: no cancellation near 2**64
        passes = _fixed_passes(length, count)
        if out_of_range == 0.0:
            assert passes == 1
            return
        if passes == _MAX_FIXED_PASSES:
            assert count * out_of_range ** (passes - 1) > _FALLBACK_CHANCE
            return
        assert count * out_of_range**passes <= _FALLBACK_CHANCE
        assert passes == 1 or count * out_of_range ** (passes - 1) > _FALLBACK_CHANCE

    def test_a_domain_one_short_of_full_takes_one_pass(self) -> None:
        """At ``2**64 - 1`` records one value of the 64-bit domain is out of range, 2**-64 of
        them; a subtraction ``1 - length / 2**64`` in floating point would round that to 0."""
        assert _fixed_passes(MAX_RECORDS, 4096) == 1
        assert _fixed_passes((1 << 63) + 1, 4096) > 1

    @pytest.mark.parametrize(
        "length", [1 << 7, (1 << 7) + 1, 255, 1000, 65537, 1_000_003, *_WIDE, (1 << 63) + 1]
    )
    def test_the_cap_does_not_bind_from_half_the_smallest_domain(self, length: int) -> None:
        """From 2**7 records a value leaves the range with probability at most 1/2, so even
        2**22 values meet the fallback bound within the capped passes, at every width."""
        domain = 1 << _block_bits(length)
        out_of_range = (domain - length) / domain
        count = 1 << 22
        assert count * out_of_range ** _fixed_passes(length, count) <= _FALLBACK_CHANCE

    def test_a_shorter_source_continues_in_the_loop_past_the_cap(self) -> None:
        assert _fixed_passes(10, 256) == _MAX_FIXED_PASSES


_LARGE = 1 << 20


def _shuffled_pipeline(num_epochs: int | None) -> Pipeline:
    source = MemorySource(MemorySourceConfig(), data={"x": np.zeros((_LARGE, 1), dtype=np.float32)})
    return Pipeline(
        source=source,
        stages=[],
        batch_size=8,
        num_epochs=num_epochs,
        rngs=nnx.Rngs(0),
        shuffle=True,
    )


class TestBatchCost:
    """A shuffled batch does no work, and writes no state, proportional to the dataset."""

    @pytest.mark.parametrize("num_epochs", [1, None], ids=["bounded", "continuous"])
    def test_no_operation_produces_a_dataset_sized_array(self, num_epochs: int | None) -> None:
        closed, _ = traced_step(_shuffled_pipeline(num_epochs))
        sizes = [
            (eqn.primitive.name, int(np.prod(var.aval.shape)))
            for jaxpr in sub_jaxprs(closed.jaxpr)
            for eqn in jaxpr.eqns
            for var in eqn.outvars
            if isinstance(var.aval, ShapedArray)
        ]
        assert sizes, "the walk found no equations"
        assert [entry for entry in sizes if entry[1] >= _LARGE] == []

    @pytest.mark.parametrize("num_epochs", [1, None], ids=["bounded", "continuous"])
    def test_a_step_writes_no_dataset_sized_state(self, num_epochs: int | None) -> None:
        _, (batch, writes) = traced_step(_shuffled_pipeline(num_epochs))
        assert batch["x"].shape == (8, 1)
        written = [int(np.prod(leaf.shape)) for leaf in jax.tree.leaves(writes)]
        assert written, "the step wrote nothing, not even its position"
        assert max(written) < _LARGE
