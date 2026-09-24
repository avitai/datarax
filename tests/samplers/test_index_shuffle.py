"""The shuffled order: a keyed bijection of the records, computed per batch in O(batch).

``shuffle_positions`` ports CCCL's ``__feistel_bijection`` (VariablePhilox, Mitchell et al. 2022)
with cycle-walking. The port is checked value for value against the reference operator
transcribed with Python integers, the order is checked to be a bijection at every block-size
boundary, and its quality with statistics that fail on a known-structured map. No permutation is
materialized or stored, so a batch costs the same at every dataset size and a step writes no
order state.
"""

from __future__ import annotations

import random
from math import comb

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.core import ShapedArray
from jax.extend.core import ClosedJaxpr, Jaxpr

from datarax.pipeline import Pipeline
from datarax.pipeline.iteration import _is_per_batch_state, _run_tracking_writes, _Writes
from datarax.samplers.index_shuffle import (
    _encrypt,
    _ROUNDS,
    index_shuffle,
    shuffle_positions,
    shuffle_positions_host,
)
from datarax.sources.memory_source import MemorySource, MemorySourceConfig


def _reference_encrypt(value: int, bits: int, keys: list[int]) -> int:
    """``cuda::__feistel_bijection::operator()`` transcribed with Python integers.

    From ``libcudacxx/include/cuda/__random/feistel_bijection.h`` (CCCL).
    """
    left_bits = bits // 2
    right_bits = bits - left_bits
    left, right = value >> right_bits, value & ((1 << right_bits) - 1)
    for key in keys:
        product = (0xD2B74407B1CE6E93 * left) & (2**64 - 1)
        new_left = ((product >> 32) ^ key) ^ right
        new_right = (((product & 0xFFFFFFFF) << (right_bits - left_bits)) & 0xFFFFFFFF) | (
            right >> left_bits
        )
        left, right = new_left & ((1 << left_bits) - 1), new_right & ((1 << right_bits) - 1)
    return (left << right_bits) | right


# Sizes at the block boundaries: the smallest block, lengths one past a power of two (where
# the last index needs one more bit), and a prime.
_SIZES = (1, 2, 7, 255, 256, 257, 65536, 65537, 262145, 1_000_003)


def _order(length: int, key: jax.Array) -> np.ndarray:
    return np.asarray(shuffle_positions(jnp.arange(length, dtype=jnp.int32), length, key))


def _orders(length: int, seed: int, count: int) -> np.ndarray:
    """``count`` orders of ``length`` records under independent keys derived from ``seed``."""
    keys = jax.random.split(jax.random.key(seed), count)
    positions = jnp.arange(length, dtype=jnp.int32)
    return np.asarray(
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
    flipped = (outputs[:, 1:] ^ outputs[:, :-1]).ravel().astype(np.int64)
    weights = np.bitwise_count(flipped)
    expected = np.array([comb(bits, weight) for weight in range(bits + 1)], dtype=float)
    expected[0] = 0.0
    expected *= weights.size / expected.sum()
    return _chi_square_p(np.bincount(weights, minlength=bits + 1).astype(float), expected)


def _pair_p(orders: np.ndarray) -> float:
    """p-value that the first two records served are a uniform ordered pair of distinct records."""
    length = orders.shape[1]
    counts = np.bincount(orders[:, 0] * length + orders[:, 1], minlength=length * length)
    cells = [i * length + j for i in range(length) for j in range(length) if i != j]
    observed = counts[cells].astype(float)
    return _chi_square_p(observed, np.full(observed.shape, observed.sum() / observed.size))


class TestPort:
    """The JAX port computes exactly what the reference operator computes."""

    @pytest.mark.parametrize("bits", range(8, 32))
    def test_every_width_matches_the_reference(self, bits: int) -> None:
        rng = random.Random(bits)
        keys = [rng.getrandbits(32) for _ in range(_ROUNDS)]
        values = [rng.getrandbits(bits) for _ in range(64)]
        ported = _encrypt(jnp.asarray(values, jnp.uint32), bits, jnp.asarray(keys, jnp.uint32))
        expected = [_reference_encrypt(value, bits, keys) for value in values]
        np.testing.assert_array_equal(np.asarray(ported), expected)


class TestBijection:
    @pytest.mark.parametrize("length", _SIZES)
    def test_every_record_is_served_exactly_once(self, length: int) -> None:
        order = _order(length, jax.random.key(0))
        np.testing.assert_array_equal(np.sort(order), np.arange(length))

    def test_the_same_key_gives_the_same_order(self) -> None:
        np.testing.assert_array_equal(
            _order(1000, jax.random.key(3)), _order(1000, jax.random.key(3))
        )

    def test_different_keys_give_different_orders(self) -> None:
        assert not np.array_equal(_order(1000, jax.random.key(3)), _order(1000, jax.random.key(4)))

    def test_a_position_maps_to_the_same_record_however_it_is_batched(self) -> None:
        key = jax.random.key(5)
        part = shuffle_positions(jnp.arange(300, 340, dtype=jnp.int32), 1000, key)
        np.testing.assert_array_equal(np.asarray(part), _order(1000, key)[300:340])

    @pytest.mark.parametrize("length", [0, 2**31])
    def test_a_length_outside_int32_positions_is_refused(self, length: int) -> None:
        with pytest.raises(ValueError, match="length"):
            shuffle_positions(jnp.arange(1, dtype=jnp.int32), length, jax.random.key(0))


class TestQuality:
    """Over many keys the order shows no structure; a structured bijection is the control."""

    @pytest.mark.parametrize("bits", range(8, 32))
    def test_consecutive_positions_serve_unrelated_records(self, bits: int) -> None:
        # The raw cipher over its whole domain: no walking, so the statistic sees each round.
        keys = jax.random.split(jax.random.key(11), 2000)
        positions = jnp.arange(256, dtype=jnp.uint32)
        outputs = jax.vmap(
            lambda key: _encrypt(positions, bits, jax.random.bits(key, (_ROUNDS,), jnp.uint32))
        )(keys)
        assert _avalanche_p(np.asarray(outputs), bits) > 1e-4

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
        np.testing.assert_array_equal(host, _order(length, key))

    @pytest.mark.parametrize("length", _SIZES)
    def test_every_record_is_served_exactly_once(self, length: int) -> None:
        order = shuffle_positions_host(np.arange(length), length, seed=3)
        np.testing.assert_array_equal(np.sort(order), np.arange(length))

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

    @pytest.mark.parametrize("index", [-1, 10])
    def test_an_index_outside_the_records_is_refused(self, index: int) -> None:
        with pytest.raises(IndexError, match="out of range"):
            index_shuffle(index, 0, 10)


class TestTransforms:
    def test_under_vmap_over_keys_matches_each_key(self) -> None:
        keys = jax.random.split(jax.random.key(2), 3)
        positions = jnp.arange(50, dtype=jnp.int32)
        batched = jax.vmap(lambda key: shuffle_positions(positions, 50, key))(keys)
        for row, key in zip(np.asarray(batched), keys, strict=True):
            np.testing.assert_array_equal(row, _order(50, key))

    def test_under_jit_with_traced_positions(self) -> None:
        key = jax.random.key(2)
        jitted = jax.jit(lambda start: shuffle_positions(start + jnp.arange(8), 50, key))
        np.testing.assert_array_equal(np.asarray(jitted(jnp.int32(10))), _order(50, key)[10:18])


_LARGE = 1 << 20


def _shuffled_pipeline(num_epochs: int | None) -> Pipeline:
    source = MemorySource(
        MemorySourceConfig(shuffle=True),
        data={"x": np.zeros((_LARGE, 1), dtype=np.float32)},
        rngs=nnx.Rngs(0),
    )
    return Pipeline(source=source, stages=[], batch_size=8, num_epochs=num_epochs, rngs=nnx.Rngs(0))


def _traced_step(pipeline: Pipeline) -> tuple[ClosedJaxpr, tuple[dict, _Writes]]:
    """The jaxpr of one batch, and the shapes of the batch and the state it writes."""
    graphdef, per_batch, staged = nnx.split(pipeline, _is_per_batch_state, ...)

    def step(per_batch: nnx.State, staged: nnx.State) -> tuple[dict, _Writes]:
        return _run_tracking_writes(
            graphdef, (per_batch, staged), lambda module: module._next_batch()
        )

    return jax.make_jaxpr(step)(per_batch, staged), jax.eval_shape(step, per_batch, staged)


def _sub_jaxprs(jaxpr: Jaxpr) -> list[Jaxpr]:
    """``jaxpr`` and every jaxpr nested in its equations (cond branches, loop bodies)."""
    found = [jaxpr]
    for eqn in jaxpr.eqns:
        for param in eqn.params.values():
            for value in param if isinstance(param, tuple | list) else (param,):
                inner = getattr(value, "jaxpr", value)
                if isinstance(inner, Jaxpr):
                    found.extend(_sub_jaxprs(inner))
    return found


class TestBatchCost:
    """A shuffled batch does no work, and writes no state, proportional to the dataset."""

    @pytest.mark.parametrize("num_epochs", [1, None], ids=["bounded", "continuous"])
    def test_no_operation_produces_a_dataset_sized_array(self, num_epochs: int | None) -> None:
        closed, _ = _traced_step(_shuffled_pipeline(num_epochs))
        sizes = [
            (eqn.primitive.name, int(np.prod(var.aval.shape)))
            for jaxpr in _sub_jaxprs(closed.jaxpr)
            for eqn in jaxpr.eqns
            for var in eqn.outvars
            if isinstance(var.aval, ShapedArray)
        ]
        assert sizes, "the walk found no equations"
        assert [entry for entry in sizes if entry[1] >= _LARGE] == []

    @pytest.mark.parametrize("num_epochs", [1, None], ids=["bounded", "continuous"])
    def test_a_step_writes_no_dataset_sized_state(self, num_epochs: int | None) -> None:
        _, (batch, writes) = _traced_step(_shuffled_pipeline(num_epochs))
        assert batch["x"].shape == (8, 1)
        written = [int(np.prod(leaf.shape)) for leaf in jax.tree.leaves(writes)]
        assert written, "the step wrote nothing, not even its position"
        assert max(written) < _LARGE
