"""Tests for random utility functions."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from substrax.rng import rngs_from_seed
from substrax.testing.compiles import expect_compiles

from datarax.core.prng import DEFAULT_RNG_STREAMS, per_record_keys


def _two_words(indices: list[int]) -> jax.Array:
    """64-bit record indices as the ``(hi, lo)`` uint32 words a ``Batch`` holds."""
    return jnp.asarray([[i >> 32, i & 0xFFFFFFFF] for i in indices], dtype=jnp.uint32)


def _key_data(keys: jax.Array) -> np.ndarray:
    return np.asarray(jax.random.key_data(keys))


def _keys(base: jax.Array, indices: list[int], epoch: int = 0, draw: int = 0) -> np.ndarray:
    n = len(indices)
    return _key_data(
        per_record_keys(
            base,
            _two_words(indices),
            jnp.full((n,), epoch, jnp.int32),
            jnp.full((n,), draw, jnp.int32),
        )
    )


class TestPerRecordKeys:
    """A record's key is a function of (base, epoch, draw, index) and of nothing else."""

    def test_the_key_folds_epoch_draw_and_both_index_words_in_that_order(self) -> None:
        base = jax.random.key(0)
        index = 2**32 * 3 + 17

        expected = jax.random.fold_in(
            jax.random.fold_in(jax.random.fold_in(jax.random.fold_in(base, 5), 2), 3), 17
        )

        np.testing.assert_array_equal(_keys(base, [index], epoch=5, draw=2)[0], _key_data(expected))

    def test_a_record_keeps_its_key_in_any_batch_position_or_size(self) -> None:
        base = jax.random.key(0)

        first = _keys(base, [3, 4, 5, 6], epoch=1)
        second = _keys(base, [5, 9], epoch=1)

        np.testing.assert_array_equal(first[2], second[0])

    def test_the_word_boundary_gives_distinct_keys(self) -> None:
        """``2^32 - 1`` and ``2^32`` differ only across the word boundary; an int32 index wraps."""
        keys = _keys(jax.random.key(0), [2**32 - 1, 2**32, 0, 1])

        assert len({row.tobytes() for row in keys}) == 4

    def test_each_epoch_and_each_draw_is_a_fresh_key(self) -> None:
        base = jax.random.key(0)

        plain = _keys(base, [7])[0]

        assert not np.array_equal(plain, _keys(base, [7], epoch=1)[0])
        assert not np.array_equal(plain, _keys(base, [7], draw=1)[0])
        assert not np.array_equal(_keys(base, [7], epoch=1)[0], _keys(base, [7], draw=1)[0])

    def test_two_draws_of_one_record_in_one_batch_differ(self) -> None:
        """A record served twice in one batch draws twice: the draw ordinal separates the keys."""
        keys = per_record_keys(
            jax.random.key(0),
            _two_words([9, 9]),
            jnp.zeros(2, jnp.int32),
            jnp.array([0, 1], jnp.int32),
        )

        assert not np.array_equal(_key_data(keys)[0], _key_data(keys)[1])

    def test_keys_do_not_depend_on_how_many_processes_share_the_batch(self) -> None:
        """Each process keys its own rows; together they equal one process keying the whole."""
        base = jax.random.key(3)
        indices = list(range(100, 108))

        whole = _keys(base, indices, epoch=2)
        halves = np.concatenate(
            [_keys(base, indices[:4], epoch=2), _keys(base, indices[4:], epoch=2)]
        )
        quarters = np.concatenate(
            [_keys(base, indices[i : i + 2], epoch=2) for i in range(0, 8, 2)]
        )

        np.testing.assert_array_equal(whole, halves)
        np.testing.assert_array_equal(whole, quarters)

    def test_a_different_base_key_changes_every_key(self) -> None:
        assert not np.array_equal(_keys(jax.random.key(0), [7]), _keys(jax.random.key(1), [7]))

    def test_keys_trace_once_for_any_values(self) -> None:
        derive = jax.jit(per_record_keys)
        base = jax.random.key(0)
        epochs, draws = jnp.zeros(4, jnp.int32), jnp.zeros(4, jnp.int32)
        derive(base, _two_words([0, 1, 2, 3]), epochs, draws)
        other = (_two_words([2**40, 5, 6, 7]), jnp.full(4, 3, jnp.int32), jnp.ones(4, jnp.int32))

        with expect_compiles(0):
            derive(base, *other)


class TestDefaultStreams:
    """The datarax stream names, as substrax derives them."""

    def test_every_stream_present_and_distinct(self):
        """Each named stream exists and draws its own keys."""
        rngs = rngs_from_seed(42, DEFAULT_RNG_STREAMS)

        assert set(rngs) == set(DEFAULT_RNG_STREAMS)
        assert not jnp.array_equal(rngs.augment(), rngs.dropout())


class TestRngUsage:
    """Tests for using Rngs in practice."""

    def test_rngs_in_module(self):
        """Test using Rngs in a module."""

        class TestModule(nnx.Module):
            def __init__(self, rngs: nnx.Rngs):
                super().__init__()
                self.rngs = rngs

            def get_random(self):
                key = self.rngs.dropout()
                return jax.random.uniform(key)

        rngs = rngs_from_seed(42, DEFAULT_RNG_STREAMS)
        module = TestModule(rngs)

        # Should produce different values each call
        val1 = module.get_random()
        val2 = module.get_random()
        assert val1 != val2

    def test_multiple_streams(self):
        """Test using multiple RNG streams."""
        rngs = rngs_from_seed(42, DEFAULT_RNG_STREAMS)

        # Different streams should produce different keys
        aug_key = rngs.augment()
        drop_key = rngs.dropout()
        param_key = rngs.params()

        assert not jnp.array_equal(aug_key, drop_key)
        assert not jnp.array_equal(drop_key, param_key)
        assert not jnp.array_equal(aug_key, param_key)

    def test_stream_iteration(self):
        """Test iterating with RNG streams."""
        rngs = rngs_from_seed(42, DEFAULT_RNG_STREAMS)

        keys = []
        for i in range(5):
            keys.append(rngs.augment())

        assert len(keys) == 5
        # All keys should be different
        for i in range(5):
            for j in range(i + 1, 5):
                assert not jnp.array_equal(keys[i], keys[j])

    def test_fork_rngs(self):
        """Test forking RNG streams."""
        rngs = rngs_from_seed(42, DEFAULT_RNG_STREAMS)

        # Fork for different purposes
        key1 = rngs.augment()
        forked_key = jax.random.split(key1, 2)[0]

        # Forked key should be different
        assert not jnp.array_equal(key1, forked_key)

        # Next key from stream should also be different
        key2 = rngs.augment()
        assert not jnp.array_equal(key1, key2)
        assert not jnp.array_equal(key2, forked_key)

    def test_vmap_with_rngs(self):
        """Test using Rngs with vmap."""

        def random_fn(key):
            return jax.random.uniform(key)

        rngs = rngs_from_seed(42, DEFAULT_RNG_STREAMS)
        keys = jax.random.split(rngs.augment(), 4)

        # Vmap the function
        results = jax.vmap(random_fn)(keys)

        assert results.shape == (4,)
        # All results should be different
        for i in range(4):
            for j in range(i + 1, 4):
                assert results[i] != results[j]

    def test_jit_with_rngs(self):
        """Test using Rngs with JIT compilation."""

        @jax.jit
        def random_fn(key):
            return jax.random.normal(key, shape=(3,))

        rngs = rngs_from_seed(42, DEFAULT_RNG_STREAMS)

        # Should work with JIT
        result1 = random_fn(rngs.augment())
        result2 = random_fn(rngs.augment())

        assert result1.shape == (3,)
        assert result2.shape == (3,)
        assert not jnp.array_equal(result1, result2)
