"""Tests for random utility functions."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from substrax.rng import rngs_from_seed

from datarax.core.prng import DEFAULT_RNG_STREAMS, per_record_keys


class TestPerRecordKeys:
    """Per-record key derivation: randomness keyed on stable global index."""

    def test_shape_and_one_key_per_record(self):
        base = jax.random.key(0)
        keys = per_record_keys(base, jnp.arange(8))
        assert keys.shape[0] == 8

    def test_same_index_same_key_regardless_of_position(self):
        """A record's key depends only on its global index, not its batch slot."""
        base = jax.random.key(0)
        # Global index 5 appears at different positions in two different "batches".
        batch_a = per_record_keys(base, jnp.array([3, 4, 5, 6]))
        batch_b = per_record_keys(base, jnp.array([5, 9, 10, 11]))
        # index 5 -> position 2 in batch_a, position 0 in batch_b; keys must match.
        assert jnp.array_equal(jax.random.key_data(batch_a[2]), jax.random.key_data(batch_b[0]))

    def test_distinct_indices_give_distinct_keys(self):
        base = jax.random.key(0)
        keys = per_record_keys(base, jnp.array([0, 1, 2]))
        data = [tuple(jax.random.key_data(keys[i]).tolist()) for i in range(3)]
        assert len(set(data)) == 3

    def test_different_base_key_changes_keys(self):
        keys0 = per_record_keys(jax.random.key(0), jnp.array([7]))
        keys1 = per_record_keys(jax.random.key(1), jnp.array([7]))
        assert not jnp.array_equal(jax.random.key_data(keys0[0]), jax.random.key_data(keys1[0]))

    def test_the_epoch_folds_in_before_the_record(self):
        """Each epoch derives its own key for a record: fold_in(fold_in(base, epoch), record)."""
        base = jax.random.key(0)
        records = jnp.array([4, 9])

        epoch0 = per_record_keys(base, records, epoch=jnp.int32(0))
        epoch1 = per_record_keys(base, records, epoch=jnp.int32(1))

        expected = jax.random.fold_in(jax.random.fold_in(base, 1), 9)
        assert not jnp.array_equal(jax.random.key_data(epoch0), jax.random.key_data(epoch1))
        assert jnp.array_equal(jax.random.key_data(epoch1[1]), jax.random.key_data(expected))

    def test_each_record_may_carry_its_own_epoch(self):
        """A batch crossing an epoch boundary keys each record on the epoch it belongs to."""
        base = jax.random.key(0)
        records = jnp.array([8, 9, 0, 1])

        mixed = per_record_keys(base, records, epoch=jnp.array([0, 0, 1, 1], dtype=jnp.int32))

        tail = per_record_keys(base, records[:2], epoch=jnp.int32(0))
        head = per_record_keys(base, records[2:], epoch=jnp.int32(1))
        expected = jnp.concatenate([jax.random.key_data(tail), jax.random.key_data(head)])
        assert jnp.array_equal(jax.random.key_data(mixed), expected)


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
