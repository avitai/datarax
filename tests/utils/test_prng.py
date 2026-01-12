"""Tests for random utility functions."""

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from datarax.utils.prng import create_rngs, DEFAULT_RNG_STREAMS


class TestCreateRngs:
    """Tests for create_rngs function."""

    def test_create_default(self):
        """Test creating Rngs with default streams."""
        rngs = create_rngs(seed=42)

        # Check all default streams are present
        for stream in DEFAULT_RNG_STREAMS:
            assert stream in rngs

        # Check that streams produce different keys
        key1 = rngs.augment()
        key2 = rngs.dropout()
        assert not jnp.array_equal(key1, key2)

    def test_create_custom_streams(self):
        """Test creating Rngs with custom streams."""
        custom_streams = ["stream1", "stream2", "stream3"]
        rngs = create_rngs(seed=42, streams=custom_streams)

        # Check custom streams are present
        for stream in custom_streams:
            assert stream in rngs

        # Check default streams are not present
        assert "augment" not in rngs

    def test_create_no_seed(self):
        """Test creating Rngs without specifying seed."""
        rngs = create_rngs()  # Should use seed=0

        # Should still have default streams
        for stream in DEFAULT_RNG_STREAMS:
            assert stream in rngs

    def test_create_reproducibility(self):
        """Test that creation is reproducible."""
        rngs1 = create_rngs(seed=42)
        rngs2 = create_rngs(seed=42)

        # Same seed should produce same keys
        key1 = rngs1.augment()
        key2 = rngs2.augment()
        assert jnp.array_equal(key1, key2)


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

        rngs = create_rngs(seed=42)
        module = TestModule(rngs)

        # Should produce different values each call
        val1 = module.get_random()
        val2 = module.get_random()
        assert val1 != val2

    def test_multiple_streams(self):
        """Test using multiple RNG streams."""
        rngs = create_rngs(seed=42)

        # Different streams should produce different keys
        aug_key = rngs.augment()
        drop_key = rngs.dropout()
        param_key = rngs.params()

        assert not jnp.array_equal(aug_key, drop_key)
        assert not jnp.array_equal(drop_key, param_key)
        assert not jnp.array_equal(aug_key, param_key)

    def test_stream_iteration(self):
        """Test iterating with RNG streams."""
        rngs = create_rngs(seed=42)

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
        rngs = create_rngs(seed=42)

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

        rngs = create_rngs(seed=42)
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

        rngs = create_rngs(seed=42)

        # Should work with JIT
        result1 = random_fn(rngs.augment())
        result2 = random_fn(rngs.augment())

        assert result1.shape == (3,)
        assert result2.shape == (3,)
        assert not jnp.array_equal(result1, result2)
