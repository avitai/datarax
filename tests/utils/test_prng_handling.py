"""Tests for PRNG handling utilities in Datarax.

This module tests the PRNG utility functions for creating and managing
Rngs objects, forking, reseeding, and ensuring required streams exist.

See tests/operators/test_element_operator.py for ElementOperator tests
covering reproducibility, batch augmentation, and batch consistency.
"""

import os

import flax.nnx as nnx
import jax

from datarax.utils.prng import DEFAULT_RNG_STREAMS, create_rngs


# Force CPU mode for consistent testing
os.environ["JAX_PLATFORM_NAME"] = "cpu"


def test_create_rngs():
    """Test create_rngs creates proper Rngs objects."""
    # Test with default streams
    rngs = create_rngs(seed=42)
    assert isinstance(rngs, nnx.Rngs)
    for stream in DEFAULT_RNG_STREAMS:
        # Check if stream exists by indexing into rngs
        assert stream in rngs

    # Test with custom streams
    custom_streams = ["test1", "test2"]
    rngs = create_rngs(seed=42, streams=custom_streams)
    assert isinstance(rngs, nnx.Rngs)
    assert set(rngs) == set(custom_streams)


def test_fork_rngs():
    """Test using fork() to create independent Rngs objects."""
    rngs = create_rngs(seed=42)

    # Create 3 independent Rngs using fork()
    forked_rngs_list = [rngs.fork() for _ in range(3)]

    # Check we got 3 distinct Rngs objects
    assert len(forked_rngs_list) == 3
    for r in forked_rngs_list:
        assert isinstance(r, nnx.Rngs)
        assert set(r) == set(rngs)

    # Check that using the forked RNGs produces different results
    def sample_uniform(r):
        # Use the stream properly by calling it
        return jax.random.uniform(r["augment"]())

    values = [sample_uniform(r) for r in forked_rngs_list]
    # All values should be different
    assert len(set(float(v) for v in values)) == 3


def test_nnx_reseed():
    """Test using nnx.reseed to reseed Rngs objects.

    This test verifies that reseeding an Rngs object changes its internal state
    and produces different random values than before reseeding.
    """
    # Create Rngs and sample a value
    rngs = create_rngs(seed=42)

    # Get a value before reseeding
    value_before = float(jax.random.uniform(rngs["augment"]()))

    # Reseed with a different seed
    nnx.reseed(rngs, augment=99)

    # Get a value after reseeding - should be different
    value_after = float(jax.random.uniform(rngs["augment"]()))

    # Values should differ after reseeding to a different seed
    assert value_before != value_after

    # Verify that reseeding to the same seed twice produces consistent results
    # Create two fresh Rngs objects and reseed both to the same seed
    rngs_a = create_rngs(seed=1)
    rngs_b = create_rngs(seed=2)

    # Reseed both to seed 123
    nnx.reseed(rngs_a, augment=123)
    nnx.reseed(rngs_b, augment=123)

    # After reseeding to the same seed, both should produce the same first value
    val_a = float(jax.random.uniform(rngs_a["augment"]()))
    val_b = float(jax.random.uniform(rngs_b["augment"]()))
    assert val_a == val_b
