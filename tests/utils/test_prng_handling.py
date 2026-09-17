"""Tests for PRNG handling in Datarax.

This module tests forking and reseeding ``nnx.Rngs`` built over the datarax streams.

See tests/operators/test_element_operator.py for ElementOperator tests
covering reproducibility, batch augmentation, and batch consistency.
"""

import flax.nnx as nnx
import jax
from substrax.rng import rngs_from_seed

from datarax.core.prng import DEFAULT_RNG_STREAMS


def test_fork_rngs():
    """Test using fork() to create independent Rngs objects."""
    rngs = rngs_from_seed(42, DEFAULT_RNG_STREAMS)

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
    rngs = rngs_from_seed(42, DEFAULT_RNG_STREAMS)

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
    rngs_a = rngs_from_seed(1, DEFAULT_RNG_STREAMS)
    rngs_b = rngs_from_seed(2, DEFAULT_RNG_STREAMS)

    # Reseed both to seed 123
    nnx.reseed(rngs_a, augment=123)
    nnx.reseed(rngs_b, augment=123)

    # After reseeding to the same seed, both should produce the same first value
    val_a = float(jax.random.uniform(rngs_a["augment"]()))
    val_b = float(jax.random.uniform(rngs_b["augment"]()))
    assert val_a == val_b
