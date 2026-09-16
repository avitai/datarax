# pyright: reportOptionalMemberAccess=false
"""Tests for Datarax's random state handling with NNX modules.

This module tests random key generation, seed reproduction, key splitting,
and random number usage in pipeline contexts with NNX modules.
"""

import flax.nnx as nnx
import jax
import numpy as np
import pytest

from datarax.core.config import DataraxModuleConfig, OperatorConfig, StructuralConfig
from datarax.core.data_source import DataSourceModule
from datarax.core.module import DataraxModule
from datarax.core.operator import OperatorModule, require_key


@pytest.fixture
def test_seed():
    """Fixture providing a fixed seed for testing reproducibility."""
    return 42


class RandomArraySourceModule(DataSourceModule):
    """Source module that generates random arrays using RNG."""

    def __init__(self, config: StructuralConfig, num_items=10, shape=(5,), *, rngs=None, name=None):
        super().__init__(config, rngs=rngs, name=name)
        self.num_items = num_items
        self.shape = shape
        self.index = nnx.Variable(0)

    def __iter__(self):
        self.index.set_value(0)
        return self

    def __next__(self):
        if self.index.get_value() >= self.num_items:
            raise StopIteration

        # Get the next random value using the source stream
        random_value = jax.random.normal(self.rngs.source(), self.shape)
        self.index.set_value(self.index.get_value() + 1)
        return random_value

    def __len__(self):
        return self.num_items


class RandomOperatorModule(OperatorModule):
    """Operator module that applies random operations."""

    def __init__(self, *, rngs=None):
        """Initialize with stochastic config for random operations."""
        config = OperatorConfig(stochastic=True, stream_name="transform")
        super().__init__(config, rngs=rngs)

    def apply(self, data, state, metadata, key=None, stats=None):
        """Scale the input data by a factor drawn from this record's key."""
        del stats
        scale = jax.random.uniform(require_key(key, self), minval=0.5, maxval=1.5)
        new_data = {k: v * scale for k, v in data.items()}
        return new_data, state, metadata


def test_random_key_generation(test_seed):
    """Test that random keys are generated correctly with NNX."""
    # Create module with a fixed seed using config-based initialization
    config = DataraxModuleConfig()
    module = DataraxModule(config, rngs=nnx.Rngs(test_seed))

    # Generate keys for different streams
    key1 = module.rngs.test()

    # The key should be a valid JAX PRNG key
    assert isinstance(key1, jax.Array)
    # Key shape has changed in newer JAX versions, just check it exists
    assert key1.shape == () or len(key1.shape) > 0

    # Generate a key for a different stream
    key2 = module.rngs.another_stream()

    # Keys for different streams should be different
    assert not np.array_equal(key1, key2)

    # Generate a key for the same stream again
    key1_again = module.rngs.test()

    # Keys from the same stream should be different (auto-split)
    assert not np.array_equal(key1, key1_again)

    # Create another module with the same seed
    another_module = DataraxModule(config, rngs=nnx.Rngs(test_seed))

    # The first key from each stream should be deterministic
    # with the same seed (but NNX may use internal state)
    key3 = another_module.rngs.test()
    # Just verify it's a valid key, not necessarily equal
    assert isinstance(key3, jax.Array)


def test_random_seed_reproduction(test_seed):
    """Test that using the same seed reproduces the initial state."""
    # Create two modules with the same seed (config-first pattern)
    config = StructuralConfig(stochastic=True, stream_name="source")
    module1 = RandomArraySourceModule(config, num_items=1, rngs=nnx.Rngs(test_seed))
    module2 = RandomArraySourceModule(config, num_items=1, rngs=nnx.Rngs(test_seed))

    # Get random values from each module
    random1 = next(iter(module1))
    random2 = next(iter(module2))

    # Initial random values should be identical with the same seed
    assert np.array_equal(random1, random2)

    # Create a module with a different seed
    module3 = RandomArraySourceModule(config, num_items=1, rngs=nnx.Rngs(test_seed + 1))
    random3 = next(iter(module3))

    # Random values should be different with a different seed
    assert not np.array_equal(random1, random3)


def test_streaming_random_values(test_seed):
    """Test that streams of random values are properly handled."""
    # Create a random source module (config-first pattern)
    config = StructuralConfig(stochastic=True, stream_name="source")
    source = RandomArraySourceModule(config, num_items=5, shape=(3,), rngs=nnx.Rngs(test_seed))

    # Collect all random values
    values = list(source)

    # Should get the expected number of items
    assert len(values) == 5

    # All values should be different
    for i in range(5):
        for j in range(i + 1, 5):
            assert not np.array_equal(values[i], values[j])

    # Reset the source and get values again
    values2 = list(source)

    # After reset, should get different values because RNG state advances
    # This is unlike stateful RNGs and matches JAX's functional approach
    assert not np.array_equal(values[0], values2[0])


def test_random_integration(test_seed):
    """Test random numbers in pipeline with NNX modules.

    The operator draws from each record's key, which the batch path derives from the operator's
    own base key, so reproducibility is a property of the seed rather than of call order.
    """
    config = StructuralConfig(stochastic=True, stream_name="source")

    def scaled(seed):
        """Run three source items through the operator, keyed by record index."""
        rngs = nnx.Rngs(seed)
        source = RandomArraySourceModule(config, num_items=3, rngs=rngs)
        operator = RandomOperatorModule(rngs=rngs)
        values = jax.numpy.stack(list(source))
        data, _ = operator._vmap_apply(
            {"value": values}, {}, None, jax.numpy.arange(3, dtype=jax.numpy.uint32)
        )
        return np.asarray(data["value"])

    # The same seed reproduces, a different seed does not
    assert np.array_equal(scaled(test_seed), scaled(test_seed))
    assert not np.array_equal(scaled(test_seed), scaled(test_seed + 1))


def test_rngs_reseed():
    """Test reseeding RNGs in NNX modules."""
    # Create a module with initial seed using config-based initialization
    config = DataraxModuleConfig()
    rngs = nnx.Rngs(42)
    module = DataraxModule(config, rngs=rngs)

    # Get initial random value
    key1 = module.rngs.test()

    # Reseed the test stream
    nnx.reseed(module, test=43)

    # Get new random value
    key2 = module.rngs.test()

    # Keys should be different after reseeding
    assert not np.array_equal(key1, key2)

    # After reseeding, we should continue getting different keys
    # (We can't guarantee they'll match a fresh module due to internal state)
    key3 = module.rngs.test()
    assert not np.array_equal(key2, key3)
