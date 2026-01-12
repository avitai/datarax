"""Test fixtures for common NNX module patterns in Datarax."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from datarax.typing import Batch, Element


class SimpleDataSourceModule(nnx.Module):
    """A simple data source module for testing."""

    def __init__(self, data: list[Element], *, rngs=None):
        """Initialize the mock data source."""
        super().__init__()
        self.data = data
        self.index = nnx.Variable(0)

    def __iter__(self):
        """Return iterator."""
        self.index.set_value(0)
        return self

    def __next__(self) -> Element:
        """Get next element."""
        if self.index.get_value() >= len(self.data):
            raise StopIteration
        item = self.data[self.index.get_value()]
        self.index.set_value(self.index.get_value() + 1)
        return item

    def __len__(self) -> int:
        """Return length of data."""
        return len(self.data)


class SimpleDeterministicOperator(nnx.Module):
    """A simple deterministic operator module for testing.

    Provides element-level transformation without randomness.
    """

    def __init__(self, scale: float = 1.0, offset: float = 0.0, *, rngs=None):
        """Initialize the deterministic operator module."""
        super().__init__()
        self.scale = scale
        self.offset = offset

    def __call__(self, element: Element) -> Element:
        """Apply transformation to a single element."""
        result = {}
        for key, value in element.items():
            if hasattr(value, "shape"):  # Only transform arrays
                result[key] = value * self.scale + self.offset
            else:
                result[key] = value
        return result

    def apply_batch(self, batch: Batch) -> Batch:
        """Apply transformation to a batch of elements."""

        # Use proper NNX vmap pattern
        @nnx.vmap(in_axes=(None, 0), out_axes=0)
        def batch_apply(module, batch_element):
            return module(batch_element)

        return batch_apply(self, batch)


class SimpleStochasticOperator(nnx.Module):
    """A simple stochastic operator module for testing.

    Provides element-level augmentation with randomness.
    """

    def __init__(self, magnitude: float = 0.1, *, rngs=None):
        """Initialize the stochastic operator module."""
        super().__init__()
        self.magnitude = magnitude
        self.rngs = rngs

    def __call__(self, element: Element, *, rngs=None) -> Element:
        """Apply augmentation to a single element."""
        # Require "augment" stream in Rngs
        if rngs is None:
            rngs = self.rngs

        if not hasattr(rngs, "augment"):
            raise ValueError("RNG stream 'augment' is required")
        aug_rng = rngs.augment()

        result = {}
        for key, value in element.items():
            if hasattr(value, "shape") and value.dtype in (jnp.float32, jnp.float64):
                # Add random noise to float arrays
                noise = jax.random.normal(aug_rng, value.shape) * self.magnitude
                result[key] = value + noise
            else:
                result[key] = value
        return result

    def apply_batch(self, batch: Batch, *, rngs=None) -> Batch:
        """Apply augmentation to a batch of elements."""
        if rngs is None:
            rngs = self.rngs

        # Use proper NNX vmap pattern
        @nnx.vmap(in_axes=(None, 0, None), out_axes=0)
        def batch_apply(module, batch_element, rngs):
            return module(batch_element, rngs=rngs)

        return batch_apply(self, batch, rngs)


class SimpleSamplerModule(nnx.Module):
    """A simple sampler module for testing."""

    def __init__(self, size: int, replacement: bool = False, *, rngs=None):
        """Initialize the sampler module."""
        super().__init__()
        self.size = size
        self.replacement = replacement
        self.rngs = rngs
        self.indices = nnx.Variable(jnp.arange(0))
        self._has_generated = nnx.Variable(False)

    def generate_indices(self, length: int, *, rngs=None) -> None:
        """Generate indices for sampling."""
        if rngs is None:
            rngs = self.rngs

        if not hasattr(rngs, "sampling"):
            raise ValueError("RNG stream 'sampling' is required")
        sampling_rng = rngs.sampling()

        if self.replacement:
            indices = jax.random.randint(sampling_rng, shape=(self.size,), minval=0, maxval=length)
        else:
            actual_size = min(self.size, length)
            indices = jax.random.permutation(sampling_rng, length)[:actual_size]

        self.indices[...] = indices
        self._has_generated.set_value(True)

    def get_indices(self, length: int, *, rngs=None) -> jax.Array:
        """Get the sampling indices."""
        if not self._has_generated.get_value():
            self.generate_indices(length, rngs=rngs)
        return self.indices.value


@pytest.fixture
def simple_data():
    """Sample data for testing."""
    return [{"x": jnp.array([1.0, 2.0, 3.0]), "y": i} for i in range(10)]


@pytest.fixture
def simple_batch():
    """Sample batch for testing."""
    return {
        "x": jnp.ones((8, 3)),
        "y": jnp.arange(8),
    }


@pytest.fixture
def data_source(simple_data):
    """Create a simple data source module for testing."""
    return SimpleDataSourceModule(simple_data)


@pytest.fixture
def transformer():
    """Create a simple deterministic operator module for testing.

    Note: Fixture name kept as 'transformer' for backward compatibility
    with existing tests that use this fixture.
    """
    return SimpleDeterministicOperator(scale=2.0, offset=1.0)


@pytest.fixture
def augmenter():
    """Create a simple stochastic operator module for testing.

    Note: Fixture name kept as 'augmenter' for backward compatibility
    with existing tests that use this fixture.
    """
    # Initialize with RNGs directly in constructor
    rngs = nnx.Rngs(0, augment=1)
    module = SimpleStochasticOperator(magnitude=0.1, rngs=rngs)
    return module


@pytest.fixture
def sampler():
    """Create a simple sampler module for testing."""
    # Initialize with RNGs directly in constructor
    rngs = nnx.Rngs(0, sampling=1)
    module = SimpleSamplerModule(size=5, rngs=rngs)
    return module
