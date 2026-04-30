# File: tests/unit/samplers/test_sequential_sampler.py

import flax.nnx as nnx

from datarax.core.sampler import SamplerModule
from datarax.samplers.sequential_sampler import (
    SequentialSamplerConfig,
    SequentialSamplerModule,
)


class TestSequentialSamplerModule:
    """Test suite for SequentialSamplerModule."""

    def test_initialization(self):
        """Test SequentialSamplerModule initialization."""
        config = SequentialSamplerConfig(
            stochastic=False,  # Sequential is deterministic
            num_records=100,
            num_epochs=1,
        )
        sampler = SequentialSamplerModule(config, rngs=nnx.Rngs(0))

        assert isinstance(sampler, nnx.Module)
        assert isinstance(sampler, SamplerModule)
        assert sampler.num_records.get_value() == 100
        assert sampler.num_epochs.get_value() == 1

    def test_sequential_iteration(self):
        """Test sequential iteration without shuffling."""
        config = SequentialSamplerConfig(stochastic=False, num_records=10, num_epochs=1)
        sampler = SequentialSamplerModule(config, rngs=nnx.Rngs(0))

        indices = list(sampler)

        # Should return sequential indices
        assert indices == list(range(10))

    def test_multiple_epochs(self):
        """Test iteration over multiple epochs."""
        config = SequentialSamplerConfig(stochastic=False, num_records=5, num_epochs=3)
        sampler = SequentialSamplerModule(config, rngs=nnx.Rngs(0))

        indices = list(sampler)

        # Should have 3 epochs worth of indices
        assert len(indices) == 15

        # Verify sequential order in each epoch
        expected = list(range(5)) * 3
        assert indices == expected

    def test_infinite_epochs(self):
        """Test infinite epoch iteration."""
        config = SequentialSamplerConfig(
            stochastic=False,
            num_records=5,
            num_epochs=-1,  # Infinite
        )
        sampler = SequentialSamplerModule(config, rngs=nnx.Rngs(0))

        indices = []
        for i, idx in enumerate(sampler):
            indices.append(idx)
            if i >= 24:  # Stop after 5 epochs
                break

        assert len(indices) == 25
        # Should repeat pattern
        assert indices[:5] == indices[5:10] == indices[10:15]

    def test_state_management(self):
        """Test state saving and restoration."""
        config = SequentialSamplerConfig(stochastic=False, num_records=10, num_epochs=2)
        sampler = SequentialSamplerModule(config, rngs=nnx.Rngs(0))

        # Iterate partway
        iterator = iter(sampler)
        for _ in range(5):
            next(iterator)

        # Save state
        state = sampler.get_state()

        assert state["current_index"] == 5
        assert state["current_epoch"] == 0

        # Create new sampler and restore
        new_config = SequentialSamplerConfig(stochastic=False, num_records=10, num_epochs=2)
        new_sampler = SequentialSamplerModule(new_config, rngs=nnx.Rngs(0))
        new_sampler.set_state(state)

        # Continue from checkpoint
        next_idx = next(iter(new_sampler))
        assert next_idx == 5

    def test_length(self):
        """Test length calculation."""
        config = SequentialSamplerConfig(stochastic=False, num_records=10, num_epochs=1)
        sampler = SequentialSamplerModule(config, rngs=nnx.Rngs(0))

        assert len(sampler) == 10

        # With multiple epochs
        config_multi = SequentialSamplerConfig(stochastic=False, num_records=10, num_epochs=3)
        sampler_multi = SequentialSamplerModule(config_multi, rngs=nnx.Rngs(0))
        assert len(sampler_multi) == 30

    def test_reset(self):
        """Test resetting sampler state."""
        config = SequentialSamplerConfig(stochastic=False, num_records=5, num_epochs=1)
        sampler = SequentialSamplerModule(config, rngs=nnx.Rngs(0))

        # Iterate through some elements
        iterator = iter(sampler)
        for _ in range(3):
            next(iterator)

        assert sampler.current_index.get_value() == 3

        # Reset by creating new iterator
        iterator = iter(sampler)
        assert sampler.current_index.get_value() == 0

        # First element should be 0
        assert next(iterator) == 0


class TestSequentialSamplerIntegration:
    """Integration tests for SequentialSamplerModule."""

    def test_emits_sequential_indices(self):
        """The sampler emits indices 0..N-1 in order.

        SequentialSamplerModule is a standalone primitive — Pipeline does
        not consume samplers as stages, so the integration check is the
        sampler's own iteration order.
        """
        config = SequentialSamplerConfig(stochastic=False, num_records=20, num_epochs=1)
        sampler = SequentialSamplerModule(config, rngs=nnx.Rngs(0))
        ids = list(sampler)
        assert ids == list(range(20))
