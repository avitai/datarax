"""Complete integration tests for NNX checkpointing system.

This module tests the complete checkpointing workflow across all Datarax
module types, ensuring state consistency and proper integration with Orbax.
"""

import warnings

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from datarax.batching.default_batcher import DefaultBatcher, DefaultBatcherConfig
from datarax.checkpoint import IteratorCheckpoint
from datarax.core.config import DataraxModuleConfig
from datarax.core.module import DataraxModule
from datarax.samplers.range_sampler import RangeSampler, RangeSamplerConfig
from datarax.samplers.shuffle_sampler import ShuffleSampler, ShuffleSamplerConfig
from datarax.sharding.array_sharder import ArraySharder
from datarax.sources.memory_source import MemorySource, MemorySourceConfig


class ComplexModule(DataraxModule):
    """Complex test module with nested state for integration testing."""

    def __init__(self, config: DataraxModuleConfig, features: int, layers: int, *, rngs: nnx.Rngs):
        """Initialize with multiple layers and complex state."""
        super().__init__(config, rngs=rngs)
        self.features = features
        self.layers = layers

        # Create multiple layers - use nnx.List for proper state tracking
        linear_layers_list = []
        for i in range(layers):
            layer = nnx.Linear(features, features, rngs=rngs)
            linear_layers_list.append(layer)
            setattr(self, f"layer_{i}", layer)
        self.linear_layers = nnx.List(linear_layers_list)

        # Complex state variables
        self.epoch = nnx.Variable(0)
        self.step_count = nnx.Variable(0)
        self.loss_history: nnx.Variable[list[float]] = nnx.Variable([])
        self.accuracy = nnx.Variable(0.0)  # Simplified from dict

        # Nested module
        self.batcher = DefaultBatcher(DefaultBatcherConfig(), rngs=rngs)

    def __call__(self, x):
        """Forward pass through all layers."""
        for layer in self.linear_layers:
            x = layer(x)
            x = nnx.relu(x)

        # Update state
        self.step_count.set_value(self.step_count.get_value() + 1)
        current_loss = float(jnp.mean(jnp.square(x)))
        self.loss_history.set_value([*self.loss_history.get_value(), current_loss])
        self.accuracy.set_value(0.95)  # Simplified assignment

        return x

    def start_epoch(self):
        """Start a new epoch."""
        self.epoch.set_value(self.epoch.get_value() + 1)
        self.step_count.set_value(0)
        self.loss_history.set_value([])


class LargeStateModule(DataraxModule):
    """Module with large state for stress testing."""

    def __init__(self, config: DataraxModuleConfig, state_size: int, *, rngs: nnx.Rngs):
        """Initialize with large state arrays."""
        super().__init__(config, rngs=rngs)
        self.state_size = state_size

        # Large state arrays
        self.large_weights = nnx.Variable(jnp.ones((state_size, state_size)))
        self.large_bias = nnx.Variable(jnp.zeros(state_size))
        self.large_cache = nnx.Variable(jnp.zeros((state_size, state_size, 10)))

        # Linear layer for processing
        self.linear = nnx.Linear(state_size, state_size, rngs=rngs)

    def __call__(self, x):
        """Process input with large state."""
        # Use large weights
        x = jnp.dot(x, self.large_weights[...]) + self.large_bias[...]
        x = self.linear(x)

        # Update cache
        self.large_cache[...] = self.large_cache[...].at[:, :, 0].set(self.large_weights[...])

        return x


# ShuffleSampler raises this flag in set_state so its next iteration resumes from the
# restored position; it marks that a restore happened and is not part of the saved state.
_RESTORE_MARKERS = {"_resume_next_iter"}


def _plain(state):
    """Turn a pure state dict into something ``==`` compares leaf by leaf."""
    state = {key: value for key, value in state.items() if key not in _RESTORE_MARKERS}
    return jax.tree.map(lambda leaf: leaf.tolist() if hasattr(leaf, "tolist") else leaf, state)


class TestNNXCheckpointingIntegration:
    """Complete integration tests for NNX checkpointing."""

    def setup_method(self):
        """Set up warning filters for each test method."""
        warnings.filterwarnings(
            "ignore", "Type handler registry type.*overriding.*Module", UserWarning
        )

    def test_simple_module_checkpointing(self, tmp_path):
        """A module's variables and nested layers come back into a fresh module."""
        module = ComplexModule(DataraxModuleConfig(), features=64, layers=3, rngs=nnx.Rngs(42))
        x = jnp.ones((10, 64))
        module(x)
        module.start_epoch()
        y2 = module(x)

        with IteratorCheckpoint(tmp_path) as checkpoint:
            checkpoint.save(module, step=0)
            restored_module = ComplexModule(
                DataraxModuleConfig(), features=64, layers=3, rngs=nnx.Rngs(999)
            )
            assert restored_module.epoch.get_value() != module.epoch.get_value()
            checkpoint.restore(restored_module, step=0)

        assert restored_module.epoch.get_value() == module.epoch.get_value()
        assert restored_module.step_count.get_value() == module.step_count.get_value()
        assert restored_module.accuracy.get_value() == module.accuracy.get_value()
        assert restored_module(x).shape == y2.shape

    @pytest.mark.parametrize(
        "build",
        [
            lambda rngs: MemorySource(MemorySourceConfig(), [1, 2, 3, 4, 5], rngs=rngs),
            lambda rngs: RangeSampler(RangeSamplerConfig(start=0, stop=5, step=1), rngs=rngs),
            lambda rngs: ShuffleSampler(ShuffleSamplerConfig(dataset_size=5), rngs=rngs),
            lambda rngs: DefaultBatcher(DefaultBatcherConfig(), rngs=rngs),
            lambda rngs: ArraySharder(rngs=rngs),
        ],
        ids=["source", "range_sampler", "shuffle_sampler", "batcher", "sharder"],
    )
    def test_every_module_type_round_trips(self, tmp_path, build):
        """Each Datarax module type restores into a freshly built instance."""
        module = build(nnx.Rngs(42))
        if hasattr(module, "__iter__"):
            for _ in zip(range(2), module):
                pass

        with IteratorCheckpoint(tmp_path) as checkpoint:
            checkpoint.save(module, step=0)
            fresh = build(nnx.Rngs(999))
            checkpoint.restore(fresh, step=0)

        assert _plain(fresh.get_state()) == _plain(module.get_state())

    def test_nested_module_checkpointing(self, tmp_path):
        """A module with nested layers, variables and a batcher round-trips."""
        module = ComplexModule(DataraxModuleConfig(), features=32, layers=2, rngs=nnx.Rngs(42))
        x = jnp.ones((5, 32))
        for _ in range(3):
            module(x)
            module.start_epoch()

        with IteratorCheckpoint(tmp_path) as checkpoint:
            checkpoint.save(module, step=0)
            fresh_module = ComplexModule(
                DataraxModuleConfig(), features=32, layers=2, rngs=nnx.Rngs(999)
            )
            checkpoint.restore(fresh_module, step=0)

        assert fresh_module.epoch.get_value() == module.epoch.get_value()
        assert fresh_module.accuracy.get_value() == module.accuracy.get_value()
        assert _plain(fresh_module.batcher.get_state()) == _plain(module.batcher.get_state())
        # nnx.List children are keyed by integer; the restore keeps them integers.
        assert jnp.array_equal(
            fresh_module.linear_layers[1].kernel.get_value(),
            module.linear_layers[1].kernel.get_value(),
        )

    def test_modules_saved_side_by_side_do_not_interfere(self, tmp_path):
        """Three modules saved under three directories each restore their own state."""
        modules = [
            ComplexModule(DataraxModuleConfig(), features=16, layers=1, rngs=nnx.Rngs(42))
            for _ in range(3)
        ]
        for i, module in enumerate(modules):
            for _ in range(i + 1):
                module(jnp.ones((2, 16)))
                module.start_epoch()

        for i, module in enumerate(modules):
            with IteratorCheckpoint(tmp_path / f"module_{i}") as checkpoint:
                checkpoint.save(module, step=i)

        for i, module in enumerate(modules):
            fresh = ComplexModule(DataraxModuleConfig(), features=16, layers=1, rngs=nnx.Rngs(999))
            with IteratorCheckpoint(tmp_path / f"module_{i}") as checkpoint:
                checkpoint.restore(fresh, step=i)
            assert fresh.epoch.get_value() == module.epoch.get_value()

    def test_state_consistency_across_cycles(self, tmp_path):
        """Saving, advancing and restoring the same module returns it to the saved state."""
        module = ComplexModule(DataraxModuleConfig(), features=16, layers=1, rngs=nnx.Rngs(42))
        x = jnp.ones((2, 16))
        module(x)
        module.start_epoch()

        with IteratorCheckpoint(tmp_path, max_to_keep=3) as checkpoint:
            for cycle in range(3):
                checkpoint.save(module, step=cycle)
                saved_epoch = module.epoch.get_value()
                saved_accuracy = module.accuracy.get_value()

                module(x)
                module.start_epoch()
                assert module.epoch.get_value() != saved_epoch

                checkpoint.restore(module, step=cycle)

                assert module.epoch.get_value() == saved_epoch
                assert module.accuracy.get_value() == saved_accuracy

    def test_large_state_checkpointing(self, tmp_path):
        """Large array state restores exactly."""
        module = LargeStateModule(DataraxModuleConfig(), state_size=100, rngs=nnx.Rngs(42))
        module(jnp.ones((10, 100)))

        with IteratorCheckpoint(tmp_path) as checkpoint:
            checkpoint.save(module, step=0)
            fresh_module = LargeStateModule(
                DataraxModuleConfig(), state_size=100, rngs=nnx.Rngs(999)
            )
            checkpoint.restore(fresh_module, step=0)

        assert jnp.array_equal(fresh_module.large_weights[...], module.large_weights[...])
        assert jnp.array_equal(fresh_module.large_cache[...], module.large_cache[...])


class TestCheckpointingErrorHandling:
    """Test error handling and recovery scenarios in checkpointing."""

    def setup_method(self):
        """Set up warning filters for each test method."""
        warnings.filterwarnings(
            "ignore", "Type handler registry type.*overriding.*Module", UserWarning
        )

    def test_directory_without_checkpoints_raises(self, tmp_path):
        """A directory holding no Orbax steps cannot be restored from."""
        (tmp_path / "invalid_file.txt").write_text("not a checkpoint")
        module = ComplexModule(DataraxModuleConfig(), features=16, layers=1, rngs=nnx.Rngs(42))

        with IteratorCheckpoint(tmp_path) as checkpoint:
            with pytest.raises(ValueError, match="No checkpoints found"):
                checkpoint.restore(module)

    def test_incompatible_module_cannot_receive_the_state(self, tmp_path):
        """A checkpoint saved from one shape does not restore into another."""
        original = ComplexModule(DataraxModuleConfig(), features=16, layers=1, rngs=nnx.Rngs(42))
        incompatible = ComplexModule(
            DataraxModuleConfig(), features=32, layers=1, rngs=nnx.Rngs(42)
        )
        before = _plain(incompatible.get_state())

        with IteratorCheckpoint(tmp_path) as checkpoint:
            checkpoint.save(original, step=0)
            with pytest.raises(ValueError):
                checkpoint.restore(incompatible, step=0)

        assert _plain(incompatible.get_state()) == before

    def test_partial_state_restoration(self):
        """Test restoration when some state components are missing."""
        rngs = nnx.Rngs(42)
        module = ComplexModule(DataraxModuleConfig(), features=16, layers=1, rngs=rngs)

        # Get complete state
        complete_state = module.get_state()

        # Create partial state (remove some keys)
        partial_state = {
            k: v for k, v in complete_state.items() if not k.startswith("loss_history")
        }

        # Create fresh module
        fresh_module = ComplexModule(
            DataraxModuleConfig(), features=16, layers=1, rngs=nnx.Rngs(999)
        )

        with pytest.raises(ValueError, match="structurally incompatible"):
            fresh_module.set_state(partial_state)
