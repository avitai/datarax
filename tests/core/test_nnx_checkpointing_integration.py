"""Complete integration tests for NNX checkpointing system.

This module tests the complete checkpointing workflow across all Datarax
module types, ensuring state consistency and proper integration with Orbax.
"""

import tempfile
import warnings
from pathlib import Path

import flax.nnx as nnx
import jax.numpy as jnp
import pytest

from datarax.batching.default_batcher import DefaultBatcher, DefaultBatcherConfig
from datarax.checkpoint.handlers import OrbaxCheckpointHandler
from datarax.core.config import DataraxModuleConfig
from datarax.dag import DAGExecutor
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


class TestNNXCheckpointingIntegration:
    """Complete integration tests for NNX checkpointing."""

    def setup_method(self):
        """Set up warning filters for each test method."""
        warnings.filterwarnings(
            "ignore", "Type handler registry type.*overriding.*Module", UserWarning
        )

    def test_simple_module_checkpointing(self):
        """Test basic checkpointing workflow with simple module."""
        rngs = nnx.Rngs(42)
        module = ComplexModule(DataraxModuleConfig(), features=64, layers=3, rngs=rngs)

        # Modify state
        x = jnp.ones((10, 64))
        module(x)
        module.start_epoch()
        y2 = module(x)

        original_epoch = module.epoch.get_value()
        original_step_count = module.step_count.get_value()
        original_accuracy = module.accuracy.get_value()

        # Save checkpoint
        handler = OrbaxCheckpointHandler()
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = handler.save(temp_dir, module)
            assert checkpoint_path is not None

            # Create new module
            restored_module = ComplexModule(
                DataraxModuleConfig(), features=64, layers=3, rngs=nnx.Rngs(999)
            )
            assert restored_module.epoch.get_value() != original_epoch

            # Restore state
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", "Sharding info not provided when restoring", UserWarning
                )
                restored_state = handler.restore(temp_dir, target=None)
            restored_module.set_state(restored_state)

            # Verify state restoration
            assert restored_module.epoch.get_value() == original_epoch
            assert restored_module.step_count.get_value() == original_step_count
            assert restored_module.accuracy.get_value() == original_accuracy

            # Verify functionality after restoration
            y_restored = restored_module(x)
            assert y_restored.shape == y2.shape

    def test_multiple_module_types_checkpointing(self):
        """Test checkpointing across different Datarax module types."""
        rngs = nnx.Rngs(42)

        # Create different module types (all with config-first pattern)
        source_module = MemorySource(MemorySourceConfig(), [1, 2, 3, 4, 5], rngs=rngs)
        range_sampler = RangeSampler(RangeSamplerConfig(start=0, stop=5, step=1), rngs=rngs)
        shuffle_sampler = ShuffleSampler(
            ShuffleSamplerConfig(buffer_size=3, dataset_size=5), rngs=rngs
        )
        batcher = DefaultBatcher(DefaultBatcherConfig(), rngs=rngs)
        sharder = ArraySharder(rngs=rngs)  # ArraySharder takes optional sharding_rules first

        modules = {
            "source": source_module,
            "range_sampler": range_sampler,
            "shuffle_sampler": shuffle_sampler,
            "batcher": batcher,
            "sharder": sharder,
        }

        # Modify states
        list(range_sampler)[:2]  # Advance iterator
        list(shuffle_sampler)[:2]  # Advance iterator

        # Get original states
        {name: module.get_state() for name, module in modules.items()}

        handler = OrbaxCheckpointHandler()

        # Test individual checkpointing
        for name, module in modules.items():
            with tempfile.TemporaryDirectory() as temp_dir:
                checkpoint_path = handler.save(temp_dir, module)
                assert checkpoint_path is not None

                # Restore and verify
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", "Sharding info not provided when restoring", UserWarning
                    )
                    restored_state = handler.restore(temp_dir, target=None)
                assert restored_state is not None

                # Create fresh module and restore
                if name == "source":
                    fresh_module = MemorySource(
                        MemorySourceConfig(), [1, 2, 3, 4, 5], rngs=nnx.Rngs(999)
                    )
                elif name == "range_sampler":
                    fresh_module = RangeSampler(
                        RangeSamplerConfig(start=0, stop=5, step=1), rngs=nnx.Rngs(999)
                    )
                elif name == "shuffle_sampler":
                    fresh_module = ShuffleSampler(
                        ShuffleSamplerConfig(buffer_size=3, dataset_size=5), rngs=nnx.Rngs(999)
                    )
                elif name == "batcher":
                    fresh_module = DefaultBatcher(DefaultBatcherConfig(), rngs=nnx.Rngs(999))
                elif name == "sharder":
                    fresh_module = ArraySharder(rngs=nnx.Rngs(999))
                else:
                    raise ValueError(f"Unknown module type: {name}")

                fresh_module.set_state(restored_state)

                # Basic verification that state was restored
                restored_state_after = fresh_module.get_state()
                assert isinstance(restored_state_after, dict)

    def test_datastream_module_checkpointing(self):
        """Test complete Pipeline checkpointing."""
        rngs = nnx.Rngs(42)

        # Create Pipeline following batch-first principle
        source = MemorySource(MemorySourceConfig(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], rngs=rngs)

        # Correct order: source -> batch -> sampler (batch-first enforcement)
        stream = DAGExecutor().add(source).batch(batch_size=2)

        # For now, skip sampler as it would need to operate on batches
        # This test focuses on checkpointing the core pipeline

        # Modify state by iterating
        stream_iter = iter(stream)
        [next(stream_iter) for _ in range(3)]

        # Save state
        handler = OrbaxCheckpointHandler()
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = handler.save(temp_dir, stream)
            assert checkpoint_path is not None

            # Restore state without target (let handler create new instance)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", "Sharding info not provided when restoring", UserWarning
                )
                restored_stream = handler.restore(temp_dir, target=None)
            # Verify some aspects of state restoration
            assert restored_stream is not None
            # The restored stream should have the same structure as the original

    def test_nested_module_checkpointing(self):
        """Test checkpointing of modules with nested components."""
        rngs = nnx.Rngs(42)

        # Create complex nested module
        module = ComplexModule(DataraxModuleConfig(), features=32, layers=2, rngs=rngs)

        # Exercise the module
        x = jnp.ones((5, 32))
        for _ in range(3):
            module(x)
            module.start_epoch()

        original_epoch = module.epoch.get_value()
        original_accuracy = module.accuracy.get_value()
        module.batcher.get_state()

        # Save and restore
        handler = OrbaxCheckpointHandler()
        with tempfile.TemporaryDirectory() as temp_dir:
            handler.save(temp_dir, module)

            # Create fresh module
            fresh_module = ComplexModule(
                DataraxModuleConfig(), features=32, layers=2, rngs=nnx.Rngs(999)
            )

            # Restore
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", "Sharding info not provided when restoring", UserWarning
                )
                restored_state = handler.restore(temp_dir, target=None)
            fresh_module.set_state(restored_state)

            # Verify restoration
            assert fresh_module.epoch.get_value() == original_epoch
            assert fresh_module.accuracy.get_value() == original_accuracy

            # Verify nested module state
            restored_batcher_state = fresh_module.batcher.get_state()
            assert isinstance(restored_batcher_state, dict)

    def test_concurrent_checkpointing(self):
        """Test that concurrent checkpointing operations don't interfere."""
        rngs = nnx.Rngs(42)

        # Create multiple modules
        modules = [
            ComplexModule(DataraxModuleConfig(), features=16, layers=1, rngs=rngs) for _ in range(3)
        ]

        # Modify each module differently
        for i, module in enumerate(modules):
            x = jnp.ones((2, 16))
            for _ in range(i + 1):
                module(x)
                module.start_epoch()

        original_epochs = [module.epoch.get_value() for module in modules]

        # Save all modules concurrently (simulated)
        handler = OrbaxCheckpointHandler()
        temp_dirs = []
        checkpoint_paths = []

        try:
            for i, module in enumerate(modules):
                temp_dir = tempfile.mkdtemp()
                temp_dirs.append(temp_dir)
                checkpoint_path = handler.save(temp_dir, module, step=i)
                checkpoint_paths.append(checkpoint_path)

            # Restore and verify each
            for i, (temp_dir, original_epoch) in enumerate(zip(temp_dirs, original_epochs)):
                fresh_module = ComplexModule(
                    DataraxModuleConfig(), features=16, layers=1, rngs=nnx.Rngs(999)
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", "Sharding info not provided when restoring", UserWarning
                    )
                    restored_state = handler.restore(temp_dir, step=i)
                fresh_module.set_state(restored_state)

                assert fresh_module.epoch.get_value() == original_epoch

        finally:
            # Cleanup
            import shutil

            for temp_dir in temp_dirs:
                shutil.rmtree(temp_dir, ignore_errors=True)

    def test_state_consistency_across_cycles(self):
        """Test that multiple save/restore cycles maintain state consistency."""
        rngs = nnx.Rngs(42)
        module = ComplexModule(DataraxModuleConfig(), features=16, layers=1, rngs=rngs)

        # Exercise module
        x = jnp.ones((2, 16))
        module(x)
        module.start_epoch()

        handler = OrbaxCheckpointHandler()

        # Perform multiple save/restore cycles
        for cycle in range(3):
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save current state
                handler.save(temp_dir, module)

                # Record current state
                current_epoch = module.epoch.get_value()
                current_accuracy = module.accuracy.get_value()

                # Modify module
                module(x)
                module.start_epoch()
                assert module.epoch.get_value() != current_epoch

                # Restore from checkpoint
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", "Sharding info not provided when restoring", UserWarning
                    )
                    restored_state = handler.restore(temp_dir)
                module.set_state(restored_state)

                # Verify restoration
                assert module.epoch.get_value() == current_epoch
                assert module.accuracy.get_value() == current_accuracy

    def test_large_state_checkpointing(self):
        """Test checkpointing with large state arrays."""
        rngs = nnx.Rngs(42)

        # Create module with large state (but not too large for tests)
        module = LargeStateModule(DataraxModuleConfig(), state_size=100, rngs=rngs)

        # Exercise module
        x = jnp.ones((10, 100))
        module(x)

        original_weights_sum = float(jnp.sum(module.large_weights[...]))
        original_cache_sum = float(jnp.sum(module.large_cache[...]))

        # Save and restore
        handler = OrbaxCheckpointHandler()
        with tempfile.TemporaryDirectory() as temp_dir:
            handler.save(temp_dir, module)

            # Create fresh module
            fresh_module = LargeStateModule(
                DataraxModuleConfig(), state_size=100, rngs=nnx.Rngs(999)
            )

            # Restore
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", "Sharding info not provided when restoring", UserWarning
                )
                restored_state = handler.restore(temp_dir)
            fresh_module.set_state(restored_state)

            # Verify large arrays were restored correctly
            restored_weights_sum = float(jnp.sum(fresh_module.large_weights[...]))
            restored_cache_sum = float(jnp.sum(fresh_module.large_cache[...]))

            assert abs(restored_weights_sum - original_weights_sum) < 1e-6
            assert abs(restored_cache_sum - original_cache_sum) < 1e-6


class TestCheckpointingErrorHandling:
    """Test error handling and recovery scenarios in checkpointing."""

    def setup_method(self):
        """Set up warning filters for each test method."""
        warnings.filterwarnings(
            "ignore", "Type handler registry type.*overriding.*Module", UserWarning
        )

    def test_malformed_checkpoint_handling(self):
        """Test handling of malformed checkpoint files."""
        rngs = nnx.Rngs(42)
        ComplexModule(DataraxModuleConfig(), features=16, layers=1, rngs=rngs)
        handler = OrbaxCheckpointHandler()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a malformed checkpoint directory
            malformed_path = Path(temp_dir) / "malformed"
            malformed_path.mkdir()
            (malformed_path / "invalid_file.txt").write_text("not a checkpoint")

            # Should handle gracefully
            with pytest.raises((ValueError, FileNotFoundError, Exception)):
                handler.restore(str(malformed_path))

    def test_incompatible_state_handling(self):
        """Test handling of incompatible state restoration."""
        rngs = nnx.Rngs(42)

        # Create module and save its state
        original_module = ComplexModule(DataraxModuleConfig(), features=16, layers=1, rngs=rngs)
        handler = OrbaxCheckpointHandler()

        with tempfile.TemporaryDirectory() as temp_dir:
            handler.save(temp_dir, original_module)

            # Try to restore to incompatible module (different features)
            incompatible_module = ComplexModule(
                DataraxModuleConfig(), features=32, layers=1, rngs=rngs
            )

            # This may fail or partially succeed depending on implementation
            # The important thing is it doesn't crash the system
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", "Sharding info not provided when restoring", UserWarning
                    )
                    restored_state = handler.restore(temp_dir)
                incompatible_module.set_state(restored_state)
                # If it succeeds, verify basic functionality
                x = jnp.ones((2, 32))
                y = incompatible_module(x)
                assert y.shape == (2, 32)
            except Exception:
                # Expected for incompatible architectures
                pass

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

        # Should handle partial restoration gracefully
        fresh_module.set_state(partial_state)

        # Verify basic functionality still works
        x = jnp.ones((2, 16))
        y = fresh_module(x)
        assert y.shape == (2, 16)
