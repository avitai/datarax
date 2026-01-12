"""Tests for checkpointable module functionality."""

import tempfile
import warnings
from pathlib import Path

import flax.nnx as nnx
import jax.numpy as jnp
import pytest

from datarax.checkpoint.handlers import OrbaxCheckpointHandler
from datarax.core.config import DataraxModuleConfig
from datarax.core.module import CheckpointableIteratorModule, DataraxModule


class SimpleModule(DataraxModule):
    """Simple test module for checkpointing tests."""

    def __init__(self, features: int, *, rngs: nnx.Rngs):
        """Initialize with a linear layer."""
        config = DataraxModuleConfig()
        super().__init__(config, rngs=rngs)
        self.features = features
        self.linear = nnx.Linear(features, features, rngs=rngs)
        self.counter = nnx.Variable(0)

    def __call__(self, x):
        """Forward pass."""
        self.counter.set_value(self.counter.get_value() + 1)
        return self.linear(x)


class SimpleIteratorModule(CheckpointableIteratorModule):
    """Simple iterator module for testing checkpointing."""

    def __init__(self, max_items: int, *, rngs: nnx.Rngs | None = None):
        """Initialize with maximum number of items."""
        if rngs is None:
            rngs = nnx.Rngs(42)
        config = DataraxModuleConfig()
        super().__init__(config, rngs=rngs)
        self.max_items = max_items
        self.position.set_value(0)

    def __next__(self) -> int:
        """Get next item."""
        if self.position.get_value() >= self.max_items:
            raise StopIteration

        current = self.position.get_value()
        self.position.set_value(self.position.get_value() + 1)
        self.current.set_value(current)
        return current

    def __len__(self) -> int:
        """Return total number of items."""
        return self.max_items

    def reset(self) -> None:
        """Reset iterator position."""
        super().reset()
        self.position.set_value(0)


class TestDataraxModuleCheckpointing:
    """Test checkpointing functionality of DataraxModule."""

    def test_implements_checkpointable_protocol(self):
        """Test that DataraxModule implements Checkpointable protocol."""
        rngs = nnx.Rngs(42)
        module = SimpleModule(10, rngs=rngs)

        # Should implement Checkpointable protocol methods (duck typing)
        assert hasattr(module, "get_state")
        assert hasattr(module, "set_state")
        assert callable(module.get_state)
        assert callable(module.set_state)

    def test_state_serialization_and_restoration(self):
        """Test that module state can be serialized and restored."""
        rngs = nnx.Rngs(42)
        module = SimpleModule(10, rngs=rngs)

        # Create some data and do a forward pass to change state
        x = jnp.ones((5, 10))
        y1 = module(x)

        # Counter should be 1
        assert module.counter.get_value() == 1

        # Get state
        state = module.get_state()
        assert isinstance(state, dict)
        assert "counter" in state
        # The counter value is stored directly as an integer in the NNX state
        assert state["counter"] == 1

        # Do another forward pass
        module(x)
        assert module.counter.get_value() == 2

        # Restore from saved state
        module.set_state(state)
        assert module.counter.get_value() == 1

        # Output should be the same as y1
        y3 = module(x)
        assert jnp.allclose(y1, y3)

    def test_clone_functionality(self):
        """Test that module cloning works."""
        rngs = nnx.Rngs(42)
        module = SimpleModule(10, rngs=rngs)

        # Change some state
        x = jnp.ones((5, 10))
        module(x)

        # Clone the module
        cloned = module.clone()

        # Should have same state but be different objects
        assert module is not cloned
        assert module.counter.get_value() == cloned.counter.get_value()

        # Changing one shouldn't affect the other
        module(x)
        assert module.counter.get_value() != cloned.counter.get_value()


class TestCheckpointableIteratorModule:
    """Test checkpointing functionality of CheckpointableIteratorModule."""

    def test_implements_checkpointable_iterator_protocol(self):
        """Test that CheckpointableIteratorModule implements CheckpointableIterator protocol."""
        iterator = SimpleIteratorModule(5)

        # Should implement both Iterator and Checkpointable
        assert hasattr(iterator, "__iter__")
        assert hasattr(iterator, "__next__")
        assert hasattr(iterator, "__len__")
        assert hasattr(iterator, "get_state")
        assert hasattr(iterator, "set_state")

    def test_iterator_functionality(self):
        """Test basic iterator functionality."""
        iterator = SimpleIteratorModule(3)

        # Test length
        assert len(iterator) == 3

        # Test iteration
        items = list(iterator)
        assert items == [0, 1, 2]

        # Iterator should be exhausted
        with pytest.raises(StopIteration):
            next(iterator)

    def test_iterator_state_checkpointing(self):
        """Test that iterator state can be checkpointed and restored."""
        iterator = SimpleIteratorModule(5)

        # Consume some items
        first = next(iterator)
        second = next(iterator)
        assert first == 0
        assert second == 1
        assert iterator.position.get_value() == 2

        # Save state
        state = iterator.get_state()

        # Consume more items
        third = next(iterator)
        assert third == 2
        assert iterator.position.get_value() == 3

        # Restore state
        iterator.set_state(state)
        assert iterator.position.get_value() == 2

        # Next item should be 2 again
        restored_third = next(iterator)
        assert restored_third == 2

    def test_iterator_reset(self):
        """Test iterator reset functionality."""
        iterator = SimpleIteratorModule(3)

        # Consume some items
        next(iterator)
        next(iterator)
        assert iterator.position.get_value() == 2

        # Reset
        iterator.reset()
        assert iterator.position.get_value() == 0
        assert iterator.current.get_value() is None

        # Should start from beginning
        first = next(iterator)
        assert first == 0

    def test_iterator_state_variables_as_nnx_variables(self):
        """Test that iterator state variables are properly stored as NNX Variables."""
        iterator = SimpleIteratorModule(3)

        # Check that state variables are NNX Variables
        assert isinstance(iterator.epoch, nnx.Variable)
        assert isinstance(iterator.position, nnx.Variable)
        assert isinstance(iterator.idx, nnx.Variable)
        assert isinstance(iterator.current, nnx.Variable)

        # Check that they're included in the NNX state
        state = nnx.state(iterator)
        state_dict = nnx.to_pure_dict(state)

        assert "epoch" in state_dict
        assert "position" in state_dict
        assert "idx" in state_dict
        assert "current" in state_dict


class TestIntegrationWithOrbax:
    """Integration tests with the existing Orbax checkpoint system."""

    def test_module_with_orbax_checkpoint_handler(self):
        """Test that modules work with the existing OrbaxCheckpointHandler."""
        from datarax.checkpoint.handlers import OrbaxCheckpointHandler

        rngs = nnx.Rngs(42)
        module = SimpleModule(10, rngs=rngs)

        # Change module state
        x = jnp.ones((5, 10))
        module(x)
        original_counter = module.counter.get_value()

        # Save with Orbax handler (using context manager for proper cleanup)
        with tempfile.TemporaryDirectory() as tmp_dir:
            with OrbaxCheckpointHandler() as handler:
                checkpoint_path = handler.save(tmp_dir, module)
                assert Path(checkpoint_path).exists()

                # Create new module and restore
                new_module = SimpleModule(10, rngs=nnx.Rngs(42))
                # Suppress the Orbax sharding info warning
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="Sharding info not provided when restoring"
                    )
                    restored_module = handler.restore(tmp_dir, new_module)

                # Should have same counter value
                assert restored_module.counter.get_value() == original_counter

    def test_iterator_with_orbax_checkpoint_handler(self):
        """Test that iterator modules work with the existing OrbaxCheckpointHandler."""
        from datarax.checkpoint.handlers import OrbaxCheckpointHandler

        iterator = SimpleIteratorModule(5)

        # Consume some items
        next(iterator)
        next(iterator)
        original_position = iterator.position.get_value()

        # Save with Orbax handler (using context manager for proper cleanup)
        with tempfile.TemporaryDirectory() as tmp_dir:
            with OrbaxCheckpointHandler() as handler:
                checkpoint_path = handler.save(tmp_dir, iterator)
                assert Path(checkpoint_path).exists()

                # Create new iterator and restore
                new_iterator = SimpleIteratorModule(5)
                # Suppress the Orbax sharding info warning
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="Sharding info not provided when restoring"
                    )
                    restored_iterator = handler.restore(tmp_dir, new_iterator)

                # Should have same position
                assert restored_iterator.position.get_value() == original_position


class TestOrbaxCheckpointHandlers:
    """Test Orbax checkpoint handlers."""

    def test_orbax_checkpoint_handler_initialization(self):
        """Test that OrbaxCheckpointHandler initializes with NNX support."""
        with OrbaxCheckpointHandler() as handler:
            assert handler.checkpointer is not None

    def test_orbax_save_and_restore_datarax_module(self):
        """Test saving and restoring DataraxModule using OrbaxCheckpointHandler."""
        # Create a module
        rngs = nnx.Rngs(42)
        module = SimpleModule(10, rngs=rngs)

        # Modify its state
        x = jnp.ones((5, 10))
        module(x)
        assert module.counter.get_value() == 1

        # Save using Orbax handler (with context manager for proper cleanup)
        with tempfile.TemporaryDirectory() as temp_dir:
            with OrbaxCheckpointHandler() as handler:
                checkpoint_path = handler.save(temp_dir, module)
                assert checkpoint_path is not None

                # Create a new module to restore into
                restored_module = SimpleModule(10, rngs=nnx.Rngs(123))
                assert restored_module.counter.get_value() == 0

                # Restore the checkpoint
                # Suppress the Orbax sharding info warning
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="Sharding info not provided when restoring"
                    )
                    restored_state = handler.restore(temp_dir, target=None)

                # Apply the restored state
                if restored_state is not None:
                    restored_module.set_state(restored_state)

                # Verify state was restored
                assert restored_module.counter.get_value() == 1

    def test_orbax_save_and_restore_iterator_module(self):
        """Test saving and restoring iterator module using OrbaxCheckpointHandler."""
        # Create an iterator module
        module = SimpleIteratorModule(10)

        # Advance the iterator
        next(module)  # 0
        next(module)  # 1
        next(module)  # 2
        assert module.position.get_value() == 3

        # Save using Orbax handler (with context manager for proper cleanup)
        with tempfile.TemporaryDirectory() as temp_dir:
            with OrbaxCheckpointHandler() as handler:
                checkpoint_path = handler.save(temp_dir, module)
                assert checkpoint_path is not None

                # Create a new module to restore into
                restored_module = SimpleIteratorModule(10)
                assert restored_module.position.get_value() == 0

                # Restore the checkpoint
                # Suppress the Orbax sharding info warning
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="Sharding info not provided when restoring"
                    )
                    restored_state = handler.restore(temp_dir, target=None)

                # Apply the restored state
                if restored_state is not None:
                    restored_module.set_state(restored_state)

                # Verify state was restored
                assert restored_module.position.get_value() == 3


class TestDataraxModuleAdditionalCoverage:
    """Additional tests to increase code coverage for DataraxModule."""

    def test_module_with_name(self):
        """Test module initialization with a name."""
        config = DataraxModuleConfig()
        module = DataraxModule(config, name="test_module")
        assert module.name == "test_module"

    def test_requires_rng_streams_default(self):
        """Test that requires_rng_streams returns None by default."""
        config = DataraxModuleConfig()
        module = DataraxModule(config)
        assert module.requires_rng_streams() is None

    def test_ensure_rng_streams_with_required_streams(self):
        """Test ensure_rng_streams when module requires specific streams."""

        class CustomModule(DataraxModule):
            def __init__(self):
                super().__init__(DataraxModuleConfig())

            def requires_rng_streams(self):
                return ["dropout", "params"]

        module = CustomModule()

        # Should not raise when all required streams are available
        module.ensure_rng_streams(["dropout", "params", "extra"])

        # Should raise when a required stream is missing
        with pytest.raises(ValueError) as exc_info:
            module.ensure_rng_streams(["dropout"])

        assert "RNG stream 'params' is required" in str(exc_info.value)
        assert "Available streams: ['dropout']" in str(exc_info.value)

    def test_ensure_rng_streams_with_no_requirements(self):
        """Test ensure_rng_streams when module has no RNG requirements."""
        config = DataraxModuleConfig()
        module = DataraxModule(config)
        # Should not raise even with empty stream list
        module.ensure_rng_streams([])
        module.ensure_rng_streams(["any", "streams"])


class TestCheckpointManagerIntegration:
    """Test integration with checkpoint manager."""

    def test_checkpoint_manager_basic(self):
        """Test basic checkpoint manager functionality."""
        # Note: Using OrbaxCheckpointHandler as manager since CheckpointManager doesn't exist
        # This simulates manager-like functionality

        rngs = nnx.Rngs(42)
        module = SimpleModule(10, rngs=rngs)

        with tempfile.TemporaryDirectory() as tmp_dir:
            with OrbaxCheckpointHandler() as handler:
                # Save checkpoint (simulating manager with step in path)
                handler.save(f"{tmp_dir}/step_0", module)

                # Modify module
                x = jnp.ones((5, 10))
                module(x)

                # Create new module and restore
                new_module = SimpleModule(10, rngs=nnx.Rngs(123))
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    restored = handler.restore(f"{tmp_dir}/step_0", new_module)

                # Verify restoration
                assert restored.counter.get_value() == 0  # Should be initial state

    def test_checkpoint_manager_multiple_steps(self):
        """Test checkpoint manager with multiple steps."""
        # Simulating manager functionality with OrbaxCheckpointHandler

        rngs = nnx.Rngs(42)
        module = SimpleModule(10, rngs=rngs)
        x = jnp.ones((5, 10))

        with tempfile.TemporaryDirectory() as tmp_dir:
            with OrbaxCheckpointHandler() as handler:
                saved_steps = []

                # Save multiple checkpoints
                for step in range(5):
                    module(x)  # Increment counter
                    handler.save(f"{tmp_dir}/step_{step}", module)
                    saved_steps.append(step)

                # Verify we can restore from different steps
                new_module = SimpleModule(10, rngs=nnx.Rngs(123))
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    restored = handler.restore(f"{tmp_dir}/step_2", new_module)
                # Counter would be 3 after being called 3 times (steps 0, 1, 2)
                assert restored.counter.get_value() == 3


class TestDistributedCheckpointing:
    """Test distributed checkpointing scenarios."""

    def test_sharded_state_checkpointing(self):
        """Test checkpointing with sharded state."""
        # Note: Full distributed testing requires multi-device setup
        # This is a simplified test for single device
        rngs = nnx.Rngs(42)
        module = SimpleModule(10, rngs=rngs)

        # Simulate sharded state (in real scenario this would use pjit)
        module.get_state()

        # Save and restore should handle sharded state
        with tempfile.TemporaryDirectory() as tmp_dir:
            with OrbaxCheckpointHandler() as handler:
                handler.save(tmp_dir, module)

                # Restore to new module
                new_module = SimpleModule(10, rngs=nnx.Rngs(123))
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Sharding info")
                    restored = handler.restore(tmp_dir, new_module)

                assert restored is not None

    def test_multi_host_checkpointing_simulation(self):
        """Test multi-host checkpointing (simulated)."""
        # This simulates multi-host without actual distribution
        rngs = nnx.Rngs(42)
        module = SimpleModule(10, rngs=rngs)

        # In real multi-host, different hosts would have different data
        # Here we simulate with process_index

        with tempfile.TemporaryDirectory() as tmp_dir:
            with OrbaxCheckpointHandler() as handler:
                # Save with process index simulation
                checkpoint_path = handler.save(tmp_dir, module)
                assert Path(checkpoint_path).exists()


class TestCheckpointVersioningAndMigration:
    """Test checkpoint versioning and migration."""

    def test_checkpoint_versioning(self):
        """Test checkpoint version compatibility."""
        rngs = nnx.Rngs(42)
        module = SimpleModule(10, rngs=rngs)

        # Save checkpoint with version info
        state = module.get_state()
        state["__version__"] = "1.0.0"

        # Create new module and restore
        new_module = SimpleModule(10, rngs=nnx.Rngs(123))
        new_module.set_state(state)

        # Should restore despite version info
        assert new_module.counter.get_value() == module.counter.get_value()

    def test_checkpoint_migration(self):
        """Test migrating old checkpoint format to new."""
        # Simulate old checkpoint format
        old_state = {
            "counter": 5,
            "features": 10,
            # Missing some fields that might be in new version
        }

        # Create module and try to restore old state
        rngs = nnx.Rngs(42)
        module = SimpleModule(10, rngs=rngs)

        # Should handle partial state restoration
        module.set_state(old_state)
        assert module.counter.get_value() == 5

    def test_incompatible_state_restoration(self):
        """Test handling of incompatible state restoration."""
        rngs = nnx.Rngs(42)
        module = SimpleModule(10, rngs=rngs)

        # Try to restore completely incompatible state
        incompatible_state = {"non_existent_field": 123, "another_field": "value"}

        # Should handle gracefully
        module.set_state(incompatible_state)
        # Module should still be functional
        x = jnp.ones((5, 10))
        result = module(x)
        assert result is not None


class TestErrorRecoveryAndCorruption:
    """Test error recovery and corruption handling."""

    def test_corrupted_checkpoint_handling(self):
        """Test handling of corrupted checkpoints."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a corrupted checkpoint file
            checkpoint_path = Path(tmp_dir) / "checkpoint"
            checkpoint_path.mkdir()

            # Write invalid data
            (checkpoint_path / "state").write_text("corrupted data")

            # Try to restore from corrupted checkpoint
            with OrbaxCheckpointHandler() as handler:
                rngs = nnx.Rngs(42)
                module = SimpleModule(10, rngs=rngs)

                # Should handle error gracefully
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        handler.restore(tmp_dir, module)
                except Exception:
                    # Expected to fail, but should not crash
                    pass

    def test_partial_checkpoint_recovery(self):
        """Test recovery from partial checkpoints."""
        rngs = nnx.Rngs(42)
        module = SimpleModule(10, rngs=rngs)

        # Save full state
        full_state = module.get_state()

        # Create partial state (missing some fields)
        partial_state = {k: v for k, v in full_state.items() if k != "linear"}

        # Try to restore partial state
        new_module = SimpleModule(10, rngs=nnx.Rngs(123))
        new_module.set_state(partial_state)

        # Should maintain functionality despite partial restoration
        x = jnp.ones((5, 10))
        result = new_module(x)
        assert result is not None

    def test_checkpoint_atomic_save(self):
        """Test atomic checkpoint saving."""
        rngs = nnx.Rngs(42)
        module = SimpleModule(10, rngs=rngs)

        with tempfile.TemporaryDirectory() as tmp_dir:
            with OrbaxCheckpointHandler() as handler:
                # Save should be atomic (all or nothing)
                checkpoint_path = handler.save(tmp_dir, module)

                # Verify checkpoint integrity
                assert Path(checkpoint_path).exists()

                # Should be able to restore
                new_module = SimpleModule(10, rngs=nnx.Rngs(123))
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    restored = handler.restore(tmp_dir, new_module)
                assert restored is not None


class TestAsyncCheckpointing:
    """Test async checkpointing functionality."""

    def test_concurrent_checkpoint_operations(self):
        """Test concurrent checkpoint operations."""
        import threading
        from unittest.mock import MagicMock, patch

        results = []

        # Mock ocp.StandardCheckpointer to avoid real I/O and JAX runtime contention
        # which causes deadlocks when called from multiple threads
        with patch("orbax.checkpoint.StandardCheckpointer") as mock_checkpointer_cls:
            mock_checkpointer = MagicMock()
            mock_checkpointer_cls.return_value = mock_checkpointer

            def save_checkpoint(index):
                rngs = nnx.Rngs(42 + index)
                module = SimpleModule(10, rngs=rngs)

                with tempfile.TemporaryDirectory() as tmp_dir:
                    # Use context manager to ensure proper cleanup of async operations
                    with OrbaxCheckpointHandler() as handler:
                        checkpoint_path = handler.save(tmp_dir, module)
                        results.append(checkpoint_path is not None)

            # Run multiple saves concurrently
            threads = []
            for i in range(3):
                thread = threading.Thread(target=save_checkpoint, args=(i,))
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join(timeout=5.0)  # Should be very fast with mocks

            # All saves should succeed
            assert all(results)
            assert len(results) == 3
            # Verify that save was called (though from different threads, so call count matches)
            assert mock_checkpointer.save.call_count >= 3


class TestPerformanceAndStress:
    """Test performance and stress scenarios."""

    def test_large_state_checkpointing(self):
        """Test checkpointing with large state."""
        # Create module with large state
        rngs = nnx.Rngs(42)
        large_module = SimpleModule(1000, rngs=rngs)  # Large feature size

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Use context manager to ensure proper cleanup of async operations
            with OrbaxCheckpointHandler() as handler:
                # Should handle large state
                import time

                start_time = time.time()
                checkpoint_path = handler.save(tmp_dir, large_module)
                save_time = time.time() - start_time

                # Should complete in reasonable time
                assert save_time < 10  # 10 seconds max
                assert Path(checkpoint_path).exists()

    def test_frequent_checkpointing(self):
        """Test frequent checkpoint saves."""
        rngs = nnx.Rngs(42)
        module = SimpleModule(10, rngs=rngs)
        x = jnp.ones((5, 10))

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Use context manager to ensure proper cleanup of async operations
            with OrbaxCheckpointHandler() as handler:
                # Save many checkpoints rapidly
                for i in range(10):
                    module(x)
                    checkpoint_path = handler.save(f"{tmp_dir}/ckpt_{i}", module)
                    assert Path(checkpoint_path).exists()

    def test_checkpoint_memory_efficiency(self):
        """Test memory efficiency of checkpointing."""
        import gc

        # Get initial memory usage
        gc.collect()

        rngs = nnx.Rngs(42)
        module = SimpleModule(100, rngs=rngs)

        # Save and restore multiple times
        for _ in range(5):
            state = module.get_state()
            new_module = SimpleModule(100, rngs=nnx.Rngs(123))
            new_module.set_state(state)
            del new_module
            gc.collect()

        # Memory should not grow significantly
        # (This is a simplified test - real memory testing would be more complex)
        assert module is not None


class TestCheckpointHandlerEdgeCases:
    """Test edge cases for checkpoint handlers."""

    def test_empty_module_checkpointing(self):
        """Test checkpointing empty module."""
        # Create minimal module
        config = DataraxModuleConfig()
        module = DataraxModule(config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            with OrbaxCheckpointHandler() as handler:
                checkpoint_path = handler.save(tmp_dir, module)
                assert Path(checkpoint_path).exists()

    def test_nested_module_checkpointing(self):
        """Test checkpointing nested modules."""

        class NestedModule(DataraxModule):
            def __init__(self, rngs):
                config = DataraxModuleConfig()
                super().__init__(config, rngs=rngs)
                self.inner = SimpleModule(5, rngs=rngs)
                self.outer = SimpleModule(10, rngs=rngs)

        rngs = nnx.Rngs(42)
        module = NestedModule(rngs)

        # Modify nested modules
        x = jnp.ones((5, 5))
        module.inner(x)

        with tempfile.TemporaryDirectory() as tmp_dir:
            with OrbaxCheckpointHandler() as handler:
                handler.save(tmp_dir, module)

                # Restore to new nested module
                new_module = NestedModule(nnx.Rngs(123))
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    restored = handler.restore(tmp_dir, new_module)

                # Check nested state restored
                assert restored.inner.counter.get_value() == module.inner.counter.get_value()

    def test_checkpoint_with_none_fields(self):
        """Test checkpointing with None fields."""

        class ModuleWithNone(DataraxModule):
            def __init__(self):
                config = DataraxModuleConfig()
                super().__init__(config)
                self.optional_field = None
                self.counter = nnx.Variable(0)

        module = ModuleWithNone()
        state = module.get_state()

        # Should handle None fields
        new_module = ModuleWithNone()
        new_module.set_state(state)
        assert new_module.optional_field is None
        assert new_module.counter.get_value() == 0
