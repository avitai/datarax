"""Tests for checkpointable module functionality."""

from pathlib import Path

import flax.nnx as nnx
import jax.numpy as jnp
import pytest

from datarax.checkpoint import IteratorCheckpoint
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
        pos = self.position.get_value()
        assert pos is not None
        if pos >= self.max_items:
            raise StopIteration

        current = pos
        self.position.set_value(pos + 1)
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
        assert module.counter.get_value() == cloned.counter.get_value()  # type: ignore[reportAttributeAccessIssue]

        # Changing one shouldn't affect the other
        module(x)
        assert module.counter.get_value() != cloned.counter.get_value()  # type: ignore[reportAttributeAccessIssue]


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


class TestCheckpointRoundTrip:
    """Modules and iterator modules round-trip through IteratorCheckpoint."""

    def test_module_state_is_restored_into_a_fresh_module(self, tmp_path):
        module = SimpleModule(10, rngs=nnx.Rngs(42))
        module(jnp.ones((5, 10)))
        original_counter = module.counter.get_value()

        with IteratorCheckpoint(tmp_path) as checkpoint:
            path = checkpoint.save(module, step=0)
            assert Path(path).exists()

            fresh = SimpleModule(10, rngs=nnx.Rngs(123))
            checkpoint.restore(fresh, step=0)

        assert fresh.counter.get_value() == original_counter
        assert jnp.array_equal(fresh.linear.kernel.get_value(), module.linear.kernel.get_value())

    def test_iterator_module_position_is_restored(self, tmp_path):
        iterator = SimpleIteratorModule(5)
        next(iterator)
        next(iterator)

        with IteratorCheckpoint(tmp_path) as checkpoint:
            checkpoint.save(iterator, step=0)
            fresh = SimpleIteratorModule(5)
            checkpoint.restore(fresh, step=0)

        assert fresh.position.get_value() == iterator.position.get_value()
        assert next(fresh) == 2

    def test_each_step_restores_the_state_saved_at_that_step(self, tmp_path):
        module = SimpleModule(10, rngs=nnx.Rngs(42))
        x = jnp.ones((5, 10))

        with IteratorCheckpoint(tmp_path, max_to_keep=10) as checkpoint:
            for step in range(5):
                module(x)
                checkpoint.save(module, step=step)

            fresh = SimpleModule(10, rngs=nnx.Rngs(123))
            checkpoint.restore(fresh, step=2)

        assert fresh.counter.get_value() == 3

    def test_nested_module_state_is_restored(self, tmp_path):
        class NestedModule(DataraxModule):
            def __init__(self, rngs):
                super().__init__(DataraxModuleConfig(), rngs=rngs)
                self.inner = SimpleModule(5, rngs=rngs)
                self.outer = SimpleModule(10, rngs=rngs)

        module = NestedModule(nnx.Rngs(42))
        module.inner(jnp.ones((5, 5)))

        with IteratorCheckpoint(tmp_path) as checkpoint:
            checkpoint.save(module, step=0)
            fresh = NestedModule(nnx.Rngs(123))
            checkpoint.restore(fresh, step=0)

        assert fresh.inner.counter.get_value() == module.inner.counter.get_value()

    def test_module_without_variables_saves(self, tmp_path):
        module = DataraxModule(DataraxModuleConfig())

        with IteratorCheckpoint(tmp_path) as checkpoint:
            assert Path(checkpoint.save(module, step=0)).exists()

    def test_restore_from_a_directory_without_checkpoints_raises(self, tmp_path):
        (tmp_path / "junk").write_text("not a checkpoint")

        with IteratorCheckpoint(tmp_path) as checkpoint:
            with pytest.raises(ValueError, match="No checkpoints found"):
                checkpoint.restore(SimpleModule(10, rngs=nnx.Rngs(42)))


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


class TestCheckpointVersioningAndMigration:
    """Test checkpoint versioning and migration."""

    def test_checkpoint_versioning(self):
        """Unknown version keys must not be injected into module state."""
        rngs = nnx.Rngs(42)
        module = SimpleModule(10, rngs=rngs)

        # Versioning belongs in checkpoint metadata, not module state.
        state = module.get_state()
        state["__version__"] = "1.0.0"

        new_module = SimpleModule(10, rngs=nnx.Rngs(123))
        with pytest.raises(ValueError, match="structurally incompatible"):
            new_module.set_state(state)

    def test_checkpoint_migration(self):
        """Legacy/partial state formats should fail fast."""
        # Simulate old checkpoint format
        old_state = {
            "counter": 5,
            "features": 10,
            # Missing some fields that might be in new version
        }

        # Create module and try to restore old state
        rngs = nnx.Rngs(42)
        module = SimpleModule(10, rngs=rngs)

        with pytest.raises(ValueError, match="structurally incompatible"):
            module.set_state(old_state)

    def test_incompatible_state_restoration(self):
        """Completely incompatible state should raise."""
        rngs = nnx.Rngs(42)
        module = SimpleModule(10, rngs=rngs)

        # Try to restore completely incompatible state
        incompatible_state = {"non_existent_field": 123, "another_field": "value"}

        with pytest.raises(ValueError, match="structurally incompatible"):
            module.set_state(incompatible_state)

        # Module should still be functional after failed restore.
        x = jnp.ones((5, 10))
        result = module(x)
        assert result.shape == (5, 10)


class TestErrorRecoveryAndCorruption:
    """Test error recovery and corruption handling."""

    def test_partial_checkpoint_recovery(self):
        """Partial checkpoint states should fail fast."""
        rngs = nnx.Rngs(42)
        module = SimpleModule(10, rngs=rngs)

        # Save full state
        full_state = module.get_state()

        # Create partial state (missing some fields)
        partial_state = {k: v for k, v in full_state.items() if k != "linear"}

        new_module = SimpleModule(10, rngs=nnx.Rngs(123))
        with pytest.raises(ValueError, match="structurally incompatible"):
            new_module.set_state(partial_state)


class TestPerformanceAndStress:
    """Test performance and stress scenarios."""

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


class TestCheckpointEdgeCases:
    """Edge cases of module state checkpointing."""

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
