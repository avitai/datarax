"""Tests for checkpointable module functionality."""

from dataclasses import dataclass
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from datarax.checkpoint import IteratorCheckpoint
from datarax.core.config import DataraxModuleConfig, ElementOperatorConfig, StructuralConfig
from datarax.core.data_source import DataSourceModule
from datarax.core.module import DataraxModule
from datarax.core.operator import DIRECT_CALL_STREAM
from datarax.operators import ElementOperator
from datarax.typing import CheckpointableIterator


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


@dataclass(frozen=True)
class _RecordReaderConfig(StructuralConfig):
    """Configuration of a reader over in-memory records."""


class _RecordReader(DataSourceModule):
    """A checkpointable host iterator: records are construction data, the position is state."""

    def __init__(self, records: list[str]) -> None:
        super().__init__(_RecordReaderConfig())
        self.records = nnx.data(records)
        self.position = nnx.Variable(jnp.int32(0))

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> "_RecordReader":
        return self

    def __next__(self) -> str:
        position = int(self.position[...])
        if position >= len(self.records):
            raise StopIteration
        self.position[...] = jnp.int32(position + 1)
        return self.records[position]


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


class TestSourceIteratorCheckpointing:
    """A DataSourceModule that iterates on the host is the checkpointable iterator pattern."""

    def test_it_is_a_checkpointable_iterator(self):
        assert isinstance(_RecordReader(["a"]), CheckpointableIterator)

    def test_its_state_holds_the_position_not_the_records(self):
        reader = _RecordReader([f"line {i}" for i in range(5)])
        next(reader)

        state = reader.get_state()

        assert "records" not in state
        assert int(state["position"]) == 1


class TestCheckpointRoundTrip:
    """Modules and source iterators round-trip through IteratorCheckpoint."""

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

    def test_a_source_iterator_resumes_where_it_was_saved(self, tmp_path):
        records = [f"line {i}" for i in range(5)]
        reader = _RecordReader(records)
        next(reader)
        next(reader)

        with IteratorCheckpoint(tmp_path) as checkpoint:
            checkpoint.save(reader, step=0)
            fresh = _RecordReader(records)
            checkpoint.restore(fresh, step=0)

        assert next(fresh) == "line 2"

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

    def test_module_without_variables_is_refused(self, tmp_path):
        """A module holding no variables has nothing to write, and Orbax rejects an empty tree.

        Orbax raises a bare ``Found empty item.`` from inside its own save path, so the state is
        checked first and the refusal names the module that produced it.
        """
        module = DataraxModule(DataraxModuleConfig())

        with IteratorCheckpoint(tmp_path) as checkpoint:
            with pytest.raises(ValueError, match="returned nothing to checkpoint"):
                checkpoint.save(module, step=0)

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


class TestOperatorStateUpgradeOnLoad:
    """A module checkpoint written under the earlier operator layout still restores.

    An operator used to keep the caller's ``Rngs`` and to hold its statistics under
    ``_computed_stats``. It now derives a private ``_rng_stream`` from ``_base_key`` and stores
    statistics under ``_statistics``. The saved dict identifies its own layout — ``rngs`` and
    ``_computed_stats`` appear only in the earlier one — so the upgrade needs no version field,
    which is what keeps a variable-free module's state empty and refused.
    """

    @staticmethod
    def _operator() -> ElementOperator:
        """A stochastic operator, which is the only kind that carries RNG state."""
        return ElementOperator(
            ElementOperatorConfig(stochastic=True, stream_name="augment"),
            fn=lambda element, key=None: element,
            rngs=nnx.Rngs(augment=1),
        )

    def test_the_current_layout_names_the_stream_and_the_statistics(self):
        assert set(self._operator().get_state()) == {"_base_key", "_rng_stream", "_statistics"}

    def test_a_state_saved_under_the_earlier_layout_restores(self):
        current = self._operator().get_state()
        base_key = current["_base_key"]
        legacy = {
            "_base_key": base_key,
            "_computed_stats": None,
            "rngs": {"augment": {"count": jnp.zeros((), jnp.uint32), "key": base_key}},
        }

        restored = self._operator()
        restored.set_state(legacy)

        assert restored.rngs is None
        assert int(restored._rng_stream.count[...]) == 0
        expected = jax.random.fold_in(base_key, DIRECT_CALL_STREAM)
        np.testing.assert_array_equal(
            jax.random.key_data(restored._rng_stream.key[...]), jax.random.key_data(expected)
        )

    def test_an_upgraded_state_leaves_the_operator_drawing_as_a_fresh_one_does(self):
        reference = self._operator()
        current = reference.get_state()
        base_key = current["_base_key"]
        legacy = {
            "_base_key": base_key,
            "_computed_stats": None,
            "rngs": {"augment": {"count": jnp.zeros((), jnp.uint32), "key": base_key}},
        }

        restored = self._operator()
        restored.set_state(legacy)

        np.testing.assert_array_equal(
            jax.random.key_data(restored._rng_stream()),
            jax.random.key_data(reference._rng_stream()),
        )

    def test_a_deterministic_operator_carries_no_rng_state_to_upgrade(self):
        operator = ElementOperator(
            ElementOperatorConfig(stochastic=False), fn=lambda element, key=None: element
        )
        state = operator.get_state()
        assert "_rng_stream" not in state
        assert "_base_key" not in state
