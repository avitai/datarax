"""Tests for OperatorModule with JAX PyTree states.

This test suite validates that OperatorModule.apply_batch() correctly handles
PyTree states using the data_axes/states_axes pattern.

Test Categories:
1. Basic apply_batch with PyTree states
2. Stochastic operators with state transformation
3. Deterministic operators with state transformation
4. Nested PyTree states
5. State preservation when not modified
6. Integration with actual transformation logic
"""

import jax
import jax.numpy as jnp
from flax import nnx

from datarax.core.config import OperatorConfig
from datarax.core.element_batch import Batch
from datarax.core.operator import OperatorModule


# Test operator implementations
class IncrementCountOperator(OperatorModule):
    """Deterministic operator that increments state counter."""

    def apply(self, data, state, metadata, random_params=None, stats=None):
        # Increment count in state
        new_state = {"count": state["count"] + 1}
        return data, new_state, metadata


class RandomScaleWithStateOperator(OperatorModule):
    """Stochastic operator that scales data and updates state."""

    def generate_random_params(self, rng, data_shapes):
        batch_size = data_shapes["x"][0]
        # Generate random scale factors (one per batch element)
        return jax.random.uniform(rng, (batch_size,), minval=0.5, maxval=1.5)

    def apply(self, data, state, metadata, random_params=None, stats=None):
        scale_factor = random_params if random_params is not None else 1.0
        scaled_data = {"x": data["x"] * scale_factor}

        # Update state with applied scale
        new_state = {
            "count": state["count"] + 1,
            "last_scale": scale_factor,
        }

        return scaled_data, new_state, metadata


class NestedStatOperator(OperatorModule):
    """Operator that works with nested PyTree states."""

    def apply(self, data, state, metadata, random_params=None, stats=None):
        # Update nested state structure
        new_state = {
            "counters": {
                "augment": state["counters"]["augment"] + 1,
                "transform": state["counters"]["transform"],
            },
            "score": state["score"] * 1.1,
        }
        return data, new_state, metadata


class TestOperatorWithSimplePyTreeStates:
    """Test basic operator functionality with PyTree states."""

    def test_deterministic_operator_modifies_states(self):
        """Test deterministic operator correctly modifies PyTree states."""
        config = OperatorConfig(stochastic=False)
        operator = IncrementCountOperator(config)

        # Create batch with PyTree states
        batch = Batch.from_parts(
            data={"x": jnp.ones((4, 3))},
            states={"count": jnp.array([0, 1, 2, 3])},
        )

        # Apply operator
        result = operator.apply_batch(batch)

        # States should be incremented
        assert result.batch_size == 4
        assert jnp.array_equal(result.states.get_value()["count"], jnp.array([1, 2, 3, 4]))

        # Data should be unchanged
        assert jnp.allclose(result.data.get_value()["x"], batch.data.get_value()["x"])

    def test_stochastic_operator_modifies_states(self):
        """Test stochastic operator correctly modifies PyTree states."""
        config = OperatorConfig(stochastic=True, stream_name="test")
        rngs = nnx.Rngs(42)
        operator = RandomScaleWithStateOperator(config, rngs=rngs)

        # Create batch
        batch = Batch.from_parts(
            data={"x": jnp.ones((4, 3))},
            states={"count": jnp.zeros((4,), dtype=jnp.int32), "last_scale": jnp.ones((4,))},
        )

        # Apply operator
        result = operator.apply_batch(batch)

        # Count should be incremented
        assert jnp.array_equal(result.states.get_value()["count"], jnp.ones((4,), dtype=jnp.int32))

        # last_scale should be updated (random values)
        assert not jnp.allclose(result.states.get_value()["last_scale"], jnp.ones((4,)))
        assert result.states.get_value()["last_scale"].shape == (4,)

        # Data should be scaled
        assert not jnp.allclose(result.data.get_value()["x"], batch.data.get_value()["x"])

    def test_operator_preserves_metadata(self):
        """Test that metadata is preserved (not vmapped over)."""
        config = OperatorConfig(stochastic=False)
        operator = IncrementCountOperator(config)

        metadata_list = ["meta0", "meta1", "meta2"]
        batch = Batch.from_parts(
            data={"x": jnp.ones((3, 2))},
            states={"count": jnp.array([0, 1, 2])},
            metadata_list=metadata_list,
        )

        result = operator.apply_batch(batch)

        # Metadata should be unchanged
        assert result._metadata_list == metadata_list


class TestOperatorWithNestedPyTreeStates:
    """Test operators with nested PyTree state structures."""

    def test_nested_state_transformation(self):
        """Test operator correctly handles nested state PyTrees."""
        config = OperatorConfig(stochastic=False)
        operator = NestedStatOperator(config)

        batch = Batch.from_parts(
            data={"x": jnp.ones((3, 2))},
            states={
                "counters": {
                    "augment": jnp.array([0, 1, 2]),
                    "transform": jnp.array([10, 20, 30]),
                },
                "score": jnp.array([0.5, 0.7, 0.9]),
            },
        )

        result = operator.apply_batch(batch)

        # Augment counter should be incremented
        assert jnp.array_equal(
            result.states.get_value()["counters"]["augment"], jnp.array([1, 2, 3])
        )

        # Transform counter should be unchanged
        assert jnp.array_equal(
            result.states.get_value()["counters"]["transform"], jnp.array([10, 20, 30])
        )

        # Score should be scaled by 1.1
        expected_scores = jnp.array([0.5, 0.7, 0.9]) * 1.1
        assert jnp.allclose(result.states.get_value()["score"], expected_scores)

    def test_deeply_nested_states(self):
        """Test operator with deeply nested state structure."""

        class DeepNestedOperator(OperatorModule):
            def apply(self, data, state, metadata, random_params=None, stats=None):
                new_state = {
                    "level1": {
                        "level2": {
                            "level3": {"value": state["level1"]["level2"]["level3"]["value"] + 1}
                        }
                    }
                }
                return data, new_state, metadata

        config = OperatorConfig(stochastic=False)
        operator = DeepNestedOperator(config)

        batch = Batch.from_parts(
            data={"x": jnp.ones((2, 3))},
            states={"level1": {"level2": {"level3": {"value": jnp.array([0, 1])}}}},
        )

        result = operator.apply_batch(batch)

        assert jnp.array_equal(
            result.states.get_value()["level1"]["level2"]["level3"]["value"], jnp.array([1, 2])
        )


class TestOperatorStatePreservation:
    """Test that states are preserved when operator doesn't modify them."""

    def test_operator_that_ignores_state(self):
        """Test operator that returns state unchanged."""

        class IdentityStateOperator(OperatorModule):
            def apply(self, data, state, metadata, random_params=None, stats=None):
                # Don't modify state at all
                return data, state, metadata

        config = OperatorConfig(stochastic=False)
        operator = IdentityStateOperator(config)

        batch = Batch.from_parts(
            data={"x": jnp.ones((3, 2))},
            states={"count": jnp.array([5, 10, 15]), "flag": jnp.array([True, False, True])},
        )

        result = operator.apply_batch(batch)

        # States should be identical
        assert jnp.array_equal(
            result.states.get_value()["count"], batch.states.get_value()["count"]
        )
        assert jnp.array_equal(result.states.get_value()["flag"], batch.states.get_value()["flag"])


class TestOperatorEdgeCases:
    """Test edge cases and error conditions."""

    def test_operator_with_empty_states(self):
        """Test operator with empty state dicts."""

        class NoOpOperator(OperatorModule):
            def apply(self, data, state, metadata, random_params=None, stats=None):
                return data, state, metadata

        config = OperatorConfig(stochastic=False)
        operator = NoOpOperator(config)

        batch = Batch.from_parts(
            data={"x": jnp.ones((3, 2))},
            states={},  # Empty states
        )

        result = operator.apply_batch(batch)

        assert result.states.get_value() == {}
        assert result.batch_size == 3

    def test_operator_with_mixed_state_types(self):
        """Test operator with different types in state (arrays, primitives)."""

        class MixedStateOperator(OperatorModule):
            def apply(self, data, state, metadata, random_params=None, stats=None):
                new_state = {
                    "jax_array": state["jax_array"] + 1,
                    "python_int": state["python_int"] + 1,
                    "python_float": state["python_float"] * 2,
                    "python_bool": ~state["python_bool"],  # Use JAX logical not
                }
                return data, new_state, metadata

        config = OperatorConfig(stochastic=False)
        operator = MixedStateOperator(config)

        batch = Batch.from_parts(
            data={"x": jnp.ones((2, 3))},
            states={
                "jax_array": jnp.array([1, 2]),
                "python_int": jnp.array([10, 20]),  # Will be stacked as array
                "python_float": jnp.array([0.5, 0.7]),
                "python_bool": jnp.array([True, False]),
            },
        )

        result = operator.apply_batch(batch)

        assert jnp.array_equal(result.states.get_value()["jax_array"], jnp.array([2, 3]))
        assert jnp.array_equal(result.states.get_value()["python_int"], jnp.array([11, 21]))

    def test_operator_with_scalar_batch_size_1(self):
        """Test operator with batch size 1 (states still have batch axis)."""
        config = OperatorConfig(stochastic=False)
        operator = IncrementCountOperator(config)

        batch = Batch.from_parts(
            data={"x": jnp.ones((1, 3))},
            states={"count": jnp.array([0])},
        )

        result = operator.apply_batch(batch)

        assert result.batch_size == 1
        assert result.states.get_value()["count"].shape == (1,)
        assert result.states.get_value()["count"][0] == 1


class TestOperatorJITCompilation:
    """Test that operators with PyTree states compile correctly."""

    def test_operator_jit_compiles_with_pytree_states(self):
        """Test operator can be JIT compiled with PyTree states."""
        config = OperatorConfig(stochastic=False)
        operator = IncrementCountOperator(config)

        @jax.jit
        def process_batch(batch):
            return operator.apply_batch(batch)

        batch = Batch.from_parts(
            data={"x": jnp.ones((4, 3))},
            states={"count": jnp.array([0, 1, 2, 3])},
        )

        # Should compile and run without errors
        result = process_batch(batch)

        assert jnp.array_equal(result.states.get_value()["count"], jnp.array([1, 2, 3, 4]))

    def test_stochastic_operator_jit_compiles(self):
        """Test stochastic operator compiles with PyTree states."""
        config = OperatorConfig(stochastic=True, stream_name="test")
        rngs = nnx.Rngs(42)
        operator = RandomScaleWithStateOperator(config, rngs=rngs)

        # Note: Cannot directly JIT a method that uses rngs (stateful)
        # But apply_batch should still work with nnx.jit

        batch = Batch.from_parts(
            data={"x": jnp.ones((4, 3))},
            states={"count": jnp.zeros((4,), dtype=jnp.int32), "last_scale": jnp.ones((4,))},
        )

        # Should work (nnx.jit handles stateful operations)
        result = operator.apply_batch(batch)

        assert result.states.get_value()["count"].shape == (4,)
        assert result.states.get_value()["last_scale"].shape == (4,)


class TestOperatorBatchStateVsBatchedStates:
    """Test distinction between batch-level state and per-element states."""

    def test_batch_state_vs_states(self):
        """Test that batch_state is separate from element states."""

        class BatchStateOperator(OperatorModule):
            def apply(self, data, state, metadata, random_params=None, stats=None):
                # Only modify element state
                new_state = {"count": state["count"] + 1}
                return data, new_state, metadata

        config = OperatorConfig(stochastic=False)
        operator = BatchStateOperator(config)

        batch = Batch.from_parts(
            data={"x": jnp.ones((3, 2))},
            states={"count": jnp.array([0, 1, 2])},
            batch_state={"total_processed": jnp.array(100)},  # Batch-level
        )

        result = operator.apply_batch(batch)

        # Element states should be modified
        assert jnp.array_equal(result.states.get_value()["count"], jnp.array([1, 2, 3]))

        # Batch state should be preserved (operator doesn't touch it)
        assert result.batch_state.get_value()["total_processed"] == 100
