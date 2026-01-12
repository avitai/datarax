"""Tests for Dynamic Sequential composition strategy.

This module tests the DYNAMIC_SEQUENTIAL strategy, which allows runtime modification
of the operator list.

Test Coverage:
- Adding operators to the sequence
- Removing operators from the sequence
- Clearing all operators
- Reordering operators
- Getting operator count
- State preservation during modification
"""

import jax.numpy as jnp
from flax import nnx

# GREEN phase - imports enabled
from datarax.operators.composite_operator import (
    CompositeOperatorModule,
    CompositeOperatorConfig,
    CompositionStrategy,
)
from datarax.operators.map_operator import MapOperator, MapOperatorConfig
from datarax.core.element_batch import Batch, Element


class TestDynamicSequential:
    """Test dynamic modification of sequential operator list."""

    def test_add_operator_to_sequence(self):
        """Test adding a new operator to the sequence."""
        rngs = nnx.Rngs(0)

        # Start with 2 operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x + 10, rngs=rngs)

        # Create dynamic sequential
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.DYNAMIC_SEQUENTIAL,
            operators=[op1, op2],
        )
        composite = CompositeOperatorModule(composite_config)

        # Test before adding: (x * 2) + 10
        batch = Batch([Element(data={"value": jnp.array([5.0])})])
        result_batch = composite(batch)
        result_data = result_batch.get_data()
        assert jnp.allclose(result_data["value"], jnp.array([[20.0]]))  # (5 * 2) + 10
        assert len(composite.operators) == 2

        # Add a third operator
        config3 = MapOperatorConfig(stochastic=False)
        op3 = MapOperator(config3, fn=lambda x, key: x * 3, rngs=rngs)
        composite.add_operator(op3)

        # Test after adding: ((x * 2) + 10) * 3
        result_batch2 = composite(batch)
        result_data2 = result_batch2.get_data()
        assert jnp.allclose(result_data2["value"], jnp.array([[60.0]]))  # ((5 * 2) + 10) * 3
        assert len(composite.operators) == 3

    def test_remove_operator_from_sequence(self):
        """Test removing an operator from the sequence."""
        rngs = nnx.Rngs(0)

        # Start with 3 operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x + 100, rngs=rngs)  # Will remove this

        config3 = MapOperatorConfig(stochastic=False)
        op3 = MapOperator(config3, fn=lambda x, key: x * 3, rngs=rngs)

        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.DYNAMIC_SEQUENTIAL,
            operators=[op1, op2, op3],
        )
        composite = CompositeOperatorModule(composite_config)

        # Test before removing: ((x * 2) + 100) * 3
        batch = Batch([Element(data={"value": jnp.array([1.0])})])
        result_batch = composite(batch)
        result_data = result_batch.get_data()
        assert jnp.allclose(result_data["value"], jnp.array([[306.0]]))  # ((1 * 2) + 100) * 3
        assert len(composite.operators) == 3

        # Remove middle operator (index 1)
        removed_op = composite.remove_operator(1)
        assert removed_op is op2

        # Test after removing: (x * 2) * 3
        result_batch2 = composite(batch)
        result_data2 = result_batch2.get_data()
        assert jnp.allclose(result_data2["value"], jnp.array([[6.0]]))  # (1 * 2) * 3
        assert len(composite.operators) == 2

    def test_clear_all_operators(self):
        """Test clearing all operators from the sequence."""
        rngs = nnx.Rngs(0)

        # Start with 3 operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x + 10, rngs=rngs)

        config3 = MapOperatorConfig(stochastic=False)
        op3 = MapOperator(config3, fn=lambda x, key: x * 3, rngs=rngs)

        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.DYNAMIC_SEQUENTIAL,
            operators=[op1, op2, op3],
        )
        composite = CompositeOperatorModule(composite_config)

        assert len(composite.operators) == 3

        # Clear all operators
        composite.clear_operators()

        assert len(composite.operators) == 0

        # After clearing, apply should pass data through unchanged (no ops to run)
        batch = Batch([Element(data={"value": jnp.array([5.0])})])
        result_batch = composite(batch)
        result_data = result_batch.get_data()
        assert jnp.allclose(result_data["value"], jnp.array([[5.0]]))  # Unchanged

    def test_reorder_operators(self):
        """Test reordering operators in the sequence."""
        rngs = nnx.Rngs(0)

        # Create 3 operators with different effects: [A, B, C]
        config_a = MapOperatorConfig(stochastic=False)
        op_a = MapOperator(config_a, fn=lambda x, key: x + 1, rngs=rngs)

        config_b = MapOperatorConfig(stochastic=False)
        op_b = MapOperator(config_b, fn=lambda x, key: x * 10, rngs=rngs)

        config_c = MapOperatorConfig(stochastic=False)
        op_c = MapOperator(config_c, fn=lambda x, key: x + 100, rngs=rngs)

        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.DYNAMIC_SEQUENTIAL,
            operators=[op_a, op_b, op_c],
        )
        composite = CompositeOperatorModule(composite_config)

        # Test original order [A, B, C]: ((x + 1) * 10) + 100
        batch = Batch([Element(data={"value": jnp.array([2.0])})])
        result_batch = composite(batch)
        result_data = result_batch.get_data()
        assert jnp.allclose(
            result_data["value"], jnp.array([[130.0]])
        )  # ((2 + 1) * 10) + 100 = 130

        # Reorder to [C, A, B]: ((x + 100) + 1) * 10
        composite.reorder_operators([2, 0, 1])

        # Test new order
        result_batch2 = composite(batch)
        result_data2 = result_batch2.get_data()
        assert jnp.allclose(
            result_data2["value"], jnp.array([[1030.0]])
        )  # ((2 + 100) + 1) * 10 = 1030

    def test_get_operator_count(self):
        """Test getting the number of operators in the sequence."""
        rngs = nnx.Rngs(0)

        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x + 10, rngs=rngs)

        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.DYNAMIC_SEQUENTIAL,
            operators=[op1, op2],
        )
        composite = CompositeOperatorModule(composite_config)

        # Check initial count
        assert len(composite.operators) == 2

        # Add operator
        config3 = MapOperatorConfig(stochastic=False)
        op3 = MapOperator(config3, fn=lambda x, key: x * 3, rngs=rngs)
        composite.add_operator(op3)
        assert len(composite.operators) == 3

        # Remove operator
        composite.remove_operator(0)
        assert len(composite.operators) == 2

    def test_dynamic_modification_preserves_state(self):
        """Test that module state is preserved during dynamic modification."""
        rngs = nnx.Rngs(0)

        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x + 10, rngs=rngs)

        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.DYNAMIC_SEQUENTIAL,
            operators=[op1, op2],
        )
        composite = CompositeOperatorModule(composite_config)

        # Apply once to initialize statistics
        batch = Batch([Element(data={"value": jnp.array([5.0])})])
        composite(batch)

        # Verify composite has statistics attribute
        assert hasattr(composite, "operator_statistics")

        # Modify operators
        config3 = MapOperatorConfig(stochastic=False)
        op3 = MapOperator(config3, fn=lambda x, key: x * 3, rngs=rngs)
        composite.add_operator(op3)

        # Verify statistics still exist after modification
        assert hasattr(composite, "operator_statistics")

        # Apply again to verify it still works
        result_batch2 = composite(batch)
        result_data2 = result_batch2.get_data()
        assert jnp.allclose(result_data2["value"], jnp.array([[60.0]]))  # ((5 * 2) + 10) * 3
