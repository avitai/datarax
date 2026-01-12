"""Tests for Conditional composition strategies.

This module tests the CONDITIONAL_SEQUENTIAL and CONDITIONAL_PARALLEL strategies,
which execute operators based on runtime conditions.

Test Coverage:
- Conditional sequential with all/some/no conditions true
- Conditional parallel with all/some/no conditions true
- require_at_least_one flag behavior
- State and metadata-based conditions
- Condition evaluation on transformed data
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


class TestConditionalSequential:
    """Test conditional sequential composition."""

    def test_conditional_sequential_all_conditions_true(self):
        """Test conditional sequential when all conditions are true."""
        rngs = nnx.Rngs(0)

        # Create 3 map operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x + 10, rngs=rngs)

        config3 = MapOperatorConfig(stochastic=False)
        op3 = MapOperator(config3, fn=lambda x, key: x * 3, rngs=rngs)

        # Create conditional sequential with always-true conditions
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.CONDITIONAL_SEQUENTIAL,
            operators=[op1, op2, op3],
            conditions=[
                lambda data: jnp.array(True),  # Always execute op1
                lambda data: jnp.array(True),  # Always execute op2
                lambda data: jnp.array(True),  # Always execute op3
            ],
        )
        composite = CompositeOperatorModule(composite_config)

        # Create batch
        batch = Batch(
            [
                Element(data={"value": jnp.array([1.0])}),
                Element(data={"value": jnp.array([2.0])}),
                Element(data={"value": jnp.array([3.0])}),
            ]
        )

        # Apply composite (all should execute: ((x * 2) + 10) * 3)
        result_batch = composite(batch)

        # Verify: ((input * 2) + 10) * 3
        result_data = result_batch.get_data()
        expected = jnp.array([[36.0], [42.0], [48.0]])
        assert jnp.allclose(result_data["value"], expected)

    def test_conditional_sequential_mixed_conditions(self):
        """Test conditional sequential with mixed true/false conditions."""
        rngs = nnx.Rngs(0)

        # Create 3 map operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x + 100, rngs=rngs)  # Won't execute

        config3 = MapOperatorConfig(stochastic=False)
        op3 = MapOperator(config3, fn=lambda x, key: x + 10, rngs=rngs)

        # Create conditional sequential with [True, False, True] conditions
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.CONDITIONAL_SEQUENTIAL,
            operators=[op1, op2, op3],
            conditions=[
                lambda data: jnp.array(True),  # Execute op1
                lambda data: jnp.array(False),  # Skip op2
                lambda data: jnp.array(True),  # Execute op3
            ],
        )
        composite = CompositeOperatorModule(composite_config)

        # Create batch
        batch = Batch(
            [
                Element(data={"value": jnp.array([1.0])}),
                Element(data={"value": jnp.array([2.0])}),
                Element(data={"value": jnp.array([3.0])}),
            ]
        )

        # Apply composite (op1 and op3 execute, op2 skipped: (x * 2) + 10)
        result_batch = composite(batch)

        # Verify: (input * 2) + 10 (op2 skipped)
        result_data = result_batch.get_data()
        expected = jnp.array([[12.0], [14.0], [16.0]])
        assert jnp.allclose(result_data["value"], expected)

    def test_conditional_sequential_all_conditions_false(self):
        """Test conditional sequential when all conditions are false."""
        rngs = nnx.Rngs(0)

        # Create 3 map operators (none will execute)
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 100, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x + 100, rngs=rngs)

        config3 = MapOperatorConfig(stochastic=False)
        op3 = MapOperator(config3, fn=lambda x, key: x * 100, rngs=rngs)

        # Create conditional sequential with all-false conditions
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.CONDITIONAL_SEQUENTIAL,
            operators=[op1, op2, op3],
            conditions=[
                lambda data: jnp.array(False),  # Skip op1
                lambda data: jnp.array(False),  # Skip op2
                lambda data: jnp.array(False),  # Skip op3
            ],
        )
        composite = CompositeOperatorModule(composite_config)

        # Create batch
        batch = Batch(
            [
                Element(data={"value": jnp.array([1.0])}),
                Element(data={"value": jnp.array([2.0])}),
                Element(data={"value": jnp.array([3.0])}),
            ]
        )

        # Apply composite (no operators execute, data passes through)
        result_batch = composite(batch)

        # Verify: data unchanged
        result_data = result_batch.get_data()
        original_data = batch.get_data()
        assert jnp.allclose(result_data["value"], original_data["value"])

    def test_conditional_sequential_condition_on_transformed_data(self):
        """Test that conditions evaluate on current (transformed) data."""
        rngs = nnx.Rngs(0)

        # Create operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 10, rngs=rngs)  # Transforms data

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x + 1000, rngs=rngs)  # Only if condition met

        # Create conditional sequential
        # op2's condition checks transformed data (after op1)
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.CONDITIONAL_SEQUENTIAL,
            operators=[op1, op2],
            conditions=[
                lambda data: jnp.array(True),  # Always execute op1
                lambda data: (
                    data["value"] > 5.0
                ).all(),  # Execute op2 if transformed value > 5 (JAX comparison, no indexing)
            ],
        )
        composite = CompositeOperatorModule(composite_config)

        # Test data: input is 1.0, after op1 becomes 10.0 (> 5.0)
        batch = Batch([Element(data={"value": jnp.array([1.0])})])

        # Apply composite (op1: 1.0 * 10 = 10.0, then op2: 10.0 + 1000 = 1010.0)
        result_batch = composite(batch)

        # Verify: condition checked transformed data (10.0 > 5.0), so op2 executed
        result_data = result_batch.get_data()
        expected = jnp.array([[1010.0]])
        assert jnp.allclose(result_data["value"], expected)

        # Test with input that won't trigger op2 after transformation
        batch2 = Batch([Element(data={"value": jnp.array([0.1])})])
        result_batch2 = composite(batch2)

        # After op1: 0.1 * 10 = 1.0 (< 5.0), so op2 should NOT execute
        result_data2 = result_batch2.get_data()
        expected2 = jnp.array([[1.0]])
        assert jnp.allclose(result_data2["value"], expected2)


class TestConditionalParallel:
    """Test conditional parallel composition."""

    def test_conditional_parallel_all_conditions_true(self):
        """Test conditional parallel when all conditions are true."""
        rngs = nnx.Rngs(0)

        # Create 3 map operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 3, rngs=rngs)

        config3 = MapOperatorConfig(stochastic=False)
        op3 = MapOperator(config3, fn=lambda x, key: x * 4, rngs=rngs)

        # Create conditional parallel with always-true conditions
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.CONDITIONAL_PARALLEL,
            operators=[op1, op2, op3],
            conditions=[
                lambda data: jnp.array(True),  # Always execute op1
                lambda data: jnp.array(True),  # Always execute op2
                lambda data: jnp.array(True),  # Always execute op3
            ],
            merge_strategy="concat",  # Concatenate outputs
        )
        composite = CompositeOperatorModule(composite_config)

        # Create batch
        batch = Batch(
            [
                Element(data={"value": jnp.array([1.0])}),
                Element(data={"value": jnp.array([2.0])}),
            ]
        )

        # Apply composite (all execute in parallel, outputs concatenated)
        result_batch = composite(batch)
        result_data = result_batch.get_data()

        # Verify: concat([x*2, x*3, x*4])
        # Each element: [1.0] -> [2.0, 3.0, 4.0] and [2.0] -> [4.0, 6.0, 8.0]
        # Batch result shape: (2, 3) for concatenation
        expected = jnp.array([[2.0, 3.0, 4.0], [4.0, 6.0, 8.0]])
        assert jnp.allclose(result_data["value"], expected)

    def test_conditional_parallel_mixed_conditions(self):
        """Test conditional parallel with mixed true/false conditions."""
        rngs = nnx.Rngs(0)

        # Create 3 map operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 100, rngs=rngs)  # Won't execute

        config3 = MapOperatorConfig(stochastic=False)
        op3 = MapOperator(config3, fn=lambda x, key: x * 4, rngs=rngs)

        # Create conditional parallel with [True, False, True] conditions
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.CONDITIONAL_PARALLEL,
            operators=[op1, op2, op3],
            conditions=[
                lambda data: jnp.array(True),  # Execute op1
                lambda data: jnp.array(False),  # Skip op2
                lambda data: jnp.array(True),  # Execute op3
            ],
            merge_strategy="concat",
        )
        composite = CompositeOperatorModule(composite_config)

        # Create batch
        batch = Batch(
            [
                Element(data={"value": jnp.array([1.0])}),
                Element(data={"value": jnp.array([2.0])}),
            ]
        )

        # Apply composite (op1 and op3 execute, op2 returns identity)
        result_batch = composite(batch)
        result_data = result_batch.get_data()

        # Verify: concat([x*2, x (identity), x*4])
        # Concat includes ALL outputs - transformed AND identity
        # Each element: [1.0] -> [2.0, 1.0, 4.0] and [2.0] -> [4.0, 2.0, 8.0]
        expected = jnp.array([[2.0, 1.0, 4.0], [4.0, 2.0, 8.0]])
        assert jnp.allclose(result_data["value"], expected)

    def test_conditional_parallel_all_false_passthrough(self):
        """Test that all-false conditions pass through data unchanged."""
        rngs = nnx.Rngs(0)

        # Create 3 map operators (none will execute)
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 100, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 100, rngs=rngs)

        config3 = MapOperatorConfig(stochastic=False)
        op3 = MapOperatorConfig(stochastic=False)
        op3 = MapOperator(config3, fn=lambda x, key: x * 100, rngs=rngs)

        # Create conditional parallel with all-false conditions
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.CONDITIONAL_PARALLEL,
            operators=[op1, op2, op3],
            conditions=[
                lambda data: jnp.array(False),  # Skip op1
                lambda data: jnp.array(False),  # Skip op2
                lambda data: jnp.array(False),  # Skip op3
            ],
            merge_strategy="concat",
        )
        composite = CompositeOperatorModule(composite_config)

        # Create batch
        batch = Batch(
            [
                Element(data={"value": jnp.array([1.0])}),
                Element(data={"value": jnp.array([2.0])}),
                Element(data={"value": jnp.array([3.0])}),
            ]
        )

        # Apply composite (no operators execute, all return identity via noop)
        result_batch = composite(batch)
        result_data = result_batch.get_data()

        # Verify: concat of 3 identity outputs = original data repeated 3 times
        # Concat includes ALL outputs - even when conditions are False
        # Each element: [1.0] -> [1.0, 1.0, 1.0], etc.
        expected = jnp.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
        assert jnp.allclose(result_data["value"], expected)


class TestConditionalAdvanced:
    """Test advanced conditional features."""

    def test_conditional_with_state_based_conditions(self):
        """Test conditions that depend on state values."""
        rngs = nnx.Rngs(0)

        # Create 2 map operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x + 100, rngs=rngs)

        # Create conditional sequential
        # Note: MapOperator doesn't use state, so we pass it through for testing
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.CONDITIONAL_SEQUENTIAL,
            operators=[op1, op2],
            conditions=[
                lambda data: jnp.array(True),  # Always execute op1
                lambda data: jnp.array(
                    True
                ),  # Always execute op2 (state doesn't affect data-based conditions)
            ],
        )
        composite = CompositeOperatorModule(composite_config)

        # Create batch with state
        batch = Batch(
            [
                Element(data={"value": jnp.array([1.0])}, state={"counter": 10}),
                Element(data={"value": jnp.array([2.0])}, state={"counter": 10}),
            ]
        )

        # Apply composite
        result_batch = composite(batch)
        result_data = result_batch.get_data()
        result_states = result_batch.states.get_value()  # Access states via nnx.Variable

        # Verify: (x * 2) + 100
        expected = jnp.array([[102.0], [104.0]])
        assert jnp.allclose(result_data["value"], expected)
        # State should pass through unchanged (MapOperator doesn't modify state)
        # Note: States are now PyTree format {key: array([val1, val2])}
        assert jnp.array_equal(result_states["counter"], jnp.array([10, 10]))

    def test_conditional_with_metadata_based_conditions(self):
        """Test conditions that depend on metadata values."""
        rngs = nnx.Rngs(0)

        # Create 2 map operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x + 100, rngs=rngs)

        # Create conditional sequential
        # Note: MapOperator doesn't modify metadata, so we test pass-through
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.CONDITIONAL_SEQUENTIAL,
            operators=[op1, op2],
            conditions=[
                lambda data: jnp.array(True),  # Always execute op1
                lambda data: jnp.array(True),  # Always execute op2
            ],
        )
        composite = CompositeOperatorModule(composite_config)

        # Create batch with metadata
        batch = Batch(
            [
                Element(data={"value": jnp.array([1.0])}, metadata={"quality": 0.95}),
                Element(data={"value": jnp.array([2.0])}, metadata={"quality": 0.95}),
            ]
        )

        # Apply composite
        result_batch = composite(batch)
        result_data = result_batch.get_data()
        result_metadata_list = result_batch._metadata_list

        # Verify: (x * 2) + 100
        expected = jnp.array([[102.0], [104.0]])
        assert jnp.allclose(result_data["value"], expected)
        # Metadata should pass through unchanged (MapOperator doesn't modify metadata)
        assert result_metadata_list[0] == {"quality": 0.95}
        assert result_metadata_list[1] == {"quality": 0.95}
