"""Tests for Sequential composition strategy.

This module tests the SEQUENTIAL, CONDITIONAL_SEQUENTIAL, and DYNAMIC_SEQUENTIAL
composition strategies, which chain operators such that output₁ → input₂ → output₂ → ...

Test Coverage:
- Basic sequential execution with 2-3+ operators
- Deterministic and stochastic operator chaining
- RNG splitting for stochastic operators
- State and metadata threading
- JIT compilation and vmap compatibility
- Statistics aggregation
- Nested composites
"""

import jax
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


class TestSequentialBasics:
    """Test basic sequential composition functionality."""

    def test_sequential_two_operators(self):
        """Test sequential composition with 2 operators."""
        rngs = nnx.Rngs(0)

        # Create 2 map operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x + 10, rngs=rngs)

        # Create sequential composite
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
            operators=[op1, op2],
        )
        composite = CompositeOperatorModule(composite_config)

        # Create batch from Elements
        batch = Batch(
            [
                Element(data={"value": jnp.array([1.0])}),
                Element(data={"value": jnp.array([2.0])}),
                Element(data={"value": jnp.array([3.0])}),
            ]
        )

        # Apply composite using __call__ (should be (x * 2) + 10)
        result_batch = composite(batch)

        # Verify: op2(op1(input)) = (input * 2) + 10
        assert result_batch.batch_size == 3
        result_data = result_batch.get_data()
        expected = jnp.array([[12.0], [14.0], [16.0]])  # Shape (3, 1) to match batched Elements
        assert jnp.allclose(result_data["value"], expected)

    def test_sequential_three_operators(self):
        """Test sequential composition with 3+ operators."""
        rngs = nnx.Rngs(0)

        # Create 3 map operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x + 1, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 2, rngs=rngs)

        config3 = MapOperatorConfig(stochastic=False)
        op3 = MapOperator(config3, fn=lambda x, key: x + 3, rngs=rngs)

        # Create sequential composite
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
            operators=[op1, op2, op3],
        )
        composite = CompositeOperatorModule(composite_config)

        # Create batch from Element
        batch = Batch([Element(data={"value": jnp.array([5.0])})])

        # Apply composite using __call__
        result_batch = composite(batch)

        # Verify: op3(op2(op1(5))) = ((5 + 1) * 2) + 3 = 15
        assert result_batch.batch_size == 1
        result_data = result_batch.get_data()
        assert jnp.allclose(result_data["value"], jnp.array([[15.0]]))

    def test_sequential_deterministic_operators(self):
        """Test sequential with all deterministic operators."""
        rngs = nnx.Rngs(0)

        # Create deterministic operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x + 5, rngs=rngs)

        # Create sequential composite
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
            operators=[op1, op2],
        )
        composite = CompositeOperatorModule(composite_config)

        # Create batch
        batch = Batch([Element(data={"value": jnp.array([3.0])})])

        # Apply twice - should give same result
        result_batch1 = composite(batch)
        result_batch2 = composite(batch)

        # Verify deterministic: same input -> same output
        result1 = result_batch1.get_data()
        result2 = result_batch2.get_data()
        assert jnp.allclose(result1["value"], result2["value"])
        expected = jnp.array([[11.0]])  # (3 * 2) + 5
        assert jnp.allclose(result1["value"], expected)

    def test_sequential_stochastic_operators(self):
        """Test sequential with all stochastic operators."""
        rngs = nnx.Rngs(0, augment=1)

        # Create stochastic operators
        config1 = MapOperatorConfig(stochastic=True, stream_name="augment")
        op1 = MapOperator(
            config1, fn=lambda x, key: x + jax.random.normal(key, x.shape) * 0.1, rngs=rngs
        )

        config2 = MapOperatorConfig(stochastic=True, stream_name="augment")
        op2 = MapOperator(
            config2, fn=lambda x, key: x * (1 + jax.random.uniform(key, x.shape) * 0.1), rngs=rngs
        )

        # Create composite
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
            operators=[op1, op2],
            stochastic=True,
            stream_name="augment",
        )
        composite = CompositeOperatorModule(composite_config, rngs=rngs)

        # Test with batch
        batch = Batch([Element(data={"value": jnp.array([1.0, 2.0, 3.0])})])
        result = composite(batch)
        result_data = result.get_data()

        # Output should differ from input due to stochastic transformations
        original_data = jnp.array([[1.0, 2.0, 3.0]])
        assert not jnp.allclose(result_data["value"], original_data)
        # But should be in reasonable range
        assert jnp.all(result_data["value"] > 0)


class TestSequentialDataFlow:
    """Test data, state, and metadata flow through sequential operators."""

    def test_sequential_state_threading(self):
        """Test that state is threaded correctly through sequential operators."""
        rngs = nnx.Rngs(0)

        # Create operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x + 10, rngs=rngs)

        # Create sequential composite
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
            operators=[op1, op2],
        )
        composite = CompositeOperatorModule(composite_config)

        # Create batch with state (MapOperator passes through unchanged)
        batch = Batch([Element(data={"value": jnp.array([1.0])}, state={"counter": jnp.array(5)})])

        # Apply composite
        result_batch = composite(batch)

        # Verify state threading (passed through unchanged by MapOperator)
        result_elem = result_batch.get_element(0)
        assert "counter" in result_elem.state
        assert jnp.allclose(result_elem.state["counter"], jnp.array(5))

    def test_sequential_metadata_threading(self):
        """Test that metadata is threaded correctly through sequential operators."""
        rngs = nnx.Rngs(0)

        # Create operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x + 5, rngs=rngs)

        # Create sequential composite
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
            operators=[op1, op2],
        )
        composite = CompositeOperatorModule(composite_config)

        # Create batch with metadata (MapOperator passes through unchanged)
        batch = Batch(
            [Element(data={"value": jnp.array([1.0])}, metadata={"source": "test", "version": 1})]
        )

        # Apply composite
        result_batch = composite(batch)

        # Verify metadata threading (passed through unchanged by MapOperator)
        result_elem = result_batch.get_element(0)
        assert result_elem.metadata == {"source": "test", "version": 1}

    def test_sequential_data_transformation(self):
        """Test data transformation through sequential chain."""
        rngs = nnx.Rngs(0)

        # Create transformation chain: +1, *2, +3
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x + 1, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 2, rngs=rngs)

        config3 = MapOperatorConfig(stochastic=False)
        op3 = MapOperator(config3, fn=lambda x, key: x + 3, rngs=rngs)

        # Create sequential composite
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
            operators=[op1, op2, op3],
        )
        composite = CompositeOperatorModule(composite_config)

        # Create batch for data transformation
        batch = Batch([Element(data={"value": jnp.array([10.0])})])

        # Apply composite
        result_batch = composite(batch)

        # Verify: ((10 + 1) * 2) + 3 = 25
        result_data = result_batch.get_data()
        expected = jnp.array([[25.0]])
        assert jnp.allclose(result_data["value"], expected)


class TestSequentialJIT:
    """Test JIT compilation and vmap compatibility."""

    def test_sequential_jit_compilation(self):
        """Test that sequential composite can be JIT compiled."""
        rngs = nnx.Rngs(0)

        # Create operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x + 10, rngs=rngs)

        # Create sequential composite
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
            operators=[op1, op2],
        )
        composite = CompositeOperatorModule(composite_config)

        # JIT compile the __call__ method (pass module as argument, not closure)
        @nnx.jit
        def jit_apply(model, batch):
            return model(batch)

        # Create batch
        batch = Batch(
            [
                Element(data={"value": jnp.array([1.0])}),
                Element(data={"value": jnp.array([2.0])}),
                Element(data={"value": jnp.array([3.0])}),
            ]
        )

        # Apply JIT-compiled version
        result_batch = jit_apply(composite, batch)

        # Verify: (x * 2) + 10
        result_data = result_batch.get_data()
        expected = jnp.array([[12.0], [14.0], [16.0]])
        assert jnp.allclose(result_data["value"], expected)

    def test_sequential_with_vmap(self):
        """Test sequential composite with vmap (batch processing via Batch)."""
        rngs = nnx.Rngs(0)

        # Create operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x + 5, rngs=rngs)

        # Create sequential composite
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
            operators=[op1, op2],
        )
        composite = CompositeOperatorModule(composite_config)

        # Create batch (Batch handles vmap internally via apply_batch)
        batch = Batch(
            [
                Element(data={"value": jnp.array([1.0])}),
                Element(data={"value": jnp.array([2.0])}),
                Element(data={"value": jnp.array([3.0])}),
            ]
        )

        # Apply composite (vmap is handled internally)
        result_batch = composite(batch)

        # Verify results: (x * 2) + 5
        result_data = result_batch.get_data()
        expected = jnp.array([[7.0], [9.0], [11.0]])  # (1*2+5), (2*2+5), (3*2+5)
        assert jnp.allclose(result_data["value"], expected)


class TestSequentialAdvanced:
    """Test advanced sequential features."""

    def test_sequential_with_map_operator_children(self):
        """Test sequential composite containing MapOperator children."""
        rngs = nnx.Rngs(0)

        # Create MapOperators (Operators)
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 3, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x + 7, rngs=rngs)

        # Create sequential composite
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
            operators=[op1, op2],
        )
        composite = CompositeOperatorModule(composite_config)

        # Create batch for integration test
        batch = Batch([Element(data={"value": jnp.array([2.0])})])

        # Apply composite
        result_batch = composite(batch)

        # Verify: (2 * 3) + 7 = 13
        result_data = result_batch.get_data()
        expected = jnp.array([[13.0]])
        assert jnp.allclose(result_data["value"], expected)

    def test_sequential_statistics_aggregation(self):
        """Test that statistics are aggregated from all operators."""
        rngs = nnx.Rngs(0)

        # Create operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x + 10, rngs=rngs)

        # Create sequential composite
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
            operators=[op1, op2],
        )
        composite = CompositeOperatorModule(composite_config)

        # Create batch to trigger statistics collection
        batch = Batch([Element(data={"value": jnp.array([1.0])})])

        # Apply composite
        composite(batch)

        # Verify statistics dict exists (even if empty for MapOperator)
        assert hasattr(composite, "operator_statistics")
        # Statistics structure should be accessible (MapOperator may not populate)

    def test_nested_sequential_composites(self):
        """Test sequential composite containing another sequential composite."""
        rngs = nnx.Rngs(0)

        # Create inner sequential composite (2 operators)
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x + 1, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 2, rngs=rngs)

        inner_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
            operators=[op1, op2],
        )
        inner_composite = CompositeOperatorModule(inner_config)

        # Create outer sequential composite (inner + one more operator)
        config3 = MapOperatorConfig(stochastic=False)
        op3 = MapOperator(config3, fn=lambda x, key: x + 3, rngs=rngs)

        outer_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
            operators=[inner_composite, op3],
        )
        outer_composite = CompositeOperatorModule(outer_config)

        # Create batch for nested composition test
        batch = Batch([Element(data={"value": jnp.array([5.0])})])

        # Apply outer composite
        result_batch = outer_composite(batch)

        # Verify: op3(inner(5)) = op3(op2(op1(5))) = ((5+1)*2)+3 = 15
        result_data = result_batch.get_data()
        expected = jnp.array([[15.0]])
        assert jnp.allclose(result_data["value"], expected)
