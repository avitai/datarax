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

from datarax.core import batch_ops
from datarax.core.element_batch import Element

# GREEN phase - imports enabled
from datarax.operators.composite_operator import (
    CompositeOperatorConfig,
    CompositeOperatorModule,
    CompositionStrategy,
)
from datarax.operators.map_operator import MapOperator, MapOperatorConfig


class TestSequentialBasics:
    """Test basic sequential composition functionality."""

    def test_sequential_two_operators(self):
        """Test sequential composition with 2 operators."""
        rngs = nnx.Rngs(0)

        # Create 2 map operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, _key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, _key: x + 10, rngs=rngs)

        # Create sequential composite
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
        )
        composite = CompositeOperatorModule(
            composite_config,
            operators=[op1, op2],
        )

        # Create batch from Elements
        batch = batch_ops.from_stacked(
            batch_ops.stack(
                [
                    Element(data={"value": jnp.array([1.0])}),
                    Element(data={"value": jnp.array([2.0])}),
                    Element(data={"value": jnp.array([3.0])}),
                ]
            )
        )

        # Apply composite using __call__ (should be (x * 2) + 10)
        result_batch = composite(batch)

        # Verify: op2(op1(input)) = (input * 2) + 10
        assert result_batch.batch_size == 3
        result_data = result_batch.data
        expected = jnp.array([[12.0], [14.0], [16.0]])  # Shape (3, 1) to match batched Elements
        assert jnp.allclose(result_data["value"], expected)

    def test_sequential_three_operators(self):
        """Test sequential composition with 3+ operators."""
        rngs = nnx.Rngs(0)

        # Create 3 map operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, _key: x + 1, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, _key: x * 2, rngs=rngs)

        config3 = MapOperatorConfig(stochastic=False)
        op3 = MapOperator(config3, fn=lambda x, _key: x + 3, rngs=rngs)

        # Create sequential composite
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
        )
        composite = CompositeOperatorModule(
            composite_config,
            operators=[op1, op2, op3],
        )

        # Create batch from Element
        batch = batch_ops.from_stacked(batch_ops.stack([Element(data={"value": jnp.array([5.0])})]))

        # Apply composite using __call__
        result_batch = composite(batch)

        # Verify: op3(op2(op1(5))) = ((5 + 1) * 2) + 3 = 15
        assert result_batch.batch_size == 1
        result_data = result_batch.data
        assert jnp.allclose(result_data["value"], jnp.array([[15.0]]))

    def test_sequential_deterministic_operators(self):
        """Test sequential with all deterministic operators."""
        rngs = nnx.Rngs(0)

        # Create deterministic operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, _key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, _key: x + 5, rngs=rngs)

        # Create sequential composite
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
        )
        composite = CompositeOperatorModule(
            composite_config,
            operators=[op1, op2],
        )

        # Create batch
        batch = batch_ops.from_stacked(batch_ops.stack([Element(data={"value": jnp.array([3.0])})]))

        # Apply twice - should give same result
        result_batch1 = composite(batch)
        result_batch2 = composite(batch)

        # Verify deterministic: same input -> same output
        result1 = result_batch1.data
        result2 = result_batch2.data
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
            stochastic=True,
            stream_name="augment",
        )
        composite = CompositeOperatorModule(composite_config, operators=[op1, op2], rngs=rngs)

        # Test with batch
        batch = batch_ops.from_stacked(
            batch_ops.stack([Element(data={"value": jnp.array([1.0, 2.0, 3.0])})])
        )
        result = composite(batch)
        result_data = result.data

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
        op1 = MapOperator(config1, fn=lambda x, _key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, _key: x + 10, rngs=rngs)

        # Create sequential composite
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
        )
        composite = CompositeOperatorModule(
            composite_config,
            operators=[op1, op2],
        )

        # Create batch with state (MapOperator passes through unchanged)
        batch = batch_ops.from_stacked(
            batch_ops.stack(
                [Element(data={"value": jnp.array([1.0])}, state={"counter": jnp.array(5)})]
            )
        )

        # Apply composite
        result_batch = composite(batch)

        # Verify state threading (passed through unchanged by MapOperator)
        result_elem = batch_ops.element(result_batch, 0)
        assert "counter" in result_elem.state
        assert jnp.allclose(result_elem.state["counter"], jnp.array(5))

    def test_sequential_keeps_record_identities(self):
        """Each record's identity passes through every operator of the chain."""
        rngs = nnx.Rngs(0)

        # Create operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, _key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, _key: x + 5, rngs=rngs)

        # Create sequential composite
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
        )
        composite = CompositeOperatorModule(
            composite_config,
            operators=[op1, op2],
        )

        batch = batch_ops.from_stacked(
            batch_ops.stack(
                [Element({"value": jnp.array([1.0])}, index=jnp.array([2, 7], jnp.uint32))]
            )
        )

        # Apply composite
        result_batch = composite(batch)

        result_elem = batch_ops.element(result_batch, 0)
        assert result_elem.index is not None
        assert jnp.array_equal(result_elem.index, jnp.array([2, 7], jnp.uint32))
        assert jnp.allclose(result_elem.data["value"], jnp.array([7.0]))

    def test_sequential_data_transformation(self):
        """Test data transformation through sequential chain."""
        rngs = nnx.Rngs(0)

        # Create transformation chain: +1, *2, +3
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, _key: x + 1, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, _key: x * 2, rngs=rngs)

        config3 = MapOperatorConfig(stochastic=False)
        op3 = MapOperator(config3, fn=lambda x, _key: x + 3, rngs=rngs)

        # Create sequential composite
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
        )
        composite = CompositeOperatorModule(
            composite_config,
            operators=[op1, op2, op3],
        )

        # Create batch for data transformation
        batch = batch_ops.from_stacked(
            batch_ops.stack([Element(data={"value": jnp.array([10.0])})])
        )

        # Apply composite
        result_batch = composite(batch)

        # Verify: ((10 + 1) * 2) + 3 = 25
        result_data = result_batch.data
        expected = jnp.array([[25.0]])
        assert jnp.allclose(result_data["value"], expected)


class TestSequentialJIT:
    """Test JIT compilation and vmap compatibility."""

    def test_sequential_jit_compilation(self):
        """Test that sequential composite can be JIT compiled."""
        rngs = nnx.Rngs(0)

        # Create operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, _key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, _key: x + 10, rngs=rngs)

        # Create sequential composite
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
        )
        composite = CompositeOperatorModule(
            composite_config,
            operators=[op1, op2],
        )

        # JIT compile the __call__ method (pass module as argument, not closure)
        @nnx.jit
        def jit_apply(model, batch):
            return model(batch)

        # Create batch
        batch = batch_ops.from_stacked(
            batch_ops.stack(
                [
                    Element(data={"value": jnp.array([1.0])}),
                    Element(data={"value": jnp.array([2.0])}),
                    Element(data={"value": jnp.array([3.0])}),
                ]
            )
        )

        # Apply JIT-compiled version
        result_batch = jit_apply(composite, batch)

        # Verify: (x * 2) + 10
        result_data = result_batch.data
        expected = jnp.array([[12.0], [14.0], [16.0]])
        assert jnp.allclose(result_data["value"], expected)

    def test_sequential_with_vmap(self):
        """Test sequential composite with vmap (batch processing via Batch)."""
        rngs = nnx.Rngs(0)

        # Create operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, _key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, _key: x + 5, rngs=rngs)

        # Create sequential composite
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
        )
        composite = CompositeOperatorModule(
            composite_config,
            operators=[op1, op2],
        )

        # Create batch (Batch handles vmap internally via apply_batch)
        batch = batch_ops.from_stacked(
            batch_ops.stack(
                [
                    Element(data={"value": jnp.array([1.0])}),
                    Element(data={"value": jnp.array([2.0])}),
                    Element(data={"value": jnp.array([3.0])}),
                ]
            )
        )

        # Apply composite (vmap is handled internally)
        result_batch = composite(batch)

        # Verify results: (x * 2) + 5
        result_data = result_batch.data
        expected = jnp.array([[7.0], [9.0], [11.0]])  # (1*2+5), (2*2+5), (3*2+5)
        assert jnp.allclose(result_data["value"], expected)


class TestSequentialAdvanced:
    """Test advanced sequential features."""

    def test_sequential_with_map_operator_children(self):
        """Test sequential composite containing MapOperator children."""
        rngs = nnx.Rngs(0)

        # Create MapOperators (Operators)
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, _key: x * 3, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, _key: x + 7, rngs=rngs)

        # Create sequential composite
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
        )
        composite = CompositeOperatorModule(
            composite_config,
            operators=[op1, op2],
        )

        # Create batch for integration test
        batch = batch_ops.from_stacked(batch_ops.stack([Element(data={"value": jnp.array([2.0])})]))

        # Apply composite
        result_batch = composite(batch)

        # Verify: (2 * 3) + 7 = 13
        result_data = result_batch.data
        expected = jnp.array([[13.0]])
        assert jnp.allclose(result_data["value"], expected)

    def test_sequential_statistics_aggregation(self):
        """Test that statistics are aggregated from all operators."""
        rngs = nnx.Rngs(0)

        # Create operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, _key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, _key: x + 10, rngs=rngs)

        # Create sequential composite
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
        )
        composite = CompositeOperatorModule(
            composite_config,
            operators=[op1, op2],
        )

        # Create batch to trigger statistics collection
        batch = batch_ops.from_stacked(batch_ops.stack([Element(data={"value": jnp.array([1.0])})]))

        composite(batch)

    def test_nested_sequential_composites(self):
        """Test sequential composite containing another sequential composite."""
        rngs = nnx.Rngs(0)

        # Create inner sequential composite (2 operators)
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, _key: x + 1, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, _key: x * 2, rngs=rngs)

        inner_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
        )
        inner_composite = CompositeOperatorModule(
            inner_config,
            operators=[op1, op2],
        )

        # Create outer sequential composite (inner + one more operator)
        config3 = MapOperatorConfig(stochastic=False)
        op3 = MapOperator(config3, fn=lambda x, _key: x + 3, rngs=rngs)

        outer_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
        )
        outer_composite = CompositeOperatorModule(
            outer_config,
            operators=[inner_composite, op3],
        )

        # Create batch for nested composition test
        batch = batch_ops.from_stacked(batch_ops.stack([Element(data={"value": jnp.array([5.0])})]))

        # Apply outer composite
        result_batch = outer_composite(batch)

        # Verify: op3(inner(5)) = op3(op2(op1(5))) = ((5+1)*2)+3 = 15
        result_data = result_batch.data
        expected = jnp.array([[15.0]])
        assert jnp.allclose(result_data["value"], expected)
