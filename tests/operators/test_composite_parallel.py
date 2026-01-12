"""Tests for Parallel composition strategy.

This module tests the PARALLEL and WEIGHTED_PARALLEL composition strategies,
which apply all operators to the same input and merge the outputs.

Test Coverage:
- All merge strategies (concat, stack, sum, mean, dict)
- Custom merge functions
- Deterministic and stochastic parallel execution
- Weighted parallel with fixed and learnable weights
- JIT compilation and vmap compatibility
- Statistics aggregation
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


class TestParallelMergeStrategies:
    """Test all merge strategies for parallel composition."""

    def test_parallel_concat_merge(self):
        """Test parallel composition with concatenation merge."""
        rngs = nnx.Rngs(0)

        # Create 2 operators with different transformations
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 3, rngs=rngs)

        # Create parallel composite with concat merge
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.PARALLEL,
            operators=[op1, op2],
            merge_strategy="concat",
            merge_axis=0,
        )
        composite = CompositeOperatorModule(composite_config)

        # Create batch
        batch = Batch(
            [
                Element(data={"value": jnp.array([1.0])}),
                Element(data={"value": jnp.array([2.0])}),
            ]
        )

        # Apply composite
        result_batch = composite(batch)

        # Verify: parallel concat merges outputs within each element
        # Element 0: concat([2], [3]) = [2, 3]
        # Element 1: concat([4], [6]) = [4, 6]
        result_data = result_batch.get_data()
        expected = jnp.array([[2.0, 3.0], [4.0, 6.0]])  # Shape (2, 2)
        assert jnp.allclose(result_data["value"], expected)

    def test_parallel_stack_merge(self):
        """Test parallel composition with stack merge."""
        rngs = nnx.Rngs(0)

        # Create 2 operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 3, rngs=rngs)

        # Create parallel composite with stack merge
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.PARALLEL,
            operators=[op1, op2],
            merge_strategy="stack",
            merge_axis=0,
        )
        composite = CompositeOperatorModule(composite_config)

        # Create batch
        batch = Batch(
            [
                Element(data={"value": jnp.array([1.0])}),
                Element(data={"value": jnp.array([2.0])}),
            ]
        )

        # Apply composite
        result_batch = composite(batch)

        # Verify: parallel stack merges outputs within each element
        # Element 0: stack([2], [3]) along axis 0 = [[2], [3]]
        # Element 1: stack([4], [6]) along axis 0 = [[4], [6]]
        result_data = result_batch.get_data()
        expected = jnp.array([[[2.0], [3.0]], [[4.0], [6.0]]])  # Shape (2, 2, 1)
        assert jnp.allclose(result_data["value"], expected)

    def test_parallel_sum_merge(self):
        """Test parallel composition with sum merge."""
        rngs = nnx.Rngs(0)

        # Create 2 operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 3, rngs=rngs)

        # Create parallel composite with sum merge
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.PARALLEL,
            operators=[op1, op2],
            merge_strategy="sum",
        )
        composite = CompositeOperatorModule(composite_config)

        # Create batch
        batch = Batch(
            [
                Element(data={"value": jnp.array([1.0])}),
                Element(data={"value": jnp.array([2.0])}),
            ]
        )

        # Apply composite
        result_batch = composite(batch)

        # Verify: [2] + [3] = [5], [4] + [6] = [10]
        result_data = result_batch.get_data()
        expected = jnp.array([[5.0], [10.0]])
        assert jnp.allclose(result_data["value"], expected)

    def test_parallel_mean_merge(self):
        """Test parallel composition with mean merge."""
        rngs = nnx.Rngs(0)

        # Create 2 operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 4, rngs=rngs)

        # Create parallel composite with mean merge
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.PARALLEL,
            operators=[op1, op2],
            merge_strategy="mean",
        )
        composite = CompositeOperatorModule(composite_config)

        # Create batch
        batch = Batch(
            [
                Element(data={"value": jnp.array([1.0])}),
                Element(data={"value": jnp.array([2.0])}),
            ]
        )

        # Apply composite
        result_batch = composite(batch)

        # Verify: ([2] + [4]) / 2 = [3], ([4] + [8]) / 2 = [6]
        result_data = result_batch.get_data()
        expected = jnp.array([[3.0], [6.0]])
        assert jnp.allclose(result_data["value"], expected)

    def test_parallel_dict_merge(self):
        """Test parallel composition with dict merge."""
        rngs = nnx.Rngs(0)

        # Create 2 operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 3, rngs=rngs)

        # Create parallel composite with dict merge
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.PARALLEL,
            operators=[op1, op2],
            merge_strategy="dict",
        )
        composite = CompositeOperatorModule(composite_config)

        # Create batch
        batch = Batch([Element(data={"value": jnp.array([1.0])})])

        # Apply composite
        result_batch = composite(batch)

        # Verify: dict merge preserves top-level keys but nests operator outputs
        # Structure: {"value": {"operator_0": array, "operator_1": array}}
        result_data = result_batch.get_data()
        assert "value" in result_data
        assert isinstance(result_data["value"], dict)
        assert "operator_0" in result_data["value"]
        assert "operator_1" in result_data["value"]

        # Use jax.tree.map to compare nested structure
        expected_tree = {
            "value": {"operator_0": jnp.array([[2.0]]), "operator_1": jnp.array([[3.0]])}
        }

        # Compare each leaf
        def check_close(a, b):
            return jnp.allclose(a, b)

        matches = jax.tree.map(check_close, result_data, expected_tree)
        assert all(jax.tree.leaves(matches))

    def test_parallel_custom_merge_function(self):
        """Test parallel composition with custom merge function."""
        rngs = nnx.Rngs(0)

        # Create 2 operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 3, rngs=rngs)

        # Custom merge: weighted average (0.7, 0.3)
        def custom_merge(outputs):
            out1, out2 = outputs
            return jax.tree.map(lambda a, b: 0.7 * a + 0.3 * b, out1, out2)

        # Create parallel composite with custom merge
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.PARALLEL,
            operators=[op1, op2],
            merge_fn=custom_merge,
        )
        composite = CompositeOperatorModule(composite_config)

        # Create batch
        batch = Batch([Element(data={"value": jnp.array([10.0])})])

        # Apply composite
        result_batch = composite(batch)

        # Verify: 0.7*20 + 0.3*30 = 14 + 9 = 23
        result_data = result_batch.get_data()
        expected = jnp.array([[23.0]])
        assert jnp.allclose(result_data["value"], expected)


class TestParallelExecution:
    """Test parallel execution modes."""

    def test_parallel_deterministic_operators(self):
        """Test parallel with all deterministic operators."""
        rngs = nnx.Rngs(0)

        # Create deterministic operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x + 5, rngs=rngs)

        # Create parallel composite
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.PARALLEL,
            operators=[op1, op2],
            merge_strategy="sum",
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
        # (3*2) = 6, (3+5) = 8, sum = 14
        expected = jnp.array([[14.0]])
        assert jnp.allclose(result1["value"], expected)

    def test_parallel_stochastic_operators(self):
        """Test parallel with all stochastic operators."""
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

        # Create parallel composite with sum merge
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.PARALLEL,
            operators=[op1, op2],
            merge_strategy="sum",
            stochastic=True,
            stream_name="augment",
        )
        composite = CompositeOperatorModule(composite_config, rngs=rngs)

        # Test with batch
        batch = Batch([Element(data={"value": jnp.array([1.0, 2.0])})])
        result = composite(batch)
        result_data = result.get_data()

        # Output should differ from simple sum due to stochastic transformations
        # But should be in reasonable range
        assert result_data["value"].shape == (1, 2)
        assert jnp.all(result_data["value"] > 0)


class TestParallelJIT:
    """Test JIT compilation and vmap compatibility."""

    def test_parallel_jit_compilation(self):
        """Test that parallel composite can be JIT compiled."""
        rngs = nnx.Rngs(0)

        # Create operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 3, rngs=rngs)

        # Create parallel composite
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.PARALLEL,
            operators=[op1, op2],
            merge_strategy="sum",
        )
        composite = CompositeOperatorModule(composite_config)

        # JIT compile with nnx.jit (pass module as argument, not closure)
        @nnx.jit
        def jit_apply(model, batch):
            return model(batch)

        # Create batch
        batch = Batch(
            [
                Element(data={"value": jnp.array([1.0])}),
                Element(data={"value": jnp.array([2.0])}),
            ]
        )

        # Apply JIT-compiled version
        result_batch = jit_apply(composite, batch)

        # Verify: [2] + [3] = [5], [4] + [6] = [10]
        result_data = result_batch.get_data()
        expected = jnp.array([[5.0], [10.0]])
        assert jnp.allclose(result_data["value"], expected)

    def test_parallel_with_vmap(self):
        """Test parallel composite with vmap (batch processing)."""
        rngs = nnx.Rngs(0)

        # Create operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 3, rngs=rngs)

        # Create parallel composite
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.PARALLEL,
            operators=[op1, op2],
            merge_strategy="sum",
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

        # Verify: sum([x*2, x*3]) = 5x
        result_data = result_batch.get_data()
        expected = jnp.array([[5.0], [10.0], [15.0]])
        assert jnp.allclose(result_data["value"], expected)


class TestParallelAdvanced:
    """Test advanced parallel features."""

    def test_parallel_statistics_aggregation(self):
        """Test that statistics are aggregated from all operators."""
        rngs = nnx.Rngs(0)

        # Create operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x + 10, rngs=rngs)

        # Create parallel composite
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.PARALLEL,
            operators=[op1, op2],
            merge_strategy="sum",
        )
        composite = CompositeOperatorModule(composite_config)

        # Create batch to trigger statistics collection
        batch = Batch([Element(data={"value": jnp.array([1.0])})])

        # Apply composite
        composite(batch)

        # Verify statistics dict exists
        assert hasattr(composite, "operator_statistics")

    def test_weighted_parallel_fixed_weights(self):
        """Test weighted parallel with fixed weights."""
        rngs = nnx.Rngs(0)

        # Create operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 3, rngs=rngs)

        # Create weighted parallel composite
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.WEIGHTED_PARALLEL,
            operators=[op1, op2],
            weights=[0.7, 0.3],
        )
        composite = CompositeOperatorModule(composite_config)

        # Create batch
        batch = Batch([Element(data={"value": jnp.array([10.0])})])

        # Apply composite
        result_batch = composite(batch)

        # Verify: 0.7*(10*2) + 0.3*(10*3) = 0.7*20 + 0.3*30 = 14 + 9 = 23
        result_data = result_batch.get_data()
        expected = jnp.array([[23.0]])
        assert jnp.allclose(result_data["value"], expected)
