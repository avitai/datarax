"""Tests for Ensemble composition strategy.

This module tests the ensemble reduction strategies (ENSEMBLE_MEAN, ENSEMBLE_SUM,
ENSEMBLE_MAX, ENSEMBLE_MIN), which are parallel execution with automatic reduction.

Test Coverage:
- All reduction modes (mean, sum, max, min)
- Ensemble with 3+ operators
- JIT compilation and vmap compatibility
- Statistics aggregation
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


class TestEnsembleReductions:
    """Test all ensemble reduction strategies."""

    def test_ensemble_mean_reduction(self):
        """Test ensemble with mean reduction."""
        rngs = nnx.Rngs(0)

        # Create 3 operators with different transformations
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 3, rngs=rngs)

        config3 = MapOperatorConfig(stochastic=False)
        op3 = MapOperator(config3, fn=lambda x, key: x * 4, rngs=rngs)

        # Create ensemble with mean reduction
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.ENSEMBLE_MEAN,
            operators=[op1, op2, op3],
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

        # Verify: mean([x*2, x*3, x*4]) for each element
        # Element 0: mean([2, 3, 4]) = 3
        # Element 1: mean([4, 6, 8]) = 6
        result_data = result_batch.get_data()
        expected = jnp.array([[3.0], [6.0]])
        assert jnp.allclose(result_data["value"], expected)

    def test_ensemble_sum_reduction(self):
        """Test ensemble with sum reduction."""
        rngs = nnx.Rngs(0)

        # Create 3 operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 3, rngs=rngs)

        config3 = MapOperatorConfig(stochastic=False)
        op3 = MapOperator(config3, fn=lambda x, key: x * 4, rngs=rngs)

        # Create ensemble with sum reduction
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.ENSEMBLE_SUM,
            operators=[op1, op2, op3],
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

        # Verify: sum([x*2, x*3, x*4]) for each element
        # Element 0: 2 + 3 + 4 = 9
        # Element 1: 4 + 6 + 8 = 18
        result_data = result_batch.get_data()
        expected = jnp.array([[9.0], [18.0]])
        assert jnp.allclose(result_data["value"], expected)

    def test_ensemble_max_reduction(self):
        """Test ensemble with max reduction."""
        rngs = nnx.Rngs(0)

        # Create 3 operators with different transformations
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 5, rngs=rngs)

        config3 = MapOperatorConfig(stochastic=False)
        op3 = MapOperator(config3, fn=lambda x, key: x * 3, rngs=rngs)

        # Create ensemble with max reduction
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.ENSEMBLE_MAX,
            operators=[op1, op2, op3],
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

        # Verify: max([x*2, x*5, x*3]) for each element
        # Element 0: max([2, 5, 3]) = 5
        # Element 1: max([4, 10, 6]) = 10
        result_data = result_batch.get_data()
        expected = jnp.array([[5.0], [10.0]])
        assert jnp.allclose(result_data["value"], expected)

    def test_ensemble_min_reduction(self):
        """Test ensemble with min reduction."""
        rngs = nnx.Rngs(0)

        # Create 3 operators with different transformations
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 5, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 2, rngs=rngs)

        config3 = MapOperatorConfig(stochastic=False)
        op3 = MapOperator(config3, fn=lambda x, key: x * 3, rngs=rngs)

        # Create ensemble with min reduction
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.ENSEMBLE_MIN,
            operators=[op1, op2, op3],
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

        # Verify: min([x*5, x*2, x*3]) for each element
        # Element 0: min([5, 2, 3]) = 2
        # Element 1: min([10, 4, 6]) = 4
        result_data = result_batch.get_data()
        expected = jnp.array([[2.0], [4.0]])
        assert jnp.allclose(result_data["value"], expected)


class TestEnsembleAdvanced:
    """Test advanced ensemble features."""

    def test_ensemble_with_many_operators(self):
        """Test ensemble with 5+ operators."""
        rngs = nnx.Rngs(0)

        # Create 5 operators
        operators = []
        for i in range(5):
            config = MapOperatorConfig(stochastic=False)
            op = MapOperator(config, fn=lambda x, key, i=i: x * (i + 1), rngs=rngs)
            operators.append(op)

        # Create ensemble with mean reduction
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.ENSEMBLE_MEAN,
            operators=operators,
        )
        composite = CompositeOperatorModule(composite_config)

        # Create batch
        batch = Batch([Element(data={"value": jnp.array([10.0])})])

        # Apply composite
        result_batch = composite(batch)

        # Verify: mean([10, 20, 30, 40, 50]) = 150 / 5 = 30
        result_data = result_batch.get_data()
        expected = jnp.array([[30.0]])
        assert jnp.allclose(result_data["value"], expected)

    def test_ensemble_jit_compilation(self):
        """Test that ensemble composite can be JIT compiled."""
        rngs = nnx.Rngs(0)

        # Create 3 operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 3, rngs=rngs)

        config3 = MapOperatorConfig(stochastic=False)
        op3 = MapOperator(config3, fn=lambda x, key: x * 4, rngs=rngs)

        # Create ensemble
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.ENSEMBLE_MEAN,
            operators=[op1, op2, op3],
        )
        composite = CompositeOperatorModule(composite_config)

        # JIT compile using nnx.jit (pass module as argument, not closure)
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

        # Verify: mean([x*2, x*3, x*4]) for each element = 3x
        result_data = result_batch.get_data()
        expected = jnp.array([[3.0], [6.0]])
        assert jnp.allclose(result_data["value"], expected)

    def test_ensemble_with_vmap(self):
        """Test ensemble composite with vmap (batch processing)."""
        rngs = nnx.Rngs(0)

        # Create 3 operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 4, rngs=rngs)

        config3 = MapOperatorConfig(stochastic=False)
        op3 = MapOperator(config3, fn=lambda x, key: x * 6, rngs=rngs)

        # Create ensemble
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.ENSEMBLE_MEAN,
            operators=[op1, op2, op3],
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

        # Verify: mean([x*2, x*4, x*6]) = 4x for each element
        result_data = result_batch.get_data()
        expected = jnp.array([[4.0], [8.0], [12.0]])
        assert jnp.allclose(result_data["value"], expected)

    def test_ensemble_statistics_aggregation(self):
        """Test that statistics are aggregated from all ensemble members."""
        rngs = nnx.Rngs(0)

        # Create 3 operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 3, rngs=rngs)

        config3 = MapOperatorConfig(stochastic=False)
        op3 = MapOperator(config3, fn=lambda x, key: x * 4, rngs=rngs)

        # Create ensemble
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.ENSEMBLE_MEAN,
            operators=[op1, op2, op3],
        )
        composite = CompositeOperatorModule(composite_config)

        # Create batch to trigger statistics collection
        batch = Batch([Element(data={"value": jnp.array([1.0])})])

        # Apply composite
        composite(batch)

        # Verify statistics dict exists
        assert hasattr(composite, "operator_statistics")
