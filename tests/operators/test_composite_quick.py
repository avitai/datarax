"""Quick integration test for CompositeOperatorModule core functionality.

This test validates that the core implementation works before we implement
all 94 comprehensive tests. Tests config, sequential, parallel, and ensemble.
"""

import pytest
import jax.numpy as jnp
from flax import nnx

from datarax.operators.composite_operator import (
    CompositeOperatorModule,
    CompositeOperatorConfig,
    CompositionStrategy,
)
from datarax.operators.map_operator import MapOperator, MapOperatorConfig
from datarax.core.element_batch import Batch, Element


class TestCompositeQuickIntegration:
    """Quick integration tests for core functionality."""

    def test_sequential_with_two_map_operators(self):
        """Test sequential composition with 2 MapOperators."""
        # Create RNGs (required even for deterministic operators)
        rngs = nnx.Rngs(0)

        # Create two simple map operators
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

        # Test data
        batch = Batch(
            [
                Element(data={"value": jnp.array([1.0])}),
                Element(data={"value": jnp.array([2.0])}),
                Element(data={"value": jnp.array([3.0])}),
            ]
        )

        # Apply composite (should be (x * 2) + 10)
        result_batch = composite(batch)
        result_data = result_batch.get_data()

        # Verify result
        expected = jnp.array([[12.0], [14.0], [16.0]])  # (1*2+10, 2*2+10, 3*2+10)
        assert jnp.allclose(result_data["value"], expected)

    def test_parallel_with_concat_merge(self):
        """Test parallel composition with concat merge."""
        rngs = nnx.Rngs(0)

        # Create two map operators that transform differently
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 3, rngs=rngs)

        # Create parallel composite with concat
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.PARALLEL,
            operators=[op1, op2],
            merge_strategy="concat",
            merge_axis=0,
        )
        composite = CompositeOperatorModule(composite_config)

        # Test data
        batch = Batch(
            [
                Element(data={"value": jnp.array([1.0, 2.0])}),
            ]
        )

        # Apply composite
        result_batch = composite(batch)
        result_data = result_batch.get_data()

        # Verify result (concat of [2, 4] and [3, 6])
        expected = jnp.array([[2.0, 4.0, 3.0, 6.0]])
        assert jnp.allclose(result_data["value"], expected)

    def test_parallel_with_stack_merge(self):
        """Test parallel composition with stack merge."""
        rngs = nnx.Rngs(0)

        # Create two map operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 3, rngs=rngs)

        # Create parallel composite with stack
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.PARALLEL,
            operators=[op1, op2],
            merge_strategy="stack",
            merge_axis=0,
        )
        composite = CompositeOperatorModule(composite_config)

        # Test data
        batch = Batch(
            [
                Element(data={"value": jnp.array([1.0, 2.0])}),
            ]
        )

        # Apply composite
        result_batch = composite(batch)
        result_data = result_batch.get_data()

        # Verify result (stack of [[2, 4], [3, 6]])
        expected = jnp.array([[[2.0, 4.0], [3.0, 6.0]]])
        assert jnp.allclose(result_data["value"], expected)

    def test_ensemble_mean_reduction(self):
        """Test ensemble with mean reduction."""
        rngs = nnx.Rngs(0)

        # Create three map operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 1.0, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 2.0, rngs=rngs)

        config3 = MapOperatorConfig(stochastic=False)
        op3 = MapOperator(config3, fn=lambda x, key: x * 3.0, rngs=rngs)

        # Create ensemble composite with mean
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.ENSEMBLE_MEAN,
            operators=[op1, op2, op3],
        )
        composite = CompositeOperatorModule(composite_config)

        # Test data
        batch = Batch([Element(data={"value": jnp.array([10.0])})])

        # Apply composite
        result_batch = composite(batch)
        result_data = result_batch.get_data()

        # Verify result (mean of [10, 20, 30] = 20)
        expected = jnp.array([[20.0]])
        assert jnp.allclose(result_data["value"], expected)

    def test_ensemble_sum_reduction(self):
        """Test ensemble with sum reduction."""
        rngs = nnx.Rngs(0)

        # Create two map operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2.0, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 3.0, rngs=rngs)

        # Create ensemble composite with sum
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.ENSEMBLE_SUM,
            operators=[op1, op2],
        )
        composite = CompositeOperatorModule(composite_config)

        # Test data
        batch = Batch([Element(data={"value": jnp.array([5.0])})])

        # Apply composite
        result_batch = composite(batch)
        result_data = result_batch.get_data()

        # Verify result (sum of [10, 15] = 25)
        expected = jnp.array([[25.0]])
        assert jnp.allclose(result_data["value"], expected)

    def test_config_validation_empty_operators_fails(self):
        """Test that empty operators list raises ValueError."""
        with pytest.raises(ValueError, match="operators list cannot be empty"):
            CompositeOperatorConfig(
                strategy=CompositionStrategy.SEQUENTIAL,
                operators=[],
            )

    def test_config_validation_branching_requires_router(self):
        """Test that branching strategy requires router function."""
        rngs = nnx.Rngs(0)

        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 2, rngs=rngs)

        # Should raise if router is missing
        with pytest.raises(ValueError, match="BRANCHING strategy requires router"):
            CompositeOperatorConfig(
                strategy=CompositionStrategy.BRANCHING,
                operators=[op1, op2],  # List without router
            )

    def test_config_auto_stochastic_detection(self):
        """Test that config auto-detects stochastic from children."""
        rngs = nnx.Rngs(0)

        # Create one deterministic and one stochastic operator
        det_config = MapOperatorConfig(stochastic=False)
        det_op = MapOperator(det_config, fn=lambda x, key: x, rngs=rngs)

        # Note: MapOperator doesn't support stochastic yet, so we'll just
        # test with deterministic for now
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
            operators=[det_op, det_op],
        )

        # Should auto-detect as False (all deterministic)
        assert not composite_config.stochastic
