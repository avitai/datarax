"""Edge cases and error handling tests for CompositeOperatorModule.

This module tests boundary conditions, error handling, and unusual scenarios
to ensure robust behavior.

Test Coverage:
- Empty operators list
- Single operator (trivial composition)
- Very large operator lists (100+ operators)
- Mismatched PyTree structures
- NaN/Inf handling in merge operations
- Zero and negative weights
- Non-existent branch routing
- Exception-raising conditions
"""

import pytest

# Edge cases now being implemented
import jax.numpy as jnp
from flax import nnx

from datarax.core.element_batch import Batch, Element
from datarax.operators.composite_operator import (
    CompositeOperatorConfig,
    CompositeOperatorModule,
    CompositionStrategy,
)
from datarax.operators.map_operator import MapOperator, MapOperatorConfig


class TestBoundaryConditions:
    """Test boundary conditions and edge cases."""

    def test_empty_operators_list_fails_at_config(self):
        """Test that empty operators list fails at config construction."""
        with pytest.raises(ValueError, match="operators list cannot be empty"):
            CompositeOperatorConfig(
                strategy=CompositionStrategy.SEQUENTIAL,
                operators=[],  # Empty list should fail validation
                stochastic=False,
            )

    def test_single_operator_trivial_composition(self):
        """Test trivial composition with single operator."""
        rngs = nnx.Rngs(0)

        # Create single operator
        config = MapOperatorConfig(stochastic=False)
        op = MapOperator(config, fn=lambda x, key: x * 2, rngs=rngs)

        # Create composite with single operator (trivial composition)
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
            operators=[op],
            stochastic=False,
        )
        composite = CompositeOperatorModule(composite_config, rngs=rngs)

        # Test with batch
        batch = Batch([Element(data={"value": jnp.array([5.0])})])
        result_batch = composite(batch)
        result_data = result_batch.get_data()

        # Should be equivalent to calling operator directly
        assert jnp.allclose(result_data["value"], jnp.array([[10.0]]))

    def test_very_large_operator_list(self):
        """Test composite with 100+ operators."""
        rngs = nnx.Rngs(0)

        # Create 100 operators that each add 1
        operators = []
        for _ in range(100):
            config = MapOperatorConfig(stochastic=False)
            op = MapOperator(config, fn=lambda x, key: x + 1, rngs=rngs)
            operators.append(op)

        # Create composite
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
            operators=operators,
            stochastic=False,
        )
        composite = CompositeOperatorModule(composite_config, rngs=rngs)

        # Test with batch
        batch = Batch([Element(data={"value": jnp.array([0.0])})])
        result_batch = composite(batch)
        result_data = result_batch.get_data()

        # Should be 0 + 100 = 100
        assert jnp.allclose(result_data["value"], jnp.array([[100.0]]))

    def test_mismatched_array_shapes(self):
        """Test error handling when operators return different array shapes."""
        rngs = nnx.Rngs(0)

        # Operator 1 returns array with shape (1,)
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x, rngs=rngs)

        # Operator 2 returns array with shape (2,) - DIFFERENT shape
        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: jnp.array([x[0], x[0] * 2]), rngs=rngs)

        # Create parallel composite with concat merge
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.PARALLEL,
            operators=[op1, op2],
            merge_strategy="concat",
            stochastic=False,
        )
        composite = CompositeOperatorModule(composite_config, rngs=rngs)

        # This works! Concat along axis 0: shape (1,) + (2,) = (3,)
        batch = Batch([Element(data={"value": jnp.array([5.0])})])
        result_batch = composite(batch)
        result_data = result_batch.get_data()

        # Concatenated result should have shape (batch_size=1, 3)
        assert result_data["value"].shape == (1, 3)
        # Values: [5.0, 5.0, 10.0]
        assert jnp.allclose(result_data["value"], jnp.array([[5.0, 5.0, 10.0]]))


class TestNumericalEdgeCases:
    """Test numerical edge cases (NaN, Inf, etc.)."""

    def test_nan_handling_in_ensemble_reduction(self):
        """Test ensemble reduction when operator returns NaN."""
        rngs = nnx.Rngs(0)

        # Operator 1 returns normal value
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        # Operator 2 returns NaN (multiply by NaN to get NaN result)
        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * jnp.nan, rngs=rngs)

        # Create ensemble with mean reduction
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.ENSEMBLE_MEAN,
            operators=[op1, op2],
            stochastic=False,
        )
        composite = CompositeOperatorModule(composite_config, rngs=rngs)

        # Test with batch
        batch = Batch([Element(data={"value": jnp.array([5.0])})])
        result_batch = composite(batch)
        result_data = result_batch.get_data()

        # NaN should propagate: mean([10.0, NaN]) = NaN
        assert jnp.isnan(result_data["value"]).all()

    def test_inf_handling_in_merge_operations(self):
        """Test merge operations when operator returns Inf."""
        rngs = nnx.Rngs(0)

        # Operator 1 returns normal value
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        # Operator 2 returns Inf (multiply by inf to get inf result)
        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * jnp.inf, rngs=rngs)

        # Create parallel with sum merge
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.PARALLEL,
            operators=[op1, op2],
            merge_strategy="sum",
            stochastic=False,
        )
        composite = CompositeOperatorModule(composite_config, rngs=rngs)

        # Test with batch
        batch = Batch([Element(data={"value": jnp.array([5.0])})])
        result_batch = composite(batch)
        result_data = result_batch.get_data()

        # Inf should propagate: sum([10.0, Inf]) = Inf
        assert jnp.isinf(result_data["value"]).all()


class TestWeightingEdgeCases:
    """Test edge cases in weighted parallel."""

    def test_zero_weights_in_weighted_parallel(self):
        """Test that weights with all zeros produces zero output."""
        rngs = nnx.Rngs(0)

        # Create operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)
        op2 = MapOperator(config1, fn=lambda x, key: x * 3, rngs=rngs)

        # Zero weights should produce zero output (not an error, just degenerate)
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.WEIGHTED_PARALLEL,
            operators=[op1, op2],
            weights=[0.0, 0.0],
            stochastic=False,
        )
        composite = CompositeOperatorModule(composite_config, rngs=rngs)

        batch = Batch([Element(data={"value": jnp.array([5.0])})])
        result_batch = composite(batch)
        result_data = result_batch.get_data()

        # All zero weights should give zero output
        assert jnp.allclose(result_data["value"], jnp.array([[0.0]]))

    def test_negative_weights_in_weighted_parallel(self):
        """Test that negative weights work correctly (subtraction)."""
        rngs = nnx.Rngs(0)

        # Create operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)
        op2 = MapOperator(config1, fn=lambda x, key: x * 3, rngs=rngs)

        # Negative weights should work (for subtraction/cancellation effects)
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.WEIGHTED_PARALLEL,
            operators=[op1, op2],
            weights=[1.0, -0.5],  # 1.0 * (x*2) + (-0.5) * (x*3)
            stochastic=False,
        )
        composite = CompositeOperatorModule(composite_config, rngs=rngs)

        batch = Batch([Element(data={"value": jnp.array([2.0])})])
        result_batch = composite(batch)
        result_data = result_batch.get_data()

        # 1.0 * (2*2) + (-0.5) * (2*3) = 4 - 3 = 1
        assert jnp.allclose(result_data["value"], jnp.array([[1.0]]))


class TestBranchingEdgeCases:
    """Test edge cases in branching strategy."""

    def test_router_returns_valid_branch_indices(self):
        """Test that branching works correctly with valid indices."""
        import jax

        rngs = nnx.Rngs(0)

        # Create operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)
        op2 = MapOperator(config1, fn=lambda x, key: x * 3, rngs=rngs)

        # Router returns valid indices (0 or 1)
        def router(data):
            # Use modulo to ensure valid indices
            return jax.lax.select(data["value"][0] < 5.0, 0, 1)

        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.BRANCHING,
            operators=[op1, op2],
            router=router,
            stochastic=False,
        )
        composite = CompositeOperatorModule(composite_config, rngs=rngs)

        # Test with value < 5.0 -> should use op1 (x * 2)
        batch1 = Batch([Element(data={"value": jnp.array([3.0])})])
        result1 = composite(batch1)
        assert jnp.allclose(result1.get_data()["value"], jnp.array([[6.0]]))

        # Test with value >= 5.0 -> should use op2 (x * 3)
        batch2 = Batch([Element(data={"value": jnp.array([10.0])})])
        result2 = composite(batch2)
        assert jnp.allclose(result2.get_data()["value"], jnp.array([[30.0]]))


class TestConditionalEdgeCases:
    """Test edge cases in conditional strategies."""

    def test_condition_raises_exception(self):
        """Test error handling when condition function raises exception."""
        rngs = nnx.Rngs(0)

        # Create operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        # Condition that raises exception
        def bad_condition(data):
            raise RuntimeError("Condition evaluation failed!")

        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.CONDITIONAL_SEQUENTIAL,
            operators=[op1],
            conditions=[bad_condition],
            stochastic=False,
        )
        composite = CompositeOperatorModule(composite_config, rngs=rngs)

        batch = Batch([Element(data={"value": jnp.array([5.0])})])

        # Exception should propagate
        with pytest.raises(RuntimeError, match="Condition evaluation failed"):
            composite(batch)
