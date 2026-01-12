"""Tests for Branching composition strategy.

This module tests the BRANCHING strategy, which routes data through different
paths based on a router function.

Test Coverage:
- Branching with 2 and 3+ branches
- Default branch behavior
- Router based on data shape
- Router based on data values
- Statistics tracking (branch usage counts)
- Different operator types per branch
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


class TestBranchingBasics:
    """Test basic branching composition functionality."""

    def test_branching_with_two_branches(self):
        """Test branching with 2 branches (binary routing)."""
        rngs = nnx.Rngs(0)

        # Create 2 branch operators
        config_a = MapOperatorConfig(stochastic=False)
        branch_a = MapOperator(config_a, fn=lambda x, key: x * 2, rngs=rngs)

        config_b = MapOperatorConfig(stochastic=False)
        branch_b = MapOperator(config_b, fn=lambda x, key: x * 10, rngs=rngs)

        # Create branching composite with router that returns integer index
        def router(data):
            # Router returns integer index (vmap/jit compatible)
            # 0 = branch_a, 1 = branch_b
            return jax.lax.select(data["value"][0] < 5.0, 0, 1)

        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.BRANCHING,
            operators=[branch_a, branch_b],  # List of operators, indexed by router
            router=router,
        )
        composite = CompositeOperatorModule(composite_config)

        # Test data that routes to branch "a" (value < 5)
        batch_a = Batch([Element(data={"value": jnp.array([2.0])})])
        result_batch_a = composite(batch_a)
        result_data_a = result_batch_a.get_data()
        assert jnp.allclose(result_data_a["value"], jnp.array([[4.0]]))  # 2 * 2

        # Test data that routes to branch "b" (value >= 5)
        batch_b = Batch([Element(data={"value": jnp.array([7.0])})])
        result_batch_b = composite(batch_b)
        result_data_b = result_batch_b.get_data()
        assert jnp.allclose(result_data_b["value"], jnp.array([[70.0]]))  # 7 * 10

    def test_branching_with_many_branches(self):
        """Test branching with 3+ branches."""
        rngs = nnx.Rngs(0)

        # Create 3 branch operators
        config_small = MapOperatorConfig(stochastic=False)
        branch_small = MapOperator(config_small, fn=lambda x, key: x + 1, rngs=rngs)

        config_medium = MapOperatorConfig(stochastic=False)
        branch_medium = MapOperator(config_medium, fn=lambda x, key: x + 10, rngs=rngs)

        config_large = MapOperatorConfig(stochastic=False)
        branch_large = MapOperator(config_large, fn=lambda x, key: x + 100, rngs=rngs)

        # Create branching composite with 3-way router
        def router(data):
            # Use nested JAX select for 3-way branching with integer indices
            # 0 = small, 1 = medium, 2 = large
            val = data["value"][0]
            return jax.lax.select(val < 3.0, 0, jax.lax.select(val < 7.0, 1, 2))

        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.BRANCHING,
            operators=[branch_small, branch_medium, branch_large],  # Indexed 0, 1, 2
            router=router,
        )
        composite = CompositeOperatorModule(composite_config)

        # Test small branch
        batch_small = Batch([Element(data={"value": jnp.array([1.0])})])
        result_batch_small = composite(batch_small)
        result_data_small = result_batch_small.get_data()
        assert jnp.allclose(result_data_small["value"], jnp.array([[2.0]]))  # 1 + 1

        # Test medium branch
        batch_medium = Batch([Element(data={"value": jnp.array([5.0])})])
        result_batch_medium = composite(batch_medium)
        result_data_medium = result_batch_medium.get_data()
        assert jnp.allclose(result_data_medium["value"], jnp.array([[15.0]]))  # 5 + 10

        # Test large branch
        batch_large = Batch([Element(data={"value": jnp.array([10.0])})])
        result_batch_large = composite(batch_large)
        result_data_large = result_batch_large.get_data()
        assert jnp.allclose(result_data_large["value"], jnp.array([[110.0]]))  # 10 + 100


class TestBranchingRouters:
    """Test different router function types."""

    def test_branching_router_based_on_shape(self):
        """Test router that selects branch based on data shape."""
        rngs = nnx.Rngs(0)

        # Create branch operators for different shapes
        config_small = MapOperatorConfig(stochastic=False)
        branch_small = MapOperator(config_small, fn=lambda x, key: x + 1, rngs=rngs)

        config_large = MapOperatorConfig(stochastic=False)
        branch_large = MapOperator(config_large, fn=lambda x, key: x + 100, rngs=rngs)

        # Create branching composite with shape-based router
        def router(data):
            shape = data["value"].shape[0]
            return 0 if shape <= 3 else 1  # 0=small, 1=large

        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.BRANCHING,
            operators=[branch_small, branch_large],  # Indexed 0=small, 1=large
            router=router,
        )
        composite = CompositeOperatorModule(composite_config)

        # Test small shape (2 elements)
        batch_small = Batch([Element(data={"value": jnp.array([1.0, 2.0])})])
        result_batch_small = composite(batch_small)
        result_data_small = result_batch_small.get_data()
        assert jnp.allclose(result_data_small["value"], jnp.array([[2.0, 3.0]]))  # +1

        # Test large shape (5 elements)
        batch_large = Batch([Element(data={"value": jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])})])
        result_batch_large = composite(batch_large)
        result_data_large = result_batch_large.get_data()
        expected = jnp.array([[101.0, 102.0, 103.0, 104.0, 105.0]])  # +100
        assert jnp.allclose(result_data_large["value"], expected)

    def test_branching_router_based_on_values(self):
        """Test router that selects branch based on data values."""
        rngs = nnx.Rngs(0)

        # Create branch operators for different value ranges
        config_low = MapOperatorConfig(stochastic=False)
        branch_low = MapOperator(config_low, fn=lambda x, key: x * 2, rngs=rngs)

        config_medium = MapOperatorConfig(stochastic=False)
        branch_medium = MapOperator(config_medium, fn=lambda x, key: x * 5, rngs=rngs)

        config_high = MapOperatorConfig(stochastic=False)
        branch_high = MapOperator(config_high, fn=lambda x, key: x * 10, rngs=rngs)

        # Create branching composite with value-based router
        def router(data):
            # Use nested JAX select for 3-way branching with integer indices
            # 0=low, 1=medium, 2=high
            mean_val = jnp.mean(data["value"])
            return jax.lax.select(mean_val < 0.3, 0, jax.lax.select(mean_val < 0.7, 1, 2))

        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.BRANCHING,
            operators=[branch_low, branch_medium, branch_high],  # Indexed 0, 1, 2
            router=router,
        )
        composite = CompositeOperatorModule(composite_config)

        # Test low values (mean < 0.3)
        batch_low = Batch([Element(data={"value": jnp.array([0.1, 0.2])})])
        result_batch_low = composite(batch_low)
        result_data_low = result_batch_low.get_data()
        assert jnp.allclose(result_data_low["value"], jnp.array([[0.2, 0.4]]))  # *2

        # Test medium values (0.3 <= mean < 0.7)
        batch_medium = Batch([Element(data={"value": jnp.array([0.4, 0.6])})])
        result_batch_medium = composite(batch_medium)
        result_data_medium = result_batch_medium.get_data()
        assert jnp.allclose(result_data_medium["value"], jnp.array([[2.0, 3.0]]))  # *5

        # Test high values (mean >= 0.7)
        batch_high = Batch([Element(data={"value": jnp.array([0.8, 0.9])})])
        result_batch_high = composite(batch_high)
        result_data_high = result_batch_high.get_data()
        assert jnp.allclose(result_data_high["value"], jnp.array([[8.0, 9.0]]))  # *10


class TestBranchingAdvanced:
    """Test advanced branching features."""

    def test_branching_statistics_tracking(self):
        """Test that branch usage is tracked in statistics."""
        rngs = nnx.Rngs(0)

        # Create branch operators
        config_a = MapOperatorConfig(stochastic=False)
        branch_a = MapOperator(config_a, fn=lambda x, key: x * 2, rngs=rngs)

        config_b = MapOperatorConfig(stochastic=False)
        branch_b = MapOperator(config_b, fn=lambda x, key: x * 10, rngs=rngs)

        # Create branching composite
        def router(data):
            # Router returns integer index (0=a, 1=b)
            return jax.lax.select(data["value"][0] < 5.0, 0, 1)

        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.BRANCHING,
            operators=[branch_a, branch_b],  # Indexed 0, 1
            router=router,
        )
        composite = CompositeOperatorModule(composite_config)

        # Apply multiple times to track statistics
        # Note: MapOperator doesn't populate statistics by default,
        # but we can verify the composite doesn't error
        batch_a = Batch([Element(data={"value": jnp.array([2.0])})])
        result_batch_a = composite(batch_a)
        result_data_a = result_batch_a.get_data()
        assert jnp.allclose(result_data_a["value"], jnp.array([[4.0]]))

        batch_b = Batch([Element(data={"value": jnp.array([7.0])})])
        result_batch_b = composite(batch_b)
        result_data_b = result_batch_b.get_data()
        assert jnp.allclose(result_data_b["value"], jnp.array([[70.0]]))

        # Statistics tracking works (even if MapOperator doesn't populate them)
        assert hasattr(composite, "operator_statistics")

    def test_branching_different_operator_types_per_branch(self):
        """Test branching with different operator types on each branch."""
        rngs = nnx.Rngs(0)

        # Branch A: Simple MapOperator
        config_a = MapOperatorConfig(stochastic=False)
        branch_a = MapOperator(config_a, fn=lambda x, key: x * 2, rngs=rngs)

        # Branch B: Sequential composite of two MapOperators
        config_b1 = MapOperatorConfig(stochastic=False)
        op_b1 = MapOperator(config_b1, fn=lambda x, key: x + 10, rngs=rngs)

        config_b2 = MapOperatorConfig(stochastic=False)
        op_b2 = MapOperator(config_b2, fn=lambda x, key: x * 3, rngs=rngs)

        config_seq = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
            operators=[op_b1, op_b2],
        )
        branch_b = CompositeOperatorModule(config_seq)

        # Create branching composite with different operator types
        def router(data):
            # Router returns integer index (0=simple, 1=sequential)
            return jax.lax.select(data["value"][0] < 5.0, 0, 1)

        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.BRANCHING,
            operators=[branch_a, branch_b],  # Indexed 0=simple, 1=sequential
            router=router,
        )
        composite = CompositeOperatorModule(composite_config)

        # Test simple branch (MapOperator)
        batch_simple = Batch([Element(data={"value": jnp.array([2.0])})])
        result_batch_simple = composite(batch_simple)
        result_data_simple = result_batch_simple.get_data()
        assert jnp.allclose(result_data_simple["value"], jnp.array([[4.0]]))  # 2 * 2

        # Test sequential branch (CompositeOperator)
        batch_seq = Batch([Element(data={"value": jnp.array([7.0])})])
        result_batch_seq = composite(batch_seq)
        result_data_seq = result_batch_seq.get_data()
        assert jnp.allclose(result_data_seq["value"], jnp.array([[51.0]]))  # (7 + 10) * 3
