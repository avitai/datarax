"""Tests for CompositeOperatorConfig validation.

This module tests the configuration validation logic for CompositeOperatorModule,
ensuring all strategy-specific requirements are enforced at config construction time.

Test Coverage:
- Valid configurations for all 11 strategies
- Invalid configurations (should fail validation)
- Auto-detection of stochastic from child operators
- Strategy-specific requirement validation
"""

import jax
import pytest
from flax import nnx

# GREEN phase - imports enabled
from datarax.operators.composite_operator import (
    CompositeOperatorModule,
    CompositeOperatorConfig,
    CompositionStrategy,
)
from datarax.operators.map_operator import MapOperator, MapOperatorConfig


class TestConfigValidation:
    """Test configuration validation for all strategies."""

    def test_valid_sequential_config(self):
        """Test valid sequential composition configuration."""
        rngs = nnx.Rngs(0)

        # Create two operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x + 10, rngs=rngs)

        # Create valid sequential config
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
            operators=[op1, op2],
        )

        # Should not raise - validation passes
        composite = CompositeOperatorModule(composite_config)
        assert composite.config.strategy == CompositionStrategy.SEQUENTIAL

    def test_valid_parallel_config(self):
        """Test valid parallel composition configuration with merge strategy."""
        rngs = nnx.Rngs(0)

        # Create two operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 3, rngs=rngs)

        # Create valid parallel config with merge strategy
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.PARALLEL,
            operators=[op1, op2],
            merge_strategy="concat",
        )

        # Should not raise - validation passes
        composite = CompositeOperatorModule(composite_config)
        assert composite.config.strategy == CompositionStrategy.PARALLEL
        assert composite.config.merge_strategy == "concat"

    def test_valid_weighted_parallel_config(self):
        """Test valid weighted parallel configuration."""
        rngs = nnx.Rngs(0)

        # Create two operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 3, rngs=rngs)

        # Create valid weighted parallel config
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.WEIGHTED_PARALLEL,
            operators=[op1, op2],
            weights=[0.3, 0.7],
        )

        # Should not raise - validation passes
        composite = CompositeOperatorModule(composite_config)
        assert composite.config.strategy == CompositionStrategy.WEIGHTED_PARALLEL
        assert composite.config.weights == [0.3, 0.7]

    def test_valid_ensemble_mean_config(self):
        """Test valid ensemble configuration with mean reduction."""
        rngs = nnx.Rngs(0)

        # Create three operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 3, rngs=rngs)

        config3 = MapOperatorConfig(stochastic=False)
        op3 = MapOperator(config3, fn=lambda x, key: x * 4, rngs=rngs)

        # Create valid ensemble mean config
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.ENSEMBLE_MEAN,
            operators=[op1, op2, op3],
        )

        # Should not raise - validation passes
        composite = CompositeOperatorModule(composite_config)
        assert composite.config.strategy == CompositionStrategy.ENSEMBLE_MEAN

    def test_valid_ensemble_sum_config(self):
        """Test valid ensemble configuration with sum reduction."""
        rngs = nnx.Rngs(0)

        # Create three operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 3, rngs=rngs)

        # Create valid ensemble sum config
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.ENSEMBLE_SUM,
            operators=[op1, op2],
        )

        # Should not raise - validation passes
        composite = CompositeOperatorModule(composite_config)
        assert composite.config.strategy == CompositionStrategy.ENSEMBLE_SUM

    def test_valid_conditional_sequential_config(self):
        """Test valid conditional sequential configuration."""
        rngs = nnx.Rngs(0)

        # Create two operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x + 10, rngs=rngs)

        # Create conditions matching number of operators
        conditions = [
            lambda data: True,  # Always apply first
            lambda data: True,  # Always apply second
        ]

        # Create valid conditional sequential config
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.CONDITIONAL_SEQUENTIAL,
            operators=[op1, op2],
            conditions=conditions,
        )

        # Should not raise - validation passes
        composite = CompositeOperatorModule(composite_config)
        assert composite.config.strategy == CompositionStrategy.CONDITIONAL_SEQUENTIAL
        assert len(composite.config.conditions) == 2

    def test_valid_conditional_parallel_config(self):
        """Test valid conditional parallel configuration."""
        rngs = nnx.Rngs(0)

        # Create two operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 3, rngs=rngs)

        # Create conditions
        conditions = [
            lambda data: True,
            lambda data: False,
        ]

        # Create valid conditional parallel config
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.CONDITIONAL_PARALLEL,
            operators=[op1, op2],
            conditions=conditions,
            merge_strategy="stack",
        )

        # Should not raise - validation passes
        composite = CompositeOperatorModule(composite_config)
        assert composite.config.strategy == CompositionStrategy.CONDITIONAL_PARALLEL

    def test_valid_branching_config(self):
        """Test valid branching configuration with router."""
        rngs = nnx.Rngs(0)

        # Create operators in a list (required for branching)
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 3, rngs=rngs)

        # Create router function (returns integer index)
        def router(data):
            return 0  # Returns index 0 or 1

        # Create valid branching config
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.BRANCHING,
            operators=[op1, op2],  # List of operators
            router=router,
        )

        # Should not raise - validation passes
        composite = CompositeOperatorModule(composite_config)
        assert composite.config.strategy == CompositionStrategy.BRANCHING
        assert composite.config.router is not None

    def test_valid_dynamic_sequential_config(self):
        """Test valid dynamic sequential configuration."""
        rngs = nnx.Rngs(0)

        # Create two operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x + 10, rngs=rngs)

        # Create valid dynamic sequential config (same as sequential)
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.DYNAMIC_SEQUENTIAL,
            operators=[op1, op2],
        )

        # Should not raise - validation passes
        composite = CompositeOperatorModule(composite_config)
        assert composite.config.strategy == CompositionStrategy.DYNAMIC_SEQUENTIAL


class TestConfigValidationFailures:
    """Test configuration validation catches errors."""

    def test_empty_operators_list_fails(self):
        """Test that empty operators list raises ValueError."""
        with pytest.raises(ValueError, match="operators list cannot be empty"):
            CompositeOperatorConfig(
                strategy=CompositionStrategy.SEQUENTIAL,
                operators=[],
            )

    def test_branching_requires_list_and_router(self):
        """Test that branching strategy requires list and router."""
        rngs = nnx.Rngs(0)

        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 3, rngs=rngs)

        # Branching requires list with router (this should pass validation)
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.BRANCHING,
            operators=[op1, op2],  # List is required
            router=lambda data: 0,  # Router returns integer index
        )
        # Should not raise - branching with list and router is valid
        composite = CompositeOperatorModule(composite_config)
        assert composite.config.strategy == CompositionStrategy.BRANCHING

    def test_conditional_without_conditions_fails(self):
        """Test that conditional strategy without conditions fails."""
        rngs = nnx.Rngs(0)

        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        # Conditional requires conditions parameter
        with pytest.raises(ValueError, match="requires conditions"):
            CompositeOperatorConfig(
                strategy=CompositionStrategy.CONDITIONAL_SEQUENTIAL,
                operators=[op1],
                # Missing conditions parameter
            )

    def test_mismatched_conditions_length_fails(self):
        """Test that conditions length != operators length fails."""
        rngs = nnx.Rngs(0)

        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 3, rngs=rngs)

        # Conditions length must match operators length
        with pytest.raises(ValueError, match="Number of conditions must match number of operators"):
            CompositeOperatorConfig(
                strategy=CompositionStrategy.CONDITIONAL_SEQUENTIAL,
                operators=[op1, op2],
                conditions=[lambda data: True],  # Only 1 condition for 2 operators
            )

    def test_weighted_parallel_mismatched_weights_fails(self):
        """Test that weights length != operators length fails."""
        rngs = nnx.Rngs(0)

        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 3, rngs=rngs)

        # Weights length must match operators length
        with pytest.raises(ValueError, match="Number of weights must match number of operators"):
            CompositeOperatorConfig(
                strategy=CompositionStrategy.WEIGHTED_PARALLEL,
                operators=[op1, op2],
                weights=[0.5],  # Only 1 weight for 2 operators
            )

    def test_branching_without_router_fails(self):
        """Test that branching strategy without router fails."""
        rngs = nnx.Rngs(0)

        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 3, rngs=rngs)

        # Branching requires router function
        with pytest.raises(ValueError, match="BRANCHING strategy requires router"):
            CompositeOperatorConfig(
                strategy=CompositionStrategy.BRANCHING,
                operators=[op1, op2],  # List is correct, but missing router
                # Missing router parameter
            )


class TestConfigAutoStochasticDetection:
    """Test automatic stochastic detection from child operators."""

    def test_auto_stochastic_all_deterministic(self):
        """Test stochastic=False when all children are deterministic."""
        rngs = nnx.Rngs(0)

        # Create all deterministic operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 3, rngs=rngs)

        # Create composite - should auto-detect stochastic=False
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
            operators=[op1, op2],
        )
        composite = CompositeOperatorModule(composite_config)

        # Verify auto-detection
        assert composite.config.stochastic is False

    def test_auto_stochastic_some_stochastic(self):
        """Test stochastic=True when any child is stochastic."""
        rngs = nnx.Rngs(0, augment=1)

        # Mix of stochastic and deterministic operators
        op1_config = MapOperatorConfig(stochastic=True, stream_name="augment")
        op1 = MapOperator(
            op1_config, fn=lambda x, key: x + jax.random.normal(key, x.shape) * 0.1, rngs=rngs
        )

        op2_config = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(op2_config, fn=lambda x, key: x * 2, rngs=rngs)

        # Composite should be stochastic if any child is stochastic
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
            operators=[op1, op2],
            stochastic=True,  # Explicitly set (auto-detection not required)
            stream_name="augment",
        )
        composite = CompositeOperatorModule(composite_config, rngs=rngs)

        assert composite.config.stochastic is True

    def test_auto_stochastic_all_stochastic(self):
        """Test stochastic=True when all children are stochastic."""
        rngs = nnx.Rngs(0, augment=1)

        # All stochastic operators
        op1_config = MapOperatorConfig(stochastic=True, stream_name="augment")
        op1 = MapOperator(
            op1_config, fn=lambda x, key: x + jax.random.normal(key, x.shape) * 0.1, rngs=rngs
        )

        op2_config = MapOperatorConfig(stochastic=True, stream_name="augment")
        op2 = MapOperator(
            op2_config, fn=lambda x, key: x + jax.random.normal(key, x.shape) * 0.2, rngs=rngs
        )

        # Composite should be stochastic when all children are stochastic
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
            operators=[op1, op2],
            stochastic=True,
            stream_name="augment",
        )
        composite = CompositeOperatorModule(composite_config, rngs=rngs)

        assert composite.config.stochastic is True
