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
import jax.numpy as jnp
import pytest
from flax import nnx

# GREEN phase - imports enabled
from datarax.core import batch_ops
from datarax.operators.composite_operator import (
    CompositeOperatorConfig,
    CompositeOperatorModule,
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
        op1 = MapOperator(config1, fn=lambda x, _key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, _key: x + 10, rngs=rngs)

        # Create valid sequential config
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL,
        )

        # Should not raise - validation passes
        composite = CompositeOperatorModule(
            composite_config,
            operators=[op1, op2],
        )
        assert composite.config.strategy == CompositionStrategy.SEQUENTIAL

    def test_valid_parallel_config(self):
        """Test valid parallel composition configuration with merge strategy."""
        rngs = nnx.Rngs(0)

        # Create two operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, _key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, _key: x * 3, rngs=rngs)

        # Create valid parallel config with merge strategy
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.PARALLEL,
            merge_strategy="concat",
        )

        # Should not raise - validation passes
        composite = CompositeOperatorModule(
            composite_config,
            operators=[op1, op2],
        )
        assert composite.config.strategy == CompositionStrategy.PARALLEL
        assert composite.config.merge_strategy == "concat"

    def test_valid_weighted_parallel_config(self):
        """Test valid weighted parallel configuration."""
        rngs = nnx.Rngs(0)

        # Create two operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, _key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, _key: x * 3, rngs=rngs)

        # Create valid weighted parallel config
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.WEIGHTED_PARALLEL,
            weights=[0.3, 0.7],
            mix_fields=("value",),
        )

        # Should not raise - validation passes
        composite = CompositeOperatorModule(
            composite_config,
            operators=[op1, op2],
        )
        assert composite.config.strategy == CompositionStrategy.WEIGHTED_PARALLEL
        assert composite.config.weights == (0.3, 0.7)

    def test_valid_ensemble_mean_config(self):
        """Test valid ensemble configuration with mean reduction."""
        rngs = nnx.Rngs(0)

        # Create three operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, _key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, _key: x * 3, rngs=rngs)

        config3 = MapOperatorConfig(stochastic=False)
        op3 = MapOperator(config3, fn=lambda x, _key: x * 4, rngs=rngs)

        # Create valid ensemble mean config
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.ENSEMBLE_MEAN,
        )

        # Should not raise - validation passes
        composite = CompositeOperatorModule(
            composite_config,
            operators=[op1, op2, op3],
        )
        assert composite.config.strategy == CompositionStrategy.ENSEMBLE_MEAN

    def test_valid_ensemble_sum_config(self):
        """Test valid ensemble configuration with sum reduction."""
        rngs = nnx.Rngs(0)

        # Create three operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, _key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, _key: x * 3, rngs=rngs)

        # Create valid ensemble sum config
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.ENSEMBLE_SUM,
        )

        # Should not raise - validation passes
        composite = CompositeOperatorModule(
            composite_config,
            operators=[op1, op2],
        )
        assert composite.config.strategy == CompositionStrategy.ENSEMBLE_SUM

    def test_valid_conditional_sequential_config(self):
        """Test valid conditional sequential configuration."""
        rngs = nnx.Rngs(0)

        # Create two operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, _key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, _key: x + 10, rngs=rngs)

        # Create conditions matching number of operators
        conditions = [
            lambda _data: True,  # Always apply first
            lambda _data: True,  # Always apply second
        ]

        # Create valid conditional sequential config
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.CONDITIONAL_SEQUENTIAL,
            conditions=conditions,
        )

        # Should not raise - validation passes
        composite = CompositeOperatorModule(
            composite_config,
            operators=[op1, op2],
        )
        assert composite.config.strategy == CompositionStrategy.CONDITIONAL_SEQUENTIAL
        assert composite.config.conditions is not None
        assert len(composite.config.conditions) == 2

    def test_valid_conditional_parallel_config(self):
        """Test valid conditional parallel configuration."""
        rngs = nnx.Rngs(0)

        # Create two operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, _key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, _key: x * 3, rngs=rngs)

        # Create conditions
        conditions = [
            lambda _data: True,
            lambda _data: False,
        ]

        # Create valid conditional parallel config
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.CONDITIONAL_PARALLEL,
            conditions=conditions,
            merge_strategy="stack",
        )

        # Should not raise - validation passes
        composite = CompositeOperatorModule(
            composite_config,
            operators=[op1, op2],
        )
        assert composite.config.strategy == CompositionStrategy.CONDITIONAL_PARALLEL

    def test_valid_branching_config(self):
        """Test valid branching configuration with router."""
        rngs = nnx.Rngs(0)

        # Create operators in a list (required for branching)
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, _key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, _key: x * 3, rngs=rngs)

        # Create router function (returns integer index)
        def router(data):
            del data
            return 0  # Returns index 0 or 1

        # Create valid branching config
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.BRANCHING,
            router=router,
        )

        # Should not raise - validation passes
        composite = CompositeOperatorModule(
            composite_config,
            operators=[op1, op2],  # List of operators
        )
        assert composite.config.strategy == CompositionStrategy.BRANCHING
        assert composite.config.router is not None

    def test_valid_dynamic_sequential_config(self):
        """Test valid dynamic sequential configuration."""
        rngs = nnx.Rngs(0)

        # Create two operators
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, _key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, _key: x + 10, rngs=rngs)

        # Create valid dynamic sequential config (same as sequential)
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.DYNAMIC_SEQUENTIAL,
        )

        # Should not raise - validation passes
        composite = CompositeOperatorModule(
            composite_config,
            operators=[op1, op2],
        )
        assert composite.config.strategy == CompositionStrategy.DYNAMIC_SEQUENTIAL


class TestConfigValidationFailures:
    """Test configuration validation catches errors."""

    def test_empty_operators_list_fails(self):
        """Test that empty operators list raises ValueError."""
        with pytest.raises(ValueError, match="operators list cannot be empty"):
            CompositeOperatorModule(
                CompositeOperatorConfig(
                    strategy=CompositionStrategy.SEQUENTIAL,
                ),
                operators=[],
            )

    def test_branching_requires_list_and_router(self):
        """Test that branching strategy requires list and router."""
        rngs = nnx.Rngs(0)

        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, _key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, _key: x * 3, rngs=rngs)

        # Branching requires list with router (this should pass validation)
        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.BRANCHING,
            router=lambda _data: 0,  # Router returns integer index
        )
        # Should not raise - branching with list and router is valid
        composite = CompositeOperatorModule(
            composite_config,
            operators=[op1, op2],  # List is required
        )
        assert composite.config.strategy == CompositionStrategy.BRANCHING

    def test_conditional_without_conditions_fails(self):
        """Test that conditional strategy without conditions fails."""
        rngs = nnx.Rngs(0)

        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, _key: x * 2, rngs=rngs)

        # Conditional requires conditions parameter
        with pytest.raises(ValueError, match="requires conditions"):
            CompositeOperatorModule(
                CompositeOperatorConfig(
                    strategy=CompositionStrategy.CONDITIONAL_SEQUENTIAL,
                ),
                operators=[op1],
                # Missing conditions parameter
            )

    def test_mismatched_conditions_length_fails(self):
        """Test that conditions length != operators length fails."""
        rngs = nnx.Rngs(0)

        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, _key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, _key: x * 3, rngs=rngs)

        # Conditions length must match operators length
        with pytest.raises(ValueError, match="Number of conditions must match number of operators"):
            CompositeOperatorModule(
                CompositeOperatorConfig(
                    strategy=CompositionStrategy.CONDITIONAL_SEQUENTIAL,
                    conditions=[lambda _data: True],  # Only 1 condition for 2 operators
                ),
                operators=[op1, op2],
            )

    def test_weighted_parallel_mismatched_weights_fails(self):
        """Test that weights length != operators length fails."""
        rngs = nnx.Rngs(0)

        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, _key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, _key: x * 3, rngs=rngs)

        # Weights length must match operators length
        with pytest.raises(ValueError, match="Number of weights must match number of operators"):
            CompositeOperatorModule(
                CompositeOperatorConfig(
                    strategy=CompositionStrategy.WEIGHTED_PARALLEL,
                    weights=[0.5],  # Only 1 weight for 2 operators
                    mix_fields=("value",),
                ),
                operators=[op1, op2],
            )

    def test_branching_without_router_fails(self):
        """Test that branching strategy without router fails."""
        rngs = nnx.Rngs(0)

        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, _key: x * 2, rngs=rngs)

        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, _key: x * 3, rngs=rngs)

        # Branching requires router function
        with pytest.raises(ValueError, match="BRANCHING strategy requires router"):
            CompositeOperatorModule(
                CompositeOperatorConfig(
                    strategy=CompositionStrategy.BRANCHING,
                ),
                operators=[op1, op2],  # List is correct, but missing router
                # Missing router parameter
            )


class TestACompositeDrawsNothingItself:
    """A composite is deterministic whatever its children; each child keys its own records."""

    @staticmethod
    def _noisy() -> MapOperator:
        return MapOperator(
            MapOperatorConfig(stochastic=True, stream_name="augment"),
            fn=lambda x, key: x + jax.random.normal(key, x.shape) * 0.1,
            rngs=nnx.Rngs(augment=1),
        )

    def test_a_composite_over_stochastic_children_is_deterministic(self):
        doubled = MapOperator(MapOperatorConfig(stochastic=False), fn=lambda x, _key: x * 2)
        for config in (
            CompositeOperatorConfig(strategy=CompositionStrategy.SEQUENTIAL),
            CompositeOperatorConfig(
                strategy=CompositionStrategy.SEQUENTIAL, stochastic=True, stream_name="augment"
            ),
        ):
            composite = CompositeOperatorModule(config, operators=[self._noisy(), doubled])
            assert composite.config.stochastic is False
            assert composite.config.stream_name is None

    def test_its_stochastic_children_still_draw(self):
        composite = CompositeOperatorModule(
            CompositeOperatorConfig(strategy=CompositionStrategy.SEQUENTIAL),
            operators=[self._noisy()],
        )
        batch = batch_ops.from_arrays({"x": jnp.zeros((4, 3), jnp.float32)})

        assert not jnp.array_equal(composite(batch).data["x"], batch.data["x"])


def test_a_configuration_without_a_strategy_is_refused():
    with pytest.raises(ValueError, match="strategy is required"):
        CompositeOperatorConfig()


def test_a_weighted_parallel_with_no_mix_fields_is_refused():
    with pytest.raises(ValueError, match="mix_fields must name at least one field"):
        CompositeOperatorConfig(strategy=CompositionStrategy.WEIGHTED_PARALLEL, mix_fields=())
