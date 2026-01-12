"""Tests for ProbabilisticOperator - Wrapper for probability-based operator application.

This module tests the ProbabilisticOperator which wraps any OperatorModule and applies
it with a configured probability.

Test Coverage:
- Config validation (probability range 0-1)
- Probabilistic application behavior (p=0.0, p=0.5, p=1.0)
- Stochastic mode with random parameter generation
- JAX compatibility (jit, vmap)
- Child operator delegation
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

# TDD RED phase - imports will fail until implementation exists
from datarax.operators.probabilistic_operator import (
    ProbabilisticOperator,
    ProbabilisticOperatorConfig,
)
from datarax.operators.map_operator import MapOperator, MapOperatorConfig
from datarax.core.element_batch import Batch, Element


class TestProbabilisticOperatorConfig:
    """Test configuration validation for ProbabilisticOperator."""

    def test_valid_probability_values(self):
        """Test that valid probability values are accepted."""
        # Create a simple child operator config
        child_config = MapOperatorConfig(stochastic=False)
        child_op = MapOperator(child_config, fn=lambda x, key: x * 2, rngs=nnx.Rngs(0))

        # Test boundary values
        config_zero = ProbabilisticOperatorConfig(operator=child_op, probability=0.0)
        assert config_zero.probability == 0.0

        config_half = ProbabilisticOperatorConfig(operator=child_op, probability=0.5)
        assert config_half.probability == 0.5

        config_one = ProbabilisticOperatorConfig(operator=child_op, probability=1.0)
        assert config_one.probability == 1.0

    def test_invalid_probability_too_high(self):
        """Test that probability > 1.0 raises ValueError."""
        child_config = MapOperatorConfig(stochastic=False)
        child_op = MapOperator(child_config, fn=lambda x, key: x * 2, rngs=nnx.Rngs(0))

        with pytest.raises(ValueError, match="probability must be in \\[0.0, 1.0\\]"):
            ProbabilisticOperatorConfig(operator=child_op, probability=1.5)

    def test_invalid_probability_negative(self):
        """Test that probability < 0.0 raises ValueError."""
        child_config = MapOperatorConfig(stochastic=False)
        child_op = MapOperator(child_config, fn=lambda x, key: x * 2, rngs=nnx.Rngs(0))

        with pytest.raises(ValueError, match="probability must be in \\[0.0, 1.0\\]"):
            ProbabilisticOperatorConfig(operator=child_op, probability=-0.1)

    def test_stochastic_inferred_from_probability(self):
        """Test that stochastic=True when probability is not 0.0 or 1.0."""
        child_config = MapOperatorConfig(stochastic=False)
        child_op = MapOperator(child_config, fn=lambda x, key: x * 2, rngs=nnx.Rngs(0))

        # Stochastic when 0 < p < 1
        config_stochastic = ProbabilisticOperatorConfig(operator=child_op, probability=0.5)
        assert config_stochastic.stochastic is True

        # Deterministic when p = 0.0
        config_never = ProbabilisticOperatorConfig(operator=child_op, probability=0.0)
        assert config_never.stochastic is False

        # Deterministic when p = 1.0
        config_always = ProbabilisticOperatorConfig(operator=child_op, probability=1.0)
        assert config_always.stochastic is False


class TestProbabilisticOperatorApplication:
    """Test probabilistic application behavior."""

    def test_never_apply_p_zero(self):
        """Test that p=0.0 never applies the operator."""
        rngs = nnx.Rngs(0)

        # Child operator that multiplies by 10
        child_config = MapOperatorConfig(stochastic=False)
        child_op = MapOperator(child_config, fn=lambda x, key: x * 10, rngs=rngs)

        # Probabilistic wrapper with p=0.0
        prob_config = ProbabilisticOperatorConfig(operator=child_op, probability=0.0)
        prob_op = ProbabilisticOperator(prob_config, rngs=rngs)

        # Create batch
        batch = Batch(
            [
                Element(data={"value": jnp.array([1.0])}),
                Element(data={"value": jnp.array([2.0])}),
                Element(data={"value": jnp.array([3.0])}),
            ]
        )

        # Apply - should NOT transform (p=0.0)
        result_batch = prob_op(batch)
        result_data = result_batch.get_data()

        # Values should be unchanged
        expected = jnp.array([[1.0], [2.0], [3.0]])
        assert jnp.allclose(result_data["value"], expected)

    def test_always_apply_p_one(self):
        """Test that p=1.0 always applies the operator."""
        rngs = nnx.Rngs(0)

        # Child operator that multiplies by 10
        child_config = MapOperatorConfig(stochastic=False)
        child_op = MapOperator(child_config, fn=lambda x, key: x * 10, rngs=rngs)

        # Probabilistic wrapper with p=1.0
        prob_config = ProbabilisticOperatorConfig(operator=child_op, probability=1.0)
        prob_op = ProbabilisticOperator(prob_config, rngs=rngs)

        # Create batch
        batch = Batch(
            [
                Element(data={"value": jnp.array([1.0])}),
                Element(data={"value": jnp.array([2.0])}),
                Element(data={"value": jnp.array([3.0])}),
            ]
        )

        # Apply - should ALWAYS transform (p=1.0)
        result_batch = prob_op(batch)
        result_data = result_batch.get_data()

        # Values should all be multiplied by 10
        expected = jnp.array([[10.0], [20.0], [30.0]])
        assert jnp.allclose(result_data["value"], expected)

    def test_probabilistic_application_p_half(self):
        """Test that p=0.5 applies operator ~50% of the time."""
        rngs = nnx.Rngs(42)  # Fixed seed for reproducibility

        # Child operator that adds 100
        child_config = MapOperatorConfig(stochastic=False)
        child_op = MapOperator(child_config, fn=lambda x, key: x + 100, rngs=rngs)

        # Probabilistic wrapper with p=0.5
        prob_config = ProbabilisticOperatorConfig(operator=child_op, probability=0.5)
        prob_op = ProbabilisticOperator(prob_config, rngs=nnx.Rngs(42))

        # Run 100 trials
        n_trials = 100
        applied_count = 0

        for i in range(n_trials):
            # Create single element batch
            batch = Batch([Element(data={"value": jnp.array([1.0])})])

            # Apply probabilistic operator
            result_batch = prob_op(batch)
            result_value = float(result_batch.get_data()["value"][0, 0])

            # Check if operator was applied (value == 101.0) or not (value == 1.0)
            if jnp.isclose(result_value, 101.0):
                applied_count += 1

        # Should be roughly 50% (allow 40-60% range for randomness)
        assert 30 < applied_count < 70, f"Applied {applied_count}/{n_trials} times (expected ~50)"


class TestProbabilisticOperatorStochastic:
    """Test stochastic mode and random parameter generation."""

    def test_generate_random_params_structure(self):
        """Test that random parameters are boolean decisions per element."""
        rngs = nnx.Rngs(0)

        # Child operator
        child_config = MapOperatorConfig(stochastic=False)
        child_op = MapOperator(child_config, fn=lambda x, key: x * 2, rngs=rngs)

        # Probabilistic wrapper with p=0.5
        prob_config = ProbabilisticOperatorConfig(operator=child_op, probability=0.5)
        prob_op = ProbabilisticOperator(prob_config, rngs=rngs)

        # Generate random params for batch of 3
        rng = jax.random.key(0)
        data_shapes = {"value": (3, 1)}  # 3 elements, shape (1,)

        random_params = prob_op.generate_random_params(rng, data_shapes)

        # Should contain boolean array of shape (3,)
        assert "apply_mask" in random_params
        assert random_params["apply_mask"].shape == (3,)
        assert random_params["apply_mask"].dtype == jnp.bool_

    def test_random_params_distribution(self):
        """Test that random params follow probability distribution."""
        rngs = nnx.Rngs(42)

        # Child operator
        child_config = MapOperatorConfig(stochastic=False)
        child_op = MapOperator(child_config, fn=lambda x, key: x * 2, rngs=rngs)

        # Probabilistic wrapper with p=0.3
        prob_config = ProbabilisticOperatorConfig(operator=child_op, probability=0.3)
        prob_op = ProbabilisticOperator(prob_config, rngs=rngs)

        # Generate many random params
        n_samples = 1000
        true_count = 0

        for i in range(n_samples):
            rng = jax.random.key(i)
            data_shapes = {"value": (1, 1)}  # Single element

            random_params = prob_op.generate_random_params(rng, data_shapes)
            if random_params["apply_mask"][0]:
                true_count += 1

        # Should be roughly 30% (allow 25-35% range)
        ratio = true_count / n_samples
        assert 0.25 < ratio < 0.35, f"Apply ratio: {ratio} (expected ~0.3)"


class TestProbabilisticOperatorJAXCompatibility:
    """Test JAX compatibility (jit, vmap, grad)."""

    def test_jit_compatibility(self):
        """Test that ProbabilisticOperator works with jax.jit."""
        rngs = nnx.Rngs(0)

        # Child operator
        child_config = MapOperatorConfig(stochastic=False)
        child_op = MapOperator(child_config, fn=lambda x, key: x * 2, rngs=rngs)

        # Probabilistic wrapper
        prob_config = ProbabilisticOperatorConfig(operator=child_op, probability=1.0)
        prob_op = ProbabilisticOperator(prob_config, rngs=rngs)

        # JIT compile
        @nnx.jit
        def apply_jitted(op, batch):
            return op(batch)

        # Create batch
        batch = Batch([Element(data={"value": jnp.array([5.0])})])

        # Apply jitted function
        result_batch = apply_jitted(prob_op, batch)
        result_value = float(result_batch.get_data()["value"][0, 0])

        assert jnp.isclose(result_value, 10.0)

    def test_vmap_compatibility(self):
        """Test that operator works correctly with vmap (batch processing)."""
        rngs = nnx.Rngs(0)

        # Child operator that adds 10
        child_config = MapOperatorConfig(stochastic=False)
        child_op = MapOperator(child_config, fn=lambda x, key: x + 10, rngs=rngs)

        # Probabilistic wrapper with p=1.0 (always apply for deterministic test)
        prob_config = ProbabilisticOperatorConfig(operator=child_op, probability=1.0)
        prob_op = ProbabilisticOperator(prob_config, rngs=rngs)

        # Create batch with multiple elements
        batch = Batch(
            [
                Element(data={"value": jnp.array([1.0])}),
                Element(data={"value": jnp.array([2.0])}),
                Element(data={"value": jnp.array([3.0])}),
            ]
        )

        # Apply operator (uses vmap internally via apply_batch)
        result_batch = prob_op(batch)
        result_data = result_batch.get_data()

        # All values should have +10
        expected = jnp.array([[11.0], [12.0], [13.0]])
        assert jnp.allclose(result_data["value"], expected)


class TestProbabilisticOperatorEdgeCases:
    """Test edge cases and error handling."""

    def test_child_operator_state_passthrough(self):
        """Test that state is correctly passed through to child operator."""
        rngs = nnx.Rngs(0)

        # Child operator (state passthrough)
        child_config = MapOperatorConfig(stochastic=False)
        child_op = MapOperator(child_config, fn=lambda x, key: x * 2, rngs=rngs)

        # Probabilistic wrapper
        prob_config = ProbabilisticOperatorConfig(operator=child_op, probability=1.0)
        prob_op = ProbabilisticOperator(prob_config, rngs=rngs)

        # Create batch with state
        batch = Batch(
            [
                Element(data={"value": jnp.array([1.0])}, state={"counter": 10}),
                Element(data={"value": jnp.array([2.0])}, state={"counter": 20}),
            ]
        )

        # Apply operator
        result_batch = prob_op(batch)

        # State should pass through unchanged (MapOperator doesn't modify state)
        result_states = result_batch.states.get_value()
        assert jnp.array_equal(result_states["counter"], jnp.array([10, 20]))

    def test_child_operator_metadata_passthrough(self):
        """Test that metadata is correctly passed through to child operator."""
        rngs = nnx.Rngs(0)

        # Child operator
        child_config = MapOperatorConfig(stochastic=False)
        child_op = MapOperator(child_config, fn=lambda x, key: x * 2, rngs=rngs)

        # Probabilistic wrapper
        prob_config = ProbabilisticOperatorConfig(operator=child_op, probability=1.0)
        prob_op = ProbabilisticOperator(prob_config, rngs=rngs)

        # Create batch with metadata
        batch = Batch(
            [
                Element(data={"value": jnp.array([1.0])}, metadata={"source": "test1"}),
                Element(data={"value": jnp.array([2.0])}, metadata={"source": "test2"}),
            ]
        )

        # Apply operator
        result_batch = prob_op(batch)

        # Metadata should pass through unchanged
        assert result_batch._metadata_list[0] == {"source": "test1"}
        assert result_batch._metadata_list[1] == {"source": "test2"}
