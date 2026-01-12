"""Tests for SelectorOperator - Random selection from multiple operators.

This module tests the SelectorOperator, which wraps multiple operators and
randomly selects ONE to apply (with optional weights).

Test Coverage:
- Config validation (empty operators, weight mismatch, normalization)
- Basic selection (uniform, weighted)
- JAX compatibility (JIT, vmap)
- Edge cases (single operator, extreme weights)
- Stochastic behavior (different keys, reproducibility)
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from datarax.operators.selector_operator import (
    SelectorOperator,
    SelectorOperatorConfig,
)
from datarax.operators.map_operator import MapOperator, MapOperatorConfig
from datarax.core.element_batch import Batch, Element


class TestSelectorOperatorConfig:
    """Test SelectorOperatorConfig validation and initialization."""

    def test_config_requires_at_least_one_operator(self):
        """Empty operators list should raise ValueError."""
        with pytest.raises(ValueError, match="at least one operator"):
            SelectorOperatorConfig(operators=[])

    def test_config_weights_must_match_operators_count(self):
        """Weights list length must match operators count."""
        rngs = nnx.Rngs(0)
        config = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config, fn=lambda x, key: x * 2, rngs=rngs)
        op2 = MapOperator(config, fn=lambda x, key: x * 3, rngs=rngs)

        with pytest.raises(ValueError, match="must match"):
            SelectorOperatorConfig(operators=[op1, op2], weights=[0.5])

    def test_config_normalizes_weights(self):
        """Weights should be normalized to sum to 1.0."""
        rngs = nnx.Rngs(0)
        config = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config, fn=lambda x, key: x * 2, rngs=rngs)
        op2 = MapOperator(config, fn=lambda x, key: x * 3, rngs=rngs)

        selector_config = SelectorOperatorConfig(operators=[op1, op2], weights=[2.0, 8.0])

        # Weights should be normalized to [0.2, 0.8]
        assert jnp.allclose(selector_config.normalized_weights, jnp.array([0.2, 0.8]))

    def test_config_uniform_weights_by_default(self):
        """Default weights should be uniform."""
        rngs = nnx.Rngs(0)
        config = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config, fn=lambda x, key: x * 2, rngs=rngs)
        op2 = MapOperator(config, fn=lambda x, key: x * 3, rngs=rngs)

        selector_config = SelectorOperatorConfig(operators=[op1, op2])

        # Uniform weights: [0.5, 0.5]
        assert jnp.allclose(selector_config.normalized_weights, jnp.array([0.5, 0.5]))

    def test_config_is_always_stochastic(self):
        """Verify the operator is always stochastic (always makes random choice)."""
        rngs = nnx.Rngs(0)
        config = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config, fn=lambda x, key: x * 2, rngs=rngs)

        selector_config = SelectorOperatorConfig(operators=[op1])
        assert selector_config.stochastic is True


class TestSelectorOperatorInit:
    """Test SelectorOperator initialization."""

    def test_init_with_multiple_operators(self):
        """Initialize selector with multiple operators."""
        rngs = nnx.Rngs(0)
        config = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config, fn=lambda x, key: x * 2, rngs=rngs)
        op2 = MapOperator(config, fn=lambda x, key: x * 3, rngs=rngs)
        op3 = MapOperator(config, fn=lambda x, key: x * 4, rngs=rngs)

        selector_config = SelectorOperatorConfig(operators=[op1, op2, op3])
        selector = SelectorOperator(selector_config, rngs=rngs)

        assert len(selector.operators) == 3
        assert selector.config.stochastic is True

    def test_init_with_single_operator(self):
        """Single operator selector should work (always selects the one)."""
        rngs = nnx.Rngs(0)
        config = MapOperatorConfig(stochastic=False)
        op = MapOperator(config, fn=lambda x, key: x * 2, rngs=rngs)

        selector_config = SelectorOperatorConfig(operators=[op])
        selector = SelectorOperator(selector_config, rngs=rngs)

        assert len(selector.operators) == 1

    def test_init_with_custom_weights(self):
        """Initialize with custom weights."""
        rngs = nnx.Rngs(0)
        config = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config, fn=lambda x, key: x + 1, rngs=rngs)
        op2 = MapOperator(config, fn=lambda x, key: x + 10, rngs=rngs)

        selector_config = SelectorOperatorConfig(operators=[op1, op2], weights=[0.9, 0.1])
        selector = SelectorOperator(selector_config, rngs=rngs)

        assert jnp.allclose(selector.weights, jnp.array([0.9, 0.1]))


class TestSelectorOperatorBasic:
    """Test basic SelectorOperator functionality."""

    def test_selects_one_operator_to_apply(self):
        """Selector should apply exactly one operator from the list."""
        rngs = nnx.Rngs(0)
        config = MapOperatorConfig(stochastic=False)
        # Operators that add 1, 10, or 100
        op1 = MapOperator(config, fn=lambda x, key: x + 1, rngs=rngs)
        op2 = MapOperator(config, fn=lambda x, key: x + 10, rngs=rngs)
        op3 = MapOperator(config, fn=lambda x, key: x + 100, rngs=rngs)

        selector_config = SelectorOperatorConfig(operators=[op1, op2, op3])
        selector = SelectorOperator(selector_config, rngs=rngs)

        batch = Batch([Element(data={"value": jnp.array([5.0])})])
        result_batch = selector(batch)
        result_value = float(result_batch.get_data()["value"][0, 0])

        # Should be exactly one of: 6 (5+1), 15 (5+10), 105 (5+100)
        assert result_value in [6.0, 15.0, 105.0]

    def test_uniform_selection_distribution(self):
        """With uniform weights, all operators should be selected roughly equally."""
        rngs = nnx.Rngs(42)
        config = MapOperatorConfig(stochastic=False)
        # Operators that add identifiable values
        op1 = MapOperator(config, fn=lambda x, key: x + 1, rngs=rngs)
        op2 = MapOperator(config, fn=lambda x, key: x + 10, rngs=rngs)
        op3 = MapOperator(config, fn=lambda x, key: x + 100, rngs=rngs)

        selector_config = SelectorOperatorConfig(operators=[op1, op2, op3])
        selector = SelectorOperator(selector_config, rngs=nnx.Rngs(0))

        # Run many times and count selections
        counts = {1.0: 0, 10.0: 0, 100.0: 0}
        for i in range(300):
            # Create new selector with different seed each time
            selector = SelectorOperator(selector_config, rngs=nnx.Rngs(i))
            batch = Batch([Element(data={"value": jnp.array([0.0])})])
            result_batch = selector(batch)
            result_value = float(result_batch.get_data()["value"][0, 0])
            counts[result_value] += 1

        # Each should be selected roughly 100 times (±50 for statistical variance)
        for value, count in counts.items():
            assert 50 < count < 150, f"Selection of +{value} is {count}, expected ~100"

    def test_weighted_selection_distribution(self):
        """With skewed weights, selection should follow the distribution."""
        config = MapOperatorConfig(stochastic=False)
        rngs = nnx.Rngs(0)
        # Operators that add identifiable values
        op1 = MapOperator(config, fn=lambda x, key: x + 1, rngs=rngs)
        op2 = MapOperator(config, fn=lambda x, key: x + 10, rngs=rngs)

        # 90% weight on op1, 10% on op2
        selector_config = SelectorOperatorConfig(operators=[op1, op2], weights=[0.9, 0.1])

        # Run many times and count selections
        counts = {1.0: 0, 10.0: 0}
        for i in range(1000):
            selector = SelectorOperator(selector_config, rngs=nnx.Rngs(i))
            batch = Batch([Element(data={"value": jnp.array([0.0])})])
            result_batch = selector(batch)
            result_value = float(result_batch.get_data()["value"][0, 0])
            counts[result_value] += 1

        # op1 (+1) should be selected ~90% of the time
        ratio = counts[1.0] / 1000
        assert 0.85 < ratio < 0.95, f"Weight distribution incorrect: {ratio}"


class TestSelectorOperatorEdgeCases:
    """Test edge cases for SelectorOperator."""

    def test_single_operator_always_selected(self):
        """Single operator should always be selected."""
        config = MapOperatorConfig(stochastic=False)
        rngs = nnx.Rngs(0)
        op = MapOperator(config, fn=lambda x, key: x * 3, rngs=rngs)

        selector_config = SelectorOperatorConfig(operators=[op])
        selector = SelectorOperator(selector_config, rngs=rngs)

        batch = Batch([Element(data={"value": jnp.array([5.0])})])
        result_batch = selector(batch)
        result_value = float(result_batch.get_data()["value"][0, 0])

        assert result_value == 15.0  # 5 * 3

    def test_zero_weight_operator_never_selected(self):
        """Operator with weight 0 should never be selected."""
        config = MapOperatorConfig(stochastic=False)
        rngs = nnx.Rngs(0)
        op1 = MapOperator(config, fn=lambda x, key: x + 1, rngs=rngs)
        op2 = MapOperator(config, fn=lambda x, key: x + 100, rngs=rngs)

        # Weight 1.0 on op1, 0 on op2 - should always select op1
        selector_config = SelectorOperatorConfig(operators=[op1, op2], weights=[1.0, 0.0])

        # Run multiple times
        for i in range(50):
            selector = SelectorOperator(selector_config, rngs=nnx.Rngs(i))
            batch = Batch([Element(data={"value": jnp.array([0.0])})])
            result_batch = selector(batch)
            result_value = float(result_batch.get_data()["value"][0, 0])
            assert result_value == 1.0, "Op2 was selected unexpectedly"


class TestSelectorOperatorStochastic:
    """Test stochastic behavior of SelectorOperator."""

    def test_different_keys_produce_different_selections(self):
        """Different RNG keys should produce different selections over many runs."""
        config = MapOperatorConfig(stochastic=False)
        rngs = nnx.Rngs(0)
        op1 = MapOperator(config, fn=lambda x, key: x + 1, rngs=rngs)
        op2 = MapOperator(config, fn=lambda x, key: x + 10, rngs=rngs)
        op3 = MapOperator(config, fn=lambda x, key: x + 100, rngs=rngs)

        selector_config = SelectorOperatorConfig(operators=[op1, op2, op3])

        values_seen = set()
        for i in range(50):
            selector = SelectorOperator(selector_config, rngs=nnx.Rngs(i))
            batch = Batch([Element(data={"value": jnp.array([0.0])})])
            result_batch = selector(batch)
            result_value = float(result_batch.get_data()["value"][0, 0])
            values_seen.add(result_value)

        # Should see multiple different selections
        assert len(values_seen) > 1, "Different keys should produce different results"

    def test_same_key_produces_same_selection(self):
        """Same RNG key should produce the same selection."""
        config = MapOperatorConfig(stochastic=False)
        rngs = nnx.Rngs(0)
        op1 = MapOperator(config, fn=lambda x, key: x + 1, rngs=rngs)
        op2 = MapOperator(config, fn=lambda x, key: x + 10, rngs=rngs)

        selector_config = SelectorOperatorConfig(operators=[op1, op2])

        # Same seed should produce same result
        selector1 = SelectorOperator(selector_config, rngs=nnx.Rngs(42))
        selector2 = SelectorOperator(selector_config, rngs=nnx.Rngs(42))

        batch = Batch([Element(data={"value": jnp.array([0.0])})])

        result1 = selector1(batch)
        result2 = selector2(batch)

        assert jnp.allclose(result1.get_data()["value"], result2.get_data()["value"])


class TestSelectorOperatorJAX:
    """Test JAX compatibility of SelectorOperator."""

    def test_jit_compilation(self):
        """Verify the operator works with JIT compilation."""
        config = MapOperatorConfig(stochastic=False)
        rngs = nnx.Rngs(0)
        op1 = MapOperator(config, fn=lambda x, key: x + 1, rngs=rngs)
        op2 = MapOperator(config, fn=lambda x, key: x + 10, rngs=rngs)

        selector_config = SelectorOperatorConfig(operators=[op1, op2])
        selector = SelectorOperator(selector_config, rngs=rngs)

        @nnx.jit
        def apply_selector(model, batch):
            return model(batch)

        batch = Batch([Element(data={"value": jnp.array([5.0])})])
        result_batch = apply_selector(selector, batch)
        result_value = float(result_batch.get_data()["value"][0, 0])

        # Should be either 6 (5+1) or 15 (5+10)
        assert result_value in [6.0, 15.0]

    def test_jit_preserves_randomness(self):
        """JIT compilation should preserve randomness across calls."""
        config = MapOperatorConfig(stochastic=False)
        rngs = nnx.Rngs(0)
        op1 = MapOperator(config, fn=lambda x, key: x + 1, rngs=rngs)
        op2 = MapOperator(config, fn=lambda x, key: x + 10, rngs=rngs)
        op3 = MapOperator(config, fn=lambda x, key: x + 100, rngs=rngs)

        selector_config = SelectorOperatorConfig(operators=[op1, op2, op3])

        @nnx.jit
        def apply_selector(model, batch):
            return model(batch)

        batch = Batch([Element(data={"value": jnp.array([0.0])})])

        values_seen = set()
        for i in range(50):
            selector = SelectorOperator(selector_config, rngs=nnx.Rngs(i))
            result_batch = apply_selector(selector, batch)
            result_value = float(result_batch.get_data()["value"][0, 0])
            values_seen.add(result_value)

        # Should see multiple different selections
        assert len(values_seen) > 1, "JIT should preserve randomness"

    def test_vmap_compatibility(self):
        """Verify the operator works with vmap for batch processing."""
        config = MapOperatorConfig(stochastic=False)
        rngs = nnx.Rngs(0)
        op1 = MapOperator(config, fn=lambda x, key: x + 1, rngs=rngs)
        op2 = MapOperator(config, fn=lambda x, key: x + 10, rngs=rngs)

        selector_config = SelectorOperatorConfig(operators=[op1, op2])
        selector = SelectorOperator(selector_config, rngs=rngs)

        # Multi-element batch
        batch = Batch(
            [
                Element(data={"value": jnp.array([1.0])}),
                Element(data={"value": jnp.array([2.0])}),
                Element(data={"value": jnp.array([3.0])}),
            ]
        )

        result_batch = selector(batch)
        result_data = result_batch.get_data()

        # Each element should have been transformed
        assert result_data["value"].shape == (3, 1)
        # Values should be either original+1 or original+10
        for i in range(3):
            val = float(result_data["value"][i, 0])
            original = float(i + 1)
            assert val in [original + 1, original + 10]
