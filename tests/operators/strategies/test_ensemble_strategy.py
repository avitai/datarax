"""Unit tests for ensemble strategy."""

import jax.numpy as jnp
import pytest
from datarax.operators.strategies.base import StrategyContext
from datarax.operators.strategies.ensemble import EnsembleStrategy
from tests.test_common.mock_operators import ConstantMockOperator as MockOperator


class TestEnsembleStrategy:
    def test_ensemble_mean(self):
        op1 = MockOperator(10.0)
        op2 = MockOperator(20.0)
        strategy = EnsembleStrategy(mode="mean")

        context = StrategyContext(jnp.array([0.0]), {}, {})
        result_data, _, _ = strategy.apply([op1, op2], context)

        # Mean of 10 and 20 is 15
        assert jnp.array_equal(result_data, jnp.array([15.0]))

    def test_ensemble_sum(self):
        op1 = MockOperator(10.0)
        op2 = MockOperator(20.0)
        strategy = EnsembleStrategy(mode="sum")

        context = StrategyContext(jnp.array([0.0]), {}, {})
        result_data, _, _ = strategy.apply([op1, op2], context)

        assert jnp.array_equal(result_data, jnp.array([30.0]))

    def test_ensemble_max(self):
        op1 = MockOperator(10.0)
        op2 = MockOperator(20.0)
        strategy = EnsembleStrategy(mode="max")

        context = StrategyContext(jnp.array([0.0]), {}, {})
        result_data, _, _ = strategy.apply([op1, op2], context)

        assert jnp.array_equal(result_data, jnp.array([20.0]))

    def test_ensemble_min(self):
        op1 = MockOperator(10.0)
        op2 = MockOperator(20.0)
        strategy = EnsembleStrategy(mode="min")

        context = StrategyContext(jnp.array([0.0]), {}, {})
        result_data, _, _ = strategy.apply([op1, op2], context)

        assert jnp.array_equal(result_data, jnp.array([10.0]))

    def test_invalid_mode(self):
        strategy = EnsembleStrategy(mode="invalid")
        context = StrategyContext(jnp.array([0.0]), {}, {})
        op1 = MockOperator(10.0)

        with pytest.raises(ValueError, match="Unknown ensemble mode"):
            strategy.apply([op1], context)
