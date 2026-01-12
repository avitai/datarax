"""Unit tests for parallel strategy."""

from unittest.mock import MagicMock

import jax.numpy as jnp
from datarax.core.operator import OperatorModule
from datarax.operators.strategies.base import StrategyContext
from datarax.operators.strategies.parallel import (
    ParallelStrategy,
    WeightedParallelStrategy,
    ConditionalParallelStrategy,
)


class MockOperator(OperatorModule):
    def __init__(self, value, name="mock"):
        self.value = value
        self.name = name
        self.statistics = {f"{name}_stat": 1.0}

    def apply(self, data, state, metadata, random_params=None):
        # Return constant value based on initialization
        # Ignore input data for simplicity in testing merge
        return jnp.full_like(data, self.value), state, metadata

    def generate_random_params(self, rng, data_shapes):
        return {}


class TestParallelStrategy:
    def test_apply_parallel_concat(self):
        op1 = MockOperator(1.0)
        op2 = MockOperator(2.0)
        strategy = ParallelStrategy(merge_strategy="concat", merge_axis=0)

        data = jnp.array([0.0])  # Shape (1,)
        context = StrategyContext(data, {}, {})

        # op1 returns [1.0], op2 returns [2.0]
        # concat([1.0], [2.0]) -> [1.0, 2.0]
        result_data, _, _ = strategy.apply([op1, op2], context)

        assert jnp.array_equal(result_data, jnp.array([1.0, 2.0]))

    def test_apply_parallel_sum(self):
        op1 = MockOperator(1.0)
        op2 = MockOperator(2.0)
        strategy = ParallelStrategy(merge_strategy="sum")

        data = jnp.array([0.0])
        context = StrategyContext(data, {}, {})

        # sum([1.0], [2.0]) -> [3.0]
        result_data, _, _ = strategy.apply([op1, op2], context)

        assert jnp.array_equal(result_data, jnp.array([3.0]))

    def test_stats_callback(self):
        op1 = MockOperator(1.0, name="op1")
        callback = MagicMock()
        strategy = ParallelStrategy(merge_strategy="sum")

        context = StrategyContext(jnp.array([0]), {}, {}, stats_callback=callback)

        strategy.apply([op1], context)

        # Verify callback was called with index 0 and stats
        callback.assert_called_with(0, {"op1_stat": 1.0})


class TestWeightedParallelStrategy:
    def test_weighted_sum(self):
        op1 = MockOperator(10.0)
        op2 = MockOperator(20.0)
        strategy = WeightedParallelStrategy()

        # Weights: 0.1 and 0.9
        # Result: 10*0.1 + 20*0.9 = 1.0 + 18.0 = 19.0
        weights = jnp.array([0.1, 0.9])

        context = StrategyContext(jnp.array([0.0]), {}, {}, extra_params={"weights": weights})

        result_data, _, _ = strategy.apply([op1, op2], context)

        # Allow small float error
        assert jnp.isclose(result_data[0], 19.0)


class TestConditionalParallelStrategy:
    def test_conditional_parallel_masking(self):
        # op1: 10.0 (Condition True)
        # op2: 20.0 (Condition False)

        op1 = MockOperator(10.0)
        op2 = MockOperator(20.0)

        conditions = [
            lambda x: jnp.array(True),
            lambda x: jnp.array(False),  # Should be masked
        ]

        # Merge strategy sum
        # Ideally: 10.0 + 0.0 (masked) = 10.0
        strategy = ConditionalParallelStrategy(conditions, merge_strategy="sum")

        context = StrategyContext(jnp.array([0.0]), {}, {})

        result_data, _, _ = strategy.apply([op1, op2], context)

        assert jnp.array_equal(result_data, jnp.array([10.0]))
