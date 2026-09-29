"""Unit tests for parallel strategy."""

import jax.numpy as jnp

from datarax.core.element_batch import Element
from datarax.operators.strategies.parallel import (
    ConditionalParallelStrategy,
    ParallelStrategy,
    WeightedParallelStrategy,
)
from tests.test_common.mock_operators import (
    ConstantMockOperator as MockOperator,
    MultiplierMockOperator,
)


class TestParallelStrategy:
    def test_apply_parallel_concat(self):
        op1 = MockOperator(1.0)
        op2 = MockOperator(2.0)
        strategy = ParallelStrategy(merge_strategy="concat", merge_axis=0)

        data = jnp.array([0.0])  # Shape (1,)
        context = Element(data)

        # op1 returns [1.0], op2 returns [2.0]
        # concat([1.0], [2.0]) -> [1.0, 2.0]
        result_data = strategy.apply([op1, op2], context, None).data

        assert jnp.array_equal(result_data, jnp.array([1.0, 2.0]))

    def test_apply_parallel_sum(self):
        op1 = MockOperator(1.0)
        op2 = MockOperator(2.0)
        strategy = ParallelStrategy(merge_strategy="sum")

        data = jnp.array([0.0])
        context = Element(data)

        # sum([1.0], [2.0]) -> [3.0]
        result_data = strategy.apply([op1, op2], context, None).data

        assert jnp.array_equal(result_data, jnp.array([3.0]))


class TestWeightedParallelStrategy:
    def test_weighted_sum_of_the_mix_fields(self):
        op1 = MultiplierMockOperator(10.0)
        op2 = MultiplierMockOperator(20.0)
        # value: 0.1 * (1 * 10) + 0.9 * (1 * 20) = 1.0 + 18.0 = 19.0
        strategy = WeightedParallelStrategy(mix_fields=("value",), weights=jnp.array([0.1, 0.9]))
        data = {"value": jnp.array([1.0]), "label": jnp.array(3, dtype=jnp.int32)}

        result_data = strategy.apply([op1, op2], Element(data), None).data

        assert jnp.isclose(result_data["value"][0], 19.0)
        # Both operators scaled the label too, but it is not a mix field: it passes through.
        assert result_data["label"] == 3
        assert result_data["label"].dtype == jnp.int32

    def test_weighted_sum_of_a_nested_mix_field(self):
        strategy = WeightedParallelStrategy(
            mix_fields=("audio.signal",), weights=jnp.array([1.0, 0.1])
        )
        data = {"audio": {"signal": jnp.array([2.0]), "rate": jnp.array(16000)}}

        result_data = strategy.apply(
            [MultiplierMockOperator(1.0), MultiplierMockOperator(10.0)], Element(data), None
        ).data

        # 1.0 * 2 + 0.1 * 20 = 4.0: static weights are a linear combination, not normalized.
        assert jnp.isclose(result_data["audio"]["signal"][0], 4.0)
        assert result_data["audio"]["rate"] == 16000


class TestConditionalParallelStrategy:
    def test_conditional_parallel_masking(self):
        # op1: 10.0 (Condition True)
        # op2: 20.0 (Condition False)

        op1 = MockOperator(10.0)
        op2 = MockOperator(20.0)

        conditions = [
            lambda _x: jnp.array(True),
            lambda _x: jnp.array(False),  # Should be masked
        ]

        # Merge strategy sum
        # Ideally: 10.0 + 0.0 (masked) = 10.0
        strategy = ConditionalParallelStrategy(conditions, merge_strategy="sum")

        context = Element(jnp.array([0.0]))

        result_data = strategy.apply([op1, op2], context, None).data

        assert jnp.array_equal(result_data, jnp.array([10.0]))
