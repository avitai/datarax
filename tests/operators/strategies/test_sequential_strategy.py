"""Unit tests for sequential strategy."""

from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
from datarax.core.operator import OperatorModule
from datarax.operators.strategies.base import StrategyContext
from datarax.operators.strategies.sequential import (
    SequentialStrategy,
    ConditionalSequentialStrategy,
)


class MockOperator(OperatorModule):
    def __init__(self, multiplier=2.0, name="mock"):
        # Minimal init to satisfy base class if needed, or just mock apply
        self.multiplier = multiplier
        self.name = name
        self.statistics = {f"{name}_stat": 1.0}

    def apply(self, data, state, metadata, random_params=None):
        # Multiply data by multiplier
        # Update state by incrementing a counter if present
        new_data = jax.tree.map(lambda x: x * self.multiplier, data)

        new_state = state.copy() if state else {}
        if "count" in new_state:
            new_state["count"] += 1

        new_metadata = metadata.copy()
        new_metadata["visited"] = [*new_metadata.get("visited", []), self.name]

        return new_data, new_state, new_metadata

    def generate_random_params(self, rng, data_shapes):
        return {}


class TestSequentialStrategy:
    def test_apply_chaining(self):
        op1 = MockOperator(multiplier=2.0, name="op1")
        op2 = MockOperator(multiplier=3.0, name="op2")
        strategy = SequentialStrategy()

        data = jnp.array([1.0, 2.0])
        state = {"count": 0}
        metadata = {}
        context = StrategyContext(data, state, metadata)

        # Apply: data * 2 * 3 = data * 6
        result_data, result_state, result_metadata = strategy.apply([op1, op2], context)

        assert jnp.array_equal(result_data, jnp.array([6.0, 12.0]))
        assert result_state["count"] == 2
        assert result_metadata["visited"] == ["op1", "op2"]

    def test_apply_empty_list(self):
        strategy = SequentialStrategy()
        data = jnp.array([1.0])
        context = StrategyContext(data, {}, {})

        result_data, _, _ = strategy.apply([], context)
        assert jnp.array_equal(result_data, data)

    def test_random_params_passing(self):
        # Verify random params are extracted correctly
        op1 = MagicMock(spec=OperatorModule)
        op1.apply.return_value = (jnp.array([1]), {}, {})
        op1.statistics = {}

        strategy = SequentialStrategy()
        context = StrategyContext(
            jnp.array([1]), {}, {}, random_params={"operator_0": {"seed": 42}}
        )

        strategy.apply([op1], context)

        # Check that apply was called with extracted params
        args, _ = op1.apply.call_args
        assert args[3] == {"seed": 42}  # 4th arg is random_params


class TestConditionalSequentialStrategy:
    def test_apply_conditional(self):
        # Use JAX-compatible mock logic as control flow requires tracing
        # We can't use python branching inside jit/vmap, but strategy handles that.
        # But here we are testing strategy logic itself.

        # Op1: adds 10 (always run)
        # Op2: adds 100 (run if data > 5)

        # Simulating simple operators that work with JAX tracing
        class AddOperator(OperatorModule):
            def __init__(self, value):
                self.value = value
                self.statistics = {}

            def apply(self, data, state, meta, rp=None):
                return data + self.value, state, meta

            def generate_random_params(self, rng, shapes):
                return {}

        op1 = AddOperator(10.0)
        op2 = AddOperator(100.0)

        # Condition: data > 5
        # Since Sequential strategy re-assigns data,
        # op2 condition sees output of op1.

        # Input: 1.0 -> Op1 -> 11.0 -> (11 > 5 is True) -> Op2 -> 111.0
        # Input: -20.0 -> Op1 -> -10.0 -> (-10 > 5 is False) -> Op2 skipped -> -10.0

        conditions = [
            lambda x: jnp.array(True),  # Always run op1
            lambda x: jnp.sum(x) > 5.0,  # Run op2 if sum(x) > 5
        ]

        strategy = ConditionalSequentialStrategy(conditions)

        # Case 1: Trigger Op2
        data1 = jnp.array([1.0])
        res1, _, _ = strategy.apply([op1, op2], StrategyContext(data1, {}, {}))
        assert res1[0] == 111.0

        # Case 2: Skip Op2
        data2 = jnp.array([-20.0])
        res2, _, _ = strategy.apply([op1, op2], StrategyContext(data2, {}, {}))
        assert res2[0] == -10.0
