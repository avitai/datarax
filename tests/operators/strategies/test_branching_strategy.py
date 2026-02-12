"""Unit tests for branching strategy."""

import jax
import jax.numpy as jnp
from datarax.operators.strategies.base import StrategyContext
from datarax.operators.strategies.branching import BranchingStrategy
from tests.test_common.mock_operators import ConstantMockOperator as MockOperator


class TestBranchingStrategy:
    def test_branching_selection(self):
        op1 = MockOperator(1.0)
        op2 = MockOperator(2.0)

        # Router that checks value
        def router(data):
            return jax.lax.select(data[0] < 5.0, 0, 1)

        strategy = BranchingStrategy(router=router)

        # Case 1: Value < 5 -> select op1 (value 1.0)
        context1 = StrategyContext(jnp.array([1.0]), {}, {})
        res1, _, _ = strategy.apply([op1, op2], context1)
        assert res1[0] == 1.0

        # Case 2: Value >= 5 -> select op2 (value 2.0)
        context2 = StrategyContext(jnp.array([10.0]), {}, {})
        res2, _, _ = strategy.apply([op1, op2], context2)
        assert res2[0] == 2.0

    def test_branching_out_of_bounds(self):
        # NOTE: jax.lax.switch clamps index to valid range in some backends,
        # or behaviour is undefined.
        # JAX docs say: "If index is out of bounds, the last branch is taken."
        # Let's verify this behavior or at least ensuring it doesn't crash Python.

        op1 = MockOperator(1.0)
        op2 = MockOperator(2.0)

        def router_invalid(data):
            return 10  # Out of bounds

        strategy = BranchingStrategy(router=router_invalid)
        context = StrategyContext(jnp.array([0.0]), {}, {})

        res, _, _ = strategy.apply([op1, op2], context)

        # Expected: Last branch taken (op2 -> 2.0)
        assert res[0] == 2.0
