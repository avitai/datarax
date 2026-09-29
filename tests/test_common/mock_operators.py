"""Shared mock operators for strategy tests.

Eliminates duplicate MockOperator definitions across
test_ensemble_strategy, test_sequential_strategy, and test_parallel_strategy.
"""

from typing import Any

import jax
import jax.numpy as jnp

from datarax.core.config import OperatorConfig
from datarax.core.element_batch import Element
from datarax.core.operator import OperatorModule


class ConstantMockOperator(OperatorModule):
    """Mock operator that returns a constant value (ignoring input).

    Used by ensemble and parallel strategy tests.
    """

    def __init__(self, value: float, name: str = "mock"):
        super().__init__(OperatorConfig(stochastic=False), name=name)
        self.value = value

    def apply(
        self,
        element: Element,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> Element:
        data = element.data
        del key, stats
        return element.replace(data=jnp.full_like(data, self.value))


class MultiplierMockOperator(OperatorModule):
    """Mock operator that multiplies input data by a constant.

    Used by sequential strategy tests. Also increments a ``count`` state entry when the
    record carries one.
    """

    def __init__(self, multiplier: float = 2.0, name: str = "mock"):
        super().__init__(OperatorConfig(stochastic=False), name=name)
        self.multiplier = multiplier

    def apply(
        self,
        element: Element,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> Element:
        data = element.data
        state = element.state
        del key, stats
        new_data = jax.tree.map(lambda x: x * self.multiplier, data)

        new_state = state.copy() if state else {}
        if "count" in new_state:
            new_state["count"] += 1

        return element.replace(data=new_data, state=new_state)
