"""Shared mock operators for strategy tests.

Eliminates duplicate MockOperator definitions across
test_ensemble_strategy, test_sequential_strategy, and test_parallel_strategy.
"""

import jax
import jax.numpy as jnp
from datarax.core.operator import OperatorModule


class ConstantMockOperator(OperatorModule):
    """Mock operator that returns a constant value (ignoring input).

    Used by ensemble and parallel strategy tests.
    """

    def __init__(self, value: float, name: str = "mock"):
        self.value = value
        self.name = name
        self.statistics = {f"{name}_stat": 1.0}

    def apply(self, data, state, metadata, random_params=None, stats=None):
        return jnp.full_like(data, self.value), state, metadata

    def generate_random_params(self, rng, data_shapes):
        return {}


class MultiplierMockOperator(OperatorModule):
    """Mock operator that multiplies input data by a constant.

    Used by sequential strategy tests. Also tracks state (count)
    and metadata (visited list).
    """

    def __init__(self, multiplier: float = 2.0, name: str = "mock"):
        self.multiplier = multiplier
        self.name = name
        self.statistics = {f"{name}_stat": 1.0}

    def apply(self, data, state, metadata, random_params=None, stats=None):
        new_data = jax.tree.map(lambda x: x * self.multiplier, data)

        new_state = state.copy() if state else {}
        if "count" in new_state:
            new_state["count"] += 1

        new_metadata = metadata.copy()
        new_metadata["visited"] = [*new_metadata.get("visited", []), self.name]

        return new_data, new_state, new_metadata

    def generate_random_params(self, rng, data_shapes):
        return {}
