"""Current Flax NNX variable access contracts."""

import jax.numpy as jnp
import pytest
from flax import nnx

from datarax.core.element_batch import Batch, Element
from datarax.operators.composite_operator import (
    CompositeOperatorConfig,
    CompositeOperatorModule,
    CompositionStrategy,
)
from datarax.operators.map_operator import MapOperator, MapOperatorConfig


def test_learnable_weighted_parallel_uses_param_indexing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Array-backed NNX params should be read with ``param[...]``."""
    rngs = nnx.Rngs(0)
    op1 = MapOperator(MapOperatorConfig(stochastic=False), fn=lambda x, _key: x * 2, rngs=rngs)
    op2 = MapOperator(MapOperatorConfig(stochastic=False), fn=lambda x, _key: x * 3, rngs=rngs)
    composite = CompositeOperatorModule(
        CompositeOperatorConfig(
            strategy=CompositionStrategy.WEIGHTED_PARALLEL,
            operators=[op1, op2],
            weights=[0.7, 0.3],
            learnable_weights=True,
        )
    )

    original_get_value = nnx.Param.get_value

    def fail_direct_get_value(self: nnx.Param, *args, **kwargs) -> object:
        if "index" not in kwargs:
            raise AssertionError("array-backed Param.get_value() should not be used")
        return original_get_value(self, *args, **kwargs)

    monkeypatch.setattr(nnx.Param, "get_value", fail_direct_get_value)

    batch = Batch([Element(data={"value": jnp.array([10.0])})])
    result = composite(batch)

    assert jnp.allclose(result.get_data()["value"], jnp.array([[23.0]]))
