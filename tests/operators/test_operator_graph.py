"""Operators as NNX graphs: what a transform compares, and tree mode.

A wrapper's children are graph children, never part of its static configuration, so wrappers
built the same way share one compiled trace and one graph structure; every exported operator
splits in tree mode (flax's ``graph=False``).
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from substrax.testing import TraceCounter

from datarax.core import batch_ops
from datarax.core.config import MapOperatorConfig, OperatorConfig
from datarax.core.operator import OperatorModule
from datarax.operators import MapOperator
from datarax.operators.composite_operator import (
    CompositeOperatorConfig,
    CompositeOperatorModule,
    CompositionStrategy,
)
from datarax.operators.modality.image.brightness_operator import (
    BrightnessOperator,
    BrightnessOperatorConfig,
)
from datarax.operators.modality.image.contrast_operator import (
    ContrastOperator,
    ContrastOperatorConfig,
)
from datarax.operators.probabilistic_operator import (
    ProbabilisticOperator,
    ProbabilisticOperatorConfig,
)
from datarax.operators.selector_operator import SelectorOperator, SelectorOperatorConfig
from datarax.pipeline.dag import name_records


def _rngs() -> nnx.Rngs:
    return nnx.Rngs(augment=0)


def _brightness() -> OperatorModule:
    return BrightnessOperator(
        BrightnessOperatorConfig(
            field_key="image", stochastic=True, stream_name="augment", brightness_range=(-0.1, 0.1)
        ),
        rngs=_rngs(),
    )


def _contrast() -> OperatorModule:
    return ContrastOperator(
        ContrastOperatorConfig(
            field_key="image", stochastic=True, stream_name="augment", contrast_range=(0.8, 1.2)
        ),
        rngs=_rngs(),
    )


def _double(x: jax.Array, key: None) -> jax.Array:
    del key
    return x * 2.0


def _scale() -> OperatorModule:
    return MapOperator(MapOperatorConfig(), fn=_double)


_WRAPPERS: dict[str, Callable[[], OperatorModule]] = {
    "probabilistic": lambda: ProbabilisticOperator(
        ProbabilisticOperatorConfig(probability=0.5), operator=_brightness(), rngs=_rngs()
    ),
    "selector": lambda: SelectorOperator(
        SelectorOperatorConfig(), operators=[_brightness(), _contrast()], rngs=_rngs()
    ),
    "composite sequential": lambda: CompositeOperatorModule(
        CompositeOperatorConfig(
            strategy=CompositionStrategy.SEQUENTIAL, stochastic=True, stream_name="augment"
        ),
        operators=[_scale(), _brightness()],
        rngs=_rngs(),
    ),
    "composite weighted parallel": lambda: CompositeOperatorModule(
        CompositeOperatorConfig(
            strategy=CompositionStrategy.WEIGHTED_PARALLEL,
            weights=[0.3, 0.7],
            learnable_weights=True,
            mix_fields=("image",),
        ),
        operators=[_scale(), _scale()],
    ),
}


def _batch() -> dict[str, jax.Array]:
    return {"image": jnp.linspace(0.2, 0.8, 4 * 8 * 8 * 3).reshape(4, 8, 8, 3)}


@pytest.mark.parametrize("name", sorted(_WRAPPERS))
@pytest.mark.parametrize("graph", [True, False], ids=["graph mode", "tree mode"])
def test_identically_built_wrappers_share_one_trace(name: str, graph: bool) -> None:
    counter = TraceCounter()
    indices = jnp.arange(4, dtype=jnp.uint32)
    apply = nnx.jit(
        counter.wrap(
            lambda op, batch: op(name_records(batch_ops.from_arrays(batch), indices, 0)).data
        ),
        graph=graph,
    )

    with counter.expect(new_traces=1):
        apply(_WRAPPERS[name](), _batch())
    with counter.expect(new_traces=0):
        apply(_WRAPPERS[name](), _batch())
        apply(_WRAPPERS[name](), _batch())


@pytest.mark.parametrize("name", sorted(_WRAPPERS))
def test_identically_built_wrappers_have_equal_hashable_graphdefs(name: str) -> None:
    first, _ = nnx.split(_WRAPPERS[name]())
    second, _ = nnx.split(_WRAPPERS[name]())

    assert hash(first) == hash(second)
    assert first == second


def test_a_composite_applies_the_operator_added_to_it() -> None:
    composite = CompositeOperatorModule(
        CompositeOperatorConfig(strategy=CompositionStrategy.DYNAMIC_SEQUENTIAL),
        operators=[_scale()],
    )
    composite.add_operator(_scale())

    out = composite(batch_ops.from_arrays(_batch()))
    out = out.data

    assert jnp.allclose(out["image"], _batch()["image"] * 4.0)


@pytest.mark.parametrize(
    "build",
    [
        lambda: BrightnessOperator(
            BrightnessOperatorConfig(field_key="image", brightness_delta=0.1)
        ),
        lambda: ContrastOperator(ContrastOperatorConfig(field_key="image", contrast_factor=1.2)),
    ],
    ids=["brightness", "contrast"],
)
def test_a_deterministic_image_operator_needs_no_rngs(build: Callable[[], OperatorModule]) -> None:
    operator = build()

    out = operator(batch_ops.from_arrays(_batch()))
    out = out.data

    assert out["image"].shape == _batch()["image"].shape


def test_every_exported_operator_splits_in_tree_mode() -> None:
    """Flax's planned default refuses shared references; no operator may hold one."""
    operators: list[OperatorModule] = [
        *(build() for build in _WRAPPERS.values()),
        _brightness(),
        _contrast(),
        _scale(),
        OperatorModule(OperatorConfig(stochastic=True, stream_name="augment"), rngs=_rngs()),
    ]

    for operator in operators:
        nnx.split(operator, graph=False)
