"""Every NNX transform decides a program's reuse by a host holder's own equality.

A holder is a static of the module graph, so each transform compares the graphdefs, and with them
the holders, to decide whether a module may run a program traced for another. Two modules that
differ only in a ``HostProvenance`` (equal by kind) run one traced program; two that differ in a
``HostArray`` (equal only to itself) trace their own. A split and merge or a clone keeps the
holder object itself.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from datarax.core.host_values import HostArray, HostProvenance


class _Traces:
    count = 0


class _Holding(nnx.Module):
    def __init__(self, holder: Any) -> None:
        self.holder = holder
        self.w = nnx.Param(jnp.ones((3,)))

    def __call__(self, x: jax.Array) -> jax.Array:
        _Traces.count += 1
        return (self.w[...] * x).sum()


_X = jnp.ones((3,))
_XS = jnp.ones((2, 3))


def _through_split(jitted: Callable[..., jax.Array]) -> Callable[[_Holding], jax.Array]:
    def run(module: _Holding) -> jax.Array:
        graphdef, state = nnx.split(module)
        return jitted(graphdef, state, _X)

    return run


def _merged_call(graphdef: Any, state: Any, x: jax.Array) -> jax.Array:
    return nnx.merge(graphdef, state)(x)


def _scan(module: _Holding) -> Any:
    def body(module: _Holding, x: jax.Array) -> tuple[_Holding, jax.Array]:
        return module, module(x)

    return nnx.scan(body, in_axes=(nnx.Carry, 0))(module, _XS)


def _transforms() -> dict[str, Callable[[_Holding], Any]]:
    """A fresh program per transform, so each pair starts with an empty cache."""
    graph = nnx.jit(lambda m, x: m(x))
    tree = nnx.jit(lambda m, x: m(x), graph=False)
    return {
        "nnx.jit, graph mode": lambda m: graph(m, _X),
        "nnx.jit, tree mode": lambda m: tree(m, _X),
        "jax.jit over the split": _through_split(jax.jit(_merged_call)),
        "jax.jit, graphdef static": _through_split(jax.jit(_merged_call, static_argnums=0)),
        "nnx.cached_partial": lambda m: nnx.cached_partial(graph, m)(_X),
        "nnx.scan": nnx.jit(_scan),
        "nnx.vmap": nnx.jit(lambda m: nnx.vmap(lambda m, x: m(x), in_axes=(None, 0))(m, _XS)),
        "nnx.grad": nnx.jit(lambda m: nnx.grad(lambda m: m(_X))(m)),
        "nnx.remat": nnx.jit(lambda m: nnx.remat(lambda m, x: m(x))(m, _X)),
    }


def _second_traces(name: str, first: Any, second: Any) -> int:
    """How many times the transform traces ``second``'s module after it ran ``first``'s."""
    jax.clear_caches()
    run = _transforms()[name]
    run(_Holding(first))
    before = _Traces.count
    run(_Holding(second))
    return _Traces.count - before


def _map(path: str) -> HostArray:
    return HostArray(path, np.zeros((4, 3), np.float32))


@pytest.mark.parametrize("name", list(_transforms()))
def test_a_kind_equal_holder_reuses_the_program_another_holder_of_its_kind_traced(
    name: str,
) -> None:
    assert _second_traces(name, HostProvenance(({"id": "a"},)), HostProvenance(({"id": "b"},))) == 0


@pytest.mark.parametrize("name", list(_transforms()))
def test_an_identity_holder_traces_its_own_program(name: str) -> None:
    assert _second_traces(name, _map("a.npy"), _map("a.npy")) >= 1


@pytest.mark.parametrize("name", list(_transforms()))
def test_holders_of_different_kinds_trace_their_own_programs(name: str) -> None:
    assert _second_traces(name, HostProvenance(()), _map("a.npy")) >= 1


@pytest.mark.parametrize("holder", [HostProvenance(()), _map("a.npy")], ids=["kind", "identity"])
@pytest.mark.parametrize("graph", [True, False], ids=["graph mode", "tree mode"])
def test_a_split_and_merge_keeps_the_holder_object(holder: Any, graph: bool) -> None:
    module = _Holding(holder)
    assert nnx.merge(*nnx.split(module, graph=graph)).holder is holder


@pytest.mark.parametrize("holder", [HostProvenance(()), _map("a.npy")], ids=["kind", "identity"])
def test_a_clone_keeps_the_holder_object(holder: Any) -> None:
    assert nnx.clone(_Holding(holder)).holder is holder
