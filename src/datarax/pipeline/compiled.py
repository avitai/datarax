"""Compiled programs shared across pipelines whose structures are equal.

A program traced for one pipeline serves every pipeline built the same way, so a loop that builds
pipelines (an evaluation per epoch, a sweep) compiles once per structure rather than once per
pipeline. The caches are lists matched by equality of their keys (graphs holding lists are
unhashable), held outside the modules: a graphdef stored as a module attribute would be embedded
in the next split's graphdef, making ``GraphDef.__eq__`` recurse into itself.

A cache entry outlives the pipeline that built it, so what an entry keeps is a stand-in of that
pipeline (:func:`stand_in`): a copy holding its structure only, never its records, provenance,
readers or host stage.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.nnx import graphlib

from datarax.core.host_values import HostValue


MAX_COMPILED_PROGRAMS = 16
"""Programs one cache keeps; past it the least recently used is dropped."""


def cached_program[P](
    cache: list[tuple[Any, P]],
    key: Any,
    build: Callable[[], P],
    *,
    stored_key: Callable[[], Any] | None = None,
) -> P:
    """Return the compiled program cached under ``key``, building it on a miss.

    Keys are compared by equality, so structurally identical pipelines share one program. The
    most recently used entry moves last; the oldest is dropped past
    :data:`MAX_COMPILED_PROGRAMS`.

    Args:
        cache: The cache, a list of ``(key, program)`` entries.
        key: What the program was compiled for.
        build: Builds the program (or the programs) on a miss.
        stored_key: Gives the key to keep on a miss, equal to ``key`` but built from a stand-in
            (:func:`stand_in`), so the cache keeps no host value of the pipeline that missed.
            ``key`` is kept when it is not given.

    Returns:
        What ``build`` built for ``key``.
    """
    for index, (cached_key, program) in enumerate(cache):
        if cached_key == key:
            cache.append(cache.pop(index))
            return program
    program = build()
    cache.append((key if stored_key is None else stored_key(), program))
    if len(cache) > MAX_COMPILED_PROGRAMS:
        cache.pop(0)
    return program


def shape_only(leaf: Any) -> Any:
    """An array as its shape and dtype, holding one element; a key array or a scalar as it is."""
    dtype = getattr(leaf, "dtype", None)
    if dtype is None or jnp.issubdtype(dtype, jax.dtypes.extended):
        return leaf
    return np.broadcast_to(np.zeros((), dtype), np.shape(leaf))


def is_static(name: str, value: Any) -> bool:
    """Whether a module attribute is a static of its graph, not state, a node or NNX's own."""
    if name.startswith(("_pytree__", "_object__")):
        return False
    return not isinstance(value, nnx.Variable | np.ndarray | jax.Array) and not graphlib.is_node(
        value
    )


def stand_in[M: nnx.Module](module: M, replace: Callable[[Any], Any | None]) -> M:
    """A copy of ``module`` holding its structure only, for a cached program to keep.

    The copy is merged from a split of ``module`` with every array replaced by a view of its shape
    and dtype holding one element (:func:`shape_only`). Each static of each of its modules for
    which ``replace`` returns a value is then replaced by that value, on the copy. ``module`` is
    never changed, and the statics are read through ``vars``, so building a stand-in inside a
    trace takes no holder off a module.

    Args:
        module: The module to stand in for.
        replace: Gives a static's replacement, or ``None`` to keep it.

    Returns:
        The stand-in.
    """
    graphdef, state = nnx.split(module, graph=True)
    copy = nnx.merge(graphdef, jax.tree.map(shape_only, state))
    for _, node in nnx.iter_graph(copy, graph=True):
        if isinstance(node, nnx.Module):
            for name, value in list(vars(node).items()):
                if is_static(name, value) and (replacement := replace(value)) is not None:
                    setattr(node, name, replacement)
    return copy


def emptied(value: Any) -> HostValue | None:
    """An empty holder of ``value``'s kind for a host value, else ``None`` (kept).

    Holders of one kind compare equal whatever they hold, so a stand-in whose host values are
    emptied has the graph definition of the module it stands in for.

    Args:
        value: A static of a module.

    Returns:
        The empty holder, or ``None``.
    """
    return type(value)(None) if isinstance(value, HostValue) else None


__all__ = [
    "MAX_COMPILED_PROGRAMS",
    "cached_program",
    "emptied",
    "is_static",
    "shape_only",
    "stand_in",
]
