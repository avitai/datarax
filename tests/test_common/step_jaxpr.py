"""The traced program of one pipeline batch, for tests about what the compiled step holds."""

from __future__ import annotations

from collections.abc import Callable

import jax
from flax import nnx
from jax.extend.core import ClosedJaxpr, Jaxpr

from datarax.pipeline import Pipeline
from datarax.pipeline.dag_call import is_per_batch_state, run_tracking_writes, Writes


type _Step = Callable[[nnx.State, nnx.State], tuple[dict, Writes]]


def _one_batch(pipeline: Pipeline) -> tuple[_Step, nnx.State, nnx.State]:
    """The pure function serving one full batch, and the state partitions it takes."""
    graphdef, per_batch, staged = nnx.split(pipeline, is_per_batch_state, ..., graph=True)

    def step(per_batch: nnx.State, staged: nnx.State) -> tuple[dict, Writes]:
        return run_tracking_writes(
            graphdef, (per_batch, staged), lambda module: module._next_batch(module.batch_size)
        )

    return step, per_batch, staged


def traced_step(pipeline: Pipeline) -> tuple[ClosedJaxpr, tuple[dict, Writes]]:
    """The jaxpr of one full batch, and the shapes of the batch and the state it writes."""
    step, per_batch, staged = _one_batch(pipeline)
    return jax.make_jaxpr(step)(per_batch, staged), jax.eval_shape(step, per_batch, staged)


def compiled_step(pipeline: Pipeline) -> str:
    """The optimized HLO of one full batch, compiled for the default platform.

    Compiled rather than lowered: a platform choice (``lax.platform_dependent``) still appears
    as a conditional on a constant in the lowered StableHLO and is resolved by the compiler.
    """
    step, per_batch, staged = _one_batch(pipeline)
    text = jax.jit(step).lower(per_batch, staged).compile().as_text()
    if text is None:
        raise RuntimeError("the backend gave no text for the compiled step")
    return text


def sub_jaxprs(jaxpr: Jaxpr) -> list[Jaxpr]:
    """``jaxpr`` and every jaxpr nested in its equations (cond branches, loop bodies)."""
    found = [jaxpr]
    for eqn in jaxpr.eqns:
        for param in eqn.params.values():
            for value in param if isinstance(param, tuple | list) else (param,):
                inner = getattr(value, "jaxpr", value)
                if isinstance(inner, Jaxpr):
                    found.extend(sub_jaxprs(inner))
    return found


HOST_CALLBACKS = ("pure_callback", "io_callback", "debug_callback", "callback")
"""Primitives that run Python on the host from inside a compiled program."""


def host_callbacks(closed: ClosedJaxpr) -> list[str]:
    """The host callbacks anywhere in a traced program, nested loops and branches included."""
    return [
        eqn.primitive.name
        for jaxpr in sub_jaxprs(closed.jaxpr)
        for eqn in jaxpr.eqns
        if eqn.primitive.name in HOST_CALLBACKS
    ]
