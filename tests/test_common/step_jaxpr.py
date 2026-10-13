"""The traced program of one pipeline batch, for tests about what the compiled step holds."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
from flax import nnx
from jax.extend.core import ClosedJaxpr, Jaxpr

from datarax.core.element_batch import Batch
from datarax.pipeline import Pipeline
from datarax.pipeline.dag_call import is_per_batch_state, run_tracking_writes, Writes
from datarax.sources.eager_source import bind_columns, eager_sources


type _Step = Callable[[nnx.State, nnx.State, list[Any]], tuple[dict, Writes]]


def _one_batch(pipeline: Pipeline) -> tuple[_Step, tuple[nnx.State, nnx.State, list[Any]]]:
    """The pure function serving one full batch, and the arguments it takes.

    As datarax's own step does, it takes the state partitions and the in-memory sources' device
    columns, which are not NNX state, and binds the columns for the sources' reads.
    """
    graphdef, per_batch, staged = nnx.split(pipeline, is_per_batch_state, ..., graph=True)

    def step(per_batch: nnx.State, staged: nnx.State, columns: list[Any]) -> tuple[dict, Writes]:
        def run(module: Pipeline) -> Batch:
            with bind_columns(eager_sources(module), columns):
                return module._next_batch(module.batch_size)

        return run_tracking_writes(graphdef, (per_batch, staged), run)

    return step, (per_batch, staged, pipeline.device_columns())


def traced_step(pipeline: Pipeline) -> tuple[ClosedJaxpr, tuple[dict, Writes]]:
    """The jaxpr of one full batch, and the shapes of the batch and the state it writes."""
    step, arguments = _one_batch(pipeline)
    return jax.make_jaxpr(step)(*arguments), jax.eval_shape(step, *arguments)


def compiled_step(pipeline: Pipeline) -> str:
    """The optimized HLO of one full batch, compiled for the default platform.

    Compiled rather than lowered: a platform choice (``lax.platform_dependent``) still appears
    as a conditional on a constant in the lowered StableHLO and is resolved by the compiler.
    """
    step, arguments = _one_batch(pipeline)
    text = jax.jit(step).lower(*arguments).compile().as_text()
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
