"""The traced program of one pipeline batch, for tests about what the compiled step holds."""

from __future__ import annotations

from collections.abc import Callable

import jax
from flax import nnx
from jax.extend.core import ClosedJaxpr, Jaxpr

from datarax.pipeline import Pipeline
from datarax.pipeline.iteration import _is_per_batch_state, _run_tracking_writes, _Writes


type _Step = Callable[[nnx.State, nnx.State], tuple[dict, _Writes]]


def _one_batch(pipeline: Pipeline) -> tuple[_Step, nnx.State, nnx.State]:
    """The pure function serving one full batch, and the state partitions it takes."""
    graphdef, per_batch, staged = nnx.split(pipeline, _is_per_batch_state, ..., graph=True)

    def step(per_batch: nnx.State, staged: nnx.State) -> tuple[dict, _Writes]:
        return _run_tracking_writes(
            graphdef, (per_batch, staged), lambda module: module._next_batch(module.batch_size)
        )

    return step, per_batch, staged


def traced_step(pipeline: Pipeline) -> tuple[ClosedJaxpr, tuple[dict, _Writes]]:
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
