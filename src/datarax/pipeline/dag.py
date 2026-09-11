"""Run a pipeline's stage DAG over one batch.

``Pipeline.__call__`` and the compiled streaming step both execute the plan that
:func:`~datarax.pipeline.topo.topological_sort` builds: each node receives the
source batch, or its predecessors' outputs as positional arguments, in
topological order. The plan is static, so tracing unrolls it into one graph.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp


def _record_indices(batch: Any, start: jax.Array | int) -> jax.Array | None:
    """Global record positions ``start + arange(n)`` for ``batch``, or None without leaves.

    ``n`` is the leading axis of the first leaf, which is static under tracing;
    ``start`` may be a traced scalar.
    """
    leaves = jax.tree.leaves(batch)
    if not leaves:
        return None
    size = leaves[0].shape[0]
    return jnp.asarray(start, dtype=jnp.int32) + jnp.arange(size, dtype=jnp.int32)


def run_dag(  # noqa: PLR0913 - the static plan is four separate pipeline attributes
    stages: Mapping[str, Any],
    exec_order: Sequence[str],
    predecessors: Mapping[str, Sequence[str]],
    sink: str | None,
    batch: Any,
    start: jax.Array | int,
) -> Any:
    """Run the stages over ``batch`` in ``exec_order`` and return the sink's output.

    Stages exposing ``_apply_on_raw(data, states, metadata, global_indices)``
    (every ``OperatorModule``) take the raw dict path with states threaded
    between them and discarded at the sink; any other ``nnx.Module`` is called
    with its inputs. Stochastic operators are keyed on the records' global
    positions ``start + arange(n)``, so augmentation does not depend on how
    records are grouped into batches.

    Args:
        stages: Stage modules by node name.
        exec_order: Node names in topological order.
        predecessors: Predecessor node names for each node; an empty list means
            the node consumes the source batch.
        sink: Node whose output is returned, or None for a pipeline without stages.
        batch: Source batch.
        start: Global position of the batch's first record.

    Returns:
        The sink node's output, or ``batch`` unchanged when there are no stages.
    """
    if not exec_order or sink is None:
        return batch
    global_indices = _record_indices(batch, start)
    outputs: dict[str, Any] = {}
    states: dict[str, Any] = {}
    for name in exec_order:
        preds = predecessors[name]
        inputs = (batch,) if not preds else tuple(outputs[p] for p in preds)
        stage = stages[name]
        apply_on_raw = getattr(stage, "_apply_on_raw", None)
        if callable(apply_on_raw) and len(inputs) == 1:
            data, states = apply_on_raw(inputs[0], states, None, global_indices)  # type: ignore[reportGeneralTypeIssues]
            outputs[name] = data
        else:
            outputs[name] = stage(*inputs)
    return outputs[sink]
