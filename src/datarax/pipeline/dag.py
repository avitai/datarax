"""Run a pipeline's stage DAG over one batch.

``Pipeline.__call__``, the indexed step and the compiled streaming step all execute the plan
that :func:`~datarax.pipeline.topo.topological_sort` builds: each node receives the source
batch, or its predecessors' outputs as positional arguments, in topological order. The plan is
static, so tracing unrolls it into one graph.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp


def record_count(batch: Any) -> int | None:
    """Number of records in ``batch``: the leading axis of its first leaf, or None without leaves.

    The leading axis is static under tracing.
    """
    leaves = jax.tree.leaves(batch)
    return leaves[0].shape[0] if leaves else None


def record_positions(batch: Any, start: jax.Array | int) -> jax.Array | None:
    """Positions ``start + arange(n)`` of ``batch``'s records, or None without leaves.

    A stream serves records in order, so these positions name its records. ``start`` may be
    a traced scalar.
    """
    size = record_count(batch)
    if size is None:
        return None
    return jnp.asarray(start, dtype=jnp.int32) + jnp.arange(size, dtype=jnp.int32)


def run_dag(  # noqa: PLR0913 - the static plan is four separate pipeline attributes
    stages: Mapping[str, Any],
    exec_order: Sequence[str],
    predecessors: Mapping[str, Sequence[str]],
    sink: str | None,
    batch: Any,
    record_indices: jax.Array | None,
    epoch: jax.Array | int | None,
) -> Any:
    """Run the stages over ``batch`` in ``exec_order`` and return the sink's output.

    Stages exposing ``_apply_on_raw(data, states, stats, record_indices, epoch)``
    (every ``OperatorModule``) take the raw dict path with states threaded
    between them and discarded at the sink; any other ``nnx.Module`` is called
    with its inputs. Stochastic operators key each record on ``epoch`` and its
    entry in ``record_indices``, so within an epoch a record's augmentation does
    not depend on how records are batched, ordered or split across workers.

    Args:
        stages: Stage modules by node name.
        exec_order: Node names in topological order.
        predecessors: Predecessor node names for each node; an empty list means
            the node consumes the source batch.
        sink: Node whose output is returned, or None for a pipeline without stages.
        batch: Source batch.
        record_indices: Stable index of each record in ``batch``.
        epoch: The epoch the records belong to.

    Returns:
        The sink node's output, or ``batch`` unchanged when there are no stages.
    """
    if not exec_order or sink is None:
        return batch
    outputs: dict[str, Any] = {}
    states: dict[str, Any] = {}
    for name in exec_order:
        preds = predecessors[name]
        inputs = (batch,) if not preds else tuple(outputs[p] for p in preds)
        stage = stages[name]
        apply_on_raw = getattr(stage, "_apply_on_raw", None)
        if callable(apply_on_raw) and len(inputs) == 1:
            data, states = apply_on_raw(inputs[0], states, None, record_indices, epoch)  # type: ignore[reportGeneralTypeIssues]
            outputs[name] = data
        else:
            outputs[name] = stage(*inputs)
    return outputs[sink]
