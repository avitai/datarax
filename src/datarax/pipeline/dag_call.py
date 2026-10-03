"""One compiled call of a pipeline's DAG over a host-read batch, with the state it writes.

NNX's per-call module-graph traversal (the Python-side split/merge inside ``nnx.jit``) dominates
per-batch cost for data pipelines, so a DAG call follows Flax's documented functional pattern:
split the module once, drive batches through a plain ``jax.jit`` step, and write the state the
step changed back into the live module after every batch. The step returns only the Variables it
wrote, found while tracing, so unchanged state is never copied per batch; a step that adds or
removes state is refused. :func:`compile_dag` serves ``for batch in pipeline``; the compiled
session (:mod:`datarax.pipeline.iteration`) builds on the same pieces. Structurally identical
DAGs share one compiled step. Nothing is donated: the batch belongs to the caller.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
from flax import nnx

from datarax.core.element_batch import Batch
from datarax.pipeline.dag import OperatorDag


# Compiled steps shared across pipelines, matched by structural equality of their
# key (graphs holding lists are unhashable, so the caches are lists, not dicts).
# The most recently used entry is kept last; the oldest is dropped past the bound.
# Held OUTSIDE the modules: storing graphdefs as module attributes would embed them
# into the next split's graphdef, making GraphDef.__eq__ recurse into itself.
_MAX_COMPILED_STEPS = 16
_DAG_STEPS: list[tuple[Any, Callable[..., Any]]] = []

# A compiled step's writes: raw values by position within each state partition,
# per-batch state first and staged state second.
Writes = tuple[dict[int, Any], dict[int, Any]]


def is_per_batch_state(path: Any, value: Any) -> bool:
    """Filter for state expected to change on every batch: RNG counts and counters.

    It decides where state lives between batches, not what a step may write.
    These leaves are uploaded to the device once per session; everything else
    (source payloads, parameters, RNG keys) is staged once per pipeline and
    reused across sessions. A step may write state in either partition.
    """
    del path
    return isinstance(value, nnx.RngCount) or type(value) is nnx.Variable


def cached_step(
    cache: list[tuple[Any, Callable[..., Any]]], key: Any, build: Callable[[], Callable[..., Any]]
) -> Callable[..., Any]:
    """Return the compiled step cached under ``key``, building it on a miss.

    Keys are compared by equality, so structurally identical pipelines share one
    compiled step. The most recently used entry moves last; the oldest is dropped past
    :data:`_MAX_COMPILED_STEPS`.
    """
    for index, (cached_key, step) in enumerate(cache):
        if cached_key == key:
            cache.append(cache.pop(index))
            return step
    step = build()
    cache.append((key, step))
    if len(cache) > _MAX_COMPILED_STEPS:
        cache.pop(0)
    return step


def state_leaves(state: Any) -> list[Any]:
    """Variables of a state pytree, in deterministic traversal order."""
    return [
        leaf
        for leaf in jax.tree.leaves(state, is_leaf=lambda x: isinstance(x, nnx.Variable))
        if isinstance(leaf, nnx.Variable)
    ]


def _snapshot(variables: list[nnx.Variable]) -> list[tuple[list[Any], Any]]:
    """The raw value leaves and tree structure each Variable holds now."""
    return [jax.tree.flatten(variable.get_raw_value()) for variable in variables]


def _written(
    variables: list[nnx.Variable], snapshot: list[tuple[list[Any], Any]]
) -> dict[int, Any]:
    """Values of the Variables rebound since ``snapshot``, keyed by position.

    A write replaces a leaf object or changes the value's tree structure, so
    comparing leaves by identity finds every write, including an in-place update
    of a container value, without comparing array contents.
    """
    writes: dict[int, Any] = {}
    for index, (variable, (leaves, treedef)) in enumerate(zip(variables, snapshot, strict=True)):
        now_leaves, now_treedef = jax.tree.flatten(variable.get_raw_value())
        if now_treedef != treedef or any(
            new is not old for new, old in zip(now_leaves, leaves, strict=True)
        ):
            writes[index] = variable.get_value()
    return writes


def _state_paths(graph: Any) -> set[tuple[Any, ...]]:
    """Paths of every state leaf in ``graph``."""
    return {path for path, _ in nnx.to_flat_state(nnx.state(graph, graph=True))}


def run_tracking_writes(
    graphdef: Any, states: tuple[Any, Any], run: Callable[[Any], Any]
) -> tuple[Any, Writes]:
    """Merge ``states``, call ``run`` on the graph, and return its output with the writes.

    Called while tracing a compiled step. The writes are found by comparing each
    Variable before and after ``run``, so the step returns exactly the state it
    changed, in either partition.

    Args:
        graphdef: The graph definition the step is compiled for.
        states: The per-batch and staged state partitions.
        run: The step body, taking the merged graph.

    Returns:
        ``run``'s output and the writes for each partition.

    Raises:
        ValueError: If ``run`` added or removed state, which a step compiled for a
            fixed module graph cannot return.
    """
    graph = nnx.merge(graphdef, *states)
    partitions = [
        state_leaves(part) for part in nnx.state(graph, is_per_batch_state, ..., graph=True)
    ]
    snapshots = [_snapshot(part) for part in partitions]
    paths = _state_paths(graph)
    output = run(graph)
    if _state_paths(graph) != paths:
        raise ValueError(
            "A stage changed the module structure while the step ran: it added or removed "
            "state, which a step compiled for a fixed module graph cannot keep. Create state "
            "in __init__."
        )
    return output, (_written(partitions[0], snapshots[0]), _written(partitions[1], snapshots[1]))


def apply_writes(writes: Writes, receivers: Sequence[Sequence[list[nnx.Variable]]]) -> None:
    """Set each written value on every Variable list that mirrors its partition."""
    for partition_writes, variable_lists in zip(writes, receivers, strict=True):
        for index, value in partition_writes.items():
            for variables in variable_lists:
                variables[index].set_value(value)


def _dag_step(graphdef: Any) -> Callable[..., Any]:
    """Return the compiled step running a DAG over one streamed batch.

    Keyed by the graph definition, which holds the DAG's static plan.
    """

    def build() -> Callable[..., Any]:
        @jax.jit
        def step(mutable_state: Any, read_only_state: Any, batch: Batch) -> tuple[Batch, Writes]:
            def run(dag: OperatorDag) -> Batch:
                return dag(batch)

            return run_tracking_writes(graphdef, (mutable_state, read_only_state), run)

        return step

    return cached_step(_DAG_STEPS, graphdef, build)


def compile_dag(dag: OperatorDag) -> Callable[[Batch], Batch]:
    """Return a function running a pipeline's DAG over one ``Batch`` the host stage read.

    The batch carries its records' indices and epochs; the DAG is split once and each batch runs
    through a cached ``jax.jit`` step, so the module graph is not traversed per batch. The split
    state references the live Variables: every call reads their current values, including
    changes made between batches, and writes every Variable the step changed back into the live
    module. A stage that adds or removes state is refused while tracing.

    Args:
        dag: The pipeline's DAG.

    Returns:
        A function taking a validated host ``Batch`` and returning the DAG's ``Batch``.
    """
    graphdef, per_batch_state, staged_state = nnx.split(dag, is_per_batch_state, ..., graph=True)
    step = _dag_step(graphdef)
    receivers = ((state_leaves(per_batch_state),), (state_leaves(staged_state),))

    def apply(batch: Any) -> Batch:
        output, writes = step(per_batch_state, staged_state, batch)
        apply_writes(writes, receivers)
        return output

    return apply
