"""Compiled iteration sessions for Pipeline.

``iter(pipeline)`` over a random-access source returns a
:class:`PipelineIterator`. NNX's per-call module-graph traversal (the
Python-side split/merge inside ``nnx.jit``) dominates per-batch cost for
data pipelines, so the iterator follows Flax's documented functional
pattern instead: split the module once per session, drive batches through
a plain ``jax.jit`` step that carries the state pytree, and write the
final state back into the live module when the session ends.

Semantics:

- Outputs are identical to calling :meth:`Pipeline.step` per batch,
  including RNG streams (the full state, RNG counts included, rides
  through the session).
- The live module is synced at every yield boundary (RNG counts and
  counters — the only state a step mutates), so checkpointing the
  pipeline with ``nnx.split``/Orbax inside or after the loop always
  observes the batches already consumed.
- :meth:`PipelineIterator.get_state`/:meth:`~PipelineIterator.set_state`
  expose iterator-owned state (position and RNG counts), valid at every
  yield boundary, for exact mid-epoch resume without touching the module.

Streaming sources pull batches on the host, so :func:`compile_streaming_dag`
applies the same pattern to the stage modules and the position counter only.
The source never enters the compiled step: its state stays with the live
module, and a source that replaces its backend iterator between passes does
not force a recompile. Structurally identical pipelines share compiled steps.
"""

from __future__ import annotations

import contextlib
import weakref
from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax.core.spec import batch_length
from datarax.pipeline.dag import run_dag


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from datarax.pipeline.pipeline import Pipeline


# Compiled session steps per pipeline, keyed by graphdef structural
# equality. Held OUTSIDE the module: storing graphdefs as module
# attributes would embed them into the next split's graphdef, making
# GraphDef.__eq__ recurse into itself. Weak keys let caches die with
# their pipelines. Bounded per pipeline (structural variants such as
# train/eval flips are few; module surgery must not grow this unbounded).
_SESSION_STEP_CACHES: weakref.WeakKeyDictionary[Any, list[dict[str, Any]]] = (
    weakref.WeakKeyDictionary()
)
_MAX_SESSIONS_PER_PIPELINE = 4


def _is_step_mutable(path: Any, value: Any) -> bool:
    """Filter for state that :meth:`Pipeline.step` may mutate.

    A data-iteration step advances RNG fork counts and plain counters
    (position, source index/epoch); it never mutates parameters, RNG keys,
    or source payloads. Only these leaves are carried through — and
    returned from — the compiled session step, so everything else stays a
    single stable device buffer instead of being copied into a fresh
    output buffer every batch. The same leaves are synced back into the
    live module after every yield, keeping module state valid at yield
    boundaries (checkpointing inside the loop sees the truth).
    """
    del path
    return isinstance(value, nnx.RngCount) or type(value) is nnx.Variable


def _leaf_ids(state: Any) -> tuple[int, ...]:
    """Identity fingerprint of a state pytree's leaves.

    Used to detect source-data swaps between sessions: identical leaf
    objects mean the device-staged copy is still valid.
    """
    return tuple(
        id(leaf) for leaf in jax.tree.leaves(state, is_leaf=lambda x: isinstance(x, nnx.Variable))
    )


def _session_resources(pipeline: Pipeline, graphdef: Any, immutable_state: Any) -> tuple[Any, Any]:
    """Return the compiled session step and device-staged immutable state.

    The session step is Flax's functional hot-loop pattern: merge the
    state pytrees into a module at trace time, run one step, and return
    the batch plus the updated mutable state. Graph traversal happens
    once per trace instead of once per batch.

    The immutable state (source payloads, params, RNG keys) is staged
    onto the device once per pipeline and reused across sessions —
    re-staging would re-upload the entire dataset on every ``iter()``
    call. A leaf-identity fingerprint invalidates the staged copy when
    the module's arrays are swapped.
    """
    fingerprint = _leaf_ids(immutable_state)
    cache = _SESSION_STEP_CACHES.setdefault(pipeline, [])
    for entry in cache:
        if entry["graphdef"] == graphdef:
            if entry["immutable_ids"] != fingerprint:
                entry["staged_immutable"] = jax.device_put(immutable_state)
                entry["immutable_ids"] = fingerprint
            return entry["session_step"], entry["staged_immutable"]

    @jax.jit
    def session_step(mutable_state: Any, immutable_state: Any) -> tuple[dict, Any]:
        module = nnx.merge(graphdef, mutable_state, immutable_state)
        batch = module.step()
        return batch, nnx.state(module, _is_step_mutable)

    cache.append(
        {
            "graphdef": graphdef,
            "session_step": session_step,
            "staged_immutable": jax.device_put(immutable_state),
            "immutable_ids": fingerprint,
        }
    )
    if len(cache) > _MAX_SESSIONS_PER_PIPELINE:
        cache.pop(0)
    return session_step, cache[-1]["staged_immutable"]


def _state_leaves(state: Any) -> list[Any]:
    """Variables of a state pytree, in deterministic traversal order."""
    return [
        leaf
        for leaf in jax.tree.leaves(state, is_leaf=lambda x: isinstance(x, nnx.Variable))
        if isinstance(leaf, nnx.Variable)
    ]


def _write_back(live_variables: list[nnx.Variable], state: Any) -> None:
    """Copy each value a compiled step returned into its live Variable."""
    for variable, updated in zip(live_variables, _state_leaves(state), strict=True):
        variable.set_value(updated.get_value())


def _session_cache_size(pipeline: Pipeline) -> int:
    """Number of compiled session steps cached for ``pipeline`` (testing)."""
    return len(_SESSION_STEP_CACHES.get(pipeline, []))


# Declared element specs per source, by x64 setting. Reading a spec can open a
# backend iterator (the TFDS and HuggingFace streaming sources peek their first
# record, which fills a shuffle buffer), so it is read once per source and
# precision mode instead of once per pass. Weak keys let entries die with sources.
_DECLARED_SPECS: weakref.WeakKeyDictionary[Any, dict[bool, Any]] = weakref.WeakKeyDictionary()


def declared_spec(source: Any) -> Any:
    """Return ``source.element_spec()``, read once per source and x64 setting.

    Args:
        source: The data source whose declaration is needed.

    Returns:
        The element spec the source declared under the active x64 setting.
    """
    specs = _DECLARED_SPECS.setdefault(source, {})
    x64 = bool(jax.config.read("jax_enable_x64"))
    if x64 not in specs:
        specs[x64] = source.element_spec()
    return specs[x64]


# Compiled streaming DAG steps shared across pipelines, matched by equality of the
# stage graph and the execution plan (graphs holding lists are unhashable). The
# most recently used entry is kept last; the oldest is dropped past the bound.
_DAG_STEPS: list[tuple[Any, Any, Callable[..., Any]]] = []
_MAX_DAG_STEPS = 16


def _uncarried_state(graph: Any) -> list[tuple[tuple[Any, ...], nnx.Variable, Any]]:
    """Variables a compiled step does not return, each with the raw value it holds now."""
    return [
        (path, node, node.get_raw_value())
        for path, node in nnx.iter_graph(graph)
        if isinstance(node, nnx.Variable) and not _is_step_mutable(path, node)
    ]


def _refuse_lost_updates(
    graph: Any,
    graphdef: Any,
    uncarried: list[tuple[tuple[Any, ...], nnx.Variable, Any]],
) -> None:
    """Raise while tracing if running the DAG changed what the step cannot return.

    Args:
        graph: The merged stage graph after the DAG ran.
        graphdef: The graph definition the step was compiled for.
        uncarried: :func:`_uncarried_state` of ``graph`` before the DAG ran.

    Raises:
        ValueError: If a stage added or removed module attributes, or wrote a
            Variable other than an RNG count or a plain ``nnx.Variable``.
    """
    if nnx.graphdef(graph) != graphdef:
        raise ValueError(
            "A stage changed the module structure while the DAG ran; streaming iteration "
            "compiles the stage graph once and cannot keep added or removed attributes. "
            "Create stage state in __init__."
        )
    written = [
        f"{'.'.join(str(key) for key in path[1:])} ({type(node).__name__})"
        for path, node, raw in uncarried
        if node.get_raw_value() is not raw
    ]
    if written:
        raise ValueError(
            "Stages wrote state that streaming iteration does not carry between batches: "
            f"{', '.join(written)}. The compiled step returns RNG counts and plain "
            "nnx.Variable state only; keep per-batch state in nnx.Variable."
        )


def _dag_step(graphdef: Any, plan: tuple[Any, ...]) -> Callable[..., Any]:
    """Return the compiled step running a stage graph over one batch."""
    for index, (cached_graphdef, cached_plan, cached_step) in enumerate(_DAG_STEPS):
        if cached_plan == plan and cached_graphdef == graphdef:
            _DAG_STEPS.append(_DAG_STEPS.pop(index))
            return cached_step
    exec_order, predecessors, sink = plan

    @jax.jit
    def step(mutable_state: Any, read_only_state: Any, batch: Any) -> tuple[Any, Any]:
        graph = nnx.merge(graphdef, mutable_state, read_only_state)
        stages, position = graph
        uncarried = _uncarried_state(graph)
        output = run_dag(stages, exec_order, predecessors, sink, batch, position[...])
        position[...] = position[...] + jnp.int32(batch_length(batch))
        _refuse_lost_updates(graph, graphdef, uncarried)
        return output, nnx.state(graph, _is_step_mutable)

    _DAG_STEPS.append((graphdef, plan, step))
    if len(_DAG_STEPS) > _MAX_DAG_STEPS:
        _DAG_STEPS.pop(0)
    return step


def compile_streaming_dag(pipeline: Pipeline) -> Callable[[Any], Any]:
    """Return a function running ``pipeline``'s stage DAG over one host batch.

    The stage modules and the position counter are split once and each batch runs
    through a cached ``jax.jit`` step, so the module graph is not traversed per
    batch. The split state references the live Variables: every call reads their
    current values, including changes made between batches, and writes the RNG
    counts and plain ``nnx.Variable`` state the step returns back into the live
    module, advancing the position by the batch's record count. A stage that
    writes other state or changes the module structure is refused while tracing.

    Args:
        pipeline: The pipeline whose stage DAG runs.

    Returns:
        A function taking a validated batch and returning the sink's output.
    """
    graph = (pipeline._stage_modules, pipeline._position)
    graphdef, mutable_state, read_only_state = nnx.split(graph, _is_step_mutable, ...)
    plan = (
        tuple(pipeline._exec_order),
        {name: tuple(preds) for name, preds in pipeline._predecessors.items()},
        pipeline._sink,
    )
    step = _dag_step(graphdef, plan)
    live_variables = _state_leaves(mutable_state)

    def apply(batch: Any) -> Any:
        output, updated = step(mutable_state, read_only_state, batch)
        _write_back(live_variables, updated)
        return output

    return apply


class PipelineIterator:
    """Compiled iteration session over a random-access pipeline source."""

    def __init__(self, pipeline: Pipeline) -> None:
        """Split the pipeline once and prepare the compiled session step.

        Args:
            pipeline: The pipeline to iterate. Its state is captured at
                construction and written back when the session ends.
        """
        self._pipeline = pipeline
        graphdef, mutable_state, immutable_state = nnx.split(pipeline, _is_step_mutable, ...)
        # Canonicalize carried leaves to device arrays so every session
        # presents identical avals to the cached jax.jit step; host-typed
        # leaves on a fresh module would otherwise force one re-trace per
        # session (tens of milliseconds each).
        self._state: Any = jax.device_put(mutable_state)
        # The split state holds the module's live Variables by reference;
        # keeping them lets each yield sync the module in O(mutable leaves).
        self._live_variables = _state_leaves(mutable_state)
        self._pure_step, self._immutable_state = _session_resources(
            pipeline, graphdef, immutable_state
        )
        source = pipeline.source
        self._source_length: int | None = len(source) if hasattr(source, "__len__") else None
        self._batch_size = pipeline.batch_size
        # One host sync at session entry; termination is then pure Python
        # arithmetic, preserving JAX's asynchronous dispatch run-ahead.
        self._position = int(pipeline._position[...])
        self._rng_count_indices = [
            index
            for index, variable in enumerate(self._live_variables)
            if isinstance(variable, nnx.RngCount)
        ]
        self._position_index = next(
            index
            for index, variable in enumerate(self._live_variables)
            if variable is pipeline._position
        )
        self._epoch_index = next(
            index
            for index, variable in enumerate(self._live_variables)
            if variable is pipeline._epoch
        )
        self._closed = False

    def __iter__(self) -> Iterator[dict]:
        """Return self (iterator protocol)."""
        return self

    def __next__(self) -> dict:
        """Produce the next batch via the compiled session step."""
        if self._closed:
            raise StopIteration
        if self._source_length is not None and self._position >= self._source_length:
            self.close()
            raise StopIteration
        batch, self._state = self._pure_step(self._state, self._immutable_state)
        # Sync the live module at every yield boundary: mid-loop
        # checkpointing (nnx.state on the pipeline) must see the truth.
        _write_back(self._live_variables, self._state)
        self._position += self._batch_size
        return batch

    def get_state(self) -> dict[str, Any]:
        """Return iterator state valid at the current yield boundary.

        The state names the batches already yielded to the caller:
        ``position`` (records consumed), ``epoch`` (which permutation a
        shuffled source serves) and ``rng_counts`` (per-stream fork
        counters, which determine every stochastic draw). Shapes and types
        are stable across the iterator's lifetime.

        Returns:
            JSON-serializable dict with ``position``, ``epoch`` and ``rng_counts``.
        """
        counts = [int(self._live_variables[index].get_value()) for index in self._rng_count_indices]
        return {
            "position": np.int64(self._position),
            "epoch": int(self._live_variables[self._epoch_index].get_value()),
            "rng_counts": counts,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore iterator state produced by :meth:`get_state`.

        The pipeline must be configured identically (same structure, same
        seeds) to the one that produced the state.

        Args:
            state: Dict with ``position``, ``epoch`` and ``rng_counts`` entries.

        Raises:
            ValueError: If ``state`` carries a different number of rng counts than this
                pipeline has streams.
        """
        counts = state["rng_counts"]
        if len(counts) != len(self._rng_count_indices):
            raise ValueError(
                f"state carries {len(counts)} rng counts but this pipeline "
                f"has {len(self._rng_count_indices)} rng streams; the "
                f"pipeline structure must match the one that produced it."
            )
        position = int(state["position"])
        carried = _state_leaves(self._state)
        for index, count in zip(self._rng_count_indices, counts, strict=True):
            for target in (self._live_variables[index], carried[index]):
                target.set_value(jnp.asarray(count, dtype=target.get_value().dtype))
        for target in (self._live_variables[self._position_index], carried[self._position_index]):
            target.set_value(jnp.asarray(position, dtype=jnp.int32))
        for target in (self._live_variables[self._epoch_index], carried[self._epoch_index]):
            target.set_value(jnp.asarray(int(state["epoch"]), dtype=jnp.int32))
        self._position = position

    def close(self) -> None:
        """End the session.

        The live module is already synced at every yield boundary, so
        closing only marks the session finished. Idempotent; subsequent
        :meth:`__next__` calls raise StopIteration.
        """
        self._closed = True

    def __del__(self) -> None:
        """Best-effort write-back if the iterator is dropped without close."""
        # Never raise during GC or interpreter teardown.
        with contextlib.suppress(Exception):
            self.close()
