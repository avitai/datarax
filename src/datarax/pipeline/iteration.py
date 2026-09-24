"""Compiled iteration sessions for Pipeline.

``iter(pipeline)`` over a random-access source returns a
:class:`PipelineIterator`. NNX's per-call module-graph traversal (the
Python-side split/merge inside ``nnx.jit``) dominates per-batch cost for
data pipelines, so the iterator follows Flax's documented functional
pattern instead: split the module once per session, drive batches through
a plain ``jax.jit`` step, and write the state the step changed back into the
live module after every batch.

Semantics:

- Outputs are identical to calling :meth:`Pipeline.step` per batch,
  including RNG streams.
- Every Variable a step writes, whatever its type, reaches the live module at
  every yield boundary, so checkpointing the pipeline with ``nnx.split``/Orbax
  inside or after the loop observes the batches already consumed. The step
  returns only the Variables it wrote, found while tracing, so unchanged state
  such as source payloads and RNG keys is never copied per batch. A step that
  adds or removes state is refused.
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
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax.core.operator import OperatorModule
from datarax.core.spec import batch_length
from datarax.pipeline.dag import record_positions, run_dag
from datarax.pipeline.epochs import EpochPlan


# The traceable body a session step runs on the merged module: one batch of the given number of
# records, returned after the module's state has advanced (the pipeline's ``_next_batch``).
type StepBody = Callable[[Any, int], dict]

# A stage graph's execution plan: topological order, each node's predecessors, the sink.
type DagPlan = tuple[tuple[str, ...], Mapping[str, tuple[str, ...]], str | None]


# Compiled steps shared across pipelines, matched by structural equality of their
# key (graphs holding lists are unhashable, so the caches are lists, not dicts).
# The most recently used entry is kept last; the oldest is dropped past the bound.
# Held OUTSIDE the modules: storing graphdefs as module attributes would embed them
# into the next split's graphdef, making GraphDef.__eq__ recurse into itself.
_MAX_COMPILED_STEPS = 16
_SESSION_STEPS: list[tuple[Any, Callable[..., Any]]] = []
_DAG_STEPS: list[tuple[Any, Callable[..., Any]]] = []

# Device copies of each pipeline's host (NumPy) arrays, uploaded at first use and shared by
# every path that stages the pipeline (``step()`` and iteration sessions). Keyed weakly by the
# pipeline, then by the array's identity with a weak reference to the array, so a copy dies
# with its pipeline or its array, and an array replaced by another is uploaded at its first use.
# NumPy arrays are never tracers, so staging inside a caller's transform caches nothing traced.
type _HostCopies = dict[int, tuple[weakref.ref[np.ndarray], jax.Array]]
_HOST_COPIES: weakref.WeakKeyDictionary[Any, _HostCopies] = weakref.WeakKeyDictionary()

# A compiled step's writes: raw values by position within each state partition,
# per-batch state first and staged state second.
_Writes = tuple[dict[int, Any], dict[int, Any]]

# The layout of PipelineIterator.get_state(). Version 1 gives each stochastic operator one
# private stream count and a deterministic one none; before it, an operator carried every
# stream of the Rngs its caller passed. A state without the field predates it and is upgraded.
_ITERATOR_STATE_VERSION = 2
# Version 1 carried ``rng_counts`` in the per-operator layout; version 2 adds ``fingerprint``,
# the configuration that produced the state, which ``set_state`` checks.
_FINGERPRINT_FIELDS = ("batch_size", "length", "drop_last", "num_epochs", "shuffled")


def _is_per_batch_state(path: Any, value: Any) -> bool:
    """Filter for state expected to change on every batch: RNG counts and counters.

    It decides where state lives between batches, not what a step may write.
    These leaves are uploaded to the device once per session; everything else
    (source payloads, parameters, RNG keys) is staged once per pipeline and
    reused across sessions. A step may write state in either partition.
    """
    del path
    return isinstance(value, nnx.RngCount) or type(value) is nnx.Variable


def _cached_step(
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


def _state_leaves(state: Any) -> list[Any]:
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


def _run_tracking_writes(
    graphdef: Any, states: tuple[Any, Any], run: Callable[[Any], Any]
) -> tuple[Any, _Writes]:
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
        _state_leaves(part) for part in nnx.state(graph, _is_per_batch_state, ..., graph=True)
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


def _apply_writes(writes: _Writes, receivers: Sequence[Sequence[list[nnx.Variable]]]) -> None:
    """Set each written value on every Variable list that mirrors its partition."""
    for partition_writes, variable_lists in zip(writes, receivers, strict=True):
        for index, value in partition_writes.items():
            for variables in variable_lists:
                variables[index].set_value(value)


def _session_step(graphdef: Any, body: StepBody, size: int) -> Callable[..., Any]:
    """The compiled step running ``body`` once, for ``size`` records, on modules shaped so.

    Flax's functional hot-loop pattern: merge the state partitions into a module at
    trace time, run one batch, and return it with the state the step wrote. Graph
    traversal happens once per trace instead of once per batch, and structurally
    identical modules running the same body at the same size share the step.
    """

    def run(graph: Any) -> dict:
        return body(graph, size)

    def build() -> Callable[..., Any]:
        @jax.jit
        def session_step(mutable_state: Any, immutable_state: Any) -> tuple[dict, _Writes]:
            return _run_tracking_writes(graphdef, (mutable_state, immutable_state), run)

        return session_step

    return _cached_step(_SESSION_STEPS, (graphdef, body, size), build)


def _host_copies(module: nnx.Module) -> _HostCopies:
    """The device copies of ``module``'s host arrays, by the arrays' identities."""
    return _HOST_COPIES.setdefault(module, {})


def _on_device(module: nnx.Module, state: Any) -> Any:
    """``state`` with every NumPy leaf replaced by its device copy, uploaded once per array.

    Device arrays and tracers pass through untouched, so a step reads the source's device
    buffers in place and a step inside a caller's transform stages nothing. A NumPy leaf is
    host data by construction: its copy is uploaded at first use and reused while the array
    lives, so iteration sessions and ``step()`` share one copy, and an array replaced by
    another is uploaded when first used. Records are immutable once given to a source: an
    in-place edit of a staged array is not uploaded.
    """
    copies = _host_copies(module)
    for key in [key for key, (array, _) in copies.items() if array() is None]:
        del copies[key]

    def stage(leaf: Any) -> Any:
        if not isinstance(leaf, np.ndarray):
            return leaf
        entry = copies.get(id(leaf))
        if entry is None or entry[0]() is not leaf:
            entry = (weakref.ref(leaf), jax.device_put(leaf))
            copies[id(leaf)] = entry
        return entry[1]

    return jax.tree.map(stage, state)


def next_batch(module: nnx.Module, body: StepBody, size: int) -> dict:
    """Run ``body`` once, for ``size`` records, on ``module`` through the compiled session step.

    The body of :meth:`Pipeline.step`: split the module, stage its host arrays, run the
    step shared with iteration sessions of the same structure and body, and write back the
    state the step changed. Splitting per call sees every structural change made between
    calls; the step itself refuses one made while it runs.

    Args:
        module: The module to advance.
        body: The traceable body to run on it.
        size: The records the batch holds.

    Returns:
        The body's output.
    """
    graphdef, per_batch, staged = nnx.split(module, _is_per_batch_state, ..., graph=True)
    batch, writes = _session_step(graphdef, body, size)(per_batch, _on_device(module, staged))
    _apply_writes(writes, ((_state_leaves(per_batch),), (_state_leaves(staged),)))
    return batch


def _dag_step(graphdef: Any, plan: DagPlan) -> Callable[..., Any]:
    """Return the compiled step running a stage graph over one batch.

    Keyed by the execution plan and then the stage graph, so the cheap comparison
    decides most misses.
    """
    exec_order, predecessors, sink = plan

    def build() -> Callable[..., Any]:
        @jax.jit
        def step(mutable_state: Any, read_only_state: Any, batch: Any) -> tuple[Any, _Writes]:
            def run(graph: Any) -> Any:
                stages, position, epoch = graph
                # A stream serves records in order, so their positions name them.
                record_indices = record_positions(batch, position[...])
                output = run_dag(
                    stages, exec_order, predecessors, sink, batch, record_indices, epoch[...]
                )
                position[...] = position[...] + jnp.int32(batch_length(batch))
                return output

            return _run_tracking_writes(graphdef, (mutable_state, read_only_state), run)

        return step

    return _cached_step(_DAG_STEPS, (plan, graphdef), build)


def compile_streaming_dag(
    stages: Mapping[str, nnx.Module],
    position: nnx.Variable[jax.Array],
    epoch: nnx.Variable[jax.Array],
    plan: DagPlan,
) -> Callable[[Any], Any]:
    """Return a function running a stage DAG over one host batch.

    The stage modules and the position counter are split once and each batch runs
    through a cached ``jax.jit`` step, so the module graph is not traversed per
    batch. The split state references the live Variables: every call reads their
    current values, including changes made between batches, and writes every
    Variable the step changed back into the live module, advancing the position by
    the batch's record count. A stage that adds or removes state is refused while
    tracing.

    Args:
        stages: The stage modules by node name.
        position: The position counter the batch's records are numbered from.
        epoch: The epoch counter the stages key their randomness on.
        plan: The stage graph's execution plan.

    Returns:
        A function taking a validated batch and returning the sink's output.
    """
    graph = (stages, position, epoch)
    graphdef, per_batch_state, staged_state = nnx.split(graph, _is_per_batch_state, ..., graph=True)
    step = _dag_step(graphdef, plan)
    receivers = ((_state_leaves(per_batch_state),), (_state_leaves(staged_state),))

    def apply(batch: Any) -> Any:
        output, writes = step(per_batch_state, staged_state, batch)
        _apply_writes(writes, receivers)
        return output

    return apply


def _operator_owned_counts(
    module: nnx.Module, live_variables: list[Any], rng_count_indices: list[int]
) -> list[bool]:
    """Return whether each of a session's RNG counts belongs to an operator.

    Ownership is decided by Variable identity rather than by state path, so the answer does not
    depend on how a path happens to be spelled. An operator's counts precede the pipeline's and
    the source's, because flax orders attributes by name and ``_stage_modules`` sorts first; an
    operator reached through some other attribute, such as one a source holds, would not, and
    :meth:`PipelineIterator._upgraded_rng_counts` refuses such a pipeline rather than placing a
    saved state into it wrongly.

    Args:
        module: The module being iterated.
        live_variables: The session's per-batch Variables, in traversal order.
        rng_count_indices: Positions of the RNG counts within ``live_variables``.

    Returns:
        One flag per entry of ``rng_count_indices``, True where an operator owns that count.
    """
    owned = {
        id(variable)
        for _, node in nnx.iter_graph(module, graph=True)
        if isinstance(node, OperatorModule)
        for variable in _state_leaves(nnx.state(node, nnx.RngCount, graph=True))
    }
    return [id(live_variables[index]) in owned for index in rng_count_indices]


class PipelineIterator:
    """Compiled iteration session over a random-access pipeline source.

    Built by :meth:`Pipeline.session`, which passes the pipeline and what the session needs
    from it; the session knows nothing else about the pipeline.
    """

    def __init__(  # noqa: PLR0913 - the module, its step, its epoch rule and its counters
        self,
        module: nnx.Module,
        *,
        body: StepBody,
        plan: EpochPlan,
        position: nnx.Variable[jax.Array],
        epoch: nnx.Variable[jax.Array],
        shuffled: bool,
    ) -> None:
        """Split the module once and prepare the compiled session step.

        Args:
            module: The module to iterate. Every write a step makes reaches it at the next
                yield boundary.
            body: The traceable body serving one batch, run by the compiled step.
            plan: How the module's records divide into batches and epochs.
            position: The module's position counter.
            epoch: The module's epoch counter.
            shuffled: Whether the source serves records in a shuffled order, recorded in
                the state's fingerprint.
        """
        graphdef, mutable_state, immutable_state = nnx.split(
            module, _is_per_batch_state, ..., graph=True
        )
        # A session carries its own Variables for the per-batch state, holding the live values
        # as they are: ``step()`` passes the same values, so both paths present the shared
        # compiled step one call signature (``jax.jit`` keys its dispatch cache on arguments'
        # shardings and committedness as well as avals, so converting a Python-int counter in
        # one path only would add a second entry for the same executable).
        self._state: Any = jax.tree.map(lambda leaf: leaf, mutable_state)
        # The split state holds the module's live Variables by reference;
        # keeping them lets each yield sync the module in O(written leaves).
        self._live_variables = _state_leaves(mutable_state)
        self._carried_variables = _state_leaves(self._state)
        self._graphdef = graphdef
        self._body = body
        # The step every full batch runs, shared with ``step()``; a run's short final batch
        # looks its own up once.
        self._pure_step = _session_step(graphdef, body, plan.batch_size)
        self._immutable_state = _on_device(module, immutable_state)
        self._receivers = (
            (self._live_variables, self._carried_variables),
            (_state_leaves(immutable_state), _state_leaves(self._immutable_state)),
        )
        self._plan = plan
        self._shuffled = shuffled
        # One host sync at session entry; the counters are then mirrored on the host by the
        # rule the step follows, so termination is pure Python arithmetic, preserving JAX's
        # asynchronous dispatch run-ahead.
        self._position = int(position[...])
        self._epoch = int(epoch[...])
        self._batches_left, self._final_size = self._extent(self._position)
        self._rng_count_indices = [
            index
            for index, variable in enumerate(self._live_variables)
            if isinstance(variable, nnx.RngCount)
        ]
        self._count_is_an_operators = _operator_owned_counts(
            module, self._live_variables, self._rng_count_indices
        )
        self._position_index = next(
            index for index, variable in enumerate(self._live_variables) if variable is position
        )
        self._epoch_index = next(
            index for index, variable in enumerate(self._live_variables) if variable is epoch
        )
        self._closed = False

    def __iter__(self) -> Iterator[dict]:
        """Return self (iterator protocol)."""
        return self

    def __next__(self) -> dict:
        """Produce the next batch via the compiled session step.

        The step starts the next epoch itself when the current one cannot start another
        batch (see :class:`~datarax.pipeline.epochs.EpochPlan`). The session stops after the
        batches its epochs hold, the final one computed at its own size when the records
        left do not fill a batch; a stream never stops.
        """
        if self._closed or self._batches_left == 0:
            self.close()
            raise StopIteration
        size = self._plan.batch_size if self._batches_left != 1 else self._final_size
        step = (
            self._pure_step
            if size == self._plan.batch_size
            else _session_step(self._graphdef, self._body, size)
        )
        batch, writes = step(self._state, self._immutable_state)
        # Sync the live module and the session copies at every yield boundary:
        # mid-loop checkpointing (nnx.state on the pipeline) must see the truth.
        _apply_writes(writes, self._receivers)
        # The host mirror of the counters the step just wrote, by the same rule.
        start, epoch = self._plan.batch_start(self._position, self._epoch)
        self._position, self._epoch = self._plan.advance(start, epoch, size)
        if self._batches_left is not None:
            self._batches_left -= 1
        return batch

    def _extent(self, position: int) -> tuple[int | None, int]:
        """Batches a session from ``position`` serves (``None``: no end) and its final size."""
        extent = self._plan.run_extent(position)
        return (None, self._plan.batch_size) if extent is None else extent

    def _write_position_and_epoch(self, position: int, epoch: int) -> None:
        carried = self._carried_variables
        for target in (self._live_variables[self._position_index], carried[self._position_index]):
            target.set_value(jnp.asarray(position, dtype=jnp.int32))
        for target in (self._live_variables[self._epoch_index], carried[self._epoch_index]):
            target.set_value(jnp.asarray(epoch, dtype=jnp.int32))
        self._position = position
        self._epoch = epoch
        self._batches_left, self._final_size = self._extent(position)

    def get_state(self) -> dict[str, Any]:
        """Return iterator state valid at the current yield boundary.

        The state names the batches already yielded to the caller:
        ``position`` (records consumed), ``epoch`` (which permutation a
        shuffled source serves), ``rng_counts`` (per-stream fork counters,
        which determine every stochastic draw) and ``version``, the layout
        those counts are in. Shapes and types are stable across the
        iterator's lifetime.

        ``rng_counts`` holds one count per stochastic operator, which stays
        0 because iteration keys each record on the operator's base key and
        never draws from its private stream, followed by the pipeline's and
        the source's. A deterministic operator contributes none, so the
        list's length follows how many operators are stochastic rather than
        how many streams their caller's ``Rngs`` carried.

        Returns:
            JSON-serializable dict with ``position``, ``epoch``, ``rng_counts`` and ``version``.
        """
        counts = [int(self._live_variables[index].get_value()) for index in self._rng_count_indices]
        return {
            "position": int(self._position),
            "epoch": self._epoch,
            "rng_counts": counts,
            "version": _ITERATOR_STATE_VERSION,
            "fingerprint": self._fingerprint(),
        }

    def _fingerprint(self) -> dict[str, Any]:
        """The configuration a state is only valid for: batch rule, length, epochs, order.

        Every leaf is a number or a bool (``num_epochs`` may be ``None``), so the state
        also fits a checkpoint template of arrays.
        """
        return {
            "batch_size": self._plan.batch_size,
            "length": self._plan.length,
            "drop_last": self._plan.drop_last,
            "num_epochs": self._plan.num_epochs,
            "shuffled": self._shuffled,
        }

    def _upgraded_rng_counts(self, state: dict[str, Any]) -> list[int]:
        """Return ``state``'s rng counts in the layout this iterator holds.

        A state naming the current version is taken as it is. One saved before the field existed
        came from a pipeline whose operators each kept the caller's ``Rngs``, so it carries an
        entry per stream of every such ``Rngs`` where this pipeline carries one per stochastic
        operator. Only the entries outside operators still mean anything — iteration never draws
        from an operator's own stream, so every operator entry restores to 0 — and those entries
        keep their order at the end of the list.

        Args:
            state: The state given to :meth:`set_state`.

        Returns:
            The rng counts to restore.

        Raises:
            ValueError: If an operator's count follows one that no operator owns, which is the
                order the placement relies on, or if the state carries fewer counts than this
                pipeline holds outside its operators.
        """
        counts = list(state["rng_counts"])
        if int(state.get("version", 0)) >= _ITERATOR_STATE_VERSION:
            return counts

        outside = self._count_is_an_operators.count(False)
        owned = len(self._count_is_an_operators) - outside
        if not all(self._count_is_an_operators[:owned]) or any(self._count_is_an_operators[owned:]):
            raise ValueError(
                "This pipeline holds an operator's RNG count after a count no operator owns, so "
                "a state saved before the counts carried a version cannot be placed in it. "
                "Save the iterator state again from this pipeline."
            )
        if len(counts) < outside:
            raise ValueError(
                f"state carries {len(counts)} rng counts, fewer than the {outside} streams this "
                f"pipeline holds outside its operators; it did not come from this pipeline."
            )
        return [0] * owned + counts[len(counts) - outside :]

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore iterator state produced by :meth:`get_state`.

        The pipeline must be configured identically (same structure, same
        seeds) to the one that produced the state. A state saved before
        ``rng_counts`` carried a version is upgraded to this pipeline's
        layout first; see :meth:`_upgraded_rng_counts`.

        Args:
            state: Dict with ``position``, ``epoch``, ``rng_counts``, ``version`` (from
                version 1) and ``fingerprint`` (from version 2) entries.

        Raises:
            ValueError: If the state's ``fingerprint`` names a different batch size, length,
                last-batch rule, epoch count or order than this pipeline has, if ``state``
                carries a different number of rng counts than this pipeline has streams, or
                a negative ``position`` or ``epoch``.
        """
        self._check_fingerprint(state)
        counts = self._upgraded_rng_counts(state)
        if len(counts) != len(self._rng_count_indices):
            raise ValueError(
                f"state carries {len(counts)} rng counts but this pipeline "
                f"has {len(self._rng_count_indices)} rng streams; the "
                f"pipeline structure must match the one that produced it."
            )
        position = int(state["position"])
        epoch = int(state["epoch"])
        if position < 0 or epoch < 0:
            raise ValueError(
                f"state carries position {position} and epoch {epoch}; both count up from 0, "
                f"so a negative value is not state this pipeline produced."
            )
        carried = self._carried_variables
        for index, count in zip(self._rng_count_indices, counts, strict=True):
            for target in (self._live_variables[index], carried[index]):
                target.set_value(jnp.asarray(count, dtype=target.get_value().dtype))
        self._write_position_and_epoch(position, epoch)

    def _check_fingerprint(self, state: dict[str, Any]) -> None:
        """Refuse a state produced under a different configuration, naming the first field.

        A version-1 state has no fingerprint and is taken as it is.
        """
        recorded = state.get("fingerprint")
        if recorded is None:
            return
        mine = self._fingerprint()
        for field in _FINGERPRINT_FIELDS:
            if recorded.get(field) != mine[field]:
                raise ValueError(
                    f"state was produced with {field}={recorded.get(field)!r} but this pipeline "
                    f"has {field}={mine[field]!r}; iterator state is only valid for the "
                    "configuration that produced it"
                )

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
