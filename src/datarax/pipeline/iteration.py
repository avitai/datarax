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

The pattern, and the step running a DAG over one host batch, live in
:mod:`datarax.pipeline.dag_call`. Structurally identical pipelines share compiled steps.
"""

from __future__ import annotations

import contextlib
import weakref
from collections.abc import Callable, Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax.core.element_batch import Batch
from datarax.core.operator import OperatorModule
from datarax.pipeline.compiled import cached_program
from datarax.pipeline.dag_call import (
    apply_writes,
    is_per_batch_state,
    run_tracking_writes,
    state_leaves,
    Writes,
)
from datarax.pipeline.epochs import EpochPlan


# The traceable body a session step runs on the merged module: one batch of the given number of
# records, returned after the module's state has advanced (the pipeline's ``_next_batch``).
type StepBody = Callable[[Any, int], Batch]


# Compiled steps shared across pipelines, matched by structural equality of their
# key (graphs holding lists are unhashable, so the caches are lists, not dicts).
# The most recently used entry is kept last; the oldest is dropped past the bound.
# Held OUTSIDE the modules: storing graphdefs as module attributes would embed them
# into the next split's graphdef, making GraphDef.__eq__ recurse into itself.
_SESSION_STEPS: list[tuple[Any, Callable[..., Any]]] = []

# Device copies of each pipeline's host (NumPy) arrays, uploaded at first use and shared by
# every path that stages the pipeline (``step()`` and iteration sessions). Keyed weakly by the
# pipeline, then by the array's identity with a weak reference to the array, so a copy dies
# with its pipeline or its array, and an array replaced by another is uploaded at its first use.
# NumPy arrays are never tracers, so staging inside a caller's transform caches nothing traced.
type _HostCopies = dict[int, tuple[weakref.ref[np.ndarray], jax.Array]]
_HOST_COPIES: weakref.WeakKeyDictionary[Any, _HostCopies] = weakref.WeakKeyDictionary()

# The layout of PipelineIterator.get_state(). Operators hold no RNG counts, so ``rng_counts`` is
# the pipeline's and those of a source that holds a stream (an in-memory source holds none). A
# state without the field keeps only its counts outside operators.
_ITERATOR_STATE_VERSION = 2
# Version 1 carried ``rng_counts`` in the per-operator layout; version 2 adds ``fingerprint``,
# the configuration that produced the state, which ``set_state`` checks.
_FINGERPRINT_FIELDS = ("batch_size", "length", "drop_last", "num_epochs", "shuffled")


def _session_step(graphdef: Any, body: StepBody, size: int) -> Callable[..., Any]:
    """The compiled step running ``body`` once, for ``size`` records, on modules shaped so.

    Flax's functional hot-loop pattern: merge the state partitions into a module at
    trace time, run one batch, and return it with the state the step wrote. Graph
    traversal happens once per trace instead of once per batch, and structurally
    identical modules running the same body at the same size share the step.
    """

    def run(graph: Any) -> Batch:
        return body(graph, size)

    def build() -> Callable[..., Any]:
        @jax.jit
        def session_step(mutable_state: Any, immutable_state: Any) -> tuple[Batch, Writes]:
            return run_tracking_writes(graphdef, (mutable_state, immutable_state), run)

        return session_step

    return cached_program(_SESSION_STEPS, (graphdef, body, size), build)


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


def staged_view[M: nnx.Module](module: M) -> M:
    """``module`` rebuilt around its live Variables, with its host arrays on the device.

    For a flax transform that takes the module as an argument (``nnx.scan``): given the live
    module, it would take the module's NumPy arrays as arguments and upload them on every call.
    The view shares every Variable object with the live module, so what the transform writes is
    written to the module, and a Variable the module shares with other arguments stays shared.
    Only the arrays outside Variables (a source's records) are replaced, by the copies
    :func:`_on_device` keeps. Only ``module``'s graph is traversed, so the cost does not grow
    with the other arguments (a model).

    Args:
        module: The module to view (the pipeline).

    Returns:
        The view.
    """
    graphdef, variables, data = nnx.split(module, nnx.Variable, ..., graph=True)
    return nnx.merge(graphdef, variables, _on_device(module, data), copy=False)


def next_batch(module: nnx.Module, body: StepBody, size: int) -> Batch:
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
    graphdef, per_batch, staged = nnx.split(module, is_per_batch_state, ..., graph=True)
    batch, writes = _session_step(graphdef, body, size)(per_batch, _on_device(module, staged))
    apply_writes(writes, ((state_leaves(per_batch),), (state_leaves(staged),)))
    return batch


def _operator_owned_counts(
    module: nnx.Module, live_variables: list[Any], rng_count_indices: list[int]
) -> list[bool]:
    """Return whether each of a session's RNG counts belongs to an operator.

    Ownership is decided by Variable identity rather than by state path, so the answer does not
    depend on how a path happens to be spelled. An operator's counts precede the pipeline's and
    the source's, because flax orders attributes by name and ``dag`` sorts before ``rngs`` and
    ``source``; an
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
        for variable in state_leaves(nnx.state(node, nnx.RngCount, graph=True))
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
            shuffled: Whether the pipeline serves records in a shuffled order, recorded in
                the state's fingerprint.
        """
        graphdef, mutable_state, immutable_state = nnx.split(
            module, is_per_batch_state, ..., graph=True
        )
        # A session carries its own Variables for the per-batch state, holding the live values
        # as they are: ``step()`` passes the same values, so both paths present the shared
        # compiled step one call signature (``jax.jit`` keys its dispatch cache on arguments'
        # shardings and committedness as well as avals, so converting a Python-int counter in
        # one path only would add a second entry for the same executable).
        self._state: Any = jax.tree.map(lambda leaf: leaf, mutable_state)
        # The split state holds the module's live Variables by reference;
        # keeping them lets each yield sync the module in O(written leaves).
        self._live_variables = state_leaves(mutable_state)
        self._carried_variables = state_leaves(self._state)
        self._graphdef = graphdef
        self._body = body
        # The step every full batch runs, shared with ``step()``; a run's short final batch
        # looks its own up once.
        self._pure_step = _session_step(graphdef, body, plan.batch_size)
        self._immutable_state = _on_device(module, immutable_state)
        self._receivers = (
            (self._live_variables, self._carried_variables),
            (state_leaves(immutable_state), state_leaves(self._immutable_state)),
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

    def __iter__(self) -> Iterator[Batch]:
        """Return self (iterator protocol)."""
        return self

    def __next__(self) -> Batch:
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
        apply_writes(writes, self._receivers)
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
        shuffling pipeline serves), ``rng_counts`` (per-stream fork counters,
        which determine every stochastic draw) and ``version``, the layout
        those counts are in. Shapes and types are stable across the
        iterator's lifetime.

        ``rng_counts`` holds the pipeline's count, then the source's if it holds
        a stream (an in-memory source holds none). An operator keys each record
        on its base key and holds no count, so the list's length does not depend
        on the operators.

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
