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
"""

from __future__ import annotations

import contextlib
import weakref
from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


if TYPE_CHECKING:
    from collections.abc import Iterator

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


def _session_cache_size(pipeline: Pipeline) -> int:
    """Number of compiled session steps cached for ``pipeline`` (testing)."""
    return len(_SESSION_STEP_CACHES.get(pipeline, []))


class PipelineIterator:
    """Compiled iteration session over a random-access pipeline source.

    Args:
        pipeline: The pipeline to iterate. Its state is captured at
            construction and written back when the session ends.
    """

    def __init__(self, pipeline: Pipeline) -> None:
        """Split the pipeline once and prepare the compiled session step."""
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
        for variable, new_value in zip(
            self._live_variables, _state_leaves(self._state), strict=True
        ):
            variable.set_value(new_value.get_value())
        self._position += self._batch_size
        return batch

    def get_state(self) -> dict[str, Any]:
        """Return iterator state valid at the current yield boundary.

        The state names the batches already yielded to the caller:
        ``position`` (records consumed) and ``rng_counts`` (per-stream fork
        counters, which determine every stochastic draw). Shapes and types
        are stable across the iterator's lifetime.

        Returns:
            JSON-serializable dict with ``position`` and ``rng_counts``.
        """
        counts = [int(self._live_variables[index].get_value()) for index in self._rng_count_indices]
        return {"position": np.int64(self._position), "rng_counts": counts}

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore iterator state produced by :meth:`get_state`.

        The pipeline must be configured identically (same structure, same
        seeds) to the one that produced the state.

        Args:
            state: Dict with ``position`` and ``rng_counts`` entries.
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
