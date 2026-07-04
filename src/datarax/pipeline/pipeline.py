"""``Pipeline(nnx.Module)`` — JAX-native data pipeline with scan-based epochs.

A pipeline is a source plus a graph of stages. Each stage is an
``nnx.Module`` whose ``__call__`` takes one or more batches and returns
one batch. The pipeline drives source iteration via an ``nnx.Variable``
position counter so a full training epoch can be expressed as a single
``nnx.scan`` call, producing one XLA graph per epoch.

Three integration tiers (user picks based on speed/flexibility tradeoff):

- **Tier A — ``for batch in pipeline:``** — compiled iteration session
  (:class:`~datarax.pipeline.iteration.PipelineIterator`). The module
  graph is split once per session and batches run through a cached
  ``jax.jit`` step, so per-batch cost is one compiled dispatch. Works
  with any training framework; the recommended data-loading loop.
- **Tier B — ``Pipeline.step()``** — single JIT-traceable batch fetch
  with live module state. For single-shot use or embedding inside your
  own jitted train step, where the outer trace absorbs the call.
- **Tier C — ``Pipeline.scan(step_fn, modules=(...), length=...)``** —
  the convenience wrapper. Pipeline lifts user-supplied ``nnx.Module``
  instances (typically a model and an ``nnx.Optimizer``) via
  ``nnx.StateAxes`` so the user never writes scan boilerplate. Fuses
  the whole epoch (data + train step) into one XLA call.

Two construction shapes (both produce identical internal execution plans):

- ``Pipeline(*, source, stages, batch_size, rngs)`` — linear chain.
  Convenience for the common case ``source → s1 → s2 → ...``. Internally
  builds a trivial DAG.
- ``Pipeline.from_dag(*, source, nodes, edges, sink, batch_size, rngs)`` —
  declarative DAG. Each node's ``__call__`` receives its predecessors'
  outputs as positional arguments. ``edges`` maps each node name to the
  list of predecessor names; an empty list means "consumes the source
  batch directly." Topological sort + cycle / connectivity validation
  happen at construction.

Public surface:

- ``__init__`` and ``from_dag`` — see above.
- ``__call__(batch)`` — runs the DAG forward; returns the sink output.
  JAX-traceable.
- ``step()`` — fetches one batch from the source, runs ``__call__``,
  advances ``_position`` and ``rngs``. JAX-traceable.
- ``scan(step_fn, *, length, modules=(), init_carry=None)`` — runs
  ``length`` steps under ``nnx.scan``, lifting pipeline + ``modules``
  state via ``StateAxes``. See method docstring for the two ``step_fn``
  signatures.
- ``__iter__()`` — compiled iteration session; see Tier A above and
  ``datarax.pipeline.iteration`` for state/checkpoint semantics.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from datarax.core.data_source import DataSourceModule
from datarax.pipeline.iteration import PipelineIterator
from datarax.pipeline.topo import topological_sort, validate_dag


class Pipeline(nnx.Module):
    """JAX-native data pipeline with scan-based epoch iteration.

    Args (linear constructor):
        source: A ``DataSourceModule`` exposing
            ``get_batch_at(start, size, key)`` for stateless indexed
            access. The pipeline owns iteration position; the source
            does not advance internal state.
        stages: Ordered list of ``nnx.Module`` stages applied in order.
            Each stage's ``__call__(batch)`` takes the current batch (a
            dict of arrays) and returns the next batch. Stages may carry
            learnable parameters via ``nnx.Param`` — gradients flow
            through ``scan`` via ``StateAxes``.
        batch_size: Number of records fetched per ``step()`` call.
        rngs: ``nnx.Rngs`` consumed by stochastic stages and the source.

    Use :meth:`from_dag` for branching / merging topologies.
    """

    def __init__(
        self,
        *,
        source: DataSourceModule,
        batch_size: int,
        rngs: nnx.Rngs,
        stages: Sequence[nnx.Module] | None = None,
        nodes: Mapping[str, nnx.Module] | None = None,
        edges: Mapping[str, Sequence[str]] | None = None,
        sink: str | None = None,
    ) -> None:
        """Initialize the module."""
        super().__init__()

        # Resolve the construction shape into the unified DAG representation.
        resolved_nodes, resolved_edges, resolved_sink = self._resolve_dag_shape(
            stages, nodes, edges, sink
        )

        if resolved_sink is not None:
            validate_dag(resolved_nodes, resolved_edges, resolved_sink)
            exec_order = topological_sort(resolved_edges)
        else:
            exec_order = []

        self.source = source
        self.batch_size = batch_size
        self.rngs = rngs
        # Modules holding state across iterations must be marked as data.
        self._stage_modules = nnx.data(resolved_nodes)
        # The exec plan and predecessor map are static structure; they
        # do not change after construction.
        self._exec_order: list[str] = list(exec_order)
        self._predecessors: dict[str, list[str]] = resolved_edges
        self._sink: str | None = resolved_sink
        self._position: nnx.Variable[jax.Array] = nnx.Variable(jnp.zeros((), dtype=jnp.int32))
        # Cache of compiled @nnx.scan bodies, keyed on the call signature.
        # The decorator must be applied to a stable function identity for
        # nnx.scan's JIT cache to hit on subsequent calls; rebuilding on
        # every scan() call defeats the cache and forces re-tracing. The
        # cache is a plain dict (not nnx.Variable) so it stays static
        # under nnx.split/merge — it lives on the Python side of the
        # module boundary alongside _exec_order and _predecessors.
        self._scan_body_cache: dict[Any, Any] = {}

    @staticmethod
    def _linear_shape(
        stages: Sequence[nnx.Module],
    ) -> tuple[dict[str, nnx.Module], dict[str, list[str]], str | None]:
        """Expand a linear stage sequence into (nodes, edges, sink).

        Stage ``i`` reads the source when ``i == 0`` and stage ``i - 1`` otherwise;
        the sink is the final stage (``None`` for an empty sequence).

        Args:
            stages: Ordered stages of the linear pipeline.

        Returns:
            Tuple of (node map, predecessor-edge map, sink name).
        """
        resolved_nodes: dict[str, nnx.Module] = {f"stage_{i}": s for i, s in enumerate(stages)}
        resolved_edges: dict[str, list[str]] = {
            f"stage_{i}": ([f"stage_{i - 1}"] if i > 0 else []) for i in range(len(stages))
        }
        resolved_sink = f"stage_{len(stages) - 1}" if stages else None
        return resolved_nodes, resolved_edges, resolved_sink

    @staticmethod
    def _resolve_dag_shape(
        stages: Sequence[nnx.Module] | None,
        nodes: Mapping[str, nnx.Module] | None,
        edges: Mapping[str, Sequence[str]] | None,
        sink: str | None,
    ) -> tuple[dict[str, nnx.Module], dict[str, list[str]], str | None]:
        """Normalize the two mutually exclusive construction shapes into one DAG.

        Accepts either ``stages=`` (linear) or ``nodes=+edges=+sink=`` (explicit DAG)
        and returns the unified representation used internally.

        Args:
            stages: Linear stage sequence, or ``None`` for the DAG shape.
            nodes: Explicit node map, or ``None`` for the linear shape.
            edges: Explicit predecessor-edge map (required with ``nodes``).
            sink: Explicit sink node name (required with ``nodes``).

        Returns:
            Tuple of (node map, predecessor-edge map, sink name).

        Raises:
            ValueError: If neither shape or both shapes are supplied.
        """
        if stages is not None and nodes is not None:
            raise ValueError(
                "Provide either stages= (linear) or nodes=+edges=+sink= (DAG); not both."
            )
        if stages is None and nodes is None:
            raise ValueError("Provide either stages= (linear) or nodes=+edges=+sink= (DAG).")

        if stages is not None:
            return Pipeline._linear_shape(stages)

        assert nodes is not None and edges is not None and sink is not None  # noqa: S101
        resolved_nodes = dict(nodes)
        resolved_edges = {name: list(preds) for name, preds in edges.items()}
        return resolved_nodes, resolved_edges, sink

    @classmethod
    def from_dag(
        cls,
        *,
        source: DataSourceModule,
        nodes: Mapping[str, nnx.Module],
        edges: Mapping[str, Sequence[str]],
        sink: str,
        batch_size: int,
        rngs: nnx.Rngs,
    ) -> Pipeline:
        """Build a Pipeline whose stages execute in user-specified topological order.

        Args:
            source: Data source (same contract as the linear constructor).
            nodes: Map from node name to ``nnx.Module``. Each node's
                ``__call__`` receives its predecessors' outputs as
                positional arguments (or the source batch if it has no
                predecessors).
            edges: Map from each node name to the list of predecessor
                node names. Empty list means "consumes source directly."
            sink: Name of the node whose output is returned by
                ``__call__``.
            batch_size: Records fetched per ``step()``.
            rngs: ``nnx.Rngs`` for stochastic stages and source.

        Raises:
            ValueError: If ``edges`` describes a cycle, references
                unknown nodes, or ``sink`` is not in ``nodes``.

        Returns:
            A configured ``Pipeline`` instance.
        """
        return cls(
            source=source,
            nodes=nodes,
            edges=edges,
            sink=sink,
            batch_size=batch_size,
            rngs=rngs,
        )

    # ------------------------------------------------------------------
    # Linear-stages compatibility shim
    # ------------------------------------------------------------------

    @property
    def stages(self) -> list[nnx.Module]:
        """Ordered list of stages for the linear-pipeline shape.

        Reconstructed from the underlying execution plan in topological
        order. For DAG-shaped pipelines (``from_dag``), this returns the
        nodes in their compiled order; the order is well-defined but the
        list does not preserve "linear chain" semantics.
        """
        return [self._stage_modules[name] for name in self._exec_order]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, batch: dict) -> dict:
        """Run the DAG forward, returning the sink node's output.

        Iterates the pre-computed topological order; each node receives
        either the source ``batch`` (if it has no predecessors) or the
        outputs of its predecessor nodes as positional arguments.
        JAX/XLA traces the static composition; no runtime tree-walking.

        Stage dispatch handles two stage shapes uniformly:

        - ``OperatorModule`` subclasses (and any module exposing
          ``_apply_on_raw(data, states)``) are called via the raw
          dict-based path so they integrate without a ``Batch``
          wrapper allocation per step. States are threaded but
          discarded at the sink (consistent with the legacy
          fused-step semantics).
        - Plain ``nnx.Module`` stages with ``__call__(batch) -> batch``
          are called directly. This is the recommended shape for new
          pipelines.

        Per-record RNG: stochastic operators are keyed on the batch's global
        record positions ``self._position + arange(batch_size)`` (the Pipeline
        owns the position counter; during ``step()`` it equals the batch's start
        index), so augmentation is invariant to batch composition, host count,
        and resume point. Traceable under ``nnx.jit``/``nnx.scan``.
        """
        if not self._exec_order or self._sink is None:
            return batch

        global_indices = self._global_indices_for(batch, self._position[...])

        outputs: dict[str, Any] = {}
        states: dict[str, Any] = {}
        for name in self._exec_order:
            preds = self._predecessors[name]
            if not preds:
                inputs: tuple[Any, ...] = (batch,)
            else:
                inputs = tuple(outputs[p] for p in preds)
            stage = self._stage_modules[name]
            apply_on_raw = getattr(stage, "_apply_on_raw", None)
            if callable(apply_on_raw) and len(inputs) == 1:
                # OperatorModule fast path: dict + states threading.
                data, states = apply_on_raw(inputs[0], states, None, global_indices)  # type: ignore[reportGeneralTypeIssues]
                outputs[name] = data
            else:
                outputs[name] = stage(*inputs)
        return outputs[self._sink]

    @staticmethod
    def _global_indices_for(batch: dict, start_index: jax.Array | int | None) -> jax.Array | None:
        """Per-record global indices for ``batch``, or None for the arange fallback.

        ``batch_size`` comes from the leading axis of the first array leaf (static
        under tracing); ``start_index`` may be a traced scalar. The result is
        ``start_index + arange(batch_size)``.
        """
        if start_index is None:
            return None
        leaves = jax.tree.leaves(batch)
        if not leaves:
            return None
        batch_size = leaves[0].shape[0]
        return jnp.asarray(start_index, dtype=jnp.int32) + jnp.arange(batch_size, dtype=jnp.int32)

    @nnx.jit
    def step(self) -> dict:
        """Fetch one batch from the source and run it through the DAG.

        Reads ``self._position``, fetches via
        ``source.get_batch_at(position, batch_size, key)``, runs
        ``__call__``, advances ``self._position`` by ``batch_size``.
        The method is JAX-traceable; the DAG iteration unrolls during
        tracing.
        """
        idx = self._position[...]
        key = self.rngs()
        batch = self.source.get_batch_at(idx, self.batch_size, key)
        # __call__ reads self._position (== idx here) to key per-record RNG on
        # stable global indices (idx + arange); advance only afterwards.
        batch = self(batch)
        self._position[...] = idx + jnp.int32(self.batch_size)
        return batch

    def scan(
        self,
        step_fn: Callable,
        *,
        length: int,
        modules: tuple[Any, ...] = (),
        init_carry: Any = None,
    ) -> Any:
        """Run ``length`` steps under ``nnx.scan``, lifting pipeline + modules state.

        Pipeline generates an ``nnx.scan`` body that lifts ``self`` and
        every entry in ``modules`` via ``nnx.StateAxes({...: nnx.Carry})``.
        Mutations performed inside ``step_fn`` (parameter updates,
        optimizer state advancement, RNG consumption) survive across
        iterations, and the entire scan body compiles to one XLA graph.

        ``step_fn`` has two signatures depending on whether
        ``init_carry`` is provided:

        - Without ``init_carry``: ``step_fn(*modules, batch) -> output``.
          Returns the per-step outputs stacked along axis 0.
        - With ``init_carry``: ``step_fn(carry, *modules, batch) ->
          (new_carry, output)``. Returns ``(final_carry, stacked_outputs)``.

        Args:
            step_fn: Per-step body. Typical training shape:
                ``(model, optimizer, batch) -> loss``.
            length: Number of scan iterations.
            modules: User ``nnx.Module`` instances whose state should be
                lifted across iterations (e.g. ``(model, optimizer)``).
                Empty tuple for body functions that need no extra modules.
            init_carry: Optional carry threaded through ``step_fn``.

        Returns:
            Stacked outputs (``init_carry is None``) or
            ``(final_carry, stacked_outputs)`` (``init_carry`` provided).
        """
        n_modules = len(modules)
        has_init_carry = init_carry is not None
        # Cache key: identity of step_fn plus the structural shape of the
        # call (length, n_modules, carry presence). Identity-based
        # caching matches JAX's standard JIT-cache contract; users who
        # rebind step_fn between calls correctly get a fresh trace.
        cache_key = (id(step_fn), length, n_modules, has_init_carry)
        scan_body = self._scan_body_cache.get(cache_key)
        if scan_body is None:
            scan_body = self._compile_scan_body(
                step_fn=step_fn,
                length=length,
                n_modules=n_modules,
                has_init_carry=has_init_carry,
            )
            self._scan_body_cache[cache_key] = scan_body

        steps = jnp.arange(length, dtype=jnp.int32)
        if has_init_carry:
            return scan_body(self, *modules, init_carry, steps)
        return scan_body(self, *modules, steps)

    def _compile_scan_body(
        self,
        *,
        step_fn: Callable,
        length: int,
        n_modules: int,
        has_init_carry: bool,
    ) -> Callable:
        """Build an ``@nnx.scan``-decorated body for one (step_fn, length) signature.

        Called once per unique cache key by ``scan``. The decorated
        function is keyed in nnx.scan's internal JIT cache by its
        identity, so reusing the returned callable on subsequent
        ``scan`` invocations hits the warm cache and avoids re-tracing.
        Without this caching, every ``scan`` call rebuilds the
        decorated body, identity changes, and tracing repeats — turning
        what should be amortized epoch execution into per-call
        recompilation.
        """
        del length  # captured by the steps array passed at call time
        state_axes = nnx.StateAxes({...: nnx.Carry})

        if not has_init_carry:
            in_axes = (state_axes,) + (state_axes,) * n_modules + (0,)

            @nnx.scan(in_axes=in_axes, out_axes=0)
            def scan_body(*args: Any) -> Any:
                pipeline, *user_modules_and_step = args
                user_modules = user_modules_and_step[:-1]
                batch = pipeline.step()
                return step_fn(*user_modules, batch)

            return scan_body

        in_axes = (state_axes,) + (state_axes,) * n_modules + (nnx.Carry, 0)

        @nnx.scan(in_axes=in_axes, out_axes=(nnx.Carry, 0))
        def scan_body_with_carry(*args: Any) -> tuple[Any, Any]:
            pipeline, *rest = args
            user_modules = rest[:n_modules]
            carry = rest[n_modules]
            batch = pipeline.step()
            return step_fn(carry, *user_modules, batch)

        return scan_body_with_carry

    def __iter__(self) -> PipelineIterator | Iterator[dict]:
        """Iterate batches through a compiled session (the Tier-A fast path).

        Random-access sources return a :class:`~datarax.pipeline.iteration.
        PipelineIterator`: the module graph is split once per session and
        batches are driven through a cached ``jax.jit`` step, with module
        state written back when the session ends (exhaustion, ``close()``,
        or garbage collection after an early break). Iteration stops when
        the position exceeds the source length; sources without ``__len__``
        iterate indefinitely. Streaming sources (no ``get_batch_at``) are
        driven sequentially via :meth:`_iter_streaming` instead.
        """
        if not self.source.supports_indexed_access():
            return self._iter_streaming()
        return PipelineIterator(self)

    def _iter_streaming(self) -> Iterator[dict]:
        """Iterate a streaming source (sequential, no random access) through the DAG.

        Streaming sources have no ``get_batch_at``, so batches are pulled
        sequentially with ``get_batch`` and run through the DAG eagerly — outside
        ``step``'s jit, since the final batch may be short. Iteration ends when the
        source is exhausted (``get_batch`` returns an empty batch).

        Yields:
            One transformed batch dict per source batch, until exhaustion.
        """
        while True:
            batch = self.source.get_batch(self.batch_size)  # type: ignore[attr-defined]
            leaves = jax.tree.leaves(batch)
            if not leaves or leaves[0].shape[0] == 0:
                return
            idx = self._position[...]
            # Compiled DAG apply: one trace per batch shape (the final short
            # batch retraces once) instead of eager per-batch op dispatch.
            output = _jitted_dag_apply(self, batch)
            self._position[...] = idx + jnp.int32(leaves[0].shape[0])
            yield output


# Compiled DAG apply for the streaming iteration path. Module-level so
# nnx.jit's trace cache keys on a stable function identity; per-call graph
# traversal remains (acceptable for streaming, whose per-batch cost is
# dominated by the source pull), but the DAG body itself is compiled.
_jitted_dag_apply = nnx.jit(Pipeline.__call__)
