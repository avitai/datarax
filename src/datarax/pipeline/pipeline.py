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
  ``jax.jit`` step, so per-batch cost is one compiled dispatch. A
  streaming source's host batches run through the stage DAG compiled the
  same way. Works with any training framework; the recommended
  data-loading loop.
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

import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax.core import Tracer

from datarax.core.data_source import DataSourceModule
from datarax.core.spec import batch_length, validate_batch, validate_device_dtypes
from datarax.pipeline.dag import record_count, run_dag
from datarax.pipeline.iteration import compile_streaming_dag, declared_spec, PipelineIterator
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
        drop_last: What the last batch of an epoch is when the source's length is
            not a multiple of ``batch_size``. ``False`` (the default) serves it padded
            to ``batch_size`` with the rows past the epoch's end marked invalid in the
            batch's ``valid_mask``; ``True`` does not serve it, PyTorch's rule.
        num_epochs: How many epochs ``iter(pipeline)`` serves before stopping, each
            ending as ``drop_last`` says; ``None`` is a continuous stream whose batches
            cross epoch boundaries without padding.

    Every batch a random-access source serves carries a top-level ``valid_mask`` leaf
    of shape ``(batch_size,)`` and dtype bool, attached after the stages run, so a
    masked loss ignores padding and the stages never see the mask. A streaming source
    carries an all-true mask over the rows it yields.

    Epochs: the pipeline owns the iteration position and the epoch counter.
    Every ``step()`` of an epoch passes the same epoch key to
    ``get_batch_at``, so a shuffled source serves one permutation per epoch
    and each record is visited exactly once; :meth:`reset` starts the next
    epoch at position 0 with a new permutation. Iterating an exhausted
    pipeline yields nothing until it is reset.

    Use :meth:`from_dag` for branching / merging topologies.
    """

    def __init__(  # noqa: PLR0913 - the DAG shape, the batch rule and the streams are separate
        self,
        *,
        source: DataSourceModule,
        batch_size: int,
        rngs: nnx.Rngs,
        stages: Sequence[nnx.Module] | None = None,
        nodes: Mapping[str, nnx.Module] | None = None,
        edges: Mapping[str, Sequence[str]] | None = None,
        sink: str | None = None,
        drop_last: bool = False,
        num_epochs: int | None = 1,
    ) -> None:
        """Initialize the module.

        Args:
            source: The data source (see the class docstring).
            batch_size: Records fetched per ``step()``.
            rngs: ``nnx.Rngs`` for the epoch key and stochastic stages.
            stages: Linear stages, exclusive with ``nodes``/``edges``/``sink``.
            nodes: DAG nodes by name.
            edges: Predecessors of each DAG node.
            sink: The DAG node whose output is returned.
            drop_last: The last-batch rule (see the class docstring).
            num_epochs: Epochs ``iter`` serves, or ``None`` for a continuous stream.

        Raises:
            ValueError: If ``num_epochs`` is neither ``None`` nor at least 1, or a
                continuous stream (``num_epochs=None``) is asked for with a batch larger
                than the source, which no single boundary batch can hold.
        """
        super().__init__()
        if num_epochs is not None and num_epochs < 1:
            raise ValueError(
                f"num_epochs must be at least 1, or None for a stream; got {num_epochs}"
            )
        length = _source_length(source)
        if num_epochs is None and length is not None and batch_size > length:
            raise ValueError(
                f"a continuous stream needs batch_size <= len(source); got batch_size "
                f"{batch_size} over {length} records"
            )
        self.drop_last = drop_last
        self.num_epochs = num_epochs
        self._length: int | None = length

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
        self._epoch: nnx.Variable[jax.Array] = nnx.Variable(jnp.zeros((), dtype=jnp.int32))
        # The epoch key is derived from this base and the epoch counter, so
        # iterator state (position, epoch, rng counts) reproduces every slice.
        self._epoch_key_base: nnx.Variable[jax.Array] = nnx.Variable(jax.random.key_data(rngs()))
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
    def from_dag(  # noqa: DOC502
        cls,
        *,
        source: DataSourceModule,
        nodes: Mapping[str, nnx.Module],
        edges: Mapping[str, Sequence[str]],
        sink: str,
        batch_size: int,
        rngs: nnx.Rngs,
        drop_last: bool = False,
        num_epochs: int | None = 1,
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
            drop_last: The last-batch rule (see the class docstring).
            num_epochs: Epochs ``iter`` serves, or ``None`` for a stream.

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
            drop_last=drop_last,
            num_epochs=num_epochs,
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

        Per-record RNG: stochastic operators key each record on the epoch and on
        the index the source names for the record at that position,
        ``source.record_indices_at(self._position, batch_size, epoch_key)``. During
        ``step()`` the position is the batch's start, so these are exactly the
        records ``get_batch_at`` served. Traceable under ``nnx.jit``/``nnx.scan``.
        """
        size = record_count(batch)
        record_indices = (
            None
            if size is None
            else self.source.record_indices_at(self._position[...], size, self.epoch_key())
        )
        return run_dag(
            self._stage_modules,
            self._exec_order,
            self._predecessors,
            self._sink,
            batch,
            record_indices,
            self._epoch[...],
        )

    def epoch_key(self) -> jax.Array:
        """The key every batch of the current epoch passes to ``get_batch_at``.

        One key per epoch is what makes a shuffled epoch a single permutation.
        """
        base = jax.random.wrap_key_data(self._epoch_key_base[...])
        return jax.random.fold_in(base, self._epoch[...])

    def __len__(self) -> int:
        """Batches per epoch: ceil(N / B), or floor(N / B) under ``drop_last``.

        Returns:
            The number of batches one epoch serves.

        Raises:
            TypeError: If the source has no length.
        """
        if self._length is None:
            raise TypeError(f"{type(self.source).__name__} has no length, so the pipeline has none")
        return _batches_in(self._length, self.batch_size, self.drop_last)

    def batches_left(self) -> int | None:
        """Batches the current epoch still holds from the current position.

        Returns:
            The count, or ``None`` when it is not known: a source without a length, a
            continuous stream (which never runs out), or a call inside a traced function,
            where the position is a tracer rather than a number.
        """
        if self._length is None or self.num_epochs is None:
            return None
        position = self._position[...]
        if isinstance(position, Tracer):
            return None
        remaining = max(self._length - int(position), 0)
        return _batches_in(remaining, self.batch_size, self.drop_last)

    def reset(self) -> None:
        """Start the next epoch: position 0, epoch counter advanced.

        A shuffled source serves a new permutation; a sequential source serves
        the same order again. Sessions in progress see the change at their next
        ``iter()``.
        """
        self._position[...] = jnp.zeros((), dtype=jnp.int32)
        self._epoch[...] = self._epoch[...] + jnp.int32(1)

    @nnx.jit
    def step(self) -> dict:
        """Fetch one batch from the source and run it through the DAG.

        Reads ``self._position``, fetches via
        ``source.get_batch_at(position, batch_size, epoch_key)``, runs
        ``__call__``, advances ``self._position`` by ``batch_size``.
        The method is JAX-traceable; the DAG iteration unrolls during
        tracing.
        """
        return self._next_batch()

    def _next_batch(self) -> dict:
        """Fetch the batch at the position, run the DAG and advance the position.

        The traceable body of :meth:`step`. Compiled iteration sessions call it
        directly: nesting ``nnx.jit`` inside their ``jax.jit`` step would rebind
        every Variable and hide which ones the step wrote.

        Returns:
            The sink output for the batch at the current position.
        """
        idx = self._position[...]
        size = self.batch_size
        if self.num_epochs is None and self._length is not None:
            return self._continuous_batch(idx)
        batch = self.source.get_batch_at(idx, size, self.epoch_key())
        # __call__ reads self._position (== idx here) to ask the source which records it
        # served, so a subclass overriding __call__ still runs; advance only afterwards.
        # Under jit XLA computes the shuffle both calls share once.
        batch = self(batch)
        self._position[...] = idx + jnp.int32(size)
        if self._length is None:
            mask = jnp.ones((size,), dtype=jnp.bool_)
        else:
            mask = idx + jnp.arange(size, dtype=jnp.int32) < jnp.int32(self._length)
        return _with_valid_mask(batch, mask, self._sink)

    def _continuous_batch(self, idx: jax.Array) -> dict:
        """One batch of the continuous stream, crossing the epoch boundary when it must.

        The rows before the boundary come from this epoch's order at ``idx``; the rows
        after it are the head of the next epoch's order, so no row is padding. The
        stages key each record on the epoch the batch started in.

        Args:
            idx: The position the batch starts at, within this epoch.

        Returns:
            The sink output with an all-true ``valid_mask``.
        """
        size, length = self.batch_size, self._length
        assert length is not None  # noqa: S101 - the caller checked
        base = jax.random.wrap_key_data(self._epoch_key_base[...])
        key_now = jax.random.fold_in(base, self._epoch[...])
        key_next = jax.random.fold_in(base, self._epoch[...] + jnp.int32(1))
        offsets = jnp.arange(size, dtype=jnp.int32)
        in_epoch = idx + offsets < jnp.int32(length)
        head_rows = jnp.clip(idx + offsets - jnp.int32(length), 0, size - 1)

        def pick(tail_leaf: jax.Array, head_leaf: jax.Array) -> jax.Array:
            chosen_head = jnp.take(head_leaf, head_rows, axis=0)
            select = jnp.reshape(in_epoch, (size,) + (1,) * (tail_leaf.ndim - 1))
            return jnp.where(select, tail_leaf, chosen_head)

        tail = self.source.get_batch_at(idx, size, key_now)
        head = self.source.get_batch_at(0, size, key_next)
        batch = jax.tree.map(pick, tail, head)
        record_indices = jnp.where(
            in_epoch,
            self.source.record_indices_at(idx, size, key_now),
            jnp.take(self.source.record_indices_at(0, size, key_next), head_rows),
        )
        batch = run_dag(
            self._stage_modules,
            self._exec_order,
            self._predecessors,
            self._sink,
            batch,
            record_indices,
            self._epoch[...],
        )
        consumed = idx + jnp.int32(size)
        self._epoch[...] = self._epoch[...] + consumed // jnp.int32(length)
        self._position[...] = consumed % jnp.int32(length)
        return _with_valid_mask(batch, jnp.ones((size,), dtype=jnp.bool_), self._sink)

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

        Raises:
            ValueError: If ``length`` exceeds the batches left in the epoch, when the
                position is known (an eager call); a continuous stream
                (``num_epochs=None``) scans any length, and a ``scan`` traced inside
                another transformation is not checked.
        """
        left = self.batches_left()
        if left is not None and length > left:
            raise ValueError(
                f"scan(length={length}) exceeds the {left} batches left in the epoch "
                f"(position {int(self._position[...])} of {self._length} records, batch_size "
                f"{self.batch_size}); reset() starts the next epoch"
            )
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
        iterate indefinitely. Streaming sources (no ``get_batch_at``) pull
        batches on the host and run them through the compiled stage DAG via
        :meth:`_iter_streaming` instead.

        Returns:
            A :class:`~datarax.pipeline.iteration.PipelineIterator` for a source
            with indexed access, otherwise a generator over streamed batches.

        Raises:
            TypeError: If the source implements neither ``get_batch_at`` nor
                ``get_batch``.
        """
        if self.source.supports_indexed_access():
            return PipelineIterator(self)
        if not callable(getattr(self.source, "get_batch", None)):
            raise TypeError(
                f"{type(self.source).__name__} implements neither get_batch_at (indexed "
                "access) nor get_batch (streaming), so Pipeline cannot iterate it."
            )
        return self._iter_streaming()

    def _iter_streaming(self) -> Iterator[dict]:  # noqa: DOC502
        """Iterate a streaming source (sequential, no random access) through the DAG.

        Streaming sources have no ``get_batch_at``, so batches are pulled on the
        host with ``get_batch`` and run through the stage DAG, compiled once per
        batch shape by :func:`~datarax.pipeline.iteration.compile_streaming_dag`;
        the final batch may be short. Iteration ends when the source is exhausted
        (``get_batch`` returns an empty batch).

        The source's ``element_spec()`` is read once per source and x64 setting and
        is refused if it declares a dtype JAX arrays cannot hold as declared: a
        ``float64`` field while x64 is off would otherwise be narrowed silently
        inside the compiled DAG. Every batch is checked against it with
        :func:`~datarax.core.spec.validate_batch` before it reaches the DAG, so a
        batch whose structure, per-element shapes, dtypes or record counts disagree
        with the declaration stops iteration with the fields named. Pipeline state
        is current at every yield.

        Yields:
            One transformed batch dict per source batch, until exhaustion.

        Raises:
            SpecMismatchError: If the declared spec, or a source batch, breaks the
                contract above.
            ValueError: If a stage adds or removes state while it runs.
        """
        element_spec = declared_spec(self.source)
        validate_device_dtypes(element_spec)
        apply = compile_streaming_dag(self)
        while True:
            batch = self.source.get_batch(self.batch_size)  # type: ignore[attr-defined]
            size = batch_length(batch)
            if not size:
                return
            validate_batch(batch, element_spec, batch_size=self.batch_size)
            yield _with_valid_mask(apply(batch), jnp.ones((size,), dtype=jnp.bool_), self._sink)


def _source_length(source: DataSourceModule) -> int | None:
    """The source's length, or ``None`` when it has none."""
    try:
        return len(source)
    except (TypeError, NotImplementedError):
        return None


def _batches_in(records: int, batch_size: int, drop_last: bool) -> int:
    """Batches ``records`` records make: floor under ``drop_last``, ceil otherwise."""
    return records // batch_size if drop_last else math.ceil(records / batch_size)


def _with_valid_mask(batch: Any, mask: jax.Array, sink: str | None) -> dict:
    """Attach ``mask`` as the batch's top-level ``valid_mask`` leaf.

    A mask the batch already carries (a batcher stage's) is combined with this one, so
    a row is valid only when both say so.

    Args:
        batch: The sink output, a mapping of leaves.
        mask: Validity of each row.
        sink: The sink node's name, for the error.

    Returns:
        ``batch`` with ``valid_mask``.

    Raises:
        TypeError: If the sink output is not a mapping, which cannot carry the mask.
    """
    if not isinstance(batch, Mapping):
        raise TypeError(
            f"the sink {sink!r} returned {type(batch).__name__}, but a pipeline batch is a "
            "mapping so it can carry valid_mask"
        )
    existing = batch.get("valid_mask")
    if existing is not None:
        mask = jnp.logical_and(mask, jnp.asarray(existing, dtype=jnp.bool_))
    return {**batch, "valid_mask": mask}
