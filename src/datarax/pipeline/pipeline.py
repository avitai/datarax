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

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from datarax.core.data_source import DataSourceModule
from datarax.core.spec import batch_length, declared_spec, validate_batch, validate_device_dtypes
from datarax.pipeline.dag import record_count, Records, run_dag
from datarax.pipeline.epochs import EpochPlan
from datarax.pipeline.iteration import compile_streaming_dag, next_batch, PipelineIterator
from datarax.pipeline.topo import topological_sort, validate_dag


class Pipeline(nnx.Module):
    """JAX-native data pipeline with scan-based epoch iteration.

    Args (linear constructor):
        source: A ``DataSourceModule`` exposing ``record_indices_at`` and
            ``get_records`` for stateless indexed access. The pipeline owns
            iteration position; the source does not advance internal state.
        stages: Ordered list of ``nnx.Module`` stages applied in order.
            Each stage's ``__call__(batch)`` takes the current batch (a
            dict of arrays) and returns the next batch. Stages may carry
            learnable parameters via ``nnx.Param`` — gradients flow
            through ``scan`` via ``StateAxes``.
        batch_size: Number of records fetched per ``step()`` call.
        rngs: ``nnx.Rngs`` consumed by stochastic stages and the source.
        drop_last: What a batch reaching the end of an epoch holds when the source's
            length is not a multiple of ``batch_size``; no batch holds padding.
            ``False`` (the default) completes it from the head of the next epoch's order,
            tf.data's and Grain's ``repeat().batch()``: every epoch serves each record
            once. ``True`` skips the records short of a full batch and starts the next
            epoch, their ``batch(drop_remainder=True).repeat()``.
        num_epochs: How many epochs ``iter(pipeline)`` serves before stopping, or
            ``None`` for a stream that never stops. Under ``drop_last=False`` it stops
            after exactly ``num_epochs * len(source)`` records, so its final batch may
            be short.

    Epochs: the pipeline owns the iteration position and the epoch counter. Every
    batch passes each record's epoch key to ``record_indices_at``, so a shuffled source
    serves one permutation per epoch and each record is visited once per epoch.
    ``step()`` and ``scan`` never run out: the compiled step starts the next epoch
    when the current one cannot start another batch. :meth:`reset` starts the next
    epoch at position 0 with a new permutation. Iterating a pipeline whose run has
    ended yields nothing until it is reset.

    Use :meth:`from_dag` for branching / merging topologies.
    """

    def __init__(  # noqa: PLR0913, DOC502 - the DAG shape, the batch rule and the streams are separate
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
            num_epochs: Epochs ``iter`` serves, or ``None`` for a stream that never stops.

        Raises:
            ValueError: If ``num_epochs`` is neither ``None`` nor at least 1, the source
                has no records, or ``drop_last`` is set with a ``batch_size`` above the
                source's length, which serves no batch.
        """
        super().__init__()
        # Refuses an epoch rule no pipeline can serve.
        EpochPlan(
            length=_source_length(source),
            batch_size=batch_size,
            drop_last=drop_last,
            num_epochs=num_epochs,
        )
        self.drop_last = drop_last
        self.num_epochs = num_epochs

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

    def __call__(self, batch: dict, records: Records | None = None) -> dict:
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

        Per-record RNG: stochastic operators key each record on its index and epoch,
        ``records``. ``step()`` passes the records it served, computed once for the gather and
        the stages; a direct call without them names the records served at the current position
        by the same rule. A subclass overriding ``__call__`` accepts ``records`` and passes it on
        where it keys randomness. Traceable under ``nnx.jit``/``nnx.scan``.

        Args:
            batch: The source batch.
            records: The batch's records, or ``None`` to name them from the position.

        Returns:
            The sink node's output.
        """
        size = record_count(batch)
        if records is None and size is not None:
            start, epoch = self.epoch_plan.batch_start(self._position[...], self._epoch[...])
            records = self._records_at(start, epoch, size)
        return run_dag(
            self._stage_modules,
            self._exec_order,
            self._predecessors,
            self._sink,
            batch,
            None if records is None else records.indices,
            self._epoch[...] if records is None else records.epochs,
        )

    def epoch_key(self) -> jax.Array:
        """The key the current epoch's records are ordered by (``record_indices_at``).

        One key per epoch is what makes a shuffled epoch a single permutation.
        """
        return self._key_of(self._epoch[...])

    def _key_of(self, epoch: jax.Array) -> jax.Array:
        """The key ``record_indices_at`` orders epoch ``epoch`` by."""
        return jax.random.fold_in(jax.random.wrap_key_data(self._epoch_key_base[...]), epoch)

    def _records_at(self, start: jax.Array, epoch: jax.Array, size: int) -> Records:
        """The ``size`` records served from ``start`` of epoch ``epoch``'s order.

        When the plan crosses epochs, rows past the epoch's end are the head of the following
        epochs' orders, so no row is padding. Every epoch the batch can touch is named by one
        ``record_indices_at`` vmapped over the epochs' keys (the first from ``start``, the rest
        from their heads) and each row takes its own epoch's name: no conditional, one batched
        index computation (a shuffle's cycle-walking loop runs once for all epochs), the same
        program for every batch, and index arrays of O(``size``) per epoch touched.

        Args:
            start: Where the rows start in epoch ``epoch``.
            epoch: The epoch the first row belongs to.
            size: Rows to serve.

        Returns:
            Each row's record index and epoch.
        """
        plan = self.epoch_plan
        start = jnp.asarray(start, dtype=jnp.int32)
        epoch = jnp.asarray(epoch, dtype=jnp.int32)

        def names(first: jax.Array, key: jax.Array) -> jax.Array:
            return jnp.asarray(self.source.record_indices_at(first, size, key), jnp.int32)

        if not plan.crosses:
            return Records(names(start, self._key_of(epoch)), jnp.full((size,), epoch, jnp.int32))
        length = plan.length
        assert length is not None  # noqa: S101 - a crossing plan has a length
        offsets = jnp.arange(plan.epochs_touched(size), dtype=jnp.int32)
        starts = jnp.where(offsets == 0, start, 0)
        named = jax.vmap(names)(starts, jax.vmap(self._key_of)(epoch + offsets))
        rows = start + jnp.arange(size, dtype=jnp.int32)
        later = rows // length  # epochs after ``epoch`` each row belongs to
        # A row of the first epoch is its own row of that epoch's names; a later epoch's row is
        # the position it reaches within that epoch, counted from its head.
        column = jnp.where(later == 0, jnp.arange(size, dtype=jnp.int32), rows - later * length)
        return Records(named[later, column], epoch + later)

    @property
    def epoch_plan(self) -> EpochPlan:
        """How this pipeline's records divide into batches and epochs.

        Built from the source's current length and the pipeline's batch rule each time,
        so a length or batch size changed since construction is what it reports.
        """
        return EpochPlan(
            length=_source_length(self.source),
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            num_epochs=self.num_epochs,
        )

    def __len__(self) -> int:
        """Batches a run of ``num_epochs`` epochs serves from the start of an epoch.

        ``ceil(num_epochs * N / B)`` batches, the last possibly short, or
        ``num_epochs * floor(N / B)`` under ``drop_last``.

        Returns:
            The batch count.

        Raises:
            TypeError: If the source has no length, or the pipeline is a stream
                (``num_epochs=None``), which never ends.
        """
        extent = self.epoch_plan.run_extent(0)
        if extent is None:
            raise TypeError(
                f"a pipeline over {type(self.source).__name__} with num_epochs="
                f"{self.num_epochs} never ends, so it has no length"
            )
        return extent[0]

    def batches_left(self) -> int | None:
        """Batches a session started now would serve: the rest of the run from the position.

        Host-side: it reads the position as a number, so call it outside traced code.

        Returns:
            The count, or ``None`` for a pipeline that never ends: a source without a
            length, or a stream (``num_epochs=None``).
        """
        extent = self.epoch_plan.run_extent(int(self._position[...]))
        return None if extent is None else extent[0]

    def reset(self) -> None:
        """Start the next epoch: position 0, epoch counter advanced.

        A shuffled source serves a new permutation; a sequential source serves
        the same order again. Sessions in progress see the change at their next
        ``iter()``.
        """
        position, epoch = EpochPlan.next_epoch(self._epoch[...])
        self._position[...] = jnp.asarray(position, dtype=jnp.int32)
        self._epoch[...] = epoch

    def step(self) -> dict:
        """Serve the next batch from the source through the DAG.

        Starts where the last batch ended, or at the next epoch when the current one cannot
        start another batch (see :class:`~datarax.pipeline.epochs.EpochPlan`), names
        ``batch_size`` records with ``source.record_indices_at`` and gathers them with
        ``source.get_records``, runs ``__call__`` and advances the position and epoch. It never
        runs out.

        Runs the compiled step iteration sessions use, so it copies none of the source's
        arrays: device data is read in place, and NumPy data is uploaded once per array and
        stays NumPy in the source. Called inside a transform (``nnx.jit``, ``nnx.grad``,
        ``nnx.scan``, ``nnx.vmap``, a functional ``jax.jit``) it traces into the caller's
        program. A structural change made between calls (a new batch size, a replaced stage,
        ``train()``/``eval()``) is honored; a stage adding state while it runs is refused.

        Returns:
            The sink output for the batch.
        """
        return next_batch(self, type(self)._next_batch, self.batch_size)

    def _next_batch(self, size: int) -> dict:
        """Serve ``size`` records from where the next batch starts, and advance past them.

        The traceable body of :meth:`step`. Compiled iteration sessions call it
        directly: nesting ``nnx.jit`` inside their ``jax.jit`` step would rebind
        every Variable and hide which ones the step wrote. A session's final batch
        passes a ``size`` below ``batch_size``.

        Args:
            size: Records to serve.

        Returns:
            The sink output for the batch.
        """
        plan = self.epoch_plan
        start, epoch = plan.batch_start(self._position[...], self._epoch[...])
        # The records are named once and shared by the gather and the stages, so the source's
        # shuffle runs once per batch. Position and epoch are the batch's start while __call__
        # runs, as a direct call would see them.
        self._position[...], self._epoch[...] = start, epoch
        records = self._records_at(start, epoch, size)
        batch = self(self.source.get_records(records.indices), records)
        self._position[...], self._epoch[...] = plan.advance(start, epoch, size)
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

        Every step serves a full batch by the rule :meth:`step` follows, starting the
        next epoch when the current one cannot start another batch, so any ``length`` runs.
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

            @nnx.scan(in_axes=in_axes, out_axes=0, graph=True, graph_updates=True)
            def scan_body(*args: Any) -> Any:
                pipeline, *user_modules_and_step = args
                user_modules = user_modules_and_step[:-1]
                batch = pipeline.step()
                return step_fn(*user_modules, batch)

            return scan_body

        in_axes = (state_axes,) + (state_axes,) * n_modules + (nnx.Carry, 0)

        @nnx.scan(in_axes=in_axes, out_axes=(nnx.Carry, 0), graph=True, graph_updates=True)
        def scan_body_with_carry(*args: Any) -> tuple[Any, Any]:
            pipeline, *rest = args
            user_modules = rest[:n_modules]
            carry = rest[n_modules]
            batch = pipeline.step()
            return step_fn(carry, *user_modules, batch)

        return scan_body_with_carry

    def session(self) -> PipelineIterator:
        """A compiled, checkpointable iteration session over this pipeline.

        The session splits the pipeline once and serves batches through the compiled step
        :meth:`step` also runs; its :meth:`~PipelineIterator.get_state` and
        :meth:`~PipelineIterator.set_state` resume iteration exactly.

        Returns:
            The session, iterated with ``for batch in session`` or ``next(session)``.

        Raises:
            TypeError: If the source has no indexed access (a streaming source), which a
                session cannot drive.
        """
        if not self.source.supports_indexed_access():
            raise TypeError(
                f"{type(self.source).__name__} has no indexed access (get_records), so it "
                "cannot back a session; iterate the pipeline to stream it."
            )
        return PipelineIterator(
            self,
            body=type(self)._next_batch,
            plan=self.epoch_plan,
            position=self._position,
            epoch=self._epoch,
            shuffled=bool(getattr(self.source, "is_random_order", False)),
        )

    def __iter__(self) -> PipelineIterator | Iterator[dict]:
        """Iterate batches through a compiled session (the Tier-A fast path).

        Random-access sources return a :class:`~datarax.pipeline.iteration.
        PipelineIterator`: the module graph is split once per session and
        batches are driven through a cached ``jax.jit`` step, with module
        state written back when the session ends (exhaustion, ``close()``,
        or garbage collection after an early break). Iteration stops after
        ``num_epochs`` epochs; a stream (``num_epochs=None``) and a source without
        ``__len__`` iterate indefinitely. Streaming sources (no ``get_records``) pull
        batches on the host and run them through the compiled stage DAG via
        :meth:`_iter_streaming` instead.

        Returns:
            A :class:`~datarax.pipeline.iteration.PipelineIterator` for a source
            with indexed access, otherwise a generator over streamed batches.

        Raises:
            TypeError: If the source implements neither ``get_records`` nor
                ``get_batch``.
        """
        if self.source.supports_indexed_access():
            return self.session()
        if not callable(getattr(self.source, "get_batch", None)):
            raise TypeError(
                f"{type(self.source).__name__} implements neither get_records (indexed "
                "access) nor get_batch (streaming), so Pipeline cannot iterate it."
            )
        return self._iter_streaming()

    def _iter_streaming(self) -> Iterator[dict]:  # noqa: DOC502
        """Iterate a streaming source (sequential, no random access) through the DAG.

        Streaming sources have no ``get_records``, so batches are pulled on the
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
        plan = (
            tuple(self._exec_order),
            {name: tuple(preds) for name, preds in self._predecessors.items()},
            self._sink,
        )
        apply = compile_streaming_dag(self._stage_modules, self._position, self._epoch, plan)
        while True:
            batch = self.source.get_batch(self.batch_size)  # type: ignore[attr-defined]
            size = batch_length(batch)
            if not size:
                return
            validate_batch(batch, element_spec, batch_size=self.batch_size)
            yield apply(batch)


def _source_length(source: DataSourceModule) -> int | None:
    """The source's length, or ``None`` when it has none."""
    try:
        return len(source)
    except (TypeError, NotImplementedError):
        return None
