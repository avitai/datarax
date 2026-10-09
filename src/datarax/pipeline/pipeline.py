"""``Pipeline(nnx.Module)`` — JAX-native data pipeline with scan-based epochs.

A pipeline is a source plus a graph of stages. Each stage is an
``nnx.Module`` whose ``__call__`` takes one or more batches and returns
one batch. The pipeline drives source iteration via an ``nnx.Variable``
position counter so a full training epoch can be expressed as a single
``nnx.scan`` call, producing one XLA graph per epoch.

Three integration tiers (measured costs: ``docs/performance/index.md``):

- **Tier A — ``for batch in pipeline:``** — the data loader: the host stage
  (:class:`~datarax.pipeline.host_stage.HostStage`) reads each batch on Grain
  threads, names the run's order on the CPU device, and places batches on the
  device on the consumer's thread, one ahead on a GPU; the batch then runs
  through the stage DAG in one cached ``jax.jit`` call
  (:func:`~datarax.pipeline.dag_call.compile_dag`), split once per iteration.
  Its batches go to a train or inference step written as the Flax and JAX
  examples write it, and no dataset is uploaded.
  :meth:`Pipeline.get_state` is where iteration stands. Works with any
  framework that takes batches; the recommended path.
- **Tier B — ``Pipeline.step()``** — one batch, traceable, with live
  module state: single batches and debugging eagerly, and inside a
  transform of your own. A pipeline passed into your own jitted step is
  an argument of every call, so the default ``nnx.jit`` copies its
  dataset per call; iterate instead (Tier A).
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
- ``dag`` — the stage graph (:class:`~datarax.pipeline.dag.OperatorDag`), ``Batch`` to
  ``Batch``: what runs inside a differentiated train step.
- ``__call__(data, records)`` — names the gathered records and runs ``dag``; returns a ``Batch``.
  JAX-traceable.
- ``step()`` — fetches one batch from the source, runs ``__call__``,
  advances ``_position`` and ``rngs``. JAX-traceable.
- ``scan(step_fn, *, length, modules=(), init_carry=None)`` — runs
  ``length`` steps under ``nnx.scan``, lifting pipeline + ``modules``
  state via ``StateAxes``. See method docstring for the two ``step_fn``
  signatures.
- ``__iter__()`` — the host stage's batches through the DAG; see Tier A above.
- ``get_state()`` / ``set_state(state)`` — where iteration stands, a versioned
  cursor (:mod:`datarax.pipeline.host_stage`); the stages' parameters and
  statistics are ``nnx.state(pipeline.dag)``.
- ``session()`` — the compiled iteration session
  (:class:`~datarax.pipeline.iteration.PipelineIterator`), which keeps its
  place in the pipeline's Variables as ``step()`` does.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, Literal, overload

import jax
import jax.numpy as jnp
from flax import nnx
from substrax.typing import CheckpointState

from datarax.core import batch_ops
from datarax.core.data_source import (
    DataSourceModule,
    known_length,
    Provenance,
    RecordIdentity,
)
from datarax.core.element_batch import Batch
from datarax.pipeline.dag import name_records, OperatorDag, Records
from datarax.pipeline.dag_call import compile_dag
from datarax.pipeline.epochs import batch_records, EpochPlan
from datarax.pipeline.host_stage import HostStage, HostStageHolder
from datarax.pipeline.iteration import next_batch, PipelineIterator, staged_view
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from datarax.typing import DataDict


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
        rngs: ``nnx.Rngs`` for the epoch key and stochastic stages.
        shuffle: Whether each epoch serves the source's records in a new random order. The
            pipeline owns the order: it passes its epoch key to ``record_indices_at`` when
            ``True`` and ``None`` (the sequential order) otherwise.
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

    Epochs: the pipeline owns the iteration position, the epoch counter and the order. With
    ``shuffle=True`` every batch passes each record's epoch key to ``record_indices_at``, so
    the source serves one permutation per epoch and each record is visited once per epoch.
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
        shuffle: bool = False,
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
            shuffle: Whether each epoch serves a new random order (see the class docstring).
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
            length=known_length(source),
            batch_size=batch_size,
            drop_last=drop_last,
            num_epochs=num_epochs,
        )
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_epochs = num_epochs

        self.dag = self._build_dag(stages, nodes, edges, sink)
        self.source = source
        self.batch_size = batch_size
        self.rngs = rngs
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
        # module boundary.
        self._scan_body_cache: dict[Any, Any] = {}
        # Where host-stage iteration stands and the Grain iterator of its current run, held on
        # the host and out of NNX state, so the graph definition does not move as it advances.
        self._host = HostStageHolder(HostStage(end_epoch=num_epochs))

    @staticmethod
    def _build_dag(
        stages: Sequence[nnx.Module] | None,
        nodes: Mapping[str, nnx.Module] | None,
        edges: Mapping[str, Sequence[str]] | None,
        sink: str | None,
    ) -> OperatorDag:
        """Build the DAG from one of the two mutually exclusive construction shapes.

        Args:
            stages: Linear stage sequence, or ``None`` for the DAG shape.
            nodes: Explicit node map, or ``None`` for the linear shape.
            edges: Explicit predecessor-edge map (required with ``nodes``).
            sink: Explicit sink node name (required with ``nodes``).

        Returns:
            The pipeline's DAG.

        Raises:
            ValueError: If neither shape or both shapes are supplied, or the DAG shape is
                incomplete.
        """
        if stages is not None and nodes is not None:
            raise ValueError(
                "Provide either stages= (linear) or nodes=+edges=+sink= (DAG); not both."
            )
        if stages is not None:
            return OperatorDag.from_stages(stages)
        if nodes is None or edges is None or sink is None:
            raise ValueError("Provide either stages= (linear) or nodes=+edges=+sink= (DAG).")
        return OperatorDag(nodes=nodes, edges=edges, sink=sink)

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
        shuffle: bool = False,
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
            rngs: ``nnx.Rngs`` for the epoch key and stochastic stages.
            shuffle: Whether each epoch serves a new random order (see the class docstring).
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
            shuffle=shuffle,
            drop_last=drop_last,
            num_epochs=num_epochs,
        )

    @classmethod
    def from_arrays(  # noqa: DOC502 - the source and the constructor raise
        cls,
        data: Mapping[str, Any],
        *,
        batch_size: int,
        seed: int,
        shuffle: bool = False,
        drop_last: bool = False,
        num_epochs: int | None = 1,
    ) -> Pipeline:
        """Build a Pipeline over in-memory arrays, with no stages.

        The same pipeline as a ``MemorySource`` over ``data`` and the linear constructor
        with ``stages=[]``, seeded with ``seed``: every guarantee of a pipeline (one
        permutation per epoch, the final-batch rule, resumable position, one compiled step)
        holds for it.

        Args:
            data: Arrays by name, all of the same leading length (the record count).
            batch_size: Records fetched per ``step()``.
            seed: Seed of the pipeline's ``nnx.Rngs``; with ``shuffle`` it chooses each
                epoch's permutation.
            shuffle: Whether each epoch serves the records in a new random order.
            drop_last: The last-batch rule (see the class docstring).
            num_epochs: Epochs ``iter`` serves, or ``None`` for a stream.

        Raises:
            ValueError: If the arrays differ in length or hold no records, or the epoch
                rule serves no batch (see the constructor).

        Returns:
            A configured ``Pipeline`` instance.
        """
        return cls(
            source=MemorySource(MemorySourceConfig(), data=dict(data)),
            stages=[],
            batch_size=batch_size,
            rngs=nnx.Rngs(seed),
            shuffle=shuffle,
            drop_last=drop_last,
            num_epochs=num_epochs,
        )

    @property
    def stages(self) -> list[nnx.Module]:
        """The DAG's nodes in topological order: the stages, for a linear pipeline."""
        return [self.dag.stages[name] for name in self.dag.order]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, data: DataDict, records: Records | None = None) -> Batch:
        """Run the DAG over the gathered ``data``, its rows named as ``records``.

        The data becomes a ``Batch`` whose rows carry their records' indices and epochs, which
        stochastic operators key each record's randomness on, and :attr:`dag` runs over it.
        ``step()`` passes the records it served, named once for the gather and the DAG; a direct
        call without them names the records served at the current position by the same rule.
        Traceable under ``nnx.jit``/``nnx.scan``.

        Args:
            data: The source's values for the batch, record axis first.
            records: The batch's records, or ``None`` to name them from the position.

        Returns:
            The DAG's output.
        """
        batch = batch_ops.from_arrays(data)
        if records is None:
            start, epoch = self.epoch_plan.batch_start(self._position[...], self._epoch[...])
            records = self._records_at(start, epoch, batch.batch_size)
        return self.dag(name_records(batch, records.indices, records.epochs))

    def epoch_key(self) -> jax.Array:
        """The key the current epoch's records are ordered by when the pipeline shuffles.

        One key per epoch is what makes a shuffled epoch a single permutation.
        """
        return self._key_of(self._epoch[...])

    def _key_of(self, epoch: jax.Array) -> jax.Array:
        """The key ``record_indices_at`` orders epoch ``epoch`` by."""
        return jax.random.fold_in(jax.random.wrap_key_data(self._epoch_key_base[...]), epoch)

    def _records_at(self, start: jax.Array, epoch: jax.Array, size: int) -> Records:
        """The ``size`` records served from ``start`` of epoch ``epoch``'s order.

        :func:`~datarax.pipeline.epochs.batch_records` over this pipeline's source and plan, the
        rule the host stage names its batches by. The source receives its epoch's key when the
        pipeline shuffles and ``None`` otherwise.

        Args:
            start: Where the rows start in epoch ``epoch``.
            epoch: The epoch the first row belongs to.
            size: Rows to serve.

        Returns:
            Each row's record index and epoch.
        """
        return batch_records(
            self.source,
            self.epoch_plan,
            key_base=self._epoch_key_base[...] if self.shuffle else None,
            start=jnp.asarray(start, dtype=jnp.int32),
            epoch=jnp.asarray(epoch, dtype=jnp.int32),
            size=size,
        )

    @property
    def epoch_plan(self) -> EpochPlan:
        """How this pipeline's records divide into batches and epochs.

        Built from the source's current length and the pipeline's batch rule each time,
        so a length or batch size changed since construction is what it reports.
        """
        return EpochPlan(
            length=known_length(self.source),
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
        """Batches the rest of the run serves, from where host iteration stands.

        Returns:
            The count, or ``None`` for a run that never ends: a source without a length, or
            ``num_epochs=None``.
        """
        return self.host_stage.batches_left(self.epoch_plan)

    def reset(self) -> None:
        """Start a new run at the next epoch's start (a stream's next pass).

        A shuffling pipeline serves a new permutation; a sequential one serves the same order
        again. The run open now is closed. ``step()`` and ``session()`` start the next epoch too.
        """
        self.host_stage.reset(self.num_epochs)
        position, epoch = EpochPlan.next_epoch(self._epoch[...])
        self._position[...] = jnp.asarray(position, dtype=jnp.int32)
        self._epoch[...] = epoch

    def get_state(self) -> CheckpointState:
        """Where iteration stands: the host stage's cursor, version 3, and what it is valid for.

        ``{"version": 3, "kind", "epoch", "position", "run_end_epoch", "stream", "fingerprint"}``:
        an indexed source's next batch starts at ``position`` of ``epoch`` and the run stops
        before ``run_end_epoch``; a stream's cursor is ``stream`` (its pass, the records of the
        pass served, the arrival ordinals served and the passes left). The fingerprint names the
        batch rule, length, epochs, key and order the state fits. It names the batches taken,
        whatever the read threads read ahead. Operator parameters and statistics are not in it:
        checkpoint them with the model through ``nnx.state(pipe.dag)``.

        Returns:
            The state, ints, bools, strings, lists of ints and ``None``.
        """
        return self.host_stage.state(self)

    def set_state(self, state: CheckpointState, /) -> None:
        """Resume iteration where ``state`` stands, in a pipeline built as the saved one was.

        A resumed run serves the records the uninterrupted run would have served, and ends where
        it would have ended. Another version of the state, another kind of source, or a state
        produced under another configuration is refused, naming what differs.

        Args:
            state: A state :meth:`get_state` produced.
        """
        self.host_stage.restore(self, state)

    @property
    def host_stage(self) -> HostStage:
        """The host stage: where host iteration stands, and the Grain iterator of its run."""
        return self._host.value

    @overload
    def raw_batches(
        self,
        chunk: int | None = None,
        *,
        max_chunk_bytes: int | None = None,
        with_provenance: Literal[False] = False,
    ) -> Iterator[Batch]: ...

    @overload
    def raw_batches(
        self,
        chunk: int | None = None,
        *,
        max_chunk_bytes: int | None = None,
        with_provenance: Literal[True],
    ) -> Iterator[tuple[Batch, Provenance]]: ...

    def raw_batches(  # noqa: DOC502 - the host stage raises
        self,
        chunk: int | None = None,
        *,
        max_chunk_bytes: int | None = None,
        with_provenance: bool = False,
    ) -> Iterator[Batch] | Iterator[tuple[Batch, Provenance]]:
        """Unprocessed batches read by the host stage and placed on the device: the E2E form.

        The DAG is not applied: pass :attr:`dag` into your differentiated train step and call it
        on each batch, so its operators' parameters train with the model. Records are served in
        the pipeline's order and epoch rule, named on the CPU device, read on the host by Grain
        threads ahead of the consumer and placed on the default device by the consumer,
        uncommitted, one batch ahead of the one taken on a GPU and none on the CPU and TPU;
        nothing else of the source reaches the device. Iteration stands where the last
        batch taken ended, so a later call continues the run with the same Grain iterator; the
        run ends after ``num_epochs`` epochs.

        Each returned iterator serves the run it was created for. Iterators of one run continue
        its cursor in turn; once the run served its last batch they stop. A later call with other
        options, :meth:`reset`, :meth:`set_state` or :meth:`close` ends the run, and the
        iterators it was serving then raise ``RuntimeError`` naming that call. A pipeline and its
        clone share one host stage and cursor: they continue it in turn, not interleaved. A read
        error reaches the caller unchanged and ends the run at the batches delivered, so the next
        call resumes at the batch whose read failed.

        Args:
            chunk: Batches per unit: ``None`` serves batches ``(B, ...)``; ``K`` serves chunks
                ``(K, B, ...)`` of ``K`` full batches, each one host read and one transfer, while
                ``K`` remain, then the remaining batches singly, the run's short final batch
                last.
            max_chunk_bytes: The most bytes a chunk's data may hold; a larger chunk is refused.
            with_provenance: Whether each unit comes as ``(batch, provenance)``, one mapping of
                strings and objects per record, in row order.

        Returns:
            The units.

        Raises:
            ValueError: If ``chunk`` is below 1, or a chunk would hold more than
                ``max_chunk_bytes``.
            TypeError: If the source has no host read the stage reads it with.
        """
        return self.host_stage.batches(
            self, chunk=chunk, max_chunk_bytes=max_chunk_bytes, with_provenance=with_provenance
        )

    def close(self) -> None:
        """Close the host stage's run and its read threads; where iteration stands is kept."""
        self.host_stage.close()

    def step(self) -> Batch:
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

    def _next_batch(self, size: int) -> Batch:
        """Serve ``size`` records from where the next batch starts, and advance past them.

        The traceable body of :meth:`step`. Compiled iteration sessions call it
        directly: nesting ``nnx.jit`` inside their ``jax.jit`` step would rebind
        every Variable and hide which ones the step wrote. A session's final batch
        passes a ``size`` below ``batch_size``.

        Args:
            size: Records to serve.

        Returns:
            The DAG's output for the batch.

        Raises:
            TypeError: If ``__call__`` (overridden by a subclass) returns anything but a ``Batch``.
        """
        plan = self.epoch_plan
        start, epoch = plan.batch_start(self._position[...], self._epoch[...])
        # The records are named once and shared by the gather and the stages, so the source's
        # shuffle runs once per batch. Position and epoch are the batch's start while __call__
        # runs, as a direct call would see them.
        self._position[...], self._epoch[...] = start, epoch
        records = self._records_at(start, epoch, size)
        batch = self(self.source.get_records(records.indices), records)
        if not isinstance(batch, Batch):
            raise TypeError(
                f"{type(self).__name__}.__call__ returned {type(batch).__name__}; a pipeline "
                "serves Batches: build on super().__call__(data, records) and transform the Batch"
            )
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

        The source's records are read where :meth:`step` and iteration read them: device data
        in place, NumPy data from its device copy, uploaded once. The compiled scan is cached
        by ``step_fn``'s identity: pass the same function on every call, since a new one (a
        lambda written in the loop) compiles again.
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
        # The scan runs on a view sharing this pipeline's Variables with its host data staged
        # once, as step() and iteration stage it, so no call uploads the records again.
        view = staged_view(self)
        if has_init_carry:
            return scan_body(view, *modules, init_carry, steps)
        return scan_body(view, *modules, steps)

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
            TypeError: If the source is not ``INDEXED`` (a stream), which a session cannot drive.
        """
        kind = self.source.record_identity
        if kind is not RecordIdentity.INDEXED:
            raise TypeError(
                f"{type(self.source).__name__} is a {kind.name} stream, not an INDEXED source, "
                "so it cannot back a session; iterate the pipeline to stream it."
            )
        return PipelineIterator(
            self,
            body=type(self)._next_batch,
            plan=self.epoch_plan,
            position=self._position,
            epoch=self._epoch,
            shuffled=self.shuffle,
        )

    def __iter__(self) -> Iterator[Batch]:  # noqa: DOC502 - the host stage raises
        """Iterate processed batches: the host stage's batches through the DAG (Tier A).

        The host stage reads batches ahead on Grain threads and places them on the consumer's
        thread (:meth:`raw_batches`); each runs through :attr:`dag` in one compiled call, split
        once per call and compiled once per batch shape, the state its stages write (BatchNorm
        statistics) written back. A pipeline without stages serves the placed batches as they
        are: an empty DAG is the identity, so it compiles nothing and copies no batch. Iteration
        continues where the last batch taken ended and stops after the run's ``num_epochs``
        epochs; :meth:`reset` starts the next run. A stream is read through
        :class:`~datarax.sources.StreamingSourceBase`; any other stream is refused.

        Returns:
            The processed batches.

        Raises:
            TypeError: If the source has no host read the stage reads it with: an ``INDEXED``
                source without ``get_batch(indices, ...)``, or a stream that is not a
                ``StreamingSourceBase``.
        """
        if not self.dag.order:
            return iter(self.raw_batches())
        apply = compile_dag(self.dag)
        return (apply(batch) for batch in self.raw_batches())
