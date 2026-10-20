"""``Pipeline(nnx.Module)`` — JAX-native data pipeline with scan-based epochs.

A pipeline is a source plus a graph of stages. Each stage is an
``nnx.Module`` whose ``__call__`` takes one or more batches and returns
one batch. The pipeline drives source iteration via an ``nnx.Variable``
position counter so a full training epoch can be expressed as a single
``nnx.scan`` call, producing one XLA graph per epoch.

Three integration tiers (measured costs: ``docs/performance/index.md``):

- **Tier A — ``for batch in pipeline:``** — the data loader: the host stage
  (:class:`~datarax.pipeline.host_stage.HostStage`) reads each batch on Grain
  threads, or in Grain worker processes for a GIL-bound read given
  ``host_resources`` (a RAM budget), names the run's order on the CPU device,
  and places batches on the device on the consumer's thread, one ahead on a
  GPU; the batch then runs
  through the stage DAG in one cached ``jax.jit`` call
  (:func:`~datarax.pipeline.dag_call.compile_dag`), split once per iteration.
  Its batches go to a train or inference step written as the Flax and JAX
  examples write it, and no dataset is uploaded.
  :meth:`Pipeline.get_state` is where iteration stands. Works with any
  framework that takes batches; the recommended path.
- **Tier B — ``Pipeline.step()``** — one batch, traceable, with live
  module state: single batches and debugging eagerly, and inside a
  transform of your own. An in-memory source's columns are host data,
  not NNX state, so your transform takes them as an argument:
  ``columns = pipeline.device_columns()`` outside it, uploaded once, and
  ``pipeline.step(columns=columns)`` inside it.
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

import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, Generic, Literal, overload

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from substrax.typing import CheckpointState
from typing_extensions import TypeVar

from datarax.core import batch_ops
from datarax.core.block_order import BlockOrder
from datarax.core.data_source import (
    DataSourceModule,
    known_length,
    PackedProvenance,
    Provenance,
    RecordIdentity,
    shard_identity,
)
from datarax.core.element_batch import Batch
from datarax.core.host_resources import HostResources
from datarax.core.host_values import HostValue
from datarax.core.pack_plan import PackPlan
from datarax.core.prng import naming_key_data, NAMING_PRNG_IMPL
from datarax.pipeline.dag import name_records, OperatorDag, Records
from datarax.pipeline.dag_call import compile_dag
from datarax.pipeline.epochs import batch_records, EpochPlan
from datarax.pipeline.host_stage import HostStage, HostStageHolder
from datarax.pipeline.iteration import (
    next_batch,
    PipelineIterator,
    source_columns,
    staged_view,
)
from datarax.pipeline.packed_rows import PackedRows, plan_packing, refuse_unpackable
from datarax.pipeline.placement import check_sharding
from datarax.pipeline.read_plan import HostPlan
from datarax.pipeline.run_configuration import Packing
from datarax.sources.eager_source import bind_columns, eager_sources
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from datarax.typing import DataDict


# Compiled @nnx.scan bodies per pipeline, keyed by the call signature. The decorator must be
# applied to a stable function identity for nnx.scan's jit cache to hit on later calls;
# rebuilding it on every scan() call would trace again. The cache is host bookkeeping, held
# outside the pipeline and keyed weakly by it: no split of the pipeline carries it, filling it
# changes no graph definition, and no program takes it off the module while JAX traces. A
# pipeline merged inside a transform is a new object, with its own entry for that trace.
_SCAN_BODIES: weakref.WeakKeyDictionary[
    Pipeline[Any], dict[tuple[Any, ...], Callable[..., Any]]
] = weakref.WeakKeyDictionary()

# The type of a pipeline's ``packing`` argument, which decides its unit and so its provenance's
# form: ``Pipeline[None]`` serves records, ``Pipeline[Packing]`` packed rows. The default makes a
# bare ``Pipeline`` an unpacked one; covariance lets code taking either pipeline take
# ``Pipeline[Packing | None]``.
_Packing_co = TypeVar("_Packing_co", bound=Packing | None, default=None, covariant=True)
_Packing = TypeVar("_Packing", bound=Packing | None, default=None)


def _check_block_order(
    source: DataSourceModule, *, shuffle: bool, block_order: BlockOrder | None
) -> None:
    """Refuse a block order the pipeline cannot name an epoch in.

    Args:
        source: The pipeline's source.
        shuffle: Whether the pipeline shuffles.
        block_order: The block order asked for, or ``None``.

    Raises:
        ValueError: If a block order is given without ``shuffle``, or for a source that is not
            ``INDEXED``, whose records arrive in its own order.
    """
    if block_order is None:
        return
    if not shuffle:
        raise ValueError(
            "block_order orders a shuffled epoch, and this pipeline does not shuffle: build it "
            "with shuffle=True, or without block_order for the sequential order"
        )
    if source.record_identity is not RecordIdentity.INDEXED:
        raise ValueError(
            f"block_order orders an INDEXED source's records; {type(source).__name__} is "
            f"{source.record_identity.name}, whose records arrive in its own order"
        )


def _served_by_iteration(call: str) -> str:
    """Why ``call`` refuses a packed pipeline: its rows are planned and laid out on the host."""
    return (
        f"{call} serves batches through the compiled step, which reads records by index; a "
        "pipeline with packing serves planned rows, laid out on the host, by iteration: "
        "`for batch in pipe` or `pipe.raw_batches()`"
    )


def _refuse_packing_with(
    packing: Packing | None, *, dag: OperatorDag, block_order: BlockOrder | None
) -> None:
    """Refuse what a packed pipeline does not serve yet: stages and a block order.

    Args:
        packing: The packing asked for, or ``None``.
        dag: The pipeline's stages.
        block_order: The block order asked for, or ``None``.

    Raises:
        ValueError: If packing is asked for with stages or with a block order, naming why.
    """
    if packing is None:
        return
    if dag.order:
        raise ValueError(
            "a packed batch holds several records in each row, which the operators' per-record "
            "map does not take apart: serving operators on packed batches is open work; build "
            "the packed pipeline with stages=[]"
        )
    if block_order is not None:
        raise ValueError(
            "block_order reads blocks of consecutive records, but a packed row's records come "
            "from the cipher-ordered planning pass, so a block of rows is not a block of records: "
            "build the packed pipeline without block_order"
        )


def _placed_by_iteration(call: str) -> str:
    """Why ``call`` refuses a pipeline holding a sharding: only iteration places batches."""
    return (
        f"{call} serves batches through the compiled step, which places nothing on a mesh; "
        "a pipeline holding a sharding is served by iteration: `for batch in pipe` or "
        "`pipe.raw_batches()`"
    )


class Pipeline(nnx.Module, Generic[_Packing_co]):
    """JAX-native data pipeline with scan-based epoch iteration.

    Generic over the type of its ``packing`` argument, which the constructor infers:
    ``Pipeline[None]`` (a bare ``Pipeline``) serves records, and ``raw_batches`` gives their
    :data:`~datarax.core.data_source.Provenance`; ``Pipeline[Packing]`` serves packed rows, and
    gives each row's :data:`~datarax.core.data_source.PackedProvenance`.

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
            length is not a multiple of ``batch_size``; no batch holds a padding row.
            ``False`` (the default) completes it from the head of the next epoch's order,
            tf.data's and Grain's ``repeat().batch()``: every epoch serves each record
            once. ``True`` skips the records short of a full batch and starts the next
            epoch, their ``batch(drop_remainder=True).repeat()``.
        num_epochs: How many epochs ``iter(pipeline)`` serves before stopping, or
            ``None`` for a stream that never stops. Under ``drop_last=False`` it stops
            after exactly ``num_epochs * len(source)`` records, so its final batch may
            be short.
        host_resources: What the host stage may spend reading
            (:class:`~datarax.core.host_resources.HostResources`): a RAM budget and a cap
            on read threads or worker processes. With it, a source whose host read holds
            the GIL is read in Grain worker processes, as many as the budget admits, and a
            GIL-free read on up to four threads, 16 units ahead; without it, one read thread
            two units ahead, which can leave a fast train step waiting (see :attr:`host_plan`).
            No default budget: the training process knows its share.
        block_order: With ``shuffle=True``, a :class:`~datarax.core.block_order.BlockOrder`
            in which each epoch's records are named: a batch reads a few blocks of
            consecutive records, for an ``INDEXED`` source read from storage larger than the
            page cache. ``None`` (the default) is the global order.
        sharding: A ``jax.sharding.NamedSharding`` over a batch's rows, to place each batch on
            its data mesh: full batches with their rows on it and their batch-level state
            replicated, a chunk ``(K, B, ...)`` with its rows on the axes after the chunk axis,
            and a bounded run's short final batch replicated, so an evaluation covers every
            record once without padding. That final batch holds all of its rows, up to
            ``batch_size - 1``, on every device, against ``batch_size / N`` for a full batch on
            ``N`` devices, so its step can need up to ``N`` times a full batch's per-device
            memory. ``None`` (the default) places on the default device, uncommitted. The mesh
            may span processes (``jax.distributed``, one process per GPU): ``batch_size`` is
            then the global batch, every process builds the same pipeline over an ``INDEXED``
            source of all its records, and each reads the rows its devices hold of every
            global batch, so the batches are one process's row for row; a run whose
            configuration differs between processes is refused in every process, and worker
            processes need each process's ``/dev/shm`` share
            (``HostResources(shm_budget_bytes=...)``). Iteration places, so ``step()``,
            ``scan`` and ``session()`` refuse a pipeline holding a sharding.

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
        host_resources: HostResources | None = None,
        block_order: BlockOrder | None = None,
        sharding: NamedSharding | None = None,
        packing: _Packing_co = None,
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
            host_resources: The host stage's RAM budget and cap, or ``None`` (see the class
                docstring).
            block_order: The block order a shuffled epoch is named in, or ``None`` for the
                global order (see the class docstring).
            sharding: The rows' sharding on a data mesh, or ``None`` for the default device
                (see the class docstring).
            packing: How a source holding variable-length records packs them into rows of a
                static length (:class:`~datarax.pipeline.run_configuration.Packing`), or
                ``None``. The records are planned into rows once, and the rows are what the
                pipeline orders, batches, counts and resumes (see :attr:`pack_plan`).

        Raises:
            ValueError: If ``num_epochs`` is neither ``None`` nor at least 1, the source
                has no records, ``drop_last`` is set with a ``batch_size`` above the
                source's length, which serves no batch, a block order is given without
                ``shuffle`` or for a source that is not ``INDEXED`` or has no length, or the
                sharding cannot place this pipeline's batches (more than the rows, an
                ``AbstractMesh``, Explicit axes, or a ``batch_size`` it cannot split; across
                processes, a stream source or one serving a worker's shard), or packing is
                asked for with stages, a block order, or a field it does not name.
            TypeError: If ``sharding`` is not a ``NamedSharding``, or packing is asked of a
                stream.
            NotImplementedError: If packing is asked of a source that does not read ragged
                records.
        """
        super().__init__()
        _check_block_order(source, shuffle=shuffle, block_order=block_order)
        self.dag = self._build_dag(stages, nodes, edges, sink)
        _refuse_packing_with(packing, dag=self.dag, block_order=block_order)
        if packing is not None:
            refuse_unpackable(source, packing)
        self.packing = packing
        self.source = source
        self.rngs = rngs
        # The epoch key is derived from this base and the epoch counter, so
        # iterator state (position, epoch, rng counts) reproduces every slice. It is a naming
        # key (NAMING_PRNG_IMPL), whatever impl the caller's streams draw with.
        self._epoch_key_base: nnx.Variable[jax.Array] = nnx.Variable(naming_key_data(rngs()))
        # The rows a packed pipeline serves, held on the host and out of NNX state. Their count
        # is kept as a plain value beside the holder: the epoch plan the compiled step traces
        # reads it, and a traced path reads no holder.
        self._packed = HostValue(self._packed_rows(source, packing, shuffle=shuffle))
        self._packed_length = None if self._packed.value is None else len(self._packed.value)
        check_sharding(sharding, batch_size, self.read_source)
        # Refuses an epoch rule no pipeline can serve.
        EpochPlan(
            length=known_length(self.read_source),
            batch_size=batch_size,
            drop_last=drop_last,
            num_epochs=num_epochs,
            block_order=block_order,
        )
        self.shuffle = shuffle
        self.block_order = block_order
        self.drop_last = drop_last
        self.num_epochs = num_epochs
        self.host_resources = host_resources
        self.sharding = sharding

        self.batch_size = batch_size
        self._position: nnx.Variable[jax.Array] = nnx.Variable(jnp.zeros((), dtype=jnp.int32))
        self._epoch: nnx.Variable[jax.Array] = nnx.Variable(jnp.zeros((), dtype=jnp.int32))
        # Where host-stage iteration stands and the Grain iterator of its current run, held on
        # the host and out of NNX state, so the graph definition does not move as it advances.
        self._host = HostStageHolder(HostStage(end_epoch=num_epochs))

    def _packed_rows(
        self, source: DataSourceModule, packing: Packing | None, *, shuffle: bool
    ) -> PackedRows | None:
        """Plan ``source``'s records into rows once: the packed source the host stage reads.

        Args:
            source: The pipeline's source.
            packing: The packing asked for, or ``None``.
            shuffle: Whether the planning order is the pipeline's cipher (else sequential).

        Returns:
            The rows as an indexed source, or ``None`` without packing.
        """
        if packing is None:
            return None
        key = self._epoch_key_base.get_value() if shuffle else None
        plan, temporary = plan_packing(
            source, packing, None if key is None else jax.device_get(key)
        )
        return PackedRows(source, packing, plan, temporary=temporary)

    @property
    def read_source(self) -> DataSourceModule:
        """What the host stage reads: the packed rows with packing, else the source."""
        packed = self._packed.value
        return self.source if packed is None else packed

    @property
    def pack_plan(self) -> PackPlan:
        """The rows a packed pipeline serves, read through memory maps.

        Raises:
            ValueError: If the pipeline does not pack.
        """
        packed = self._packed.value
        if packed is None:
            raise ValueError("this pipeline does not pack: build it with packing=Packing(...)")
        return packed.plan

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
        cls: type[Pipeline[_Packing]],
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
        host_resources: HostResources | None = None,
        block_order: BlockOrder | None = None,
        sharding: NamedSharding | None = None,
        packing: _Packing = None,
    ) -> Pipeline[_Packing]:
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
            host_resources: The host stage's RAM budget and cap, or ``None`` (see the class
                docstring).
            block_order: The block order a shuffled epoch is named in, or ``None`` (see the
                class docstring).
            sharding: The rows' sharding on a data mesh, or ``None`` for the default device
                (see the class docstring).
            packing: How variable-length records pack into rows, or ``None`` (see the
                constructor).

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
            host_resources=host_resources,
            block_order=block_order,
            sharding=sharding,
            packing=packing,
        )

    @classmethod
    def from_arrays(  # noqa: DOC502 - the source and the constructor raise
        cls: type[Pipeline[None]],
        data: Mapping[str, Any],
        *,
        batch_size: int,
        seed: int,
        shuffle: bool = False,
        drop_last: bool = False,
        num_epochs: int | None = 1,
        block_order: BlockOrder | None = None,
        sharding: NamedSharding | None = None,
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
            block_order: The block order a shuffled epoch is named in, or ``None`` (see the
                class docstring).
            sharding: The rows' sharding on a data mesh, or ``None`` for the default device
                (see the class docstring).

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
            block_order=block_order,
            sharding=sharding,
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
        key = jax.random.wrap_key_data(self._epoch_key_base[...], impl=NAMING_PRNG_IMPL)
        return jax.random.fold_in(key, epoch)

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
            length=known_length(self.source) if self.packing is None else self._packed_length,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            num_epochs=self.num_epochs,
            block_order=self.block_order,
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
        """Where iteration stands: the host stage's cursor, version 4, and what it is valid for.

        ``{"version": 4, "kind", "epoch", "position", "run_end_epoch", "stream", "fingerprint"}``:
        an indexed source's next batch starts at ``position`` of ``epoch`` and the run stops
        before ``run_end_epoch``; a stream's cursor is ``stream`` (its pass, the records of the
        pass served, the arrival ordinals served and the passes left). The fingerprint names the
        batch rule, length, epochs, key and order the state fits, and a packed pipeline's packing
        and the lengths its rows were planned from (a packed pipeline's position counts rows).
        It names the batches taken,
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

    @property
    def host_plan(self) -> HostPlan:
        """How the host stage reads: the plan of the run opened last, or of single batches.

        Its path (threads, worker processes, or a stream's one thread), threads, workers and
        buffers, and with ``host_resources`` the terms it charged the budget with
        (:class:`~datarax.pipeline.read_plan.HostPlan`). Held on the host, never in the
        pipeline's state.
        """
        return self.host_stage.last_plan(self)

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
        self: Pipeline[None],
        chunk: int | None = None,
        *,
        max_chunk_bytes: int | None = None,
        with_provenance: Literal[True],
    ) -> Iterator[tuple[Batch, Provenance]]: ...

    @overload
    def raw_batches(
        self: Pipeline[Packing],
        chunk: int | None = None,
        *,
        max_chunk_bytes: int | None = None,
        with_provenance: Literal[True],
    ) -> Iterator[tuple[Batch, PackedProvenance]]: ...

    @overload
    def raw_batches(
        self: Pipeline[Packing | None],
        chunk: int | None = None,
        *,
        max_chunk_bytes: int | None = None,
        with_provenance: Literal[True],
    ) -> Iterator[tuple[Batch, Provenance | PackedProvenance]]: ...

    def raw_batches(  # noqa: DOC502 - the host stage raises
        self,
        chunk: int | None = None,
        *,
        max_chunk_bytes: int | None = None,
        with_provenance: bool = False,
    ) -> Iterator[Batch] | Iterator[tuple[Batch, Provenance | PackedProvenance]]:
        """Unprocessed batches read by the host stage and placed on the device: the E2E form.

        The DAG is not applied: pass :attr:`dag` into your differentiated train step and call it
        on each batch, so its operators' parameters train with the model. Records are served in
        the pipeline's order and epoch rule, named on the CPU device, read on the host by Grain
        threads ahead of the consumer and placed by the consumer: on the default device,
        uncommitted, or with :attr:`sharding` on its data mesh (the short final batch
        replicated), one batch ahead of the one taken on a GPU and none on the CPU and TPU;
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
            with_provenance: Whether each unit comes as ``(batch, provenance)``: one mapping of
                strings and objects per record, in row order, or for a packed pipeline one
                tuple per row of its slots' mappings, empty past its records.

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

    def device_columns(self) -> list[Any]:
        """The in-memory sources' columns on the device, for a caller's own transform.

        An in-memory source's columns are host data, not NNX state, so a transform the caller
        writes around :meth:`step` or :meth:`scan` takes them as an argument, the way it takes
        a batch: get them here, outside the transform, pass them in, and hand them to
        ``step(columns=...)`` or ``scan(..., columns=...)``. Each array is uploaded once and
        shared with the copy the pipeline's own compiled steps read.

        Returns:
            One columns pytree per in-memory source, in graph order (empty for a pipeline over
            no in-memory source).
        """
        return source_columns(self, eager_sources(self), None)

    def step(self, columns: list[Any] | None = None) -> Batch:
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

        An in-memory source's columns are not NNX state, so inside the caller's transform they
        arrive as an argument: ``columns``, from :meth:`device_columns`. Without them a traced
        call is refused.

        Args:
            columns: The in-memory sources' columns, as :meth:`device_columns` returns them,
                passed into the caller's transform. Not needed outside one.

        Returns:
            The sink output for the batch.

        Raises:
            ValueError: If the pipeline holds a sharding (iteration places on a mesh).
        """
        if self.packing is not None:
            raise ValueError(_served_by_iteration("step()"))
        if self.sharding is not None:
            raise ValueError(_placed_by_iteration("step()"))
        if columns is None:
            return next_batch(self, type(self)._next_batch, self.batch_size)
        with bind_columns(eager_sources(self), columns):
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
        columns: list[Any] | None = None,
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
            columns: The in-memory sources' columns, from :meth:`device_columns`, when the
                scan runs inside the caller's own transform (see :meth:`step`).

        Returns:
            Stacked outputs (``init_carry is None``) or
            ``(final_carry, stacked_outputs)`` (``init_carry`` provided).

        Every step serves a full batch by the rule :meth:`step` follows, starting the
        next epoch when the current one cannot start another batch, so any ``length`` runs.

        The source's records are read where :meth:`step` and iteration read them: device data
        in place, NumPy data from its device copy, uploaded once. The compiled scan is cached
        by ``step_fn``'s identity: pass the same function on every call, since a new one (a
        lambda written in the loop) compiles again.

        Raises:
            ValueError: If the pipeline holds a sharding (iteration places on a mesh).
        """
        if self.packing is not None:
            raise ValueError(_served_by_iteration("scan"))
        if self.sharding is not None:
            raise ValueError(_placed_by_iteration("scan"))
        n_modules = len(modules)
        has_init_carry = init_carry is not None
        # Cache key: identity of step_fn plus the structural shape of the
        # call (length, n_modules, carry presence). Identity-based
        # caching matches JAX's standard JIT-cache contract; users who
        # rebind step_fn between calls correctly get a fresh trace.
        cache_key = (id(step_fn), length, n_modules, has_init_carry)
        bodies = _SCAN_BODIES.setdefault(self, {})
        scan_body = bodies.get(cache_key)
        if scan_body is None:
            scan_body = self._compile_scan_body(
                step_fn=step_fn,
                length=length,
                n_modules=n_modules,
                has_init_carry=has_init_carry,
            )
            bodies[cache_key] = scan_body

        steps = jnp.arange(length, dtype=jnp.int32)
        # The scan runs on a view sharing this pipeline's Variables with its host data staged
        # once, as step() and iteration stage it, so no call uploads the records again.
        view, columns = staged_view(self, columns)
        if has_init_carry:
            return scan_body(view, *modules, columns, init_carry, steps)
        return scan_body(view, *modules, columns, steps)

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

        # The in-memory sources' columns enter every iteration whole (``in_axes`` None).
        if not has_init_carry:
            in_axes = (state_axes,) + (state_axes,) * n_modules + (None, 0)

            @nnx.scan(in_axes=in_axes, out_axes=0, graph=True, graph_updates=True)
            def scan_body(*args: Any) -> Any:
                pipeline, *rest = args
                user_modules = rest[:n_modules]
                batch = pipeline.step(columns=rest[n_modules])
                return step_fn(*user_modules, batch)

            return scan_body

        in_axes = (state_axes,) + (state_axes,) * n_modules + (None, nnx.Carry, 0)

        @nnx.scan(in_axes=in_axes, out_axes=(nnx.Carry, 0), graph=True, graph_updates=True)
        def scan_body_with_carry(*args: Any) -> tuple[Any, Any]:
            pipeline, *rest = args
            user_modules = rest[:n_modules]
            carry = rest[n_modules + 1]
            batch = pipeline.step(columns=rest[n_modules])
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
            ValueError: If the pipeline holds a sharding (iteration places on a mesh).
        """
        if self.packing is not None:
            raise ValueError(_served_by_iteration("session()"))
        if self.sharding is not None:
            raise ValueError(_placed_by_iteration("session()"))
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
            shard=shard_identity(self.source),
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
