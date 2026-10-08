"""The host stage: a pipeline's batches named, read and placed on the host, ahead of the consumer.

Every source kind reads through one stage, which does nothing but read:

- An ``INDEXED`` source's run is a :class:`~datarax.pipeline.epochs.Run` of the pipeline's plan,
  cut into units (:class:`~datarax.pipeline.run_units.RunUnits`: one batch, or a chunk of ``K``
  batches read together). Grain maps each unit's number to a read on worker threads
  (``grain.MapDataset.range(...).map(...)``) by :class:`~datarax.pipeline.run_units.IndexedRead`:
  the batch's records are named on the CPU device
  (:class:`~datarax.pipeline.epochs.HostNaming`) and read with the source's stateless host read,
  ``get_batch(indices, *, epochs, contiguous)``
  (:class:`~datarax.core.data_source.IndexedHostRead`).
- A stream with a run dataset (TFDS) reads the same units of its passes through that dataset: one
  Grain dataset for the run, numbered from the run's start, which Grain can slice across workers.
- Any other stream (HuggingFace) is read pass by pass, at a position of the host stage's own, its
  batches cut by :func:`~datarax.pipeline.epochs.stream_batches`.

How a run is read is its plan (:meth:`HostStage.plan`): without a RAM budget, an indexed source
on one Grain read thread two units ahead and a stream on its one producer thread, as C5a-4 left
them; with ``Pipeline(host_resources=...)``, a GIL-free read on threads and a GIL-bound one
(a Python decode) in Grain worker processes (``mp_prefetch``), as many as the budget admits
(:mod:`datarax.pipeline.read_plan`). A worker reads and decodes, and sends its unit back through
shared memory with its provenance as plain values (:mod:`datarax.pipeline.host_workers`); the
consumer checks every unit arrives in order.

A batch's provenance comes beside it when asked. Units are read ahead of the consumer, which places
them on the default device, uncommitted, so a jitted step taking them compiles once; nothing else
transfers. Taking a unit first places units until the platform's depth wait beside it
(:func:`default_device_buffer`: one on a GPU, none on the CPU and TPU), and ``jax.device_put``
returns before its transfer completes, so a transfer overlaps the step before it. Where iteration
stands (:class:`Cursor`) advances when the consumer takes a batch, never when a worker reads one,
so the cursor names exactly the batches served whatever the workers have read ahead. One Grain
iterator serves a run across calls; it is closed at the run's end, by ``close()``, and when
collected.
"""

from __future__ import annotations

import collections
import dataclasses
import inspect
import logging
import sys
import time
import weakref
from collections.abc import Callable, Iterator, Mapping
from types import MappingProxyType
from typing import Any, cast

import grain
import jax
import numpy as np

from datarax.core import batch_ops
from datarax.core.data_source import (
    DataSourceModule,
    HostRead,
    Provenance,
    RecordIdentity,
)
from datarax.core.element_batch import Batch
from datarax.core.host_resources import HostResources
from datarax.core.index_words import to_words
from datarax.core.prng import key_words
from datarax.core.spec import declared_spec, validate_batch, validate_device_dtypes
from datarax.pipeline import host_workers, run_configuration as configuration
from datarax.pipeline.epochs import EpochPlan, HostNaming, Run, stream_batches
from datarax.pipeline.host_workers import JaxSettings
from datarax.pipeline.read_plan import (
    budget_plan,
    default_plan,
    host_batch_bytes,
    host_read_path,
    HostPlan,
    HostReadPath,
    HostTerms,
    measure_terms,
    ReadAccess,
    spec_bytes,
)
from datarax.pipeline.run_units import HostElement, IndexedRead, RunUnits
from datarax.sources._source_base import StreamCursor, StreamingSourceBase
from datarax.sources.eager_source import HostValue


logger = logging.getLogger(__name__)


_GPU_PLATFORMS = frozenset({"gpu", "cuda", "rocm"})
"""Platform names a GPU goes by: ``Device.platform`` and ``jax.default_backend()`` say ``gpu``,
a ``JAX_PLATFORMS`` entry or a default-device string may say ``cuda`` or ``rocm``."""


def default_device_buffer(platform: str) -> int:
    """Placed batches staged on a platform's device ahead of the one the consumer holds.

    One on a GPU, the fewest that keeps an H100 as busy as batches already on the device
    (ResNet-18 and ViT-S/16 at 224 px, B 64 to 512); zero on the CPU and TPU, as Flax's
    ``prefetch_to_device`` advises ("mostly useful for GPUs, for TPUs and CPUs it should not be
    necessary": their allocators never hand out memory still in use), and on the CPU staging
    measures no faster.

    Args:
        platform: The device's platform name.

    Returns:
        The batches staged ahead.
    """
    return 1 if platform in _GPU_PLATFORMS else 0


@dataclasses.dataclass(slots=True)
class Cursor:
    """Where iteration stands: the next batch's place, and the run's end.

    Attributes:
        epoch: The epoch (an indexed source) or pass (a stream) the next batch starts in.
        position: Records of that epoch or pass already served.
        arrived: Arrival ordinals served, for a stream naming records by arrival.
        end_epoch: The epoch or pass the run stops before, or ``None`` for a run without an end.
    """

    epoch: int
    position: int
    arrived: int
    end_epoch: int | None


class _Stream(grain.IterDataset):
    """A stream's run read pass by pass at a position of the host stage's own, on one thread."""

    def __init__(self, source: StreamingSourceBase, cursor: Cursor, config: _StreamRead) -> None:
        super().__init__()
        self._source = source
        self._cursor = dataclasses.replace(cursor)
        self._config = config

    def __iter__(self) -> _StreamIterator:
        return _StreamIterator(self._source, self._cursor, self._config)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _StreamRead:
    """What a stream's run reads: its batch rule, key, chunking, spec and JAX settings."""

    batch_size: int
    drop_last: bool
    key: np.ndarray | None
    chunk: int | None
    with_provenance: bool
    check: Callable[[Batch], None]
    settings: JaxSettings


class _StreamIterator(grain.DatasetIterator):
    """Units of a stream's run, cut by :func:`stream_batches` from the host stage's position."""

    def __init__(self, source: StreamingSourceBase, cursor: Cursor, config: _StreamRead) -> None:
        super().__init__()
        self._source = source
        self._config = config
        self._stream = StreamCursor(pass_index=cursor.epoch, arrived=cursor.arrived)
        self._epoch, self._position = cursor.epoch, cursor.position
        passes = None if cursor.end_epoch is None else max(0, cursor.end_epoch - cursor.epoch)
        with config.settings.entered():
            self._skip(cursor.position)
        self._batches = stream_batches(
            self._pull, config.batch_size, drop_last=config.drop_last, num_epochs=passes
        )
        self._held: list[tuple[Batch, Provenance, tuple[int, int, int]]] = []
        self._ended = False
        self._units = 0
        weakref.finalize(self, self._stream.close)

    def _skip(self, records: int) -> None:
        """Read past ``records`` of the cursor's pass without naming them (a resumed pass).

        The stream has no position to seek to, so its pass is read again from the start to the
        saved count, as it was first read; the replay's cost is logged at ``INFO``.
        """
        if not records:
            return
        started = time.perf_counter()
        left = records
        while left:
            batch, _ = self._source.read_from(
                self._stream, left, key=self._config.key, read_size=self._config.batch_size
            )
            if not batch.batch_size:
                break
            left -= batch.batch_size
        self._stream.arrived -= records - left if self._named_by_arrival() else 0
        logger.info(
            "%s resumed mid-pass: replayed %d records of pass %d in %.3f s",
            type(self._source).__name__,
            records - left,
            self._epoch,
            time.perf_counter() - started,
        )

    def _named_by_arrival(self) -> bool:
        return self._source.record_identity is RecordIdentity.ARRIVAL

    def _pull(self, size: int) -> tuple[Batch, Provenance]:
        batch, provenance = self._source.read_from(
            self._stream, size, key=self._config.key, read_size=self._config.batch_size
        )
        if batch.batch_size:
            self._config.check(batch)
        return batch, provenance

    def _next_batch(self) -> tuple[Batch, Provenance, tuple[int, int, int]] | None:
        if self._ended:
            return None
        try:
            batch, provenance = next(self._batches)
        except StopIteration:
            self._ended = True
            return None
        epochs = np.asarray(batch.epochs)
        last = int(epochs[-1])
        rows = int(np.sum(epochs == last))
        self._position = (self._position if last == self._epoch else 0) + rows
        self._epoch = last
        return batch, provenance, (self._epoch, self._position, self._stream.arrived)

    def _fill(self, chunk: int) -> None:
        """Hold up to ``chunk`` batches, stopping at the run's end or its short final batch."""
        while len(self._held) < chunk:
            batch = self._next_batch()
            if batch is None:
                return
            self._held.append(batch)
            if batch[0].batch_size < self._config.batch_size:
                return

    def _unit(
        self, chunk: int
    ) -> tuple[list[tuple[Batch, Provenance, tuple[int, int, int]]], bool]:
        """The held batches the next unit takes, and whether they are a full chunk."""
        size = self._config.batch_size
        full = len(self._held) == chunk and all(b.batch_size == size for b, _, _ in self._held)
        parts = self._held if full else self._held[:1]
        self._held = self._held[len(parts) :]
        return parts, full

    def __next__(self) -> HostElement:
        stacked = self._config.chunk is not None
        chunk = self._config.chunk or 1
        with self._config.settings.entered():
            self._fill(chunk)
        if not self._held:
            raise StopIteration
        parts, full = self._unit(chunk)
        batch = batch_ops.stack([b for b, _, _ in parts]) if full and stacked else parts[0][0]
        provenance = tuple(record for _, part, _ in parts for record in part)
        unit, self._units = self._units, self._units + 1
        return HostElement(
            batch, provenance if self._config.with_provenance else None, parts[-1][2], unit
        )

    def get_state(self) -> dict[str, Any]:
        """Not used: the host stage's cursor is where a stream resumes."""
        return {}

    def set_state(self, state: dict[str, Any]) -> None:  # noqa: DOC502
        """Not supported: a stream resumes through the host stage's cursor.

        Args:
            state: The state to restore.

        Raises:
            NotImplementedError: Always.
        """
        del state
        raise NotImplementedError("a stream resumes through the pipeline's host stage")


def _stream_unit(
    element: tuple[dict[str, Any], tuple[Mapping[str, Any], ...], np.ndarray, np.ndarray],
    *,
    chunk: int | None,
    check: Callable[[Batch], None],
    with_provenance: bool,
) -> tuple[Batch, Provenance | None]:
    """A run dataset's decoded unit as a batch named by the stream's ids, or a chunk of them."""
    columns, provenance, ids, epochs = element
    batch = batch_ops.from_arrays(columns, indices=to_words(ids), epochs=epochs)
    if chunk is not None:
        parts = batch_ops.split(batch, chunk)
        for part in parts:
            check(part)
        batch = batch_ops.stack(parts)
    else:
        check(batch)
    return batch, (
        tuple(MappingProxyType(dict(record)) for record in provenance) if with_provenance else None
    )


def _checker(
    source: DataSourceModule, batch_size: int, settings: JaxSettings
) -> Callable[[Batch], None]:
    """A check of a stream's batches against the spec its source declares, as the device holds it.

    The spec is read once and refused if it declares a dtype JAX arrays cannot hold as declared.
    Batches are checked under the run's settings on whichever thread reads them: a context
    manager's setting is thread-local, and the read threads do not inherit it.

    Args:
        source: The stream.
        batch_size: Records per batch.
        settings: The run's JAX settings, in which the spec was read.

    Returns:
        A function refusing a batch whose structure, shapes or dtypes disagree with the spec.
    """
    element_spec = declared_spec(source)
    validate_device_dtypes(element_spec)

    def check(batch: Batch) -> None:
        with settings.entered():
            validate_batch(
                batch.data, element_spec, batch_size=batch_size, as_the_device_holds=True
            )

    return check


def _place(element: HostElement) -> tuple[Batch, Provenance | None, tuple[int, int, int]]:
    """The element's batch on the default device, uncommitted; provenance stays on the host."""
    return jax.device_put(element.batch), element.provenance, element.after


type _PlacedElement = tuple[Batch, Provenance | None, tuple[int, int, int]]
type _Placed = grain.DatasetIterator[_PlacedElement]


def _placement_platform() -> str:
    """The platform of the device a batch placed now lands on: the caller's default device's."""
    device = jax.config.jax_default_device
    if device is None:
        return jax.default_backend()
    return device if isinstance(device, str) else device.platform


class _PlacedAhead(grain.IterDataset):
    """Host elements placed by the consumer, ``depth`` placed ahead of the one it takes.

    Flax's ``prefetch_to_device`` (``flax/jax_utils.py``, flax ``d1f0d69b``): a deque of placed
    batches filled on the consumer's thread. ``jax.device_put`` returns before its transfer
    completes, so the transfers of the batches ahead overlap the step. Placing on the consumer
    keeps the caller's thread-local JAX settings (transfer guard, precision mode, default device,
    mesh), which another thread would not inherit.
    """

    def __init__(self, parent: grain.IterDataset, depth: int) -> None:
        super().__init__(parent)
        self._depth = depth

    def __iter__(self) -> _PlacedAheadIterator:
        return _PlacedAheadIterator(iter(self._parent), self._depth)


class _PlacedAheadIterator(grain.DatasetIterator):
    """The deque of :class:`_PlacedAhead`: taking an element first places until ``depth`` wait."""

    def __init__(self, parent: grain.DatasetIterator, depth: int) -> None:
        super().__init__(parent)
        self._depth = depth
        self._placed: collections.deque[_PlacedElement] = collections.deque()
        self._exhausted = False
        # An error met placing ahead, raised once the batches placed before it are taken.
        self._failed: Exception | None = None

    def _fill(self) -> None:
        """Place elements until ``depth`` wait beside the one about to be taken."""
        while len(self._placed) <= self._depth and not self._exhausted and self._failed is None:
            try:
                self._placed.append(_place(next(self._parent)))
            except StopIteration:
                self._exhausted = True
            except Exception as error:  # noqa: BLE001 - raised after the batches placed before it
                self._failed = error

    def __next__(self) -> _PlacedElement:
        self._fill()
        if self._placed:
            return self._placed.popleft()
        if self._failed is not None:
            raise self._failed
        raise StopIteration

    def get_state(self) -> dict[str, Any]:
        """Not used: the host stage's cursor is where a run resumes."""
        return {}

    def set_state(self, state: dict[str, Any]) -> None:  # noqa: DOC502
        """Not supported: a run resumes through the host stage's cursor.

        Args:
            state: The state to restore.

        Raises:
            NotImplementedError: Always.
        """
        del state
        raise NotImplementedError("a run resumes through the pipeline's host stage")

    def close(self) -> None:
        """Close the parent and drop the batches placed ahead."""
        self._parent.close()
        self._placed.clear()


class HostStage:
    """A pipeline's host stage: its cursor and the Grain iterator serving the current run.

    Attributes:
        cursor: Where iteration stands.
    """

    def __init__(self, *, end_epoch: int | None) -> None:
        """A stage at the start of the first epoch, with no run open.

        Args:
            end_epoch: The epoch the first run stops before, or ``None`` for no end.
        """
        self.cursor = Cursor(epoch=0, position=0, arrived=0, end_epoch=end_epoch)
        # How runs are read, by what shapes a run's reads (:meth:`plan`), and the plan of the
        # run opened last. The plan is the read options' only source.
        self._plans: dict[tuple[Any, ...], HostPlan] = {}
        self._plan: HostPlan | None = None
        # Placed batches staged on the device ahead of the one the consumer holds, or ``None``
        # for the default device's platform's (:func:`default_device_buffer`). Internal.
        self._device_buffer: int | None = None
        # The run's Grain iterator, in a list the run's finalizer empties, so at exit the
        # iterator is closed and released before the interpreter tears Grain's modules down.
        self._run_iterator: list[_Placed] = []
        self._opened_for: tuple[Any, ...] | None = None
        self._finalizer: weakref.finalize | None = None
        self._token: weakref.ref[_RunToken] | None = None

    @property
    def iterator(self) -> _Placed | None:
        """The run's Grain iterator, or ``None`` when no run is open."""
        return self._run_iterator[0] if self._run_iterator else None

    def close(self) -> None:
        """Close the run's Grain iterator and its threads; the cursor stays where it is.

        Iterators the run was serving are refused from then on, naming ``close()``.
        """
        self._end("close()")

    def _end(self, by: str) -> None:
        """End the open run, recording on its token what ended it, and close its iterator.

        Args:
            by: What ended the run: :data:`_RUN_END` at its last batch, else the call that ended
                it, which the run's iterators name when they are refused.
        """
        token = None if self._token is None else self._token()
        if token is not None and token.ended_by is None:
            token.ended_by = by
        if self._finalizer is not None:
            self._finalizer()
        self._opened_for, self._finalizer, self._token = None, None, None

    def _depth(self) -> int:
        """Placed batches staged ahead of the one the consumer holds, on the default device."""
        if self._device_buffer is not None:
            return self._device_buffer
        return default_device_buffer(_placement_platform())

    def plan(self, pipeline: Any, chunk: int | None = None) -> HostPlan:
        """How a run of the pipeline reads its units: threads or processes, and their buffers.

        Made once for each shape of run (source, epoch plan, order, chunk, depth, resources) and
        kept: measuring it reads the source's spec, counts what a worker is sent and, without a
        caller's ``worker_bytes``, starts one probe worker.

        Args:
            pipeline: The pipeline whose run is read.
            chunk: Batches per chunk, or ``None`` for single batches.

        Returns:
            The plan.
        """
        depth = self._depth()
        resources: HostResources | None = pipeline.host_resources
        source = pipeline.source
        shape = (id(source), pipeline.epoch_plan, pipeline.shuffle, chunk, depth, resources)
        plan = self._plans.get(shape)
        if plan is None:
            plan = self._new_plan(pipeline, chunk, depth, resources)
            self._plans[shape] = plan
        self._plan = plan
        return plan

    def last_plan(self, pipeline: Any) -> HostPlan:
        """The plan of the run opened last, or a run of single batches' when none was.

        Args:
            pipeline: The pipeline whose run is read.

        Returns:
            The plan.
        """
        return self._plan if self._plan is not None else self.plan(pipeline)

    def _new_plan(
        self, pipeline: Any, chunk: int | None, depth: int, resources: HostResources | None
    ) -> HostPlan:
        source: DataSourceModule = pipeline.source
        access = self._access(pipeline, chunk)
        path = host_read_path(resources, source.host_read, access)
        if resources is None:
            if source.host_read is HostRead.GIL_BOUND and access is not ReadAccess.STREAM:
                logger.warning(
                    "%s decodes in Python on each read, on one thread: pass "
                    "Pipeline(host_resources=HostResources(ram_budget_bytes=..., max_workers=...)) "
                    "to read it in worker processes",
                    type(source).__name__,
                )
            return default_plan(path, device_buffer=depth)
        terms = self._terms(
            source,
            batch_size=pipeline.batch_size,
            epoch_plan=pipeline.epoch_plan,
            shuffle=pipeline.shuffle,
            key=_key(pipeline),
            chunk=chunk,
            depth=depth,
            path=path,
            resources=resources,
            settings=JaxSettings.current(),
        )
        return budget_plan(resources, path, terms)

    def _access(self, pipeline: Any, chunk: int | None) -> ReadAccess:
        """How the pipeline's run can be read: by unit, as a sliceable dataset, or pass by pass."""
        source = pipeline.source
        if source.record_identity is RecordIdentity.INDEXED:
            return ReadAccess.BY_UNIT
        plan: EpochPlan = pipeline.epoch_plan
        if isinstance(source, StreamingSourceBase) and plan.length is not None:
            units = RunUnits(run=self._run(plan), chunk=chunk)
            if source.run_dataset(units, None) is not None:
                return ReadAccess.SLICEABLE_STREAM
        return ReadAccess.STREAM

    def _terms(  # noqa: PLR0913 - the terms of one run shape
        self,
        source: DataSourceModule,
        *,
        batch_size: int,
        epoch_plan: EpochPlan,
        shuffle: bool,
        key: np.ndarray | None,
        chunk: int | None,
        depth: int,
        path: HostReadPath,
        resources: HostResources,
        settings: JaxSettings,
    ) -> HostTerms:
        """What a run of ``source`` read along ``path`` costs (:func:`measure_terms`).

        Args:
            source: The source the run reads.
            batch_size: Records per batch.
            epoch_plan: The run's epoch plan.
            shuffle: Whether the run is shuffled.
            key: The pipeline's key as host words, or ``None`` when it does not shuffle.
            chunk: Batches per chunk, or ``None`` for single batches.
            depth: Placed units staged on the device ahead of the consumer's.
            path: Where the run is read.
            resources: The caller's budget and cap.
            settings: The run's JAX settings.

        Returns:
            The terms.
        """
        size = 1 if chunk is None else chunk
        return measure_terms(
            unit_bytes=size * host_batch_bytes(source, batch_size),
            depth=depth,
            path=path,
            resources=resources,
            read=lambda: self._read(
                source, epoch_plan, shuffle=shuffle, key=key, chunk=chunk, settings=settings
            ),
            first_unit=_FirstUnit,
            settings=settings,
        )

    def _read(  # noqa: PLR0913 - what one run's worker read is built from
        self,
        source: DataSourceModule,
        plan: EpochPlan,
        *,
        shuffle: bool,
        key: np.ndarray | None,
        chunk: int | None,
        settings: JaxSettings,
    ) -> Any:
        """What worker processes read a run of ``source`` with: an indexed read or a run dataset.

        Args:
            source: An indexed source, or a stream with a run dataset.
            plan: The run's epoch plan.
            shuffle: Whether the run is shuffled.
            key: The pipeline's key as host words, or ``None`` when it does not shuffle.
            chunk: Batches per chunk, or ``None`` for single batches.
            settings: The run's JAX settings.

        Returns:
            The read, which pickles to the workers.
        """
        units = RunUnits(run=self._run(plan), chunk=chunk)
        if source.record_identity is RecordIdentity.INDEXED:
            return IndexedRead(
                source,
                units,
                HostNaming(source, plan, shuffled=shuffle),
                key=key,
                with_provenance=False,
                settings=settings,
                crossing=True,
            )
        # Only an indexed source or a stream with a run dataset is read in processes.
        stream = cast(StreamingSourceBase, source)
        return stream.run_dataset(units, key, for_workers=True)

    def _open(self, dataset: grain.IterDataset, options: tuple[Any, ...], depth: int) -> _RunToken:
        """Start the run's iterator: units read ahead, placed by the consumer ``depth`` ahead.

        The run lives as long as its token: the pipelines it served and the iterators serving
        it hold the token, and the stage only a weak reference. A compiled-step cache keyed by
        a pipeline's graph keeps the stage (a static of that graph), so a run the stage owned
        would keep its threads and the batches read ahead for the life of the process.

        Args:
            dataset: The run's dataset of host elements.
            options: What the run was opened for, compared when a later call continues it.
            depth: Placed batches staged ahead of the one the consumer holds.

        Returns:
            The run's token.
        """
        self._run_iterator.append(iter(_PlacedAhead(dataset, depth)))
        self._opened_for = options
        token = _RunToken()
        self._token = weakref.ref(token)
        self._finalizer = weakref.finalize(token, _close_run, self._run_iterator)
        return token

    def batches(
        self,
        pipeline: Any,
        *,
        chunk: int | None,
        max_chunk_bytes: int | None,
        with_provenance: bool,
    ) -> Iterator[Batch] | Iterator[tuple[Batch, Provenance]]:
        """The run's units from the cursor, placed, advancing the cursor as each is taken.

        Args:
            pipeline: The pipeline whose source, plan and key the run reads.
            chunk: Batches per chunk, or ``None`` for single batches.
            max_chunk_bytes: The most bytes a chunk may hold, or ``None`` for no bound.
            with_provenance: Whether each unit comes with its records' provenance.

        Returns:
            The units, each a batch or a ``(batch, provenance)`` pair.

        Raises:
            ValueError: If ``chunk`` is below 1, or a chunk would hold more than
                ``max_chunk_bytes``.
        """
        size = 1 if chunk is None else chunk
        if size < 1:
            raise ValueError(f"a chunk holds at least one batch; got chunk={chunk}")
        source = pipeline.source
        if chunk is not None and max_chunk_bytes is not None:
            batch_bytes = _batch_bytes(source, pipeline.batch_size)
            if size * batch_bytes > max_chunk_bytes:
                raise ValueError(
                    f"a chunk of {size} batches holds {size * batch_bytes} bytes, over "
                    f"max_chunk_bytes={max_chunk_bytes}; one batch holds {batch_bytes} bytes"
                )
        # A run is read under the caller's JAX settings (precision and the PRNG settings, which
        # read threads and worker processes do not inherit) and by its plan, which holds its read
        # options and its depth on the default device's platform; a change of any opens a new run.
        settings = JaxSettings.current()
        reads = self.plan(pipeline, chunk)
        options = (
            id(source),
            pipeline.epoch_plan,
            pipeline.shuffle,
            chunk,
            with_provenance,
            settings,
            reads,
        )
        cursor = self.cursor
        opened = (*options, cursor.epoch, cursor.position)
        token = None if self._token is None else self._token()
        if token is None or self.iterator is None or self._opened_for != opened:
            self._end(_replaced_by(self._opened_for, opened))
            dataset = self._dataset(pipeline, chunk, with_provenance, reads, settings)
            token = self._open(dataset, opened, reads.device_buffer)
        _RUN_HOLDERS[pipeline] = token
        return cast(
            Iterator[Batch] | Iterator[tuple[Batch, Provenance]],
            _Served(self, (*options,), with_provenance, token),
        )

    def _dataset(  # noqa: PLR0913 - a run's dataset is built from all of them
        self,
        pipeline: Any,
        chunk: int | None,
        with_provenance: bool,
        reads: HostPlan,
        settings: JaxSettings,
    ) -> grain.IterDataset:
        """The run's dataset of host elements, from the cursor, read as ``reads`` plans."""
        source: DataSourceModule = pipeline.source
        key = _key(pipeline)
        plan: EpochPlan = pipeline.epoch_plan
        processes = reads.path is HostReadPath.PROCESSES
        if source.record_identity is RecordIdentity.INDEXED:
            if not _has_indexed_host_read(source):
                raise TypeError(
                    f"{type(source).__name__} is INDEXED but has no host read, "
                    "get_batch(indices, *, epochs, contiguous), which the host stage reads with"
                )
            units = RunUnits(run=self._run(plan), chunk=chunk)
            read = IndexedRead(
                source,
                units,
                HostNaming(source, plan, shuffled=pipeline.shuffle),
                key=key,
                with_provenance=with_provenance,
                settings=settings,
                crossing=processes,
            )
            count = units.units()
            mapped = grain.MapDataset.range(sys.maxsize if count is None else count).map(read)
            if processes:
                # Each worker reads its units itself, one at a time; its queue reads ahead.
                inline = mapped.to_iter_dataset(
                    grain.ReadOptions(num_threads=0, prefetch_buffer_size=0)
                )
                workers = host_workers.in_workers(
                    inline,
                    workers=reads.workers,
                    worker_buffer=reads.worker_buffer,
                    settings=settings,
                )
                return _Received(workers, units)
            return mapped.to_iter_dataset(
                grain.ReadOptions(num_threads=reads.threads, prefetch_buffer_size=reads.read_buffer)
            )
        if not isinstance(source, StreamingSourceBase):
            kind = source.record_identity.name
            article = "an" if kind[0] in "AEIOU" else "a"
            raise TypeError(
                f"{type(source).__name__} is {article} {kind} stream that is not a "
                "StreamingSourceBase, whose pass reader the host stage reads every stream with: "
                "subclass datarax.sources.StreamingSourceBase and implement _open_pass"
            )
        elements = self._stream_elements(pipeline, chunk, with_provenance, key, reads, settings)
        if processes:  # the workers' buffers hold the units read ahead
            return elements
        return grain.experimental.ThreadPrefetchIterDataset(
            elements, prefetch_buffer_size=reads.read_buffer
        )

    def _stream_elements(  # noqa: PLR0913 - a stream run's dataset is built from all of them
        self,
        pipeline: Any,
        chunk: int | None,
        with_provenance: bool,
        key: np.ndarray | None,
        reads: HostPlan,
        settings: JaxSettings,
    ) -> grain.IterDataset:
        """A stream's run as host elements: through its run dataset when it has one (TFDS)."""
        source: StreamingSourceBase = pipeline.source
        check = _checker(source, pipeline.batch_size, settings)
        plan: EpochPlan = pipeline.epoch_plan
        if plan.length is not None:
            units = RunUnits(run=self._run(plan), chunk=chunk)
            processes = reads.path is HostReadPath.PROCESSES
            run = source.run_dataset(units, key, for_workers=processes)
            if run is not None:
                if processes:
                    run = host_workers.in_workers(
                        run,
                        workers=reads.workers,
                        worker_buffer=reads.worker_buffer,
                        settings=settings,
                    )
                return _RunElements(run, units, check, with_provenance, settings)
        config = _StreamRead(
            batch_size=pipeline.batch_size,
            drop_last=pipeline.drop_last,
            key=key,
            chunk=chunk,
            with_provenance=with_provenance,
            check=check,
            settings=settings,
        )
        return _Stream(source, self.cursor, config)

    def _run(self, plan: EpochPlan) -> Run:
        cursor = self.cursor
        return Run(
            plan=plan, position=cursor.position, epoch=cursor.epoch, end_epoch=cursor.end_epoch
        )

    def state(self, pipeline: Any) -> dict[str, Any]:
        """Where iteration stands, as version :data:`STATE_VERSION` of the pipeline's state.

        Every leaf is an int, a bool, a string, a list of ints or ``None``; the cursor names the
        batches the consumer took, whatever the read threads have read ahead.

        Args:
            pipeline: The pipeline whose state it is.

        Returns:
            The state.
        """
        cursor = self.cursor
        kind = pipeline.source.record_identity
        indexed = kind is RecordIdentity.INDEXED
        stream = None
        if not indexed:
            stream = {
                "pass": cursor.epoch,
                "records": cursor.position,
                "arrived": cursor.arrived,
                "passes_left": None
                if cursor.end_epoch is None
                else max(0, cursor.end_epoch - cursor.epoch),
            }
        return {
            "version": STATE_VERSION,
            "kind": kind.value,
            "epoch": cursor.epoch if indexed else None,
            "position": cursor.position if indexed else None,
            "run_end_epoch": cursor.end_epoch if indexed else None,
            "stream": stream,
            "fingerprint": configuration.run_configuration(pipeline),
        }

    def restore(self, pipeline: Any, state: Mapping[str, Any]) -> None:
        """Move the cursor to where ``state`` stands, closing the run open now.

        Args:
            pipeline: The pipeline the state is restored into.
            state: A state :meth:`state` produced, as saved or as a checkpoint store returns it.

        Raises:
            ValueError: If the state is another version, another kind of source, holds no
                fingerprint, or was produced under another configuration, naming what differs;
                nothing converts a layout.
        """
        state = configuration.plain_state(state)
        version = state.get("version")
        if version != STATE_VERSION:
            if version is None:
                saved = "a state without a version (a pipeline module_state layout)"
            elif version == configuration.SESSION_VERSION:
                saved = f"version {version} (the session layout of PipelineIterator)"
            else:
                saved = f"version {version}"
            raise ValueError(
                f"the state is {saved}; this pipeline reads version {STATE_VERSION}: save it "
                "again from a pipeline built as this one is"
            )
        kind = pipeline.source.record_identity.value
        if state.get("kind") != kind:
            raise ValueError(
                f"the state's kind is {state.get('kind')!r} but this pipeline's source is {kind!r}"
            )
        configuration.refuse_another_configuration(pipeline, state.get("fingerprint"))
        self._end("set_state()")
        if kind == RecordIdentity.INDEXED.value:
            self.cursor = Cursor(
                epoch=int(state["epoch"]),
                position=int(state["position"]),
                arrived=0,
                end_epoch=None if state["run_end_epoch"] is None else int(state["run_end_epoch"]),
            )
            return
        stream = state["stream"]
        left = stream["passes_left"]
        self.cursor = Cursor(
            epoch=int(stream["pass"]),
            position=int(stream["records"]),
            arrived=int(stream["arrived"]),
            end_epoch=None if left is None else int(stream["pass"]) + int(left),
        )

    def reset(self, num_epochs: int | None) -> None:
        """Start a new run at the next epoch's (or pass's) start, closing the run open now.

        Args:
            num_epochs: Epochs the new run serves, or ``None`` for no end.
        """
        self._end("reset()")
        epoch = self.cursor.epoch + 1
        self.cursor = Cursor(
            epoch=epoch,
            position=0,
            arrived=self.cursor.arrived,
            end_epoch=None if num_epochs is None else epoch + num_epochs,
        )

    def batches_left(self, plan: EpochPlan) -> int | None:
        """Batches the rest of the run serves, or ``None`` for one without a length or an end.

        Args:
            plan: The pipeline's epoch plan.

        Returns:
            The count.
        """
        return self._run(plan).batches()

    def read_for_workers(self, pipeline: Any) -> IndexedRead:
        """The read a worker runs for the pipeline's run from the cursor, single batches.

        Args:
            pipeline: The pipeline over an ``INDEXED`` source.

        Returns:
            The read, which pickles with what it holds.
        """
        plan: EpochPlan = pipeline.epoch_plan
        return IndexedRead(
            pipeline.source,
            RunUnits(run=self._run(plan)),
            HostNaming(pipeline.source, plan, shuffled=pipeline.shuffle),
            key=_key(pipeline),
            with_provenance=False,
            settings=JaxSettings.current(),
        )


class _RunElements(grain.IterDataset):
    """A stream's run dataset with its decoded units turned into host elements."""

    def __init__(
        self,
        run: grain.IterDataset,
        units: RunUnits,
        check: Callable[[Batch], None],
        with_provenance: bool,
        settings: JaxSettings,
    ) -> None:
        super().__init__()
        self._run = run
        self._units = units
        self._check = check
        self._with_provenance = with_provenance
        self._settings = settings

    def __iter__(self) -> _RunElementsIterator:
        return _RunElementsIterator(self)


class _RunElementsIterator(grain.DatasetIterator):
    """The run dataset's units, numbered from the run's start, as host elements."""

    def __init__(self, dataset: _RunElements) -> None:
        super().__init__(iter(dataset._run))  # noqa: SLF001 - the dataset this iterator serves
        self._dataset = dataset
        self._ordinal = 0

    def __next__(self) -> HostElement:
        units = self._dataset._units  # noqa: SLF001
        try:
            with self._dataset._settings.entered():  # noqa: SLF001 - the run read on this thread
                element = next(self._parent)  # Grain's single-parent accessor
        except StopIteration:
            host_workers.check_all_arrived(self._ordinal, units.units())
            raise
        ordinal = self._ordinal
        host_workers.check_arrival(ordinal, element[4])
        self._ordinal += 1
        batches = units.unit(ordinal)
        assert batches is not None  # noqa: S101 - the dataset serves the schedule's units
        batch, provenance = _stream_unit(
            element[:4],
            chunk=len(batches) if units.is_chunk(ordinal) else None,
            check=self._dataset._check,  # noqa: SLF001
            with_provenance=self._dataset._with_provenance,  # noqa: SLF001
        )
        epoch, position = units.after(ordinal)
        return HostElement(batch, provenance, (epoch, position, 0), ordinal)

    def get_state(self) -> dict[str, Any]:
        """Units served; a stream resumes through the host stage's cursor, not this state."""
        return {"units": self._ordinal}

    def set_state(self, state: dict[str, Any]) -> None:  # noqa: DOC502
        """Not supported: a stream resumes through the host stage's cursor.

        Args:
            state: The state to restore.

        Raises:
            NotImplementedError: Always.
        """
        del state
        raise NotImplementedError("a stream resumes through the pipeline's host stage")


class _Received(grain.IterDataset):
    """An indexed run's units from worker processes, checked to arrive in turn and restored."""

    def __init__(self, parent: grain.IterDataset, units: RunUnits) -> None:
        super().__init__(parent)
        self._units = units

    def __iter__(self) -> _ReceivedIterator:
        return _ReceivedIterator(iter(self._parent), self._units)


class _ReceivedIterator(grain.DatasetIterator):
    """Units in turn, their provenance the read-only mappings a thread read serves."""

    def __init__(self, parent: grain.DatasetIterator, units: RunUnits) -> None:
        super().__init__(parent)
        self._units = units
        self._next = 0

    def __next__(self) -> HostElement:
        try:
            element: HostElement = next(self._parent)
        except StopIteration:
            host_workers.check_all_arrived(self._next, self._units.units())
            raise
        host_workers.check_arrival(self._next, element.unit)
        self._next += 1
        return element._replace(provenance=host_workers.received_provenance(element.provenance))

    def get_state(self) -> dict[str, Any]:
        """Units received; a run resumes through the host stage's cursor, not this state."""
        return {"units": self._next}

    def set_state(self, state: dict[str, Any]) -> None:  # noqa: DOC502
        """Not supported: a run resumes through the host stage's cursor.

        Args:
            state: The state to restore.

        Raises:
            NotImplementedError: Always.
        """
        del state
        raise NotImplementedError("a run resumes through the pipeline's host stage")


@dataclasses.dataclass(frozen=True, slots=True)
class _FirstUnit:
    """Reads a run's first unit: what the footprint probe worker does once.

    Attributes:
        read: An indexed read, or a stream's run dataset.
    """

    read: Any

    def __call__(self) -> Any:
        """The run's first unit, as a worker reads it."""
        if isinstance(self.read, IndexedRead):
            return self.read(0)
        return next(iter(self.read))


def _key(pipeline: Any) -> np.ndarray | None:
    """The pipeline's key as host words when it shuffles, else ``None``."""
    if not pipeline.shuffle:
        return None
    return key_words(pipeline._epoch_key_base.get_value())  # noqa: SLF001 - the pipeline's key base


class _Served:
    """The units a call of ``raw_batches`` takes from the run it was created for.

    It advances the stage's cursor as each unit is taken. Once its run reached its end it keeps
    raising ``StopIteration``; once anything else ended its run it is refused, naming what did,
    so it never serves a run opened after it. A read error ends the run where the delivered
    units end, and reaches the caller unchanged.
    """

    def __init__(
        self,
        stage: HostStage,
        options: tuple[Any, ...],
        with_provenance: bool,
        token: _RunToken,
    ) -> None:
        self._stage = stage
        self._options = options
        self._with_provenance = with_provenance
        self._token = token  # the run stays open while this iterator lives

    def __iter__(self) -> _Served:
        return self

    def __next__(self) -> Batch | tuple[Batch, Provenance]:
        ended = self._token.ended_by
        if ended is _RUN_END:
            raise StopIteration
        if ended is not None:
            raise RuntimeError(
                f"this iterator's run was ended by {ended}; call raw_batches() or iter() on the "
                "pipeline again to continue from where iteration stands"
            )
        stage = self._stage
        # A token nothing ended belongs to the stage's open run: ending a run marks its token.
        iterator = cast(_Placed, stage.iterator)
        try:
            batch, provenance, after = next(iterator)
        except StopIteration:
            stage._end(_RUN_END)  # noqa: SLF001 - the run this iterator serves
            raise
        except BaseException as error:
            # A read error (or an interrupt) leaves the Grain iterator past the unit it failed
            # on; ending the run here lets the next call reopen it at the cursor, which counts
            # only the units delivered, so iterating again loses and repeats nothing.
            stage._end(f"the error it raised ({type(error).__name__})")  # noqa: SLF001
            raise
        cursor = stage.cursor
        cursor.epoch, cursor.position, cursor.arrived = after
        stage._opened_for = (*self._options, cursor.epoch, cursor.position)  # noqa: SLF001
        if self._with_provenance:
            # The run was opened with provenance (its options say so), so every unit carries it.
            return batch, cast(Provenance, provenance)
        return batch


STATE_VERSION = 3
"""The layout of ``Pipeline.get_state()``: the host stage's cursor and the configuration it fits."""


class _RunToken:
    """A run's lifetime: the run closes when the last holder of its token is gone.

    Attributes:
        ended_by: What ended the run, or ``None`` while it is open: :data:`_RUN_END` once it
            served its last batch, else the call that ended it.
    """

    __slots__ = ("__weakref__", "ended_by")

    def __init__(self) -> None:
        """An open run's token."""
        self.ended_by: str | None = None


_RUN_END = "the run's end"
"""What a run that served its last batch was ended by; its iterators then stop."""

_OPTION_NAMES = (
    "source",
    "epoch_plan",
    "shuffle",
    "chunk",
    "with_provenance",
    "jax_settings",
    "plan",
    "epoch",
    "position",
)
"""The names of a run's options, in the order :meth:`HostStage.batches` records them."""


def _replaced_by(opened: tuple[Any, ...] | None, wanted: tuple[Any, ...]) -> str:
    """The call that ends a run opened for ``opened`` to open one for ``wanted``, named."""
    if opened is None:
        return "a later call"
    if opened[0] != wanted[0]:
        return (
            "a call over another source: a clone or merged copy of the pipeline, which shares "
            "its host stage and cursor, continues them in turn, not interleaved"
        )
    changed = ", ".join(
        f"{name}={new!r}"
        for name, old, new in zip(_OPTION_NAMES, opened, wanted, strict=True)
        if old != new
    )
    return f"a later call reading with {changed}"


_RUN_HOLDERS: weakref.WeakKeyDictionary[Any, _RunToken] = weakref.WeakKeyDictionary()
"""The token of the run each live pipeline was last served from."""


def _close_run(run: list[_Placed]) -> None:
    """Close a run's Grain iterator and the threads behind it, and let it go."""
    while run:
        run.pop().close()


def _has_indexed_host_read(source: DataSourceModule) -> bool:
    """Whether ``source`` has a ``get_batch`` taking ``epochs`` and ``contiguous`` by keyword.

    The shape :class:`~datarax.core.data_source.IndexedHostRead` declares, as far as a run's
    start can see it: a stream's forward ``get_batch(batch_size, ...)`` has neither keyword.
    """
    get_batch = getattr(source, "get_batch", None)
    if not callable(get_batch):
        return False
    parameters = inspect.signature(get_batch).parameters.values()
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return True
    names = {
        parameter.name
        for parameter in parameters
        if parameter.kind
        in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    }
    return {"epochs", "contiguous"} <= names


def _batch_bytes(source: DataSourceModule, batch_size: int) -> int:
    """The bytes of one batch's data, as the device holds it, from the source's spec."""
    return batch_size * spec_bytes(declared_spec(source))


class HostStageHolder(HostValue):
    """A pipeline's host stage, held on the host and out of NNX state (see :class:`HostValue`)."""

    __slots__ = ()
    value: HostStage


__all__ = [
    "STATE_VERSION",
    "Cursor",
    "HostStage",
    "HostStageHolder",
]
