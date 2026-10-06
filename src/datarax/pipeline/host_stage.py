"""The host stage: a pipeline's batches named, read and placed on the host, ahead of the consumer.

Every source kind reads through one stage, which does nothing but read:

- An ``INDEXED`` source's run is a :class:`~datarax.pipeline.epochs.Run` of the pipeline's plan,
  cut into units (:class:`RunUnits`: one batch, or a chunk of ``K`` batches read together).
  Grain maps each unit's number to a read on worker threads
  (``grain.MapDataset.range(...).map(...)``): the batch's records are named on the CPU device
  (:class:`~datarax.pipeline.epochs.HostNaming`) and read with the source's stateless host read,
  ``get_batch(indices, *, epochs, contiguous)``
  (:class:`~datarax.core.data_source.IndexedHostRead`).
- A stream with a run dataset (TFDS) reads the same units of its passes through that dataset: one
  Grain dataset for the run, numbered from the run's start, which Grain can slice across workers.
- Any other stream (HuggingFace) is read pass by pass, at a position of the host stage's own, its
  batches cut by :func:`~datarax.pipeline.epochs.stream_batches`.

A stream's run, either way, is read and decoded on one producer thread ahead of the consumer.

A batch's provenance comes beside it when asked. Units are read ahead of the consumer and each is
placed on the default device as it is taken, uncommitted, so a jitted step taking them compiles
once; nothing else transfers. Where iteration stands (:class:`Cursor`) advances when the consumer
takes a batch, never when a worker reads one, so the cursor names exactly the batches served
whatever the workers have read ahead. One Grain iterator serves a run across calls; it is closed
at the run's end, by ``close()``, and when collected.
"""

from __future__ import annotations

import dataclasses
import math
import sys
import weakref
from collections.abc import Callable, Iterator, Mapping
from types import MappingProxyType
from typing import Any, cast, NamedTuple

import grain
import jax
import numpy as np

from datarax.core import batch_ops
from datarax.core.data_source import (
    DataSourceModule,
    IndexedHostRead,
    IndexedHostReadWithProvenance,
    Provenance,
    RecordIdentity,
)
from datarax.core.element_batch import Batch
from datarax.core.index_words import from_words, to_words
from datarax.core.prng import key_words
from datarax.core.spec import declared_spec, validate_batch, validate_device_dtypes
from datarax.pipeline.epochs import EpochPlan, HostNaming, Run, stream_batches
from datarax.sources._source_base import StreamCursor, StreamingSourceBase
from datarax.sources.eager_source import HostValue


_READ_THREADS = 1
"""Read threads per run. One host thread serves in-memory and memory-mapped gathers (millions of
records a second); a decode-bound read gains little from a second thread (the GIL), and Grain
processes are the remedy there."""
_READ_BUFFER = 2
"""Units read ahead of the consumer. Each is placed when the consumer takes it: ``device_put`` is
asynchronous on an accelerator, so the transfer still overlaps the step before. A placement thread
running ahead holds more copies: on the P3 shape (CV-1, B=64, CPU device) peak host RSS measures
68-101 MB with one and about 22 MB without, against the 75 MiB bound."""


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


class HostElement(NamedTuple):
    """A unit as read on the host: its batch, its records' provenance, where the run stands after.

    Attributes:
        batch: The unit's batch, ``(B, ...)`` or a chunk ``(K, B, ...)``, on the host.
        provenance: One mapping per record, in row order, or ``None`` when not asked for.
        after: ``(epoch, position, arrived)`` once the unit is served.
    """

    batch: Batch
    provenance: Provenance | None
    after: tuple[int, int, int]


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class RunUnits:
    """A run cut into units: chunks of ``chunk`` full batches, then the rest one at a time.

    While ``chunk`` full batches remain, a unit holds that many; the full batches left over come
    singly, and the run's short final batch last, so a scan over chunks needs a ``chunk``-step
    body, a one-step body, and one more compile for the short batch. Units are numbered from the
    run's start. A :class:`~datarax.core.data_source.BatchSchedule`.

    Attributes:
        run: The run.
        chunk: Full batches per chunk, or ``None`` for single batches.
    """

    run: Run
    chunk: int | None = None

    @property
    def size(self) -> int:
        """Batches in a chunk; 1 when the run is served in single batches."""
        return 1 if self.chunk is None else self.chunk

    def is_chunk(self, ordinal: int) -> bool:
        """Whether unit ``ordinal`` is a chunk ``(K, B, ...)`` rather than a single batch."""
        if self.chunk is None:
            return False
        full = self._full_batches()
        return full is None or ordinal < full // self.chunk

    def _full_batches(self) -> int | None:
        """The run's full batches, or ``None`` for a run without an end."""
        total = self.run.batches()
        if total is None or total == 0:
            return total
        last = self.run.batch(total - 1)
        assert last is not None  # noqa: S101 - a batch below the run's count exists
        return total - (last[2] < self.run.plan.batch_size)

    def unit(self, ordinal: int) -> tuple[tuple[int, int, int], ...] | None:
        """The batches of unit ``ordinal``, or ``None`` past the run's end.

        Args:
            ordinal: The unit's place in the run.

        Returns:
            Its batches as ``(start, epoch, size)``.
        """
        size = self.size
        full = self._full_batches()
        chunks = None if full is None else full // size
        if chunks is None or ordinal < chunks:
            first = ordinal * size
            batches = tuple(self.run.batch(first + k) for k in range(size))
            return None if None in batches else cast(tuple[tuple[int, int, int], ...], batches)
        batch = self.run.batch(chunks * size + ordinal - chunks)
        return None if batch is None else (batch,)

    def units(self) -> int | None:
        """The run's units, or ``None`` for a run without an end."""
        full = self._full_batches()
        if full is None:
            return None
        total = self.run.batches()
        assert total is not None  # noqa: S101 - a run with full batches counted has an end
        size = self.size
        return full // size + total - full // size * size

    def after(self, ordinal: int) -> tuple[int, int]:
        """``(epoch, position)`` once unit ``ordinal`` is served.

        Args:
            ordinal: A unit of the run.

        Returns:
            Where the next unit starts.
        """
        batches = self.unit(ordinal)
        assert batches is not None  # noqa: S101 - only a served unit is asked for
        start, epoch, size = batches[-1]
        position, epoch = self.run.plan.advance(start, epoch, size)
        return epoch, position


class IndexedRead:
    """The read of an ``INDEXED`` source's run, unit by unit: what each Grain worker runs.

    A unit's batches are named on the CPU device and read with one host read; a chunk is split
    into its batches and stacked, ``(K, B, ...)``. Asked for provenance, a source implementing
    :class:`~datarax.core.data_source.IndexedHostReadWithProvenance` reads both at once; any
    other looks its records' provenance up by index. A read marks a run consecutive records as
    contiguous only when its names are (so a source reads it as views). It holds the source, the
    run's units, the naming and the key as host words, and pickles with them, so a worker process
    reads with a copy of it.
    """

    def __init__(
        self,
        source: DataSourceModule,
        units: RunUnits,
        naming: HostNaming,
        *,
        key: np.ndarray | None,
        with_provenance: bool,
    ) -> None:
        """Hold what a unit's read needs.

        Args:
            source: The indexed source.
            units: The run's units.
            naming: The pipeline's host naming.
            key: The pipeline's key as host words, or ``None`` when it does not shuffle.
            with_provenance: Whether each unit's provenance is looked up.
        """
        self.source = source
        self.units = units
        self.naming = naming
        self.key = key
        self.with_provenance = with_provenance

    def __call__(self, ordinal: int) -> HostElement:
        """Read unit ``ordinal``.

        Args:
            ordinal: The unit, from the run's start.

        Returns:
            The unit as read on the host.
        """
        batches = self.units.unit(ordinal)
        assert batches is not None  # noqa: S101 - Grain asks only for the run's units
        named = [self.naming(start, epoch, size, self.key) for start, epoch, size in batches]
        indices = np.concatenate([indices for indices, _ in named])
        epochs = np.concatenate([epochs for _, epochs in named])
        values = from_words(indices)
        contiguous = len(values) > 1 and bool(np.all(np.diff(values.astype(np.int64)) == 1))
        source = self.source
        provenance: Provenance | None = None
        if self.with_provenance and isinstance(source, IndexedHostReadWithProvenance):
            batch, provenance = source.read_with_provenance(
                indices, epochs=epochs, contiguous=contiguous
            )
        else:
            batch = cast(IndexedHostRead, source).get_batch(
                indices, epochs=epochs, contiguous=contiguous
            )
            provenance = source.provenance(indices) if self.with_provenance else None
        if self.units.is_chunk(ordinal):
            batch = batch_ops.stack(batch_ops.split(batch, len(batches)))
        epoch, position = self.units.after(ordinal)
        return HostElement(batch, provenance, (epoch, position, 0))


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
    """What a stream's run reads: its batch rule, key, chunking and spec."""

    batch_size: int
    drop_last: bool
    key: np.ndarray | None
    chunk: int | None
    with_provenance: bool
    check: Callable[[Batch], None]


class _StreamIterator(grain.DatasetIterator):
    """Units of a stream's run, cut by :func:`stream_batches` from the host stage's position."""

    def __init__(self, source: StreamingSourceBase, cursor: Cursor, config: _StreamRead) -> None:
        super().__init__()
        self._source = source
        self._config = config
        self._stream = StreamCursor(pass_index=cursor.epoch, arrived=cursor.arrived)
        self._epoch, self._position = cursor.epoch, cursor.position
        passes = None if cursor.end_epoch is None else max(0, cursor.end_epoch - cursor.epoch)
        self._skip(cursor.position)
        self._batches = stream_batches(
            self._pull, config.batch_size, drop_last=config.drop_last, num_epochs=passes
        )
        self._held: list[tuple[Batch, Provenance, tuple[int, int, int]]] = []
        self._ended = False
        weakref.finalize(self, self._stream.close)

    def _skip(self, records: int) -> None:
        """Read past ``records`` of the cursor's pass without naming them (a resumed pass)."""
        left = records
        while left:
            batch, _ = self._source.read_from(
                self._stream, left, key=self._config.key, read_size=self._config.batch_size
            )
            if not batch.batch_size:
                break
            left -= batch.batch_size
        self._stream.arrived -= records - left if self._named_by_arrival() else 0

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
        self._fill(chunk)
        if not self._held:
            raise StopIteration
        parts, full = self._unit(chunk)
        batch = batch_ops.stack([b for b, _, _ in parts]) if full and stacked else parts[0][0]
        provenance = tuple(record for _, part, _ in parts for record in part)
        return HostElement(
            batch, provenance if self._config.with_provenance else None, parts[-1][2]
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


def _checker(source: DataSourceModule, batch_size: int) -> Callable[[Batch], None]:
    """A check of a stream's batches against the spec its source declares, as the device holds it.

    The spec is read once and refused if it declares a dtype JAX arrays cannot hold as declared.
    Batches are checked in the precision mode the spec was read in, on whichever thread reads
    them: ``jax.enable_x64`` is thread-local, and the read threads do not inherit it.

    Args:
        source: The stream.
        batch_size: Records per batch.

    Returns:
        A function refusing a batch whose structure, shapes or dtypes disagree with the spec.
    """
    element_spec = declared_spec(source)
    validate_device_dtypes(element_spec)
    x64 = bool(jax.config.read("jax_enable_x64"))

    def check(batch: Batch) -> None:
        with jax.enable_x64(x64):
            validate_batch(
                batch.data, element_spec, batch_size=batch_size, as_the_device_holds=True
            )

    return check


def _place(element: HostElement) -> tuple[Batch, Provenance | None, tuple[int, int, int]]:
    """The element's batch on the default device, uncommitted; provenance stays on the host."""
    return jax.device_put(element.batch), element.provenance, element.after


type _Placed = grain.DatasetIterator[tuple[Batch, Provenance | None, tuple[int, int, int]]]


class HostStage:
    """A pipeline's host stage: its cursor and the Grain iterator serving the current run.

    Attributes:
        cursor: Where iteration stands.
        read_threads: Grain threads reading units of an indexed source's run.
        read_buffer: Units read ahead of the consumer.
    """

    def __init__(self, *, end_epoch: int | None) -> None:
        """A stage at the start of the first epoch, with no run open.

        Args:
            end_epoch: The epoch the first run stops before, or ``None`` for no end.
        """
        self.cursor = Cursor(epoch=0, position=0, arrived=0, end_epoch=end_epoch)
        self.read_threads = _READ_THREADS
        self.read_buffer = _READ_BUFFER
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

    def _open(self, dataset: grain.IterDataset, options: tuple[Any, ...]) -> _RunToken:
        """Start the run's iterator: units read ahead, each placed as the consumer takes it.

        The run lives as long as its token: the pipelines it served and the iterators serving
        it hold the token, and the stage only a weak reference. A compiled-step cache keyed by
        a pipeline's graph keeps the stage (a static of that graph), so a run the stage owned
        would keep its threads and the batches read ahead for the life of the process.

        Args:
            dataset: The run's dataset of host elements.
            options: What the run was opened for, compared when a later call continues it.

        Returns:
            The run's token.
        """
        self._run_iterator.append(iter(dataset.map(_place)))
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
        # A run is read in the caller's precision mode (a thread-local setting the read threads
        # do not inherit) and with the stage's read options; a change of any opens a new run.
        x64 = bool(jax.config.read("jax_enable_x64"))
        options = (
            id(source),
            pipeline.epoch_plan,
            pipeline.shuffle,
            chunk,
            with_provenance,
            x64,
            self.read_threads,
            self.read_buffer,
        )
        cursor = self.cursor
        opened = (*options, cursor.epoch, cursor.position)
        token = None if self._token is None else self._token()
        if token is None or self.iterator is None or self._opened_for != opened:
            self._end(_replaced_by(self._opened_for, opened))
            dataset = self._dataset(pipeline, chunk, with_provenance)
            token = self._open(dataset, opened)
        _RUN_HOLDERS[pipeline] = token
        return cast(
            Iterator[Batch] | Iterator[tuple[Batch, Provenance]],
            _Served(self, (*options,), with_provenance, token),
        )

    def _dataset(
        self, pipeline: Any, chunk: int | None, with_provenance: bool
    ) -> grain.IterDataset:
        """The run's dataset of host elements, from the cursor."""
        source: DataSourceModule = pipeline.source
        key = (
            key_words(pipeline._epoch_key_base.get_value())  # noqa: SLF001 - the pipeline's key base
            if pipeline.shuffle
            else None
        )
        plan: EpochPlan = pipeline.epoch_plan
        if source.record_identity is RecordIdentity.INDEXED:
            if not callable(getattr(source, "get_batch", None)):
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
            )
            count = units.units()
            return (
                grain.MapDataset.range(sys.maxsize if count is None else count)
                .map(read)
                .to_iter_dataset(
                    grain.ReadOptions(
                        num_threads=self.read_threads, prefetch_buffer_size=self.read_buffer
                    )
                )
            )
        if not isinstance(source, StreamingSourceBase):
            raise TypeError(
                f"{type(source).__name__} is a {source.record_identity.name} stream that is not a "
                "StreamingSourceBase, whose pass reader the host stage reads a stream with"
            )
        return grain.experimental.ThreadPrefetchIterDataset(
            self._stream_elements(pipeline, chunk, with_provenance, key),
            prefetch_buffer_size=self.read_buffer,
        )

    def _stream_elements(
        self, pipeline: Any, chunk: int | None, with_provenance: bool, key: np.ndarray | None
    ) -> grain.IterDataset:
        """A stream's run as host elements: through its run dataset when it has one (TFDS)."""
        source: StreamingSourceBase = pipeline.source
        check = _checker(source, pipeline.batch_size)
        plan: EpochPlan = pipeline.epoch_plan
        if plan.length is not None:
            units = RunUnits(run=self._run(plan), chunk=chunk)
            run = source.run_dataset(units, key)
            if run is not None:
                return _RunElements(run, units, check, with_provenance)
        config = _StreamRead(
            batch_size=pipeline.batch_size,
            drop_last=pipeline.drop_last,
            key=key,
            chunk=chunk,
            with_provenance=with_provenance,
            check=check,
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
            "fingerprint": _fingerprint(pipeline),
        }

    def restore(self, pipeline: Any, state: Mapping[str, Any]) -> None:
        """Move the cursor to where ``state`` stands, closing the run open now.

        Args:
            pipeline: The pipeline the state is restored into.
            state: A state :meth:`state` produced, as saved or as a checkpoint store returns it.

        Raises:
            ValueError: If the state is another version, another kind of source, or was produced
                under another configuration, naming what differs; nothing converts a layout.
        """
        state = _plain(state)
        version = state.get("version")
        if version != STATE_VERSION:
            if version is None:
                saved = "a state without a version (a pipeline module_state layout)"
            elif version == _SESSION_VERSION:
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
        mine = _fingerprint(pipeline)
        for field, value in mine.items():
            if state["fingerprint"].get(field) != value:
                raise ValueError(
                    f"the state was produced with {field}={state['fingerprint'].get(field)!r} "
                    f"but this pipeline has {field}={value!r}; a state is only valid for the "
                    "configuration that produced it"
                )
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
        key = (
            key_words(pipeline._epoch_key_base.get_value())  # noqa: SLF001 - the pipeline's key base
            if pipeline.shuffle
            else None
        )
        return IndexedRead(
            pipeline.source,
            RunUnits(run=self._run(plan)),
            HostNaming(pipeline.source, plan, shuffled=pipeline.shuffle),
            key=key,
            with_provenance=False,
        )


class _RunElements(grain.IterDataset):
    """A stream's run dataset with its decoded units turned into host elements."""

    def __init__(
        self,
        run: grain.IterDataset,
        units: RunUnits,
        check: Callable[[Batch], None],
        with_provenance: bool,
    ) -> None:
        super().__init__()
        self._run = run
        self._units = units
        self._check = check
        self._with_provenance = with_provenance

    def __iter__(self) -> _RunElementsIterator:
        return _RunElementsIterator(self)


class _RunElementsIterator(grain.DatasetIterator):
    """The run dataset's units, numbered from the run's start, as host elements."""

    def __init__(self, dataset: _RunElements) -> None:
        super().__init__(iter(dataset._run))  # noqa: SLF001 - the dataset this iterator serves
        self._dataset = dataset
        self._ordinal = 0

    def __next__(self) -> HostElement:
        element = next(self._parent)  # Grain's single-parent accessor
        ordinal, self._ordinal = self._ordinal, self._ordinal + 1
        units = self._dataset._units  # noqa: SLF001
        batches = units.unit(ordinal)
        assert batches is not None  # noqa: S101 - the dataset serves the schedule's units
        batch, provenance = _stream_unit(
            element,
            chunk=len(batches) if units.is_chunk(ordinal) else None,
            check=self._dataset._check,  # noqa: SLF001
            with_provenance=self._dataset._with_provenance,  # noqa: SLF001
        )
        epoch, position = units.after(ordinal)
        return HostElement(batch, provenance, (epoch, position, 0))

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
_SESSION_VERSION = 2  # the layout of ``PipelineIterator.get_state()``, named when refused


def _fingerprint(pipeline: Any) -> dict[str, Any]:
    """The configuration a state is only valid for: batch rule, length, epochs, key and order."""
    plan: EpochPlan = pipeline.epoch_plan
    return {
        "batch_size": plan.batch_size,
        "length": plan.length,
        "drop_last": plan.drop_last,
        "num_epochs": plan.num_epochs,
        "shuffled": bool(pipeline.shuffle),
        "seed": [int(word) for word in key_words(pipeline._epoch_key_base.get_value())],  # noqa: SLF001
        "order": {"kind": "global"},
    }


def _plain(value: Any) -> Any:
    """A saved state as plain Python values: a checkpoint store may return NumPy leaves."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


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
    "x64",
    "read_threads",
    "read_buffer",
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


def _batch_bytes(source: DataSourceModule, batch_size: int) -> int:
    """The bytes of one batch's data, as the device holds it, from the source's spec."""
    spec = declared_spec(source)
    return batch_size * sum(
        math.prod(leaf.shape) * np.dtype(leaf.dtype).itemsize for leaf in jax.tree.leaves(spec)
    )


class HostStageHolder(HostValue):
    """A pipeline's host stage, held on the host and out of NNX state (see :class:`HostValue`)."""

    __slots__ = ()
    value: HostStage


__all__ = [
    "STATE_VERSION",
    "Cursor",
    "HostElement",
    "HostStage",
    "HostStageHolder",
    "IndexedRead",
    "RunUnits",
]
