"""A stream's read: one resumable iterator over its passes, cut into the batches asked for.

:meth:`~datarax.sources.StreamingSourceBase.iter_batches` returns a :class:`StreamBatches`. It opens
each pass with the stream's ``_open_pass``, takes the records in the order the pass yields them,
and cuts them into ``batch_size`` host ``Batch``es named by the stream: ``STREAM_IDS`` by the ids
the pass reports, ``ARRIVAL`` by arrival ordinals counted by the iterator, never repeated within
it; ``epochs`` is each row's pass. A pass's last records are completed from the next pass's head,
each row keeping its pass, unless ``drop_last`` skips them; after ``num_passes`` passes the
iterator ends, its last batch possibly short, and with ``None`` it never ends.

The iterator is the one place a stream's pass is counted. Its state
(:meth:`StreamBatches.get_state`) is where it stands: the pass, the records of it served, the
ordinals served and the passes left. An iterator built with the same arguments and set to that
state serves exactly what the original would have: a pass resumed midway is read again from its
start to the saved count, as it was first read, its records unnamed.

An error raised while a pass is read reaches the ``next`` that met it; the iterator then refuses to
go on, naming the error, and a new iterator starts again. A pass's reader stopped midway is closed
by :meth:`StreamBatches.close`, when the iterator is collected, or at interpreter exit before module
teardown.
"""

from __future__ import annotations

import logging
import time
import weakref
from collections.abc import Generator, Iterator
from dataclasses import dataclass, field
from typing import Any, cast, NamedTuple, Protocol

import jax
import numpy as np
from jax.typing import ArrayLike
from jaxtyping import PyTree

from datarax.core import batch_ops
from datarax.core.data_source import Provenance, RecordIdentity
from datarax.core.element_batch import Batch
from datarax.core.index_words import MAX_RECORDS, to_words
from datarax.core.prng import key_words


logger = logging.getLogger(__name__)


class StreamChunk(NamedTuple):
    """Records a stream read, in the order it serves them.

    Attributes:
        columns: The records' array parts as host NumPy columns, leading axis the records.
        provenance: One mapping per record of its strings and objects.
        ids: uint64 ``(n,)`` ids the stream reports (``(shard << 32) | offset`` for
            ``STREAM_IDS``), or ``None`` for a stream naming records by arrival.
    """

    columns: PyTree
    provenance: Provenance
    ids: np.ndarray | None


def chunk_size(chunk: StreamChunk) -> int:
    """The number of records ``chunk`` holds."""
    return len(chunk.provenance)


class PassReader(Protocol):
    """What the iterator reads a stream through: its passes and how it names records."""

    @property
    def record_identity(self) -> RecordIdentity:
        """``STREAM_IDS`` or ``ARRIVAL``."""
        ...

    def _open_pass(
        self, pass_index: int, key: np.ndarray | None, read_size: int
    ) -> Iterator[StreamChunk]:
        """Read pass ``pass_index`` in its order, as chunks of records."""
        ...


def _join(chunks: list[StreamChunk]) -> StreamChunk:
    if len(chunks) == 1:
        return chunks[0]
    columns = jax.tree.map(lambda *parts: np.concatenate(parts), *(c.columns for c in chunks))
    provenance = tuple(record for c in chunks for record in c.provenance)
    ids = (
        None
        if chunks[0].ids is None
        else np.concatenate([c.ids for c in chunks if c.ids is not None])
    )
    return StreamChunk(columns, provenance, ids)


def _split(chunk: StreamChunk, size: int) -> tuple[StreamChunk, StreamChunk]:
    def part(rows: slice) -> StreamChunk:
        return StreamChunk(
            jax.tree.map(lambda column: column[rows], chunk.columns),
            chunk.provenance[rows],
            None if chunk.ids is None else chunk.ids[rows],
        )

    return part(slice(None, size)), part(slice(size, None))


class _ReaderCloser:
    """Closes a pass's open reader, holding it only weakly.

    ``weakref.finalize`` keeps this object, never the reader, so it keeps no stream alive. It runs
    when the cursor is collected or, for a cursor still alive, at interpreter exit, before module
    teardown: a backend reader left suspended until teardown is finalized with its modules half
    cleared, where HuggingFace's Parquet reader hangs.
    """

    __slots__ = ("reader",)

    def __init__(self) -> None:
        """Hold no reader."""
        self.reader: weakref.ref[Iterator[StreamChunk]] | None = None

    def __call__(self) -> None:
        """Close the reader if it is still alive."""
        reader = None if self.reader is None else self.reader()
        self.reader = None
        if isinstance(reader, Generator):
            reader.close()


@dataclass(slots=True, weakref_slot=True)
class StreamCursor:
    """Where the read is: the pass, the open pass's reader, records read ahead, arrivals.

    ``failure`` is the error that stopped the open pass's reader. The cursor's finalizer closes a
    reader still open when the cursor is collected or at exit.
    """

    pass_index: int = 0
    reader: Iterator[StreamChunk] | None = None
    ahead: StreamChunk | None = None
    arrived: int = 0
    failure: BaseException | None = None
    closer: _ReaderCloser = field(default_factory=_ReaderCloser)

    def __post_init__(self) -> None:
        """Close the reader this cursor holds when the cursor goes, never holding the cursor."""
        weakref.finalize(self, self.closer)

    def open(self, reader: Iterator[StreamChunk]) -> None:
        """Hold ``reader`` as the open pass's reader, closing any reader held before."""
        self.close()
        self.reader = reader
        self.closer.reader = weakref.ref(reader)

    def close(self) -> None:
        """Close and drop the open pass's reader, if any."""
        self.closer()
        self.reader = None


class StreamBatches[T](Iterator[T]):
    """The batches of a stream's passes, resumable (see the module docstring).

    ``T`` is what each ``next`` returns: a host ``Batch``, or asked for provenance, the batch and
    its records' provenance, one mapping per row.
    """

    def __init__(  # noqa: PLR0913 - the read's batch rule, order and reads are separate settings
        self,
        source: PassReader,
        batch_size: int,
        *,
        num_passes: int | None,
        key: ArrayLike | None,
        with_provenance: bool,
        read_size: int | None,
        drop_last: bool,
    ) -> None:
        """Start before the first record of pass 0.

        Args:
            source: The stream.
            batch_size: Records per batch.
            num_passes: Passes served, or ``None`` for no end.
            key: The key each pass's order is drawn from (typed, raw or its host words, read once,
                as host words), or ``None`` for the stream's own order.
            with_provenance: Whether each batch comes with its records' provenance.
            read_size: Records a pass reads at a time; ``None`` reads it ``batch_size`` at a time.
            drop_last: Whether a pass's records short of a full batch are skipped.

        Raises:
            ValueError: If ``batch_size``, ``read_size`` or ``num_passes`` is below one.
        """
        read_size = batch_size if read_size is None else read_size
        for name, value in (("batch_size", batch_size), ("read_size", read_size)):
            if value < 1:
                raise ValueError(f"{name} must be at least 1; got {value}")
        if num_passes is not None and num_passes < 1:
            raise ValueError(f"num_passes must be at least 1 or None; got {num_passes}")
        self._source = source
        self._batch_size = batch_size
        self._read_size = read_size
        self._key = None if key is None else key_words(key)
        self._with_provenance = with_provenance
        self._drop_last = drop_last
        self._cursor = StreamCursor()
        self._records = 0
        self._end = num_passes
        self._replay = 0
        # Records of the current pass, unknown until a pass is seen to begin: a pass read whole
        # holding none means the stream holds none, which only a pass seen from its start shows.
        self._served: int | None = None
        self._done = False

    def get_state(self) -> dict[str, Any]:
        """Where the iterator stands.

        Returns:
            ``pass``, the records of it served, ``arrived`` (ordinals served) and ``passes_left``
            counting the current pass, or ``None`` for no end.
        """
        cursor = self._cursor
        return {
            "pass": cursor.pass_index,
            "records": self._records,
            "arrived": cursor.arrived,
            "passes_left": None if self._end is None else max(0, self._end - cursor.pass_index),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Stand where ``state`` says, as :meth:`get_state` wrote it; a pass midway is read again.

        The state is where an iterator built with the same arguments stood: the batch size, key,
        read size and ``drop_last`` are this iterator's, not the state's, so an iterator built
        with others serves its own batches from that place.

        Args:
            state: The state to stand at.

        Raises:
            ValueError: If ``state`` lacks a field or holds a negative count; the iterator is
                left as it was.
        """
        try:
            place = [int(state[name]) for name in ("pass", "records", "arrived")]
            left = None if state["passes_left"] is None else int(state["passes_left"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "a stream iterator's state holds int pass, records and arrived, and passes_left "
                f"(an int or None); got {state!r}"
            ) from error
        if min(*place, 0 if left is None else left) < 0:
            raise ValueError(f"a stream iterator's state holds no negative count; got {state!r}")
        pass_index, records, arrived = place
        self.close()
        self._cursor = StreamCursor(pass_index=pass_index, arrived=arrived)
        self._records = self._replay = records
        self._end = None if left is None else pass_index + left
        self._served = None
        self._done = self._end is not None and pass_index >= self._end

    def close(self) -> None:
        """Close the open pass's reader; the iterator ends."""
        self._cursor.close()
        self._done = True

    def __iter__(self) -> StreamBatches[T]:
        """The iterator itself."""
        return self

    def __next__(self) -> T:  # noqa: DOC503 - re-raised as met
        """The next batch, with its provenance when asked for.

        Returns:
            The batch, or the batch and its records' provenance.

        Raises:
            StopIteration: After the last batch.
            RuntimeError: If an error stopped a pass's read.
        """
        if self._replay:
            self._read_past(self._replay)
        parts: list[tuple[Batch, Provenance]] = []
        held = 0
        while not self._done and held < self._batch_size:
            chunk = self._pull(self._batch_size - held)
            if chunk is None:
                self._end_pass(parts)
                held = sum(batch.batch_size for batch, _ in parts)
                continue
            size = chunk_size(chunk)
            parts.append((self._named(chunk, size), chunk.provenance))
            held += size
            self._records += size
            self._served = (self._served or 0) + size
        if not parts:
            raise StopIteration
        batch, provenance = _concatenated(parts)
        # T is fixed when the iterator is built: the pair when asked for provenance, else the batch.
        return cast(T, (batch, provenance) if self._with_provenance else batch)

    def _end_pass(self, parts: list[tuple[Batch, Provenance]]) -> None:
        """Meet a pass's end: the stream ends, its tail is dropped, or the next pass starts.

        A pass end that ends the run while a final batch is still to be served leaves the iterator
        in that pass: the state after a batch is just past its last row, as an epoch's end leaves
        an indexed source's cursor in that epoch.
        """
        cursor = self._cursor
        cursor.close()
        empty = self._served == 0
        if self._drop_last:
            parts.clear()
        if empty or (self._end is not None and cursor.pass_index + 1 >= self._end):
            self._done = True
            if parts:
                return
        cursor.pass_index += 1
        self._records, self._served = 0, 0

    def _read_past(self, records: int) -> None:
        """Read ``records`` of the cursor's pass unnamed: a pass resumed midway, read again."""
        started = time.perf_counter()
        left = records
        while left:
            chunk = self._pull(left)
            if chunk is None:
                break
            left -= chunk_size(chunk)
        self._replay = 0
        logger.info(
            "%s resumed mid-pass: replayed %d records of pass %d in %.3f s",
            type(self._source).__name__,
            records - left,
            self._cursor.pass_index,
            time.perf_counter() - started,
        )

    def _pull(self, size: int) -> StreamChunk | None:
        """Up to ``size`` records of the open pass (opening it), ``None`` at its end."""
        cursor = self._cursor
        if cursor.failure is not None:
            raise RuntimeError(
                f"{type(self._source).__name__}'s pass {cursor.pass_index} stopped at an error and "
                "cannot go on; a new iter_batches() starts again"
            ) from cursor.failure
        reader = cursor.reader
        if reader is None:
            reader = self._source._open_pass(  # noqa: SLF001 - the hook every stream implements
                cursor.pass_index, self._key, self._read_size
            )
            cursor.open(reader)
        try:
            return _take(cursor, reader, size)
        except BaseException as error:
            # Any error raised through the reader ends it (a generator raising is finished), so
            # a later next() would read as the pass's end: keep the error and refuse instead.
            cursor.close()
            cursor.ahead, cursor.failure = None, error
            raise

    def _named(self, chunk: StreamChunk, size: int) -> Batch:
        """``chunk``'s ``size`` records as a host ``Batch`` named by the stream: ids or ordinals."""
        cursor = self._cursor
        if not jax.tree.leaves(chunk.columns):
            raise ValueError(
                f"{type(self._source).__name__} read records holding no numeric field, but a "
                "batch is built from numeric fields; strings and objects are each record's "
                "provenance"
            )
        if self._source.record_identity is RecordIdentity.ARRIVAL:
            if cursor.arrived + size > MAX_RECORDS:
                raise ValueError(
                    f"{type(self._source).__name__} would name a record by ordinal "
                    f"{cursor.arrived + size - 1}, past the 2**63 records a stream can name: a "
                    "record's index keeps the top bit clear"
                )
            ordinals = np.arange(cursor.arrived, cursor.arrived + size, dtype=np.uint64)
            cursor.arrived += size
            words = to_words(ordinals)
        else:
            if chunk.ids is None:
                raise TypeError(
                    f"{type(self._source).__name__} is a STREAM_IDS stream, so each record it "
                    "reads carries the id it is named by"
                )
            past = np.asarray(chunk.ids, np.uint64) >= np.uint64(MAX_RECORDS)
            if past.any():
                raise ValueError(
                    f"{type(self._source).__name__} reported the id {int(chunk.ids[past][0])}, "
                    "past the 2**63 records a stream can name: a record's index keeps the top "
                    "bit clear"
                )
            words = to_words(chunk.ids)
        return batch_ops.from_arrays(
            chunk.columns, indices=words, epochs=np.full((size,), cursor.pass_index, np.int32)
        )


def _take(cursor: StreamCursor, reader: Iterator[StreamChunk], size: int) -> StreamChunk | None:
    """Up to ``size`` records of the open pass, keeping what is read past them for later."""
    parts, held = ([cursor.ahead], chunk_size(cursor.ahead)) if cursor.ahead else ([], 0)
    cursor.ahead = None
    while held < size:
        chunk = next(reader, None)
        if chunk is None:
            break
        if chunk_size(chunk):
            parts.append(chunk)
            held += chunk_size(chunk)
    if not parts:
        return None
    joined = _join(parts)
    if held > size:
        joined, cursor.ahead = _split(joined, size)
    return joined


def _concatenated(parts: list[tuple[Batch, Provenance]]) -> tuple[Batch, Provenance]:
    if len(parts) == 1:
        return parts[0]
    batch = batch_ops.concatenate([batch for batch, _ in parts])
    return batch, tuple(record for _, provenance in parts for record in provenance)


__all__ = ["PassReader", "StreamBatches", "StreamChunk", "StreamCursor", "chunk_size"]
