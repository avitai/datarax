"""Shared source pieces: the mixin of sources reading a named dataset, and the base of streams.

A stream reads its records forward, one pass after another, in an order it applies itself. The
:class:`StreamingSourceBase` owns everything but the reading: a subclass opens a pass
(``_open_pass``), a generator of :class:`StreamChunk` (host columns, provenance and record ids),
and the base turns pulls into host ``Batch``es named by the stream (``get_batch``). Where the stream
is, which pass and how far into it, is kept on the host in a holder NNX leaves out of module state,
so no Python value sits in a Variable and the module's graph definition never changes as the stream
advances. ``clone()`` gives a stream a position of its own; ``nnx.split``/``nnx.merge`` keep one
stream, the merged module reading on from the same place.
"""

from __future__ import annotations

import logging
import weakref
from collections.abc import Generator, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, cast, Literal, NamedTuple, overload, Self

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PyTree

from datarax.core import batch_ops
from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule, RecordIdentity
from datarax.core.element_batch import Batch
from datarax.core.index_words import to_words
from datarax.core.spec import array_to_spec_strip_leading, device_spec
from datarax.sources.eager_source import HostValue
from datarax.sources.source_ops import format_source_repr


logger = logging.getLogger(__name__)

type Provenance = tuple[Mapping[str, Any], ...]
"""One immutable mapping per record: its strings and objects, beside the batch."""


class DatasetSourceMixin:
    """What a source reading a named dataset reports: its name, split and info.

    A subclass sets ``dataset_name``, ``split_name`` and ``_dataset_info`` in its ``__init__``.
    The info is held in a :class:`~datarax.sources.eager_source.HostValue`: a backend's info
    object compares by identity (TFDS's ``DatasetInfo``), so held as an attribute it would make
    every source a graphdef of its own and every jitted step over a new source compile again.
    """

    dataset_name: str | None
    split_name: str | None
    _dataset_info: HostValue
    length: int

    def get_dataset_info(self) -> Any:
        """Return cached backend-specific dataset metadata."""
        return self._dataset_info.value

    def _repr_extra_fields(self) -> dict[str, Any]:
        """Optional additional repr fields for subclasses."""
        return {}

    def __repr__(self) -> str:
        """String representation."""
        return format_source_repr(
            type(self).__name__,
            self.dataset_name,
            self.split_name,
            self.length,
            self._repr_extra_fields(),
        )


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
    """Closes a stream's open pass reader, holding it only weakly.

    ``weakref.finalize`` keeps this object, never the reader, so it keeps no stream alive. It runs
    when the stream's position is collected or, for a stream still alive, at interpreter exit,
    before module teardown: a backend reader left suspended until teardown is finalized with its
    modules half cleared, where HuggingFace's Parquet reader hangs.
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
class _Cursor:
    """Where a stream is: its pass, the open pass's reader, records read ahead, arrivals.

    ``failure`` is the error that stopped the open pass's reader, kept until :meth:`reset`. The
    cursor's finalizer closes a reader still open when the cursor is collected or at exit.
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

    def between_passes(self) -> _Cursor:
        """A cursor of its own at this one's place, which is between passes (no reader open)."""
        return _Cursor(pass_index=self.pass_index, arrived=self.arrived, failure=self.failure)

    def open(self, reader: Iterator[StreamChunk]) -> None:
        """Hold ``reader`` as the open pass's reader, closing any reader held before."""
        self.close()
        self.reader = reader
        self.closer.reader = weakref.ref(reader)

    def close(self) -> None:
        """Close and drop the open pass's reader, if any."""
        self.closer()
        self.reader = None


class StreamPosition(HostValue):
    """A stream's position, held on the host and out of NNX state (see :class:`HostValue`)."""

    __slots__ = ()
    value: _Cursor


def empty_stream_batch() -> Batch:
    """The batch a stream returns at the end of a pass: no rows."""
    return Batch(
        {},
        states={},
        indices=np.zeros((0, 2), np.uint32),
        epochs=np.zeros((0,), np.int32),
        draws=np.zeros((0,), np.int32),
        batch_state={},
    )


class StreamingSourceBase(DataSourceModule):
    """The base of every stream: a source read forward, one pass after another.

    A subclass declares its kind (``STREAM_IDS`` or ``ARRIVAL``) and implements
    :meth:`_open_pass`, which reads one pass in its order. The base serves it:

    - :meth:`get_batch` returns up to ``batch_size`` records of the current pass as a host
      ``Batch``: NumPy columns, ``indices`` the stream's ids as two words (``STREAM_IDS``) or the
      records' arrival ordinals, never reset (``ARRIVAL``), ``epochs`` the pass from 0, draws 0.
      An empty ``Batch`` ends a pass; the next call starts the next one. Strings and objects
      never enter a batch: ``with_provenance=True`` returns them beside it.
    - The order is the stream's to apply and the pipeline's to choose: the pipeline passes its key
      when it shuffles and ``None`` otherwise, and pass ``p`` is ordered by that key and ``p``.
    - :meth:`element_spec` is the array part of the first record, as the device holds it.

    Where the stream is lives in a host holder outside NNX state, so a stream holds no Python
    value in a Variable and its graph definition does not change as it advances.
    """

    def __init__(self, config: StructuralConfig, *, name: str | None = None) -> None:
        """Create a stream at the start of its first pass.

        Args:
            config: The source's configuration.
            name: Optional module name.
        """
        super().__init__(config, name=name)
        self._position = StreamPosition(_Cursor())

    def _open_pass(
        self, pass_index: int, key: jax.Array | None, read_size: int
    ) -> Iterator[StreamChunk]:
        """Read pass ``pass_index`` in its order, as chunks of records: a subclass's generator.

        Args:
            pass_index: The pass, from 0.
            key: The pipeline's key when it shuffles, ``None`` for the stream's own order.
            read_size: Records the pass reads at a time (a backend's batched read or decode);
                chunks may hold any number.

        Returns:
            The pass's records, in the order served.

        Raises:
            NotImplementedError: Always, on the base class.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _open_pass")

    @property
    def pass_index(self) -> int:
        """The pass the stream is in, from 0."""
        return self._position.value.pass_index

    def reset(self) -> None:
        """Start again at the first pass, closing the reader of a pass stopped midway.

        Arrival ordinals keep counting, so none repeats.
        """
        cursor = self._position.value
        cursor.close()
        cursor.pass_index, cursor.ahead, cursor.failure = 0, None, None

    def clone(self) -> Self:
        """A new stream with this one's state and a position of its own.

        The clone is where this stream is: in the same pass, its arrival ordinals counting on
        from the same number. A pass's open reader cannot be copied, so a stream is cloned
        between passes: before its first pull, after a pass's end, or after :meth:`reset`.
        (``nnx.split`` then ``nnx.merge``, and ``nnx.clone`` of a module holding the stream,
        carry its position by reference: they make one stream, not two.)

        Returns:
            The clone.

        Raises:
            ValueError: If the stream is in the middle of a pass.
        """
        cursor = self._position.value
        if cursor.reader is not None:
            raise ValueError(
                f"{type(self).__name__} is in the middle of pass {cursor.pass_index}, whose "
                "reader cannot be copied; clone a stream between passes (before its first pull, "
                "after a pass's end, or after reset())"
            )
        twin = cast(Self, super().clone())
        twin._position = StreamPosition(cursor.between_passes())  # noqa: SLF001 - its own class
        return twin

    @overload
    def get_batch(
        self,
        batch_size: int,
        *,
        key: jax.Array | None = None,
        with_provenance: Literal[False] = False,
        read_size: int | None = None,
    ) -> Batch: ...

    @overload
    def get_batch(
        self,
        batch_size: int,
        *,
        key: jax.Array | None = None,
        with_provenance: Literal[True],
        read_size: int | None = None,
    ) -> tuple[Batch, Provenance]: ...

    def get_batch(  # noqa: DOC503 - a reader's error is re-raised as it came
        self,
        batch_size: int,
        *,
        key: jax.Array | None = None,
        with_provenance: bool = False,
        read_size: int | None = None,
    ) -> Batch | tuple[Batch, Provenance]:
        """Read up to ``batch_size`` records of the current pass as a host ``Batch``.

        An error raised while the pass is read reaches this call; the pass cannot go on, and
        every later call is refused, naming the error, until :meth:`reset`.

        Args:
            batch_size: The most records to return.
            key: The key the pass's order is drawn from (read when a pass starts), or ``None``
                for the stream's own order.
            with_provenance: Whether to return the records' provenance beside the batch.
            read_size: Records a pass this call opens reads at a time, the pipeline's batch
                size; ``None`` reads it ``batch_size`` at a time.

        Returns:
            The batch, empty at the end of a pass, and with ``with_provenance`` its records'
            provenance, one mapping per row.

        Raises:
            ValueError: If ``batch_size`` or ``read_size`` is not positive.
            RuntimeError: If an error stopped the current pass's read.
        """
        read_size = batch_size if read_size is None else read_size
        if min(batch_size, read_size) < 1:
            raise ValueError(
                f"batch_size and read_size must be at least 1; got {batch_size} and {read_size}"
            )
        cursor = self._position.value
        if cursor.failure is not None:
            raise RuntimeError(
                f"{type(self).__name__}'s pass {cursor.pass_index} stopped at an error and "
                "cannot go on; reset() starts the first pass again"
            ) from cursor.failure
        reader = cursor.reader
        if reader is None:
            reader = self._open_pass(cursor.pass_index, key, read_size)
            cursor.open(reader)
        try:
            chunk = self._pull(cursor, reader, batch_size)
        except BaseException as error:
            # Any error raised through the reader ends it (a generator raising is finished), so
            # a later next() would read as the pass's end: keep the error and refuse instead.
            cursor.close()
            cursor.ahead, cursor.failure = None, error
            raise
        if chunk is None:
            cursor.close()
            cursor.pass_index += 1
            batch: Batch = empty_stream_batch()
            return (batch, ()) if with_provenance else batch
        batch = self._named(chunk, cursor)
        return (batch, chunk.provenance) if with_provenance else batch

    @staticmethod
    def _pull(cursor: _Cursor, reader: Iterator[StreamChunk], size: int) -> StreamChunk | None:
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

    def _named(self, chunk: StreamChunk, cursor: _Cursor) -> Batch:
        """``chunk`` as a host ``Batch`` named by the stream: its ids or arrival ordinals."""
        size = chunk_size(chunk)
        if not jax.tree.leaves(chunk.columns):
            raise ValueError(
                f"{type(self).__name__} read records holding no numeric field, but a batch is "
                "built from numeric fields; strings and objects are each record's provenance"
            )
        if self.record_identity is RecordIdentity.ARRIVAL:
            ordinals = np.arange(cursor.arrived, cursor.arrived + size, dtype=np.uint64)
            cursor.arrived += size
            words = to_words(ordinals)
        else:
            if chunk.ids is None:
                raise TypeError(
                    f"{type(self).__name__} is a STREAM_IDS stream, so each record it reads "
                    "carries the id it is named by"
                )
            words = to_words(chunk.ids)
        return batch_ops.from_arrays(
            chunk.columns, indices=words, epochs=np.full((size,), cursor.pass_index, np.int32)
        )

    def element_spec(self) -> Any:
        """The spec of one record's array part as the device holds it, from the first record.

        A fresh first pass in the stream's own order is opened and its first record read; the
        stream's position is not touched.

        Returns:
            A pytree of ``jax.ShapeDtypeStruct``.

        Raises:
            ValueError: If the stream holds no records.
        """
        for chunk in self._open_pass(0, None, 1):
            if chunk_size(chunk):
                return device_spec(jax.tree.map(array_to_spec_strip_leading, chunk.columns))
        raise ValueError(f"{type(self).__name__} holds no records, so it declares no spec")


def pass_generator(key: jax.Array, pass_index: int) -> np.random.Generator:
    """The generator a stream draws pass ``pass_index``'s order from: keyed by the pass's key.

    The pass's key is ``fold_in(key, pass_index)``, the key an indexed pipeline orders its epoch
    ``pass_index`` by, so passes never share a key (Grain's ``seed + epoch`` collision cannot
    arise). Its key data, read on the host once per pass, keys NumPy's counter-based Philox
    generator.

    Args:
        key: The pipeline's key, typed or as raw key data.
        pass_index: The pass, from 0.

    Returns:
        The pass's generator.
    """
    return np.random.Generator(np.random.Philox(key=pass_seed(key, pass_index)))


def pass_seed(key: jax.Array, pass_index: int) -> int:
    """Pass ``pass_index``'s key, ``fold_in(key, pass_index)``, as one integer, read on the host.

    Args:
        key: The pipeline's key, typed or as raw key data.
        pass_index: The pass, from 0.

    Returns:
        The pass key's data as a non-negative integer.
    """
    return key_integer(jax.random.fold_in(typed_key(key), pass_index))


def typed_key(key: jax.Array) -> jax.Array:
    """``key`` as a typed key: raw key data is wrapped, a typed key returned as it is."""
    if jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
        return key
    return jax.random.wrap_key_data(key)


def key_integer(key: jax.Array) -> int:
    """All of ``key``'s data as one non-negative integer: a seed NumPy and HF generators take."""
    words = np.asarray(jax.random.key_data(typed_key(key)), np.uint32)
    return int.from_bytes(words.tobytes(), "little")
