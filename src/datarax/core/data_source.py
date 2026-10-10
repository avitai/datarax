"""Base module for data sources in Datarax.

This module defines the base class for all Datarax data source components
that use flax.nnx.Module for state management and JAX transformation
compatibility.
"""

import abc
import enum
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import jax
import numpy as np
from jax.typing import ArrayLike

from datarax.core.element_batch import Batch, PADDING_INDEX
from datarax.core.index_shuffle import shuffle_positions
from datarax.core.index_words import from_words, low_words, wrapped_positions
from datarax.core.structural import StructuralModule
from datarax.typing import DataDict


def known_length(source: Any) -> int | None:
    """``len(source)``, or ``None`` for a source without a length.

    A source without a length either defines no ``__len__`` or defines one that raises
    ``NotImplementedError`` (a stream whose backend reports none).

    Args:
        source: A data source.

    Returns:
        The number of records, or ``None``.
    """
    try:
        return len(source)
    except (TypeError, NotImplementedError):
        return None


def shard_identity(source: "DataSourceModule") -> dict[str, int] | None:
    """The shard a source serves, as a resume state's fingerprint records it.

    A shard's length counts its own records only, so two shards of one dataset, or shards of two
    datasets differing by fewer records than the shard count, have equal lengths. The identity
    therefore holds the shard's index and count and the records of the whole dataset
    (``index_space()``).

    Args:
        source: The pipeline's source.

    Returns:
        ``{"index", "count", "records"}``, or ``None`` for a source serving all its records.
    """
    shard = source.shard
    if shard is None:
        return None
    index, count = shard
    return {"index": index, "count": count, "records": source.index_space()}


def record_words(indices: ArrayLike) -> np.ndarray:
    """``indices`` as host record words, refusing anything that is not uint32 ``(n, 2)``.

    Args:
        indices: Record indices, each as its words ``(hi, lo)``.

    Returns:
        The words as a host array.

    Raises:
        ValueError: If ``indices`` are not uint32 ``(n, 2)`` words.
    """
    words = np.asarray(indices)
    if words.dtype != np.uint32 or words.ndim != 2 or words.shape[1] != 2:  # noqa: PLR2004
        raise ValueError(
            "record indices are uint32 (n, 2) words (hi, lo), as index_words.to_words names "
            f"positions; got {words.dtype} {words.shape}"
        )
    return words


def refuse_padding(words: np.ndarray) -> None:
    """Refuse the padding index, which names no record.

    Args:
        words: uint32 ``(n, 2)`` record indices.

    Raises:
        IndexError: If an index is the padding index (all ones).
    """
    padding = np.all(words == PADDING_INDEX, axis=1)
    if padding.any():
        raise IndexError(
            f"index {int(np.argmax(padding))} is the padding index (all ones), which names "
            "no record"
        )


def host_rows(words: np.ndarray, length: int) -> np.ndarray:
    """The rows ``words`` name in a source of ``length`` records, refusing words naming none.

    The one row check of the host reads and provenance lookups: the padding index and records
    outside the source are refused.

    Args:
        words: uint32 ``(n, 2)`` record indices.
        length: The source's record count.

    Returns:
        uint32 ``(n,)`` row numbers.

    Raises:
        IndexError: If an index is the padding index or outside the source.
    """
    refuse_padding(words)
    outside = (words[:, 0] != 0) | (words[:, 1] >= length)
    if outside.any():
        raise IndexError(
            f"record index {int(from_words(words[outside][:1])[0])} is outside [0, {length})"
        )
    return low_words(words, length)


NO_PROVENANCE: Mapping[str, Any] = MappingProxyType({})
"""The provenance of a record that carries nothing but arrays."""

type Provenance = tuple[Mapping[str, Any], ...]
"""One immutable mapping per record: its strings and objects, beside the batch."""


class RecordIdentity(enum.Enum):
    """What a source's record index means: the kind of identity its records carry.

    Every source declares one (:attr:`DataSourceModule.record_identity`); the pipeline's host
    stage reads an ``INDEXED`` source by the indices it names and pulls any other forward.
    """

    INDEXED = "indexed"
    """The record's stable position in the source, after the epoch's shuffle: the pipeline turns
    (epoch, position) into it with ``record_indices_at``. Unique, stable across passes and runs,
    and global across workers; its epoch is the pipeline's pass counter."""

    STREAM_IDS = "stream_ids"
    """The id the stream reports for the record (a shard and an offset, in the two words).
    Unique, stable and global; its epoch is the source's pass counter."""

    ARRIVAL = "arrival"
    """The record's arrival ordinal in the run, never reset, so never repeated. Unique only: the
    record's provenance travels beside the batch, and a table keyed by record is refused."""


class IndexedHostRead(Protocol):
    """The stateless host read of an ``INDEXED`` source, which the host stage reads it with.

    ``EagerSource`` (and so ``MemorySource``, ``TFDSEagerSource`` and ``HFEagerSource``),
    ``StreamingDiskSource`` and ``MixDataSourcesNode`` implement it. It is a protocol, not a base
    method: a stream's ``get_batch(batch_size, ...)`` pulls forward, a different contract under
    the same name.
    """

    def get_batch(
        self, indices: ArrayLike, *, epochs: ArrayLike = 0, contiguous: bool = False
    ) -> Batch:
        """Read the records ``indices`` names, as a ``Batch`` named with them, on the host.

        Args:
            indices: uint32 ``(n, 2)`` record indices.
            epochs: The epoch of every record, or of each ``(n,)``.
            contiguous: Whether ``indices`` is a run of consecutive records, read as views.

        Returns:
            The records as a host ``Batch``.
        """
        ...


@runtime_checkable
class IndexedHostReadWithProvenance(Protocol):
    """An indexed host read returning a batch and its records' provenance from one read.

    A source whose provenance would cost a second read and decode of the records implements it
    (``ArrayRecordSourceModule``), so the host stage reads a batch asked for with provenance
    once. Every other indexed source serves the pair by ``get_batch`` and
    :meth:`DataSourceModule.provenance`.
    """

    def read_with_provenance(
        self, indices: ArrayLike, *, epochs: ArrayLike = 0, contiguous: bool = False
    ) -> tuple[Batch, Provenance]:
        """Read the records ``indices`` names, as ``get_batch`` does, with their provenance.

        Args:
            indices: uint32 ``(n, 2)`` record indices.
            epochs: The epoch of every record, or of each ``(n,)``.
            contiguous: Whether ``indices`` is a run of consecutive records, read as views.

        Returns:
            The records as a host ``Batch``, and one mapping of strings and objects per record,
            in row order.
        """
        ...


class BatchSchedule(Protocol):
    """The batches a run serves, unit by unit: what a reader of the run is handed.

    A unit is one batch, or a chunk of consecutive batches read together; its batches are
    ``(start, epoch, size)`` of the pipeline's epoch plan, a batch crossing an epoch's end
    continuing at the next epoch's head. Units are numbered from the run's start.
    """

    def unit(self, ordinal: int) -> tuple[tuple[int, int, int], ...] | None:
        """The batches of unit ``ordinal``, or ``None`` past the run's end."""
        ...


class DataSourceModule(StructuralModule):
    """Enhanced base module for all Datarax data source components.

    This class extends StructuralModule for non-parametric structural data loading.
    Concrete data sources define their own config classes extending StructuralConfig.

    A DataSourceModule is responsible for reading data from an external source
    (e.g., files, memory, network) and yielding data elements as PyTrees. Each
    data element is typically a dictionary or other PyTree structure containing
    JAX arrays or Python primitives.

    **Important**: When subclassing, if you store data containing JAX Arrays in
    an attribute (like `self.data`), wrap the assigned value with `nnx.Param`
    or assignment-time `nnx.data(value)`:

    Examples:
        ```python
        @dataclass(frozen=True)
        class MyDataSourceConfig(StructuralConfig):
            required_param: int | None = None
            def __post_init__(self):
                super().__post_init__()
                if self.required_param is None:
                    raise ValueError("required_param is required")
        class MyDataSource(DataSourceModule):
            data: list[dict]
            def __init__(self, config: MyDataSourceConfig, data: list[dict], *,
                         rngs: nnx.Rngs | None = None, name: str | None = None):
                super().__init__(config, rngs=rngs, name=name)
                self.data = nnx.data(data)  # Mark as pytree data, not parameters.
        ```

    This prevents NNX from trying to track individual JAX Arrays within the data
    structure as trainable parameters.

    Every source declares what its record index means, :attr:`record_identity`; a subclass that
    does not is refused at construction.
    """

    @property
    @abc.abstractmethod
    def record_identity(self) -> RecordIdentity:
        """What this source's record index means (see :class:`RecordIdentity`).

        ``INDEXED`` sources implement ``record_indices_at``, the traced ``get_records`` that
        ``step()`` calls, and the host read ``get_batch(indices, *, epochs, contiguous)`` that the
        pipeline's host stage calls; ``STREAM_IDS`` and ``ARRIVAL`` sources pull
        forward with ``get_batch(batch_size)``, which the host stage calls in order. A subclass
        declares it with a property returning its kind.
        """

    @property
    def shard(self) -> tuple[int, int] | None:
        """``(index, count)`` when the source serves one shard of its records, else ``None``.

        Shard ``index`` of ``count`` serves its own part of the positions of the whole order, so
        a pipeline's resume state names the shard it was saved on (:func:`shard_identity`).
        """
        return None

    def provenance(  # noqa: DOC502 - the checks it calls raise
        self, indices: ArrayLike
    ) -> tuple[Mapping[str, Any], ...]:
        """The provenance of the records ``indices`` names: their strings and objects, by index.

        A source naming records stably (``INDEXED``, ``STREAM_IDS``) serves each record's
        non-array part by its index, as an immutable mapping, in the order named. This base
        serves a source that holds none: one empty mapping per record, after refusing the padding
        index and, for a source with a length, indices outside it. A source holding provenance
        overrides it. A source naming records by arrival refuses it (see :meth:`record_keys`).

        Args:
            indices: uint32 ``(n, 2)`` record indices, as a batch's ``indices`` holds them.

        Returns:
            One mapping per index.

        Raises:
            TypeError: If the source names its records by arrival.
            ValueError: If ``indices`` are not uint32 ``(n, 2)`` words.
            IndexError: If an index is the padding index or outside the source.
        """
        self._refuse_arrival("look a record's provenance up by its index")
        words = record_words(indices)
        length = known_length(self)
        if length is None:
            refuse_padding(words)
        else:
            host_rows(words, length)
        return (NO_PROVENANCE,) * len(words)

    def record_keys(self, batch: Batch) -> Any:  # noqa: DOC502 - _refuse_arrival raises
        """The keys a per-record table indexes this source's records by: the batch's indices.

        A table keyed by record (D7) needs identities that are stable and unique across passes,
        which a source naming records by position (``INDEXED``) or by the ids its stream reports
        (``STREAM_IDS``) gives. Only the static ``record_identity`` is read, so under ``jit`` the
        batch's indices pass through and the call adds nothing to the program. A ``Batch`` has no
        source and so no record keys: key a table with this method on a batch this source's
        pipeline served.

        Args:
            batch: A batch this source's records were served in.

        Returns:
            ``batch.indices``, uint32 ``(B, 2)``.

        Raises:
            TypeError: If the source names its records by arrival.
        """
        self._refuse_arrival("key a per-record table")
        return batch.indices

    def _refuse_arrival(self, what: str) -> None:
        """Refuse a lookup by record on a source naming its records by arrival.

        Args:
            what: What the caller asked for.

        Raises:
            TypeError: If the source's records are named by arrival.
        """
        kind = self.record_identity
        if kind is RecordIdentity.ARRIVAL:
            raise TypeError(
                f"{type(self).__name__} names its records by arrival ({kind.name}), so it cannot "
                f"{what}: an arrival ordinal names no record it can find again. A stream naming "
                "records by arrival hands each record's provenance out beside each batch it serves."
            )

    def index_space(self) -> int:
        """How many record indices the source names; its indices run from 0 to one below it.

        For most sources that is ``len(self)``, the positions one epoch serves. A source whose
        epoch serves a part of the indices it names says so: a worker's shard of in-memory data
        names positions of the whole data, and a mix names every record of its children while an
        epoch serves Grain's length of them.

        Returns:
            The size of the source's index space.

        Raises:
            TypeError: If the source has no length.
        """
        length = known_length(self)
        if length is None:
            raise TypeError(f"{type(self).__name__} has no length, so it names no index space")
        return length

    def get_records(self, indices: jax.Array) -> DataDict:
        """Gather the records at ``indices``: indexed access for ``Pipeline``-driven iteration.

        ``indices`` are the stable record indices :meth:`record_indices_at` names. The pipeline
        computes a batch's indices once and passes the same ones here and to the stages that key
        randomness on them. Implementations must be stateless (no mutation of internal counters)
        and JAX-traceable (``indices`` may be a traced array) so the call composes with
        ``nnx.jit`` and ``nnx.scan``.

        Args:
            indices: uint32 ``(n, 2)`` record indices in ``[0, index_space())``, each as its words
                ``(hi, lo)``.

        Returns:
            One array per field, with leading dim ``n``.

        Raises:
            NotImplementedError: If the source does not implement it. An ``INDEXED`` source
                without it is read on the host only (``for batch in pipe`` and
                ``Pipeline.raw_batches()``); a stream (``STREAM_IDS`` or ``ARRIVAL``) is pulled
                with ``get_batch``.
        """
        del indices
        if self.record_identity is RecordIdentity.INDEXED:
            raise NotImplementedError(
                f"{type(self).__name__} does not implement get_records(indices), the traced read "
                "step(), scan() and session() gather an INDEXED source's records with; iterate "
                "the pipeline with `for batch in pipe` or pipe.raw_batches(), which read it on "
                "the host"
            )
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_records(indices), which an INDEXED "
            "source serves its records with; a stream declares STREAM_IDS or ARRIVAL and "
            "implements get_batch(batch_size)."
        )

    def record_indices_at(
        self,
        start: int | ArrayLike,
        size: int,
        key: Any | None = None,
    ) -> Any:
        """Return the stable index of each record at positions ``start .. start + size``.

        These are the indices :meth:`get_records` gathers. Stochastic operators key each
        record's randomness on them, so within an epoch a record keeps its augmentation however
        records are batched, ordered or split across workers. The order is the one ``key``
        selects, or the sequential order when ``key`` is ``None``: the pipeline owns the shuffle
        and passes its epoch key exactly when it was built with ``shuffle=True``. The default
        names records by their wrapped position ``(start + arange(size)) % len(self)``, and with
        a key by the keyed permutation of those positions
        (:func:`~datarax.core.index_shuffle.shuffle_positions`). A source that partitions
        or mixes records overrides it. Indices are 64-bit, each a uint32 ``(hi, lo)`` pair, the
        layout of ``Batch.indices``.

        Args:
            start: Starting position: a Python int of any size, its two uint32 words ``(hi, lo)``
                (NumPy or traced; a position of the order, below its length), or a traced
                int32 ``jax.Array``.
            size: Number of records (Python int).
            key: The key selecting the order, or ``None`` for the sequential order.

        Returns:
            uint32 array of shape ``(size, 2)``.

        Raises:
            ValueError: If a key is given to a source without a length, which has no order to
                shuffle.
        """
        length = known_length(self)
        if key is None:
            return wrapped_positions(start, size, length)
        if length is None:
            raise ValueError(
                f"{type(self).__name__} has no length, so it has no order to shuffle; build its "
                "pipeline with shuffle=False"
            )
        return shuffle_positions(wrapped_positions(start, size, length), length, key)

    def element_spec(self) -> Any:
        """Return a PyTree of ``jax.ShapeDtypeStruct`` describing per-element output.

        Downstream consumers (operators, batchers, models) use this contract to
        pre-allocate buffers, auto-size learnable layers, and statically validate
        operator chains. Subclasses MUST override this method.

        Returns:
            A PyTree (typically a dict) whose leaves are ``jax.ShapeDtypeStruct``
            instances describing one emitted element.

        Raises:
            NotImplementedError: Always, on the base class. Subclasses must
                override.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement element_spec(). "
            "Return a PyTree of jax.ShapeDtypeStruct describing one emitted element."
        )
