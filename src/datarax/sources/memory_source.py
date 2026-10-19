"""In-memory data source: a dict of arrays, a list of records or an array, served from the host.

``MemorySource`` turns the data it is given into host NumPy columns and per-record provenance
once, at construction (see :mod:`datarax.sources.eager_source`), and serves it through the
eager-source base. A dict is a mapping of columns; a list is a list of records, whose numeric
values are stacked into columns and whose strings and other objects become provenance; an array
is one column. :meth:`MemorySource.of_columns` holds columns already decoded (a build's), nested
by field and with their provenance, as they are.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

import jax
import numpy as np
from jax.typing import ArrayLike
from jaxtyping import PyTree

from datarax.config.registry import register_component
from datarax.core.block_order import BlockOrder
from datarax.core.config import StructuralConfig
from datarax.core.element_batch import Element
from datarax.core.ragged import is_ragged, ragged_column, RaggedColumn
from datarax.sources.eager_source import (
    EagerSource,
    is_array_leaf,
    ragged_parts_of_records,
    stack_records,
)
from datarax.sources.source_ops import partition_length, resolve_wrapped_indices


@dataclass(frozen=True)
class MemorySourceConfig(StructuralConfig):
    """Configuration for MemorySource (in-memory data source).

    The order records are served in belongs to the pipeline (``Pipeline(shuffle=...)``), not
    to the source.

    Args:
        prefetch_size: Number of items to prefetch (0 = no prefetching)
        shard_id: Optional shard identifier for distributed processing
        num_workers: Number of parallel workers (default 1). When > 1,
            each worker (identified by shard_id) receives a disjoint
            partition of the global order. Worker k gets elements at
            global positions [k::num_workers].
    """

    # Optional parameters with defaults
    prefetch_size: int = 0
    shard_id: int | None = None
    num_workers: int = 1

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {self.num_workers}")
        if self.num_workers > 1 and self.shard_id is None:
            raise ValueError("shard_id is required when num_workers > 1")
        if self.num_workers > 1 and self.shard_id is not None and self.shard_id >= self.num_workers:
            raise ValueError(
                f"shard_id ({self.shard_id}) must be < num_workers ({self.num_workers})"
            )

        # Call parent validation
        super().__post_init__()


type _Parts = tuple[PyTree, tuple[dict[str, Any], ...], dict[str, RaggedColumn]]


@dataclass(frozen=True, slots=True)
class _HeldColumns:
    """Columns :meth:`MemorySource.of_columns` holds as given, and their provenance."""

    columns: PyTree
    provenance: tuple[Mapping[str, Any], ...]


@register_component("source", "MemorySource")
class MemorySource(EagerSource):
    """In-memory data source over a dict of arrays, a list of records or an array.

    The data becomes host NumPy columns once, at construction; device arrays are copied to the
    host. Strings, bytes and other objects a record carries are kept as its provenance, never
    served in a batch. Records are read by index (``source[i]``, iteration in order, Grain's
    batched access) and by the stateless host read ``get_batch(indices, epochs=...)``, which
    returns a ``Batch``. The order a pipeline serves them in is the pipeline's.

    Examples:
        ```python
        import numpy as np
        from datarax.core.index_words import to_words

        data = {"x": np.arange(100.0), "label": np.arange(100) % 10}
        source = MemorySource(MemorySourceConfig(), data)

        for record in source:  # records in order, each a dict of NumPy rows
            process(record)

        batch = source.get_batch(to_words(np.arange(32)))  # records 0..31 as a Batch

        pipeline = Pipeline(source=source, stages=[], batch_size=32, rngs=nnx.Rngs(0),
                            shuffle=True)
        ```
    """

    # Narrow config type for pyright (base stores via nnx.static)
    config: MemorySourceConfig  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(  # noqa: DOC503 - _parts raises the ValueError
        self,
        config: MemorySourceConfig,
        data: Mapping[str, Any] | Sequence[Any] | np.ndarray | jax.Array | _HeldColumns,
        *,
        name: str | None = None,
    ) -> None:
        """Build the source's columns and provenance from ``data``.

        Args:
            config: Configuration for the MemorySource
            data: A mapping of columns (all of one length), a sequence of records (each a
                mapping of values, an ``Element`` or a single value), or an array whose leading
                axis is the records (or the columns :meth:`of_columns` holds).
            name: Optional name for the module (defaults to "MemorySource")

        Raises:
            TypeError: If ``data`` is a string, or a column is a mapping.
            ValueError: If the columns disagree on their length or hold no array, a record's
                numeric fields or shapes differ from the others', or an ``Element`` carries an
                index or state.
        """
        super().__init__(config, name="MemorySource" if name is None else name)
        if isinstance(data, str | bytes):
            raise TypeError(
                f"MemorySource expects a list, sequence, or dictionary, got string: {type(data)}"
            )
        self.prefetch_size = config.prefetch_size
        if isinstance(data, _HeldColumns):
            self._store(data.columns, data.provenance)
            return
        self._store(*_parts(data))

    @classmethod
    def of_columns(  # noqa: DOC502 - _store raises
        cls,
        columns: PyTree,
        provenance: Sequence[Mapping[str, Any]] = (),
        *,
        name: str | None = None,
    ) -> Self:
        """A source over host columns already decoded, held as given, with their provenance.

        A build's columns (``datarax.pipeline.materialize``'s RAM result) are nested by field,
        which the constructor's mapping of columns refuses, and come with one provenance mapping
        per record. They are held as every eager source holds its columns (a host holder, out of
        NNX state; see :mod:`datarax.sources.eager_source`), not copied. Record ``r`` is row
        ``r``.

        Args:
            columns: A pytree of host arrays (nested mappings by field), one row per record.
            provenance: One mapping of strings and objects per record, or nothing.
            name: Optional name for the module (defaults to "MemorySource").

        Returns:
            The source.

        Raises:
            ValueError: If the columns disagree on their record count, or the provenance is not
                one mapping per record.
        """
        return cls(MemorySourceConfig(), _HeldColumns(columns, tuple(provenance)), name=name)

    def __len__(self) -> int:
        """Return the number of records this source serves.

        With ``num_workers > 1`` that is this worker's partition, positions
        ``[shard_id::num_workers]`` of the global order.

        Returns:
            Number of records served.
        """
        return partition_length(self.length, self.config.num_workers, self.config.shard_id or 0)

    @property
    def shard(self) -> tuple[int, int] | None:
        """``(shard_id, num_workers)`` when the records are split between workers, else ``None``."""
        if self.config.num_workers == 1:
            return None
        return (self.config.shard_id or 0, self.config.num_workers)

    def __iter__(self) -> Iterator[PyTree]:
        """Iterate over this worker's records in order, each its array part; stateless.

        With ``num_workers > 1`` that is global positions ``[shard_id::num_workers]``. When
        ``prefetch_size > 0`` the records are read ahead on a thread.

        Returns:
            Iterator over the records.
        """
        rows = range(self.config.shard_id or 0, self.length, self.config.num_workers)
        records = (self[row] for row in rows)
        if self.prefetch_size > 0:
            from datarax.control.prefetcher import Prefetcher  # noqa: PLC0415

            return Prefetcher(buffer_size=self.prefetch_size).prefetch(records)
        return records

    def record_indices_at(
        self,
        start: int | ArrayLike,
        size: int,
        key: jax.Array | None = None,
        *,
        block: BlockOrder | None = None,
    ) -> jax.Array:
        """Return the global index of each record at positions ``start .. start + size``.

        ``start`` and ``size`` are positions in the order ``key`` and ``block`` select (the
        sequential order when both are ``None``): the whole dataset, or with
        ``num_workers > 1`` this worker's positions ``[shard_id::num_workers]`` of the order. The
        indices are global, so a record has one index on every worker.

        Args:
            start: Starting logical position: a Python int, its two uint32 words ``(hi, lo)``,
                or a traced int32 ``jax.Array`` (see
                :func:`~datarax.core.index_words.wrapped_positions`).
            size: Number of records (Python int).
            key: The key selecting the order, or ``None`` for the sequential order.
            block: The block order, or ``None`` for the global order a key selects.

        Returns:
            uint32 ``jax.Array`` of shape ``(size, 2)``, each index as its words ``(hi, lo)``.
        """
        return resolve_wrapped_indices(
            start,
            size,
            self.length,
            key,
            block=block,
            num_workers=self.config.num_workers,
            shard_id=self.config.shard_id or 0,
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"MemorySource(length={self.length})"


def _parts(data: Mapping[str, Any] | Sequence[Any] | np.ndarray | jax.Array) -> _Parts:
    """The columns, provenance and ragged fields ``data`` holds.

    Args:
        data: A mapping of columns, an array, or a sequence of records.

    Returns:
        The host columns, one provenance mapping per record (or none), and the ragged fields.
    """
    if isinstance(data, Mapping):
        return _parts_of_columns(data)
    if isinstance(data, np.ndarray | jax.Array):
        return np.asarray(data), (), {}
    return _parts_of_records(data)


def _parts_of_columns(data: Mapping[str, Any]) -> _Parts:
    """A mapping of columns: numeric columns stay columns, the others become provenance.

    A value without rows (a scalar) is every record's; a string is a value, not a column. A
    column of per-record arrays differing along their first axis only is a ragged field.

    Args:
        data: The mapping of columns.

    Returns:
        The host columns, one provenance mapping per record (or none), and the ragged fields.
    """
    length = _record_count(data)
    columns: dict[str, Any] = {}
    ragged: dict[str, RaggedColumn] = {}
    provenance_columns: dict[str, Sequence[Any]] = {}
    for key, value in data.items():
        column = _column(key, value, length)
        if column is None:
            provenance_columns[key] = list(value) if _has_rows(value) else [value] * length
        elif isinstance(column, RaggedColumn):
            ragged[key] = column
        else:
            columns[key] = column
    provenance = (
        tuple({key: rows[k] for key, rows in provenance_columns.items()} for k in range(length))
        if provenance_columns
        else ()
    )
    return columns, provenance, ragged


def _record_count(data: Mapping[str, Any]) -> int:
    """The number of records a mapping of columns holds: the length its columns share.

    Args:
        data: The mapping of columns.

    Returns:
        The record count.

    Raises:
        TypeError: If a column is itself a mapping, whose key count is not a record count.
        ValueError: If the columns disagree on their length or no value has rows.
    """
    for key, value in data.items():
        if isinstance(value, Mapping):
            raise TypeError(
                f"Data column {key!r} is a mapping; columns are arrays with one row per record, "
                "in a flat mapping. Give each nested field its own top-level key."
            )
    lengths = {key: len(value) for key, value in data.items() if _has_rows(value)}
    if not lengths:
        raise ValueError("Data dictionary must contain at least one array-like value")
    if len(set(lengths.values())) > 1:
        raise ValueError(
            f"All arrays in data dictionary must have the same length. Got lengths: {lengths}"
        )
    return next(iter(lengths.values()))


def _column(key: str, value: Any, length: int) -> np.ndarray | RaggedColumn | None:  # noqa: DOC502
    """``value`` as a host column of ``length`` rows, a ragged field, or ``None`` for provenance.

    A column of per-record arrays (a list, a tuple, or an object array holding them) is stacked
    when the arrays share one shape, and is a ragged field when they differ along their first
    axis only; arrays differing past it are refused by :func:`stack_records`. A column whose rows
    are not all numeric (text, other objects) is provenance.

    Args:
        key: The column's name.
        value: The column, or a single value every record shares.
        length: The record count.

    Returns:
        The host column, the ragged field, or ``None`` for a non-numeric value.

    Raises:
        ValueError: If the column's arrays differ in more than their first axis.
    """
    if not _has_rows(value):
        if not is_array_leaf(value):
            return None
        scalar = np.asarray(value)
        return np.broadcast_to(scalar, (length, *scalar.shape))
    if isinstance(value, np.ndarray | jax.Array) and is_array_leaf(value):
        return np.asarray(value)
    rows = _numeric_rows(value)
    return None if rows is None else _rows_column(key, rows)


def _numeric_rows(value: Any) -> list[np.ndarray] | None:
    """The per-record arrays of a column of rows, or ``None`` when a row is not numeric.

    Args:
        value: A list, a tuple or an array of rows; a non-object array of a non-numeric dtype
            (text) is provenance.

    Returns:
        The rows as host arrays, or ``None`` for provenance.
    """
    if isinstance(value, np.ndarray | jax.Array) and np.asarray(value).dtype != object:
        return None
    if not all(is_array_leaf(row) for row in value):
        return None
    return [np.asarray(row) for row in value]


def _rows_column(  # noqa: DOC502 - stack_records raises
    key: str, rows: list[np.ndarray]
) -> np.ndarray | RaggedColumn:
    """Per-record arrays as one column when their shapes agree, or a ragged field.

    Args:
        key: The column's name.
        rows: One array per record.

    Returns:
        The stacked column, or the ragged field when the arrays differ along their first axis
        only.

    Raises:
        ValueError: If the arrays differ in more than their first axis.
    """
    if is_ragged([row.shape for row in rows]):
        return ragged_column(rows)
    return stack_records([{key: row} for row in rows])[key]


def _has_rows(value: Any) -> bool:
    """Whether ``value`` is a column (one row per record) rather than a single value.

    A list or tuple always is, whatever its rows hold; an array is when it has an axis.

    Args:
        value: The value.

    Returns:
        Whether it has rows.
    """
    if isinstance(value, list | tuple):
        return True
    return hasattr(value, "__len__") and not isinstance(value, str | bytes) and np.ndim(value) > 0


def _parts_of_records(records: Sequence[Any]) -> _Parts:  # noqa: DOC502 - ragged_parts_of_records
    """A sequence of records: numeric values stacked into columns, the rest provenance.

    A field whose records differ along their first axis only is a ragged field.

    Args:
        records: The records.

    Returns:
        The host columns, one provenance mapping per record (or none), and the ragged fields.

    Raises:
        ValueError: If an ``Element`` carries an index or state, no record holds a number, the
            records' numeric fields differ, or a field differs past its first axis (see
            :func:`~datarax.sources.eager_source.ragged_parts_of_records`).
    """
    return ragged_parts_of_records(
        [_record_data(record, position) for position, record in enumerate(records)]
    )


def _record_data(record: Any, position: int) -> Any:
    """A record's values: an ``Element`` contributes its ``data``, the identity is the source's.

    Args:
        record: The record.
        position: Its position, for the refusal.

    Returns:
        The record's values.

    Raises:
        ValueError: If an ``Element`` carries an index or state, which the source would drop.
    """
    if not isinstance(record, Element):
        return record
    if record.index is not None:
        raise ValueError(
            f"record {position} is an Element with an index; the source names its records by "
            "position, so an Element given to it carries data only"
        )
    if jax.tree.leaves(record.state):
        raise ValueError(
            f"record {position} is an Element with state; the source serves records' data, so "
            "an Element given to it carries data only"
        )
    return record.data
