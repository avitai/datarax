"""In-memory data source: a dict of arrays, a list of records or an array, served from the host.

``MemorySource`` turns the data it is given into host NumPy columns and per-record provenance
once, at construction (see :mod:`datarax.sources.eager_source`), and serves it through the
eager-source base. A dict is a mapping of columns; a list is a list of records, whose numeric
values are stacked into columns and whose strings and other objects become provenance; an array
is one column.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
from jax.typing import ArrayLike
from jaxtyping import PyTree

from datarax.config.registry import register_component
from datarax.core.config import StructuralConfig
from datarax.core.element_batch import Element
from datarax.sources.eager_source import (
    EagerSource,
    is_array_leaf,
    parts_of_records,
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


type _Parts = tuple[PyTree, tuple[dict[str, Any], ...]]


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
        data: Mapping[str, Any] | Sequence[Any] | np.ndarray | jax.Array,
        *,
        name: str | None = None,
    ) -> None:
        """Build the source's columns and provenance from ``data``.

        Args:
            config: Configuration for the MemorySource
            data: A mapping of columns (all of one length), a sequence of records (each a
                mapping of values, an ``Element`` or a single value), or an array whose leading
                axis is the records.
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
        columns, provenance = _parts(data)
        self._store(columns, provenance)

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
    ) -> jax.Array:
        """Return the global index of each record at positions ``start .. start + size``.

        ``start`` and ``size`` are positions in the order ``key`` selects (the sequential order
        when ``key`` is ``None``): the whole dataset, or with ``num_workers > 1`` this worker's
        positions ``[shard_id::num_workers]`` of the global order. The indices are global, so a
        record has one index on every worker.

        Args:
            start: Starting logical position: a Python int, its two uint32 words ``(hi, lo)``,
                or a traced int32 ``jax.Array`` (see
                :func:`~datarax.core.index_words.wrapped_positions`).
            size: Number of records (Python int).
            key: The key selecting the order, or ``None`` for the sequential order.

        Returns:
            uint32 ``jax.Array`` of shape ``(size, 2)``, each index as its words ``(hi, lo)``.
        """
        return resolve_wrapped_indices(
            start,
            size,
            self.length,
            key,
            num_workers=self.config.num_workers,
            shard_id=self.config.shard_id or 0,
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"MemorySource(length={self.length})"


def _parts(data: Mapping[str, Any] | Sequence[Any] | np.ndarray | jax.Array) -> _Parts:
    """The columns and provenance ``data`` holds.

    Args:
        data: A mapping of columns, an array, or a sequence of records.

    Returns:
        The host columns and one provenance mapping per record (or none).
    """
    if isinstance(data, Mapping):
        return _parts_of_columns(data)
    if isinstance(data, np.ndarray | jax.Array):
        return np.asarray(data), ()
    return _parts_of_records(data)


def _parts_of_columns(data: Mapping[str, Any]) -> _Parts:
    """A mapping of columns: numeric columns stay columns, the others become provenance.

    A value without rows (a scalar) is every record's; a string is a value, not a column.

    Args:
        data: The mapping of columns.

    Returns:
        The host columns and one provenance mapping per record (or none).
    """
    length = _record_count(data)
    columns: dict[str, Any] = {}
    provenance_columns: dict[str, Sequence[Any]] = {}
    for key, value in data.items():
        column = _column(key, value, length)
        if column is None:
            provenance_columns[key] = list(value) if _has_rows(value) else [value] * length
        else:
            columns[key] = column
    provenance = (
        tuple({key: rows[k] for key, rows in provenance_columns.items()} for k in range(length))
        if provenance_columns
        else ()
    )
    return columns, provenance


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


def _column(key: str, value: Any, length: int) -> np.ndarray | None:
    """``value`` as a host column of ``length`` rows, or ``None`` when it is provenance.

    Args:
        key: The column's name.
        value: The column, or a single value every record shares.
        length: The record count.

    Returns:
        The host column, or ``None`` for a non-numeric value.
    """
    if not _has_rows(value):
        if not is_array_leaf(value):
            return None
        scalar = np.asarray(value)
        return np.broadcast_to(scalar, (length, *scalar.shape))
    if isinstance(value, np.ndarray | jax.Array):
        return np.asarray(value) if is_array_leaf(value) else None
    if all(is_array_leaf(row) for row in value):
        return stack_records([{key: np.asarray(row)} for row in value])[key]
    return None


def _has_rows(value: Any) -> bool:
    """Whether ``value`` is a column (one row per record) rather than a single value.

    Args:
        value: The value.

    Returns:
        Whether it has rows.
    """
    return hasattr(value, "__len__") and not isinstance(value, str | bytes) and np.ndim(value) > 0


def _parts_of_records(records: Sequence[Any]) -> _Parts:  # noqa: DOC502 - from parts_of_records
    """A sequence of records: numeric values stacked into columns, the rest provenance.

    Args:
        records: The records.

    Returns:
        The host columns and one provenance mapping per record (or none).

    Raises:
        ValueError: If an ``Element`` carries an index or state, no record holds a number, or the
            records' numeric fields or shapes differ (see :func:`parts_of_records`).
    """
    return parts_of_records(
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
