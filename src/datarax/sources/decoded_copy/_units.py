"""Reading a unit of a source's records and writing it into a copy's columns: the one decode path.

:func:`write_unit` reads a unit with the source's own host read and assigns each field to its rows
of a given column, refusing a field unlike the copy's layout instead of casting it. The same
writer fills in-memory columns (:func:`build_in_memory`) and a copy's memory-mapped files, and
each unit's rows are digested (:func:`unit_digest`, :func:`content_digest`).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from datarax.core.data_source import DataSourceModule, known_length, Provenance, read_records
from datarax.core.element_batch import Batch
from datarax.core.field_paths import by_field
from datarax.core.index_words import to_words
from datarax.sources.decoded_copy._layout import (
    ColumnLayout,
    LayoutColumn,
    unit_count,
    UNIT_RECORDS,
    unit_rows,
)


type UnitRead = tuple[Batch, Provenance | None]
"""A unit's records as a host ``Batch``, and their provenance when it was read."""


def read_unit(source: DataSourceModule, rows: range, *, with_provenance: bool) -> UnitRead:
    """Read records ``rows`` with the source's own host read: one read, one decode.

    Args:
        source: An indexed source with the stateless host read; record ``r`` is index ``r``.
        rows: The records, a run.
        with_provenance: Whether the records' provenance is read too.

    Returns:
        The records as a host ``Batch``, and their provenance or ``None``.
    """
    words = to_words(np.arange(rows.start, rows.stop, dtype=np.uint64))
    return read_records(source, words, epochs=0, contiguous=True, with_provenance=with_provenance)


def first_read(
    source: DataSourceModule, *, unit_records: int, with_provenance: bool = True
) -> tuple[ColumnLayout, UnitRead]:
    """A source's layout from its first unit, and that unit's read, which is then written as is.

    Every decoded copy and in-memory build reads unit 0 once: its batch gives the stored dtypes
    and record shapes, and it is unit 0's rows (passed to :func:`write_unit` as ``read``).

    Args:
        source: An indexed source with the stateless host read.
        unit_records: Records per unit.
        with_provenance: Whether unit 0's provenance is read too.

    Returns:
        The layout, and unit 0's read.

    Raises:
        ValueError: If the source holds no record, or reports no length.
    """
    length = known_length(source)
    if not length:
        raise ValueError(
            f"{type(source).__name__} holds no record (its length: {length}), so it has no columns"
        )
    read = read_unit(source, unit_rows(0, length, unit_records), with_provenance=with_provenance)
    return ColumnLayout.of_batch(read[0], length), read


def source_layout(source: DataSourceModule) -> ColumnLayout:  # noqa: DOC502 - first_read raises
    """The layout of ``source``'s records, read from its first record as it decodes.

    The dtypes are the stored ones (an int64 label stays int64 whatever the precision mode), with
    no spec translation.

    Args:
        source: An indexed source with the stateless host read.

    Returns:
        The layout.

    Raises:
        ValueError: If the source holds no record.
    """
    return first_read(source, unit_records=1, with_provenance=False)[0]


def write_unit(  # noqa: PLR0913, DOC502 - one unit's write; its checks raise
    source: DataSourceModule,
    rows: range,
    out: Mapping[str, np.ndarray],
    *,
    layout: ColumnLayout,
    with_provenance: bool,
    read: UnitRead | None = None,
) -> tuple[dict[str, Any], ...] | None:
    """Decode records ``rows`` of ``source`` into ``out``: the one per-unit writer.

    The records are read with the source's own host read
    (:func:`~datarax.core.data_source.read_records`), and each field is assigned to its column's
    rows; it never allocates the destination.

    Args:
        source: An indexed source with the stateless host read; record ``r`` is index ``r``.
        rows: The records, a run.
        out: One array per layout column, keyed by field path, ``len(rows)`` rows each.
        layout: The columns the records decode to.
        with_provenance: Whether the records' provenance is read and returned.
        read: The records already read (unit 0's, by :func:`first_read`), written without
            reading them again; ``None`` reads them.

    Returns:
        One plain ``dict`` of strings and objects per record (a form a worker process sends), or
        ``None`` when not asked for.

    Raises:
        ValueError: If a field is missing from the layout, absent from the records, or of
            another dtype or record shape than its column, naming the field.
    """
    batch, provenance = read or read_unit(source, rows, with_provenance=with_provenance)
    leaves = by_field(batch.data)
    refuse_other_fields(leaves, layout, rows)
    for column in layout.columns:
        out[column.name][...] = checked(leaves[column.name], column, rows)
    if provenance is None:
        return None
    return tuple(dict(record) for record in provenance)


def refuse_other_fields(leaves: Mapping[str, Any], layout: ColumnLayout, rows: range) -> None:
    """Refuse records whose numeric fields are not the layout's columns, naming the difference.

    Args:
        leaves: The records' numeric fields, by path.
        layout: The copy's columns.
        rows: The records, for the refusal.

    Raises:
        ValueError: Naming the fields that differ.
    """
    expected = [column.name for column in layout.columns]
    differing = sorted(set(leaves) ^ set(expected))
    if differing:
        raise ValueError(
            f"records {rows.start}..{rows.stop} hold the fields {sorted(leaves)} where the copy's "
            f"columns are {expected}: {differing} differ; every record holds the same fields"
        )


def checked(leaf: np.ndarray, column: LayoutColumn, rows: range) -> np.ndarray:
    """``leaf``, refused unless it is ``column``'s dtype and record shape for ``rows``.

    Args:
        leaf: A field's rows, as read.
        column: Its column.
        rows: The records.

    Returns:
        The leaf.

    Raises:
        ValueError: Naming the field, what it is and what its column holds.
    """
    shape = (len(rows), *column.shape)
    if leaf.dtype.str != column.dtype or tuple(leaf.shape) != shape:
        raise ValueError(
            f"field {column.name!r} of records {rows.start}..{rows.stop} is "
            f"{leaf.dtype.str}{tuple(leaf.shape)}, where its column holds "
            f"{column.dtype}{shape}: a column is never cast or reshaped, and a batch's shapes "
            "are static, so every record holds the field at one dtype and shape"
        )
    return leaf


def unit_digest(out: Mapping[str, np.ndarray]) -> bytes:
    """The SHA-256 of a unit's columns, in layout order, each column's rows as raw bytes.

    Args:
        out: The unit's rows of every column, keyed by field path in layout order.

    Returns:
        The digest.
    """
    digest = hashlib.sha256()
    for rows in out.values():
        digest.update(memoryview(np.ascontiguousarray(rows)).cast("B"))
    return digest.digest()


def content_digest(unit_digests: Iterable[bytes]) -> str:
    """A copy's content digest: the SHA-256 of its units' digests in unit order, as hex."""
    digest = hashlib.sha256()
    for unit in unit_digests:
        digest.update(unit)
    return digest.hexdigest()


def build_in_memory(
    source: DataSourceModule, *, unit_records: int = UNIT_RECORDS
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Decode every record of ``source`` into host columns on the calling thread, unit by unit.

    Args:
        source: An indexed source with the stateless host read.
        unit_records: Records decoded at a time.

    Returns:
        The columns, nested by field path, and one provenance mapping per record, or none when
        no record carries anything but numbers.
    """
    layout, first = first_read(source, unit_records=unit_records)
    columns = layout.empty()
    provenance: list[dict[str, Any]] = []
    for unit in range(unit_count(layout.length, unit_records)):
        rows = unit_rows(unit, layout.length, unit_records)
        out = {name: column[rows.start : rows.stop] for name, column in columns.items()}
        written = write_unit(
            source, rows, out, layout=layout, with_provenance=True, read=None if unit else first
        )
        provenance.extend(written or ())
    return layout.nested(columns), tuple(provenance) if any(provenance) else ()
