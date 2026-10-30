"""Reading a unit of a source's records and writing it into a copy's columns: the one decode path.

:func:`read_unit` reads a unit with the source's own host read; a source holding a ragged field
refuses every unpacked read, so its units are read field by field through its ragged read (fixed
fields as rows, ragged ones as values and offsets). :func:`write_unit` assigns each fixed field to
its rows of a given column, and :func:`ragged_parts` hands each ragged field back for appending,
both refusing a field unlike the copy's layout instead of casting it. The same writer fills
in-memory columns (:func:`build_in_memory`) and a copy's memory-mapped files, and each unit's rows
are digested (:func:`unit_digest`, :func:`content_digest`).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any, cast

import numpy as np

from datarax.core.data_source import DataSourceModule, known_length, Provenance, read_records
from datarax.core.index_words import to_words
from datarax.core.ragged import RaggedColumn
from datarax.sources._field_paths import by_field
from datarax.sources.decoded_copy._layout import (
    ColumnLayout,
    LayoutColumn,
    unit_count,
    UNIT_RECORDS,
    unit_rows,
)


type UnitFields = dict[str, np.ndarray | RaggedColumn]
"""A unit's numeric fields by path: a fixed field's rows, or a ragged field."""

type UnitRead = tuple[UnitFields, Provenance | None]
"""A unit's records' fields, and their provenance when it was read."""


def read_unit(  # noqa: DOC502 - fixed_rows raises
    source: DataSourceModule, rows: range, *, with_provenance: bool
) -> UnitRead:
    """Read records ``rows`` with the source's own host read: one read, one decode.

    A source holding a ragged field is read through its ragged read, every field it names
    (``field_names``): its ragged fields as values and offsets, its fixed ones as rows.

    Args:
        source: An indexed source with the stateless host read; record ``r`` is index ``r``.
        rows: The records, a run.
        with_provenance: Whether the records' provenance is read too.

    Returns:
        The records' fields, and their provenance or ``None``.

    Raises:
        ValueError: If a field the source names fixed differs in length between records.
    """
    words = to_words(np.arange(rows.start, rows.stop, dtype=np.uint64))
    ragged = source.ragged_fields
    if not ragged:
        batch, provenance = read_records(
            source, words, epochs=0, contiguous=True, with_provenance=with_provenance
        )
        return by_field(batch.data), provenance
    fields: UnitFields = {}
    for name in source.field_names:
        values, offsets = source.get_ragged(words, name)
        column = RaggedColumn(values=values, offsets=offsets)
        fields[name] = column if name in ragged else fixed_rows(name, column, rows, ragged)
    return fields, (tuple(source.provenance(words)) if with_provenance else None)


def fixed_rows(name: str, column: RaggedColumn, rows: range, ragged: tuple[str, ...]) -> np.ndarray:
    """A fixed field read through the ragged read, as rows: ``(n, L, *inner)``.

    Every record's length is the field's row length ``L``, so ``(n * L, *inner)`` values are
    ``n`` rows of ``(L, *inner)`` (a scalar field's records are of length 1).

    Args:
        name: The field.
        column: Its records, as the ragged read gives them.
        rows: The records, for the refusal.
        ragged: The source's ragged fields, for the refusal.

    Returns:
        The rows.

    Raises:
        ValueError: If the records differ in length, naming the field.
    """
    lengths = np.diff(column.offsets)
    if len(set(lengths.tolist())) > 1:
        raise ValueError(
            f"field {name!r} of records {rows.start}..{rows.stop} holds records of lengths "
            f"{int(lengths.min())}..{int(lengths.max())}, though the source names it fixed (its "
            f"ragged fields are {list(ragged)}): a source names every field whose records differ "
            "in length among its ragged fields"
        )
    return column.values.reshape(len(rows), int(lengths[0]), *column.values.shape[1:])


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
    return ColumnLayout.of_fields(read[0], length), read


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

    The records are read with the source's own host read (:func:`read_unit`), and each fixed
    field is assigned to its column's rows; it never allocates the destination. Each ragged field
    is checked against its column; :func:`ragged_parts` hands it back from the same read.

    Args:
        source: An indexed source with the stateless host read; record ``r`` is index ``r``.
        rows: The records, a run.
        out: One array per fixed layout column, keyed by field path, ``len(rows)`` rows each.
        layout: The columns the records decode to.
        with_provenance: Whether the records' provenance is read and returned.
        read: The records already read (unit 0's, by :func:`first_read`, or a worker's), written
            without reading them again; ``None`` reads them.

    Returns:
        One plain ``dict`` of strings and objects per record (a form a worker process sends), or
        ``None`` when not asked for.

    Raises:
        ValueError: If a field is missing from the layout, absent from the records, or of
            another dtype or record shape than its column, naming the field.
    """
    fields, provenance = read or read_unit(source, rows, with_provenance=with_provenance)
    refuse_other_fields(fields, layout, rows)
    for column in layout.columns:
        held = checked(fields[column.name], column, rows)
        if not column.ragged:
            out[column.name][...] = held
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


def ragged_parts(read: UnitRead, layout: ColumnLayout) -> dict[str, RaggedColumn]:
    """A unit's ragged fields, in layout order, as :func:`write_unit` checked them.

    Args:
        read: The unit's read.
        layout: The copy's columns.

    Returns:
        Each ragged field's values and offsets (from 0), by path.
    """
    return {
        column.name: cast(RaggedColumn, read[0][column.name])
        for column in layout.columns
        if column.ragged
    }


def checked(  # noqa: DOC502 - the checks raise
    leaf: np.ndarray | RaggedColumn, column: LayoutColumn, rows: range
) -> np.ndarray | RaggedColumn:
    """``leaf``, refused unless it is ``column``'s kind, dtype and record shape for ``rows``.

    Args:
        leaf: A field's rows, or a ragged field, as read.
        column: Its column.
        rows: The records.

    Returns:
        The leaf.

    Raises:
        ValueError: Naming the field, what it is and what its column holds.
    """
    if isinstance(leaf, RaggedColumn) != column.ragged:
        raise ValueError(
            f"field {column.name!r} of records {rows.start}..{rows.stop} is "
            f"{'fixed' if column.ragged else 'ragged'}, where its column is "
            f"{'ragged' if column.ragged else 'fixed'}: a field is ragged in every record or "
            "in none"
        )
    if isinstance(leaf, RaggedColumn):
        return checked_ragged(leaf, column, rows)
    shape = (len(rows), *column.shape)
    if leaf.dtype.str != column.dtype or tuple(leaf.shape) != shape:
        raise ValueError(
            f"field {column.name!r} of records {rows.start}..{rows.stop} is "
            f"{leaf.dtype.str}{tuple(leaf.shape)}, where its column holds "
            f"{column.dtype}{shape}: a column is never cast or reshaped, and a batch's shapes "
            "are static, so every record holds the field at one dtype and shape"
        )
    return leaf


def checked_ragged(leaf: RaggedColumn, column: LayoutColumn, rows: range) -> RaggedColumn:
    """A ragged ``leaf``, refused unless its cells are ``column``'s and it holds ``rows``.

    Args:
        leaf: A ragged field's values and offsets, as read.
        column: Its column.
        rows: The records.

    Returns:
        The leaf.

    Raises:
        ValueError: Naming the field, what it is and what its column holds.
    """
    values, offsets = leaf.values, leaf.offsets
    if (
        values.dtype.str != column.dtype
        or tuple(values.shape[1:]) != column.shape
        or offsets.shape != (len(rows) + 1,)
        or offsets[0] != 0
        or offsets[-1] != len(values)
    ):
        raise ValueError(
            f"ragged field {column.name!r} of records {rows.start}..{rows.stop} holds "
            f"{values.dtype.str}{tuple(values.shape)} cells at offsets {offsets.shape}, where "
            f"its column holds {column.dtype} cells of shape {column.shape} at {len(rows) + 1} "
            "offsets from 0: a column is never cast or reshaped"
        )
    return leaf


def in_layout_order(
    layout: ColumnLayout, rows: Mapping[str, np.ndarray], ragged: Mapping[str, RaggedColumn]
) -> dict[str, np.ndarray | RaggedColumn]:
    """A unit's fixed rows and ragged fields as one mapping, in layout order, for its digest.

    Args:
        layout: The copy's columns.
        rows: The unit's rows of every fixed column.
        ragged: Its ragged fields.

    Returns:
        Every column's part, by path, in layout order.
    """
    return {c.name: ragged[c.name] if c.ragged else rows[c.name] for c in layout.columns}


def raw_bytes(array: np.ndarray) -> np.ndarray:
    """``array``'s bytes in C order, as one uint8 axis; an array of no element gives none."""
    return np.ascontiguousarray(array).reshape(-1).view(np.uint8)


def unit_digest(out: Mapping[str, np.ndarray | RaggedColumn]) -> bytes:
    """The SHA-256 of a unit's columns, in layout order, each as raw bytes.

    A fixed column gives its rows; a ragged field its values, then its offsets from 0, so a cell
    moved between records changes the digest as a changed cell does.

    Args:
        out: The unit's rows of every column, or its ragged fields, keyed by field path in
            layout order.

    Returns:
        The digest.
    """
    digest = hashlib.sha256()
    for held in out.values():
        parts = (held.values, held.offsets) if isinstance(held, RaggedColumn) else (held,)
        for part in parts:
            digest.update(raw_bytes(part))
    return digest.digest()


def content_digest(unit_digests: Iterable[bytes]) -> str:
    """A copy's content digest: the SHA-256 of its units' digests in unit order, as hex."""
    digest = hashlib.sha256()
    for unit in unit_digests:
        digest.update(unit)
    return digest.hexdigest()


def refuse_ragged_in_memory(source: DataSourceModule) -> None:
    """Refuse decoding a source holding a ragged field into RAM, naming the decoded copy.

    Args:
        source: The source to decode.

    Raises:
        NotImplementedError: If it holds a ragged field.
    """
    if source.ragged_fields:
        raise NotImplementedError(
            f"{type(source).__name__} holds the ragged fields {list(source.ragged_fields)}, which "
            "materialize writes into a decoded copy (materialize(..., directory=...)); decoding "
            "them into RAM is open work: a ragged field's size is known only once it is decoded"
        )


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
    refuse_ragged_in_memory(source)
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
