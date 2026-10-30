"""Ragged fields: records differing in length along their first axis, as values and offsets.

A ragged field is held as one values array, every record's cells one record after another, and
int64 offsets, one more than the records: record ``r`` is ``values[offsets[r]:offsets[r + 1]]``. It
is the layout of Arrow's list arrays, Megatron-LM's indexed dataset (``sequence_pointers`` beside
the values) and levanter's ``JaggedArrayStore`` (``data`` and cumulative ``offsets``), and the one
``batch_ops.pack_rows`` lays rows out from.

Reading records for packing counts a record's length in a field as Grain's packer does: the length
of its array along the first axis, a scalar counting 1 (``packing_packed_batch.py``,
``_get_element_lengths_flat``). A fixed column ``(N, L, ...)`` is therefore read the same way as a
ragged one, each record of length ``L``, so packing never depends on lengths happening to agree.
Host NumPy throughout.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np


@dataclasses.dataclass(frozen=True, slots=True)
class RaggedColumn:
    """One ragged field of a source's records.

    Attributes:
        values: Every record's cells, record after record, ``(T, *inner)``.
        offsets: Each record's first cell and the end, int64 ``(N + 1,)``, from 0.
    """

    values: np.ndarray
    offsets: np.ndarray

    def __len__(self) -> int:
        """The records the field holds."""
        return len(self.offsets) - 1


def ragged_column(records: Sequence[np.ndarray]) -> RaggedColumn:
    """Concatenate records' arrays, which agree past their first axis, into a ragged field.

    Args:
        records: One array per record, of one dtype kind and one shape past the first axis.

    Returns:
        The field's values and offsets.
    """
    lengths = np.fromiter((len(record) for record in records), np.int64, count=len(records))
    offsets = np.concatenate([np.zeros(1, np.int64), np.cumsum(lengths)])
    return RaggedColumn(values=np.concatenate(records), offsets=offsets)


def varies_along_first_axis_only(shapes: Sequence[tuple[int, ...]]) -> bool:
    """Whether arrays of ``shapes`` have one rank of at least 1 and agree past the first axis.

    Args:
        shapes: The records' array shapes.

    Returns:
        Whether only their first axis differs, which a ragged field holds.
    """
    first = shapes[0]
    return len(first) >= 1 and all(
        len(shape) == len(first) and shape[1:] == first[1:] for shape in shapes
    )


def is_ragged(shapes: Sequence[tuple[int, ...]]) -> bool:
    """Whether records of ``shapes`` make a ragged field: they differ, along the first axis only.

    Args:
        shapes: The records' array shapes.

    Returns:
        Whether the shapes are not all equal and agree past the first axis.
    """
    return len(set(shapes)) > 1 and varies_along_first_axis_only(shapes)


def ragged_lengths(column: RaggedColumn, rows: np.ndarray) -> np.ndarray:
    """The lengths of records ``rows`` of a ragged field.

    Args:
        column: The field.
        rows: Row numbers, any order, repeats allowed.

    Returns:
        int64 ``(n,)``.
    """
    rows = np.asarray(rows, np.int64)
    return column.offsets[rows + 1] - column.offsets[rows]


def take_ragged(column: RaggedColumn, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Records ``rows`` of a ragged field, in the order asked, as values and offsets.

    One vectorised gather: every asked cell's position in ``values`` is computed from the offsets,
    with no loop over records.

    Args:
        column: The field.
        rows: Row numbers, any order, repeats allowed.

    Returns:
        Their cells, record after record, and their offsets, int64 from 0.
    """
    rows = np.asarray(rows, np.int64)
    starts = column.offsets[rows]
    lengths = column.offsets[rows + 1] - starts
    offsets = np.concatenate([np.zeros(1, np.int64), np.cumsum(lengths)])
    cells = np.arange(offsets[-1]) - np.repeat(offsets[:-1] - starts, lengths)
    return column.values[cells], offsets


def column_lengths(column: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """The lengths of records ``rows`` of a fixed column: its row length, or 1 for scalars.

    Args:
        column: A column with one row per record.
        rows: Row numbers.

    Returns:
        int64 ``(n,)``.
    """
    length = 1 if column.ndim == 1 else column.shape[1]
    return np.full(len(np.asarray(rows)), length, np.int64)


def take_column(column: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Records ``rows`` of a fixed column, as a ragged field's values and offsets.

    Args:
        column: A column with one row per record.
        rows: Row numbers, any order, repeats allowed.

    Returns:
        Their cells, record after record (a scalar record is one cell), and int64 offsets.
    """
    taken = column[np.asarray(rows, np.int64)]
    lengths = column_lengths(column, rows)
    offsets = np.concatenate([np.zeros(1, np.int64), np.cumsum(lengths)])
    values = taken if column.ndim == 1 else taken.reshape(-1, *column.shape[2:])
    return values, offsets


def field_lengths(field: RaggedColumn | np.ndarray, rows: np.ndarray) -> np.ndarray:
    """The lengths of records ``rows`` of a field, ragged or a fixed column.

    Args:
        field: The field.
        rows: Row numbers, any order, repeats allowed.

    Returns:
        int64 ``(n,)``.
    """
    if isinstance(field, RaggedColumn):
        return ragged_lengths(field, rows)
    return column_lengths(field, rows)


def take_field(field: RaggedColumn | np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Records ``rows`` of a field, ragged or a fixed column, as values and offsets.

    Args:
        field: The field.
        rows: Row numbers, any order, repeats allowed.

    Returns:
        Their cells, record after record, and their int64 offsets from 0.
    """
    if isinstance(field, RaggedColumn):
        return take_ragged(field, rows)
    return take_column(field, rows)


def refuse_unpacked_read(ragged: Sequence[str]) -> None:
    """Refuse an unpacked read of a source holding the ragged fields ``ragged``, naming one.

    Args:
        ragged: The source's ragged fields, by name.

    Raises:
        ValueError: If there is one, naming it, packing and padding.
    """
    if not ragged:
        return
    raise ValueError(
        f"field {ragged[0]!r} is ragged (its records differ in length along their first axis), "
        "but a batch's shapes are static: pack records with segment ids, or pad the field to a "
        "fixed length along its first axis and keep its mask or length in data"
    )


__all__ = [
    "RaggedColumn",
    "column_lengths",
    "field_lengths",
    "is_ragged",
    "ragged_column",
    "ragged_lengths",
    "refuse_unpacked_read",
    "take_column",
    "take_field",
    "take_ragged",
    "varies_along_first_axis_only",
]
