"""A decoded copy's layout: its file names, its columns' layout and its units of records."""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Mapping
from typing import Any

import numpy as np

from datarax.core.ragged import RaggedColumn
from datarax.sources._field_paths import nested


COPY_FORMAT = 2
"""The copy format this module writes and reads; a copy of another format is refused."""

UNIT_RECORDS = 256
"""Records decoded and written at a time."""

MANIFEST = "manifest.json"

MANIFEST_TMP = "manifest.json.tmp"

LOCK = ".lock"

PROVENANCE = "provenance.array_record"

PARTIAL = ".partial"

COLUMN = re.compile(r"column-\d+(\.values|\.offsets)?\.npy")
"""A column's file: ``column-k.npy`` for a fixed field, ``.values.npy`` and ``.offsets.npy`` for
a ragged one."""

PLAN = re.compile(r"plan-[0-9a-f]{64}")
"""A pack plan's directory in the copy (:mod:`datarax.sources.decoded_copy._plans`)."""

DIGEST = "sha256-of-unit-sha256"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class LayoutColumn:
    """One column of a layout: a field, its stored dtype and one record's shape, or one cell's.

    Attributes:
        name: The field's path, ``/`` between levels.
        dtype: The stored dtype, as a NumPy dtype string.
        shape: One record's shape; a ragged field's records differ along their first axis, so
            its shape is one cell's, the record's shape past that axis.
        ragged: Whether the field is ragged, stored as values and offsets.
    """

    name: str
    dtype: str
    shape: tuple[int, ...]
    ragged: bool = False

    @property
    def row_bytes(self) -> int:
        """One record's bytes in this column (one cell's, for a ragged field)."""
        return math.prod(self.shape) * np.dtype(self.dtype).itemsize


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ColumnLayout:
    """The columns a source's records decode to, and how many records there are.

    Attributes:
        columns: One column per numeric field, in field order.
        length: The records.
    """

    columns: tuple[LayoutColumn, ...]
    length: int

    @classmethod
    def of_fields(
        cls, fields: Mapping[str, np.ndarray | RaggedColumn], length: int
    ) -> ColumnLayout:
        """The layout of records read as ``fields`` (a unit's), ``length`` records in all.

        Args:
            fields: A unit's fields by path, as read: a fixed field's rows, or a ragged field.
            length: The source's records.

        Returns:
            The layout, at the stored dtypes, in the fields' order.
        """
        columns = tuple(
            LayoutColumn(
                name=name,
                dtype=field.values.dtype.str,
                shape=tuple(int(s) for s in field.values.shape[1:]),
                ragged=True,
            )
            if isinstance(field, RaggedColumn)
            else LayoutColumn(
                name=name, dtype=field.dtype.str, shape=tuple(int(s) for s in field.shape[1:])
            )
            for name, field in fields.items()
        )
        return cls(columns=columns, length=length)

    @property
    def fixed(self) -> tuple[LayoutColumn, ...]:
        """The fixed fields' columns, in layout order."""
        return tuple(column for column in self.columns if not column.ragged)

    @property
    def is_ragged(self) -> bool:
        """Whether any field is ragged."""
        return any(column.ragged for column in self.columns)

    @property
    def nbytes(self) -> int:
        """The decoded fixed columns' bytes: every record of every fixed column."""
        return self.length * sum(column.row_bytes for column in self.fixed)

    def empty(self) -> dict[str, np.ndarray]:
        """Uninitialised host columns of every record of every fixed field, keyed by path."""
        return {c.name: np.empty((self.length, *c.shape), np.dtype(c.dtype)) for c in self.fixed}

    def nested(self, columns: Mapping[str, Any]) -> dict[str, Any]:
        """``columns`` keyed by field path, as the nested mappings a record holds."""
        return nested({column.name: columns[column.name] for column in self.columns})


def unit_count(length: int, unit_records: int) -> int:
    """The units ``length`` records make, ``unit_records`` at a time (the last may be short)."""
    return -(-length // unit_records)


def unit_rows(unit: int, length: int, unit_records: int) -> range:
    """The records of ``unit``."""
    return range(unit * unit_records, min(length, (unit + 1) * unit_records))
