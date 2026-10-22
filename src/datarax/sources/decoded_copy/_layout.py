"""A decoded copy's layout: its file names, its columns' layout and its units of records."""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Mapping
from typing import Any

import numpy as np

from datarax.core.element_batch import Batch
from datarax.core.field_paths import by_field, nested


COPY_FORMAT = 1
"""The copy format this module writes and reads; a copy of another format is refused."""

UNIT_RECORDS = 256
"""Records decoded and written at a time."""

MANIFEST = "manifest.json"

MANIFEST_TMP = "manifest.json.tmp"

LOCK = ".lock"

PROVENANCE = "provenance.array_record"

PARTIAL = ".partial"

COLUMN = re.compile(r"column-\d+\.npy")

DIGEST = "sha256-of-unit-sha256"


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class LayoutColumn:
    """One column of a layout: a field, its stored dtype and one record's shape.

    Attributes:
        name: The field's path, ``/`` between levels.
        dtype: The stored dtype, as a NumPy dtype string.
        shape: One record's shape.
    """

    name: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def row_bytes(self) -> int:
        """One record's bytes in this column."""
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
    def of_batch(cls, batch: Batch, length: int) -> ColumnLayout:
        """The layout of records read as ``batch``, ``length`` records in all.

        Args:
            batch: A host batch of the source's records, as its read returns it.
            length: The source's records.

        Returns:
            The layout, at the batch's stored dtypes.
        """
        columns = tuple(
            LayoutColumn(
                name=name, dtype=leaf.dtype.str, shape=tuple(int(s) for s in leaf.shape[1:])
            )
            for name, leaf in by_field(batch.data).items()
        )
        return cls(columns=columns, length=length)

    @property
    def nbytes(self) -> int:
        """The decoded columns' bytes: every record of every column."""
        return self.length * sum(column.row_bytes for column in self.columns)

    def empty(self) -> dict[str, np.ndarray]:
        """Uninitialised host columns of every record, keyed by field path."""
        return {c.name: np.empty((self.length, *c.shape), np.dtype(c.dtype)) for c in self.columns}

    def nested(self, columns: Mapping[str, Any]) -> dict[str, Any]:
        """``columns`` keyed by field path, as the nested mappings a record holds."""
        return nested({column.name: columns[column.name] for column in self.columns})


def unit_count(length: int, unit_records: int) -> int:
    """The units ``length`` records make, ``unit_records`` at a time (the last may be short)."""
    return -(-length // unit_records)


def unit_rows(unit: int, length: int, unit_records: int) -> range:
    """The records of ``unit``."""
    return range(unit * unit_records, min(length, (unit + 1) * unit_records))
