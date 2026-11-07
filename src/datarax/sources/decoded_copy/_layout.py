"""A decoded copy's layout: its file names, its columns' layout and its units of records."""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Iterator, Mapping
from typing import Any

import jax
import numpy as np

from datarax.core.element_batch import Batch
from datarax.core.field_nodes import is_field_node
from datarax.core.field_paths import (
    fields_nested,
    flat_fields,
    KeyTuple,
    label,
    record_problems,
)


COPY_FORMAT = 2
"""The copy format this module writes and reads; a copy of another format is refused.

Format 2 names each column by its field's key tuple (``"keys"`` in the manifest)."""

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
    """One column of a layout: an array field or a field node's member, its dtype and row shape.

    Attributes:
        keys: The field's key tuple.
        dtype: The stored dtype, as a NumPy dtype string.
        shape: One record's shape.
        member: For a member of a field node, the member as jax prints its path (``.present``);
            empty for an array field.
        slot: The column's key among a layout's columns, derived once: its key tuple, then a
            member's name if any (a field node holds no group, so no field has that key tuple).
    """

    keys: KeyTuple
    dtype: str
    shape: tuple[int, ...]
    member: str = ""
    slot: KeyTuple = dataclasses.field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Derive the column's slot."""
        object.__setattr__(self, "slot", (*self.keys, self.member) if self.member else self.keys)

    @property
    def label(self) -> str:
        """How a refusal names the column (``'meta/score'``, ``'depth'.present``)."""
        return label(self.keys, self.member)

    @property
    def row_bytes(self) -> int:
        """One record's bytes in this column."""
        return math.prod(self.shape) * np.dtype(self.dtype).itemsize


def _arrays(data: Any, owner: str) -> Iterator[tuple[KeyTuple, str, Any]]:
    """Each array of a batch's data: an array field's keys and ``""``, or a node's and a member."""
    for keys, field in flat_fields(data, owner=owner).items():
        if is_field_node(field):
            for path, member in jax.tree_util.tree_flatten_with_path(field)[0]:
                yield keys, jax.tree_util.keystr(path), member
        else:
            yield keys, "", field


def slots_of(data: Any, *, owner: str) -> dict[KeyTuple, Any]:
    """A batch's arrays by column slot: each array field, and each member of a field node.

    Args:
        data: A host batch's data.
        owner: Who reads the fields, for a refusal.

    Returns:
        Each array, keyed by its column's slot (:attr:`LayoutColumn.slot`), in field order.
    """
    return {
        (*keys, member) if member else keys: array for keys, member, array in _arrays(data, owner)
    }


def slot_labels(data: Any, *, owner: str) -> dict[KeyTuple, str]:
    """How a refusal names each array of a batch's data, keyed by column slot (refusals only).

    Args:
        data: A host batch's data.
        owner: Who reads the fields, for a refusal.

    Returns:
        Each array's label (``'meta/score'``, ``'depth'.present``), keyed by its slot.
    """
    return {
        (*keys, member) if member else keys: label(keys, member)
        for keys, member, _ in _arrays(data, owner)
    }


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ColumnLayout:
    """The columns a source's records decode to, and how many records there are.

    Attributes:
        columns: One column per array field and per member of a field node, in field order.
        length: The records.
        nodes: Each field node's key tuple and jax tree definition, which rebuilds the node
            from its member columns; a decoded copy stores no field node (``begin_copy``).
    """

    columns: tuple[LayoutColumn, ...]
    length: int
    nodes: tuple[tuple[KeyTuple, Any], ...] = ()

    @classmethod
    def of_batch(cls, batch: Batch, length: int) -> ColumnLayout:
        """The layout of records read as ``batch``, ``length`` records in all.

        Args:
            batch: A host batch of the source's records, as its read returns it.
            length: The source's records.

        Returns:
            The layout, at the batch's stored dtypes.

        Raises:
            ValueError: If the records break the record rules (a key empty or exotic, a list
                holding fields, two fields of one spelling), before anything is stored.
        """
        problems = record_problems(batch.data)
        if problems:
            raise ValueError(
                "the source's records cannot be stored: "
                + "; ".join(str(problem) for problem in problems)
            )
        columns = []
        nodes = []
        for keys, field in flat_fields(batch.data, owner="a decoded copy's layout").items():
            if is_field_node(field):
                members, treedef = jax.tree_util.tree_flatten_with_path(field)
                nodes.append((keys, treedef))
                columns.extend(
                    _column(keys, leaf, member=jax.tree_util.keystr(path)) for path, leaf in members
                )
            else:
                columns.append(_column(keys, field))
        return cls(columns=tuple(columns), length=length, nodes=tuple(nodes))

    @property
    def nbytes(self) -> int:
        """The decoded columns' bytes: every record of every column."""
        return self.length * sum(column.row_bytes for column in self.columns)

    def empty(self) -> dict[KeyTuple, np.ndarray]:
        """Uninitialised host columns of every record, keyed by column slot."""
        return {c.slot: np.empty((self.length, *c.shape), np.dtype(c.dtype)) for c in self.columns}

    def nested(self, columns: Mapping[KeyTuple, Any]) -> dict[str, Any]:
        """``columns``, keyed by column slot, as the record they hold.

        Nesting is rebuilt from the key tuples, and each field node from its member columns by
        its tree definition.
        """
        fields: dict[KeyTuple, Any] = {
            column.keys: columns[column.slot] for column in self.columns if not column.member
        }
        for keys, treedef in self.nodes:
            members = [columns[c.slot] for c in self.columns if c.member and c.keys == keys]
            fields[keys] = jax.tree_util.tree_unflatten(treedef, members)
        return fields_nested(fields)

    def spec(self) -> dict[str, Any]:
        """One record's spec: each column's dtype and row shape, nested as :meth:`nested` nests."""
        return self.nested(
            {c.slot: jax.ShapeDtypeStruct(c.shape, np.dtype(c.dtype)) for c in self.columns}
        )


def _column(keys: KeyTuple, leaf: Any, member: str = "") -> LayoutColumn:
    """The column of one array of a host batch: its stored dtype and one record's shape."""
    return LayoutColumn(
        keys=keys, dtype=leaf.dtype.str, shape=tuple(int(s) for s in leaf.shape[1:]), member=member
    )


def unit_count(length: int, unit_records: int) -> int:
    """The units ``length`` records make, ``unit_records`` at a time (the last may be short)."""
    return -(-length // unit_records)


def unit_rows(unit: int, length: int, unit_records: int) -> range:
    """The records of ``unit``."""
    return range(unit * unit_records, min(length, (unit + 1) * unit_records))
