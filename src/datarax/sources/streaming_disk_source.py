"""Out-of-core indexed source over columns of arrays stored raw on disk, read through memory maps.

A column is an array stored raw in a file at a byte offset, :class:`DiskColumn`
``(name, file, offset, dtype, shape)``, its leading axis the record axis: a ``.npy`` file is the
case whose offset is its header's length (:meth:`DiskColumn.from_npy`), an uncompressed contiguous
HDF5 dataset at the offset its file reports is another. A column's name is its field: a key tuple
(``("meta", "score")``) nests, and a string is one key, ``/`` and all, since a source receives no
record a spelling could resolve against. The source's columns are one declaration, checked once in
any order. Every column is opened with ``np.memmap`` for zero-copy random access, and two reads
serve the source:

- the host read, ``get_batch(indices, *, epochs=0, contiguous=False)``, the eager sources' read
  (:func:`~datarax.sources.eager_source.read_host_batch`): a NumPy gather of the named rows of
  every column (a contiguous run as views of them), returned as a ``Batch`` named with the given
  words and epochs, creating no device array;
- the traced read, ``get_records(indices)``, which issues the disk read through
  ``jax.experimental.io_callback`` for the compiled pipeline session, and wraps the rows in
  ``jax.lax.stop_gradient`` to mark them as constants: the callback is given integer indices,
  which are never differentiated, so ``jax.grad`` over parameters works with or without it.

A source may also serve its records' provenance (their strings and objects) by index from one
ArrayRecord file of encoded records (:class:`DiskProvenance`), as a decoded copy holds it
(:mod:`datarax.sources.decoded_copy`), and declares what it reads (:meth:`source_identity`).
"""

from __future__ import annotations

import dataclasses
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import jax
import numpy as np
from array_record.python.array_record_data_source import ArrayRecordDataSource
from flax import nnx
from jax.experimental import io_callback
from jax.typing import ArrayLike

from datarax.core.block_order import BlockOrder, buffer_bytes
from datarax.core.config import StructuralConfig
from datarax.core.data_source import (
    DataSourceModule,
    host_rows,
    NO_PROVENANCE,
    record_words,
    RecordIdentity,
)
from datarax.core.element_batch import Batch
from datarax.core.field_paths import (
    as_field_name,
    create_path,
    FieldName,
    fields_nested,
    KeyTuple,
    label,
    record_problems,
)
from datarax.core.host_values import HostArray, HostArrays
from datarax.core.index_words import from_words
from datarax.core.spec import device_spec, JAX_ARRAY_KINDS
from datarax.sources._disk_provenance import decode_provenance, DiskProvenance
from datarax.sources.eager_source import read_host_batch
from datarax.typing import DataDict


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DiskColumn:
    """One column of a :class:`StreamingDiskSource`: an array stored raw in a file at an offset.

    Attributes:
        name: The field: a key tuple (``("meta", "score")``, a list converts to one), or a string,
            which is one key (``"meta/score"`` is the key ``meta/score``). Held as given, so
            ``"x"`` and ``("x",)`` name one field and are two configs.
        keys: The field's key tuple, derived once from ``name`` (a string is one key).
        file: The file holding the array's bytes.
        offset: The byte offset of the array's first byte in the file.
        dtype: The stored dtype (a NumPy dtype string), numeric.
        shape: The whole array's shape, its leading axis the record axis.
    """

    name: FieldName
    file: str
    offset: int
    dtype: str
    shape: tuple[int, ...]
    keys: KeyTuple = dataclasses.field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:  # noqa: DOC503 - as_field_name raises the TypeError
        """Hold the name as a string or key tuple, and refuse a column no memory map can read.

        Raises:
            TypeError: If the name is neither a string nor a non-empty list of strings.
            ValueError: If a key is empty or exotic, the offset is negative, the shape has no
                record axis, or the dtype is not numeric.
        """
        name = as_field_name(self.name, "DiskColumn's name")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "keys", (name,) if isinstance(name, str) else name)
        create_path({}, name, owner="DiskColumn", param="name")  # the key rule
        if self.offset < 0:
            raise ValueError(f"column {self.label}: offset is at least 0; got {self.offset}")
        if not self.shape:
            raise ValueError(f"column {self.label} has no record axis: its shape is ()")
        if np.dtype(self.dtype).kind not in JAX_ARRAY_KINDS:
            raise ValueError(
                f"column {self.label} stores {np.dtype(self.dtype)}, but a column is numeric; "
                "strings and other objects are a record's provenance"
            )

    @classmethod
    def from_npy(cls, name: FieldName, path: str | os.PathLike[str]) -> DiskColumn:
        """The array a ``.npy`` file stores, as a column: its offset is the header's length.

        Args:
            name: The field: a key tuple, or a string, which is one key.
            path: The ``.npy`` file.

        Returns:
            The column.

        Raises:
            ValueError: If the file stores its array in Fortran order, or a non-numeric one.
        """
        array = np.load(path, mmap_mode="r")
        if array.ndim > 1 and not array.flags.c_contiguous:
            raise ValueError(
                f"{path} stores its array in Fortran order; a column's rows are C-ordered: save "
                "it with np.save(path, np.ascontiguousarray(array))"
            )
        column = cls(
            name=name,
            file=str(path),
            offset=int(array.offset),
            dtype=array.dtype.str,
            shape=tuple(int(size) for size in array.shape),
        )
        del array  # closes the map
        return column

    @property
    def label(self) -> str:
        """How a refusal names the column's field."""
        return label(self.keys)

    @property
    def row_spec(self) -> jax.ShapeDtypeStruct:
        """One record's row as stored: the shape past the leading axis, the stored dtype."""
        return jax.ShapeDtypeStruct(self.shape[1:], np.dtype(self.dtype))

    @property
    def length(self) -> int:
        """The column's records: its leading axis."""
        return self.shape[0]

    @property
    def nbytes(self) -> int:
        """The column's bytes in its file."""
        return math.prod(self.shape) * np.dtype(self.dtype).itemsize


@dataclasses.dataclass(frozen=True)
class StreamingDiskSourceConfig(StructuralConfig):
    """Configuration for ``StreamingDiskSource``.

    Attributes:
        columns: The source's columns, one row per record in each.
        provenance: The records' provenance on disk, or ``None`` when they carry nothing but
            arrays.
        content_digest: The content digest of the decoded copy the columns are, or ``None`` for
            columns that are no copy; part of the source's identity.
    """

    columns: tuple[DiskColumn, ...] = ()
    provenance: DiskProvenance | None = None
    content_digest: str | None = None

    def __post_init__(self) -> None:
        """Validate the config.

        Raises:
            ValueError: If there is no column, the columns' names clash (:func:`_refuse_clashes`),
                or the columns hold different record counts.
        """
        super().__post_init__()
        if not self.columns:
            raise ValueError("StreamingDiskSourceConfig needs at least one column")
        _refuse_clashes(self.columns)
        if len({column.length for column in self.columns}) > 1:
            lengths = ", ".join(f"{c.label}: {c.length}" for c in self.columns)
            raise ValueError(f"columns hold one row per record; their lengths are {lengths}")


def _refuse_clashes(columns: Sequence[DiskColumn]) -> None:
    """Refuse columns that do not make one record: the declaration checked once, in any order.

    A source receives no record, so each name creates its own field (N3a): a key tuple exactly
    its path, a string one key. Over their union: a field given twice, a key tuple passing
    through another column's field, a string extending another column's field (offering its key
    tuple), and two fields of one spelling are refused, in that order.

    Args:
        columns: The source's columns.

    Raises:
        ValueError: Naming the clash.
    """
    _refuse_a_field_twice(columns)
    fields = {column.keys: column for column in columns}
    _refuse_a_path_through_a_column(columns, fields)
    _refuse_a_spelling_beside_a_field(columns, fields)
    problems = record_problems(fields_nested({keys: c.row_spec for keys, c in fields.items()}))
    if problems:
        raise ValueError(f"the columns do not make one record: {problems[0]}")


def _refuse_a_field_twice(columns: Sequence[DiskColumn]) -> None:
    """Refuse two columns of one key tuple.

    Args:
        columns: The source's columns.

    Raises:
        ValueError: Naming the field.
    """
    twice = [keys for keys, count in Counter(c.keys for c in columns).items() if count > 1]
    if twice:
        raise ValueError(f"{label(twice[0])} names two columns; each column has its own name")


def _refuse_a_path_through_a_column(
    columns: Sequence[DiskColumn], fields: Mapping[KeyTuple, DiskColumn]
) -> None:
    """Refuse a key tuple whose path passes through another column's field.

    Args:
        columns: The source's columns.
        fields: The columns by key tuple.

    Raises:
        ValueError: Naming the key tuple and the column it passes through.
    """
    for column in columns:
        for depth in range(1, len(column.keys)):
            through = fields.get(column.keys[:depth])
            if through is not None:
                raise ValueError(
                    f"DiskColumn creates {column.keys!r}, which passes through {through.label}, "
                    f"a column ({np.dtype(through.dtype)}{list(through.shape[1:])}) and not a "
                    "group; name another column"
                )


def _refuse_a_spelling_beside_a_field(  # noqa: DOC502 - create_path raises
    columns: Sequence[DiskColumn], fields: Mapping[KeyTuple, DiskColumn]
) -> None:
    """Refuse a string name that would extend another column's root field, offering its key tuple.

    Args:
        columns: The source's columns.
        fields: The columns by key tuple.

    Raises:
        ValueError: From :func:`create_path`, naming the field and the key tuple to use.
    """
    root = {keys[0]: c.row_spec for keys, c in fields.items() if len(keys) == 1}
    for column in columns:
        if isinstance(column.name, str):
            others = {key: spec for key, spec in root.items() if key != column.name}
            create_path(others, column.name, owner="DiskColumn", param="name")


@dataclasses.dataclass(frozen=True)
class _Nesting:
    """Where each column sits in the record, derived once from the columns' key tuples.

    The record's tree is static, so a read places its arrays by JAX's own unflatten rather than
    rebuilding the tree from the key tuples each time.
    """

    structure: Any  # a jax PyTreeDef; jaxlib ships no type stub that names it
    order: tuple[int, ...]

    @classmethod
    def of(cls, columns: Sequence[DiskColumn]) -> _Nesting:
        """The record's tree of ``columns``, and the column at each of its leaves.

        Args:
            columns: The columns, checked to make one record.

        Returns:
            The nesting.
        """
        positions = fields_nested({column.keys: i for i, column in enumerate(columns)})
        order, structure = jax.tree_util.tree_flatten(positions)
        return cls(structure, tuple(order))

    def nested(self, values: Sequence[Any]) -> Any:
        """``values``, one per column in the columns' order, nested as the record.

        Args:
            values: One value per column.

        Returns:
            The record's nested dictionary of them.
        """
        return jax.tree_util.tree_unflatten(self.structure, [values[i] for i in self.order])


def _provenance_files(
    provenance: DiskProvenance | None, length: int
) -> ArrayRecordDataSource | None:
    """The provenance file's reader, held on the host, checked to hold one record per record.

    ArrayRecord's data source opens the file at its first read and pickles without an open
    reader, so a copy sent to a worker process reopens it where it reads.

    Args:
        provenance: The provenance on disk, or ``None``.
        length: The source's records.

    Returns:
        The reader, or ``None`` without provenance.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If it holds another number of records than ``length``.
    """
    if provenance is None:
        return None
    if not Path(provenance.file).is_file():
        raise FileNotFoundError(
            f"StreamingDiskSource: provenance file {provenance.file} is missing"
        )
    files = ArrayRecordDataSource([provenance.file])
    held = len(files)
    if held != length or provenance.records != length:
        raise ValueError(
            f"{provenance.file} holds {held} records' provenance ({provenance.records} declared), "
            f"but the columns hold {length} records"
        )
    return files


class StreamingDiskSource(DataSourceModule):
    """An indexed source over columns of arrays on disk, larger than RAM, read by memory maps.

    ``get_batch`` reads rows on the host; ``get_records`` reads them inside a compiled program
    through ``io_callback``.
    """

    config: StreamingDiskSourceConfig  # pyright: ignore[reportIncompatibleVariableOverride]

    @property
    def record_identity(self) -> RecordIdentity:
        """A record's index is its row in the columns."""
        return RecordIdentity.INDEXED

    def __init__(
        self,
        config: StreamingDiskSourceConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Check every column's file holds it; the maps open at first use.

        Args:
            config: The source's columns.
            rngs: Unused; accepted as every source accepts it.
            name: Optional module name.

        Raises:
            FileNotFoundError: If a column's file, or the provenance file, does not exist.
            ValueError: If a column's file is shorter than its offset plus its bytes, or the
                provenance file holds another number of records than the columns.
        """
        super().__init__(config, rngs=rngs, name=name)
        for column in config.columns:
            path = Path(column.file)
            if not path.is_file():
                raise FileNotFoundError(
                    f"StreamingDiskSource: column {column.label}'s file {path} does not exist"
                )
            size = path.stat().st_size
            if size < column.offset + column.nbytes:
                raise ValueError(
                    f"column {column.label}: {path} holds {size} bytes, below its offset "
                    f"{column.offset} plus its {column.nbytes} bytes "
                    f"({column.shape} of {np.dtype(column.dtype)})"
                )
        # The memory maps stay on the host, outside NNX state; the rest is static metadata.
        self._hosts = HostArrays(
            tuple(
                HostArray(c.file, offset=c.offset, dtype=c.dtype, shape=c.shape)
                for c in config.columns
            )
        )
        self._length = nnx.static(config.columns[0].length)
        self._nesting = nnx.static(_Nesting.of(config.columns))
        self._provenance_files = _provenance_files(config.provenance, self._length)

    def __len__(self) -> int:
        """Total number of records on disk."""
        return self._length

    def order_working_bytes(self, order: BlockOrder) -> int:
        """A buffer of ``order``'s records as stored: every column's row bytes per record.

        Args:
            order: The block order a run reads in.

        Returns:
            The bytes a buffer of records keeps in the page cache.
        """
        row_bytes = sum(column.nbytes // column.shape[0] for column in self.config.columns)
        return buffer_bytes(order, self._length, row_bytes)

    def __repr__(self) -> str:
        """Config-identifying representation for checkpoint validation.

        Names every column's field, file, offset, dtype and shape, and the record count, so a
        restore can detect a source pointed at different data.
        """
        columns = ", ".join(
            f"{c.label}: {c.file}@{c.offset} {np.dtype(c.dtype)}{list(c.shape)}"
            for c in self.config.columns
        )
        return f"StreamingDiskSource(columns=[{columns}], length={self._length})"

    def _columns(self) -> dict[str, Any]:
        """Every column's memory map, nested by the columns' key tuples."""
        return self._nesting.nested([host.array for host in self._hosts.arrays])

    def provenance(  # noqa: DOC502 - record_words and host_rows raise
        self, indices: ArrayLike
    ) -> tuple[Mapping[str, Any], ...]:
        """The provenance of the records ``indices`` names, read from the provenance file by index.

        Args:
            indices: uint32 ``(n, 2)`` record indices.

        Returns:
            Each named record's read-only mapping of strings and objects, in the order named; an
            empty mapping for every record of a source without a provenance file.

        Raises:
            ValueError: If ``indices`` are not uint32 ``(n, 2)`` words.
            IndexError: If an index is the padding index or outside the columns.
        """
        rows = host_rows(record_words(indices), self._length)
        files = self._provenance_files
        if files is None:
            return (NO_PROVENANCE,) * len(rows)
        return tuple(
            MappingProxyType(decode_provenance(record))
            for record in files.__getitems__([int(row) for row in rows])
        )

    def source_identity(self) -> dict[str, Any]:
        """What the source reads, never where: its columns, and the copy it is, if one.

        Returns:
            ``{"source": ..., "decoder": {}}``: each column's field (its key tuple), dtype, shape,
            offset, file base name and size, and the decoded copy's content digest when the
            columns are one; nothing decodes, so no decoder library.
        """
        columns = [
            {
                "keys": list(c.keys),
                "dtype": c.dtype,
                "shape": list(c.shape),
                "offset": c.offset,
                "file": Path(c.file).name,
                "bytes": Path(c.file).stat().st_size,
            }
            for c in self.config.columns
        ]
        return {
            "source": {
                "kind": "disk_columns",
                "columns": columns,
                "content": self.config.content_digest,
            },
            "decoder": {},
        }

    def element_spec(self) -> Any:
        """One record's spec as the device holds it: each column's leading axis stripped.

        Returns:
            Nested ``jax.ShapeDtypeStruct`` per field, nested by the columns' key tuples, its
            dtype as JAX arrays hold the stored one (``device_spec``).
        """
        return device_spec(
            self._nesting.nested([column.row_spec for column in self.config.columns])
        )

    def get_batch(  # noqa: DOC502 - read_host_batch raises
        self, indices: ArrayLike, *, epochs: ArrayLike = 0, contiguous: bool = False
    ) -> Batch:
        """Read the rows ``indices`` names from every column, as a ``Batch``, on the host.

        The eager sources' host read (:func:`~datarax.sources.eager_source.read_host_batch`): one
        NumPy gather of the named rows, the ``Batch`` named with the given words and epochs, no
        state read or changed and no device array created. With ``contiguous=True`` the caller
        states that ``indices`` is a run of consecutive rows, read as views of the memory maps.

        Args:
            indices: uint32 ``(n, 2)`` row indices, each as its words ``(hi, lo)``.
            epochs: The epoch of every record, or of each ``(n,)``.
            contiguous: Whether ``indices`` is a run of consecutive rows.

        Returns:
            The named rows of every column, nested by the columns' key tuples, as a host
            ``Batch``.

        Raises:
            ValueError: If ``indices`` are not uint32 ``(n, 2)`` words, or a run declared
                contiguous is not one.
            IndexError: If an index is the padding index or outside the columns.
        """
        return read_host_batch(
            self._columns(), self._length, indices, epochs=epochs, contiguous=contiguous
        )

    def get_records(self, indices: jax.Array) -> DataDict:
        """Fetch the rows at ``indices`` from disk and return them ``stop_gradient``-wrapped.

        Args:
            indices: uint32 ``(n, 2)`` indices into the columns, each as its words ``(hi, lo)``,
                so every row of a column past ``2**32`` rows is addressed.

        Returns:
            Every column's rows at ``indices``, nested by the columns' key tuples, each
            ``(len(indices), *row_shape)`` and wrapped with ``jax.lax.stop_gradient``.
        """
        columns = self.config.columns
        # A callback returns what JAX arrays hold: a stored 64-bit column at 32 bits while x64 is
        # off, as the eager sources' traced read gives it.
        dtypes = tuple(jax.dtypes.canonicalize_dtype(np.dtype(c.dtype)) for c in columns)
        result_spec = tuple(
            jax.ShapeDtypeStruct((indices.shape[0], *column.shape[1:]), dtype)
            for column, dtype in zip(columns, dtypes, strict=True)
        )
        hosts = self._hosts.arrays

        def _host_read(idx_array: np.ndarray) -> tuple[np.ndarray, ...]:
            # ``idx_array`` arrives as a numpy array on the host; each gather copies the rows.
            rows = from_words(idx_array)
            return tuple(
                np.asarray(host.array[rows], dtype=dtype)
                for host, dtype in zip(hosts, dtypes, strict=True)
            )

        raw = io_callback(_host_read, result_spec, indices)
        # The rows are constants of any gradient: stop_gradient marks them so.
        return self._nesting.nested([jax.lax.stop_gradient(rows) for rows in raw])


__all__ = ["DiskColumn", "StreamingDiskSource", "StreamingDiskSourceConfig"]
