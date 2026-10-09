"""Out-of-core indexed source over columns of arrays stored raw on disk, read through memory maps.

A column is an array stored raw in a file at a byte offset, :class:`DiskColumn`
``(name, file, offset, dtype, shape)``, its leading axis the record axis: a ``.npy`` file is the
case whose offset is its header's length (:meth:`DiskColumn.from_npy`), an uncompressed contiguous
HDF5 dataset at the offset its file reports is another. A column's name is its field's path, with
``/`` between levels, so the source's records keep their nesting. Every column is opened with
``np.memmap`` for zero-copy random access, and two reads serve the source:

- the host read, ``get_batch(indices, *, epochs=0, contiguous=False)``, the eager sources' read
  (:func:`~datarax.sources.eager_source.read_host_batch`): a NumPy gather of the named rows of
  every column (a contiguous run as views of them), returned as a ``Batch`` named with the given
  words and epochs, creating no device array;
- the traced read, ``get_records(indices)``, which issues the disk read through
  ``jax.experimental.io_callback`` for the compiled pipeline session, and wraps the rows in
  ``jax.lax.stop_gradient`` to mark them as constants: the callback is given integer indices,
  which are never differentiated, so ``jax.grad`` over parameters works with or without it.
"""

from __future__ import annotations

import dataclasses
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import jax
import numpy as np
from flax import nnx
from jax.experimental import io_callback
from jax.typing import ArrayLike

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule, RecordIdentity
from datarax.core.element_batch import Batch
from datarax.core.host_values import HostArray, HostArrays
from datarax.core.index_words import from_words
from datarax.core.spec import device_spec, JAX_ARRAY_KINDS
from datarax.sources._field_paths import LEVEL, nested
from datarax.sources.eager_source import read_host_batch
from datarax.typing import DataDict


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DiskColumn:
    """One column of a :class:`StreamingDiskSource`: an array stored raw in a file at an offset.

    Attributes:
        name: The field's path, with ``/`` between levels (``"meta/score"``).
        file: The file holding the array's bytes.
        offset: The byte offset of the array's first byte in the file.
        dtype: The stored dtype (a NumPy dtype string), numeric.
        shape: The whole array's shape, its leading axis the record axis.
    """

    name: str
    file: str
    offset: int
    dtype: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        """Refuse a column no memory map can read.

        Raises:
            ValueError: If the name has an empty level, the offset is negative, the shape has no
                record axis, or the dtype is not numeric.
        """
        if not self.name or any(not level for level in self.name.split(LEVEL)):
            raise ValueError(
                f"a column's name is its field's path, levels joined by {LEVEL!r}; "
                f"{self.name!r} has an empty level"
            )
        if self.offset < 0:
            raise ValueError(f"column {self.name!r}: offset is at least 0; got {self.offset}")
        if not self.shape:
            raise ValueError(f"column {self.name!r} has no record axis: its shape is ()")
        if np.dtype(self.dtype).kind not in JAX_ARRAY_KINDS:
            raise ValueError(
                f"column {self.name!r} stores {np.dtype(self.dtype)}, but a column is numeric; "
                "strings and other objects are a record's provenance"
            )

    @classmethod
    def from_npy(cls, name: str, path: str | os.PathLike[str]) -> DiskColumn:
        """The array a ``.npy`` file stores, as a column: its offset is the header's length.

        Args:
            name: The field's path.
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
    """

    columns: tuple[DiskColumn, ...] = ()

    def __post_init__(self) -> None:
        """Validate the config.

        Raises:
            ValueError: If there is no column, a name is given twice or is both a column and a
                level of another, or the columns hold different record counts.
        """
        super().__post_init__()
        if not self.columns:
            raise ValueError("StreamingDiskSourceConfig needs at least one column")
        _refuse_clashing_names([column.name for column in self.columns])
        lengths = {column.name: column.length for column in self.columns}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"columns hold one row per record; their lengths are {lengths}")


def _refuse_clashing_names(names: list[str]) -> None:
    """Refuse a name given twice, or a column's name that is a level of another's.

    Args:
        names: The columns' names.

    Raises:
        ValueError: Naming the clash.
    """
    twice = [name for name, count in Counter(names).items() if count > 1]
    if twice:
        raise ValueError(f"{twice[0]!r} names two columns; each column has its own name")
    given = set(names)
    for name in names:
        levels = name.split(LEVEL)
        clash = next(
            (
                LEVEL.join(levels[:depth])
                for depth in range(1, len(levels))
                if LEVEL.join(levels[:depth]) in given
            ),
            None,
        )
        if clash is not None:
            raise ValueError(
                f"{clash!r} is a column and a level of {name!r}; a field is one or the other"
            )


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
            FileNotFoundError: If a column's file does not exist.
            ValueError: If a column's file is shorter than its offset plus its bytes.
        """
        super().__init__(config, rngs=rngs, name=name)
        for column in config.columns:
            path = Path(column.file)
            if not path.is_file():
                raise FileNotFoundError(
                    f"StreamingDiskSource: column {column.name!r}'s file {path} does not exist"
                )
            size = path.stat().st_size
            if size < column.offset + column.nbytes:
                raise ValueError(
                    f"column {column.name!r}: {path} holds {size} bytes, below its offset "
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

    def __len__(self) -> int:
        """Total number of records on disk."""
        return self._length

    def __repr__(self) -> str:
        """Config-identifying representation for checkpoint validation.

        Names every column's field, file, offset, dtype and shape, and the record count, so a
        restore can detect a source pointed at different data.
        """
        columns = ", ".join(
            f"{c.name}: {c.file}@{c.offset} {np.dtype(c.dtype)}{list(c.shape)}"
            for c in self.config.columns
        )
        return f"StreamingDiskSource(columns=[{columns}], length={self._length})"

    def _columns(self) -> dict[str, Any]:
        """Every column's memory map, nested by field path."""
        return nested(
            {
                column.name: host.array
                for column, host in zip(self.config.columns, self._hosts.arrays, strict=True)
            }
        )

    def element_spec(self) -> Any:
        """One record's spec as the device holds it: each column's leading axis stripped.

        Returns:
            Nested ``jax.ShapeDtypeStruct`` per field, its dtype as JAX arrays hold the stored
            one (``device_spec``).
        """
        return device_spec(
            nested(
                {
                    column.name: jax.ShapeDtypeStruct(column.shape[1:], np.dtype(column.dtype))
                    for column in self.config.columns
                }
            )
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
            The named rows of every column, nested by field path, as a host ``Batch``.

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
            Every column's rows at ``indices``, nested by field path, each
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
        return nested(
            {
                column.name: jax.lax.stop_gradient(rows)
                for column, rows in zip(columns, raw, strict=True)
            }
        )


__all__ = ["DiskColumn", "StreamingDiskSource", "StreamingDiskSourceConfig"]
