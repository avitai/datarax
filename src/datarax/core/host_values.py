"""Host values: what a source or pipeline keeps on the host, out of NNX state and out of programs.

A holder is a plain object, so NNX treats it as a static of the module graph: it is not module
state, never a jit argument and never traced. Its equality decides whether two modules share a
graphdef, and with it a compiled program. ``HostValue`` and its subclasses compare equal by kind,
because nothing a program runs reads what they hold. ``HostArray`` and ``HostArrays`` compare
equal only to themselves, because a compiled program's host callback reads their memory maps.

Equality reflects what compiled code reads: no program takes a kind-equal holder off a module,
reads what it holds, or captures it in a host callback, and a holder a program must read compares
by identity.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

import numpy as np


class HostValue:
    """A value a source keeps on the host, out of NNX state and out of every program.

    As a plain object it is static to NNX: it is not module state, never a jit argument and never
    traced. Holders of one kind compare equal whatever they hold, so two sources differing only in
    such values share one graphdef and one compiled program; nothing traced reads them.

    Attributes:
        value: The value held.
    """

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        """Hold ``value``.

        Args:
            value: The value.
        """
        self.value = value

    def __eq__(self, other: object) -> bool:
        """Every holder of a kind is equal to every other: what it holds decides no program."""
        return type(other) is type(self)

    def __hash__(self) -> int:
        """One hash per kind of holder, as ``__eq__`` requires."""
        return hash(type(self))


class HostProvenance(HostValue):
    """The non-array part of a source's records, held on the host and out of NNX state.

    Attributes:
        value: One immutable mapping per record, aligned with the rows, or empty when the
            records carry nothing but arrays.
    """

    __slots__ = ()
    value: tuple[Mapping[str, Any], ...]


class HostColumns(HostValue):
    """An in-memory source's columns, held on the host and out of NNX state.

    Its records are never NNX state, so a split of a module holding the source carries no
    dataset-sized leaf, and sources that differ only in their data share one graph definition
    and one compiled program. Nothing a program runs reads the holder: a compiled program takes
    the columns' device copy as an explicit argument.

    Attributes:
        value: The columns, NumPy arrays sharing a leading record axis.
    """

    __slots__ = ()
    value: Any


class HostRagged(HostValue):
    """An in-memory source's ragged fields, held on the host and out of NNX state, as its columns.

    Attributes:
        value: Each ragged field's ``RaggedColumn`` (``datarax.core.ragged``), by field name.
    """

    __slots__ = ()
    value: Mapping[str, Any]


class HostArray:
    """A memory map of an array stored raw in a file at an offset, held on the host, not in NNX.

    A memory map stored as ``nnx.data`` would become a traced input of every compiled step, and
    the host callback would then close over a tracer. As a plain object it stays static and
    compares equal only to itself. It pickles as its place on disk (file, offset, dtype and
    shape), so a copy sent to a worker process carries no rows, and it opens its map at first use,
    once, however many threads read it at once.

    Attributes:
        path: The file the array is stored in.
        offset: The byte offset of the array's first byte in the file.
        dtype: The stored dtype, a NumPy dtype string.
        shape: The array's shape.
    """

    __slots__ = ("_array", "_lock", "dtype", "offset", "path", "shape")

    def __init__(self, path: str, *, offset: int, dtype: str, shape: tuple[int, ...]) -> None:
        """Hold the array's place on disk; the map opens at first use.

        Args:
            path: The file.
            offset: The byte offset of the array's first byte.
            dtype: The stored dtype.
            shape: The array's shape.
        """
        self.path = path
        self.offset = offset
        self.dtype = dtype
        self.shape = shape
        self._array: np.ndarray | None = None
        self._lock = threading.Lock()

    @property
    def array(self) -> np.ndarray:
        """The memory map, opened read-only at first use."""
        array = self._array
        if array is None:
            with self._lock:
                if self._array is None:
                    self._array = np.memmap(
                        self.path,
                        dtype=np.dtype(self.dtype),
                        mode="r",
                        offset=self.offset,
                        shape=self.shape,
                    )
                array = self._array
        return array

    def __getstate__(self) -> dict[str, Any]:
        """The array's place on disk alone: a copy maps the file again where it is used."""
        return {"path": self.path, "offset": self.offset, "dtype": self.dtype, "shape": self.shape}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Hold the place on disk; the map opens at first use."""
        self.__init__(
            state["path"], offset=state["offset"], dtype=state["dtype"], shape=state["shape"]
        )


class HostArrays:
    """A source's memory maps (:class:`HostArray`), held as one static of its graph.

    A tuple of them would be a node of the graph, its maps statics of the graph definition, which
    a program cached by source structure would keep; this holder is one static, which the host
    naming's stand-in replaces (``datarax.pipeline.epochs``). It compares equal only to itself, as
    the maps it holds do.

    Attributes:
        arrays: The maps, in the source's order.
    """

    __slots__ = ("arrays",)

    def __init__(self, arrays: tuple[HostArray, ...]) -> None:
        """Hold ``arrays``.

        Args:
            arrays: The maps.
        """
        self.arrays = arrays


__all__ = ["HostArray", "HostArrays", "HostColumns", "HostProvenance", "HostValue"]
