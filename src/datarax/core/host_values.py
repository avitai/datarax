"""Host values: what a source or pipeline keeps on the host, out of NNX state and out of programs.

A holder is a plain object, so NNX treats it as a static of the module graph: it is not module
state, never a jit argument and never traced. Its equality decides whether two modules share a
graphdef, and with it a compiled program. ``HostValue`` and its subclasses compare equal by kind,
because nothing a program runs reads what they hold. ``HostArray`` compares equal only to itself,
because a compiled program's host callback reads its memory map.
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


class HostArray:
    """A memory map of an on-disk array, kept on the host and out of NNX state.

    A memory map stored as ``nnx.data`` would become a traced input of every compiled step, and
    the host callback would then close over a tracer. As a plain object it stays static and
    compares equal only to itself. It pickles as its path, so a copy sent to a worker process
    carries no rows, and it opens its map at first use, once, however many threads read it at
    once.
    """

    __slots__ = ("_array", "_lock", "path")

    def __init__(self, path: str, array: np.ndarray | None = None) -> None:
        """Hold the map of the array at ``path``.

        Args:
            path: The ``.npy`` file.
            array: The map when the caller has already opened it; otherwise it opens at first use.
        """
        self.path = path
        self._array = array
        self._lock = threading.Lock()

    @property
    def array(self) -> np.ndarray:
        """The memory map, opened read-only at first use."""
        array = self._array
        if array is None:
            with self._lock:
                if self._array is None:
                    self._array = np.load(self.path, mmap_mode="r")
                array = self._array
        return array

    def __getstate__(self) -> dict[str, str]:
        """The path alone: a copy opens the file again where it is used."""
        return {"path": self.path}

    def __setstate__(self, state: dict[str, str]) -> None:
        """Hold the path; the map opens at first use."""
        self.path = state["path"]
        self._array = None
        self._lock = threading.Lock()


__all__ = ["HostArray", "HostProvenance", "HostValue"]
