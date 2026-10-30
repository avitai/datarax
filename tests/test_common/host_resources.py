"""What a process still holds of files a dropped source read: objects, descriptors, mappings.

A source's host resources (a reader, an open file, a memory map) must go when the source and its
pipelines go, whatever a compiled-program cache keeps. These helpers look for them in the live
objects, in the process's descriptors (``/dev/fd``, on Linux and macOS) and in its memory maps
(``/proc``, Linux only: :data:`needs_proc` marks a test that reads them).
"""

from __future__ import annotations

import fcntl
import gc
import os
import threading
from collections.abc import Callable, Iterable
from pathlib import Path

import pytest


needs_proc = pytest.mark.skipif(
    not Path("/proc/self/maps").exists(),
    reason="reads this process's memory maps from /proc, which only Linux has",
)
"""For a test that calls :func:`mapped`."""


def descriptor_path(fd: int) -> str:
    """The path open descriptor ``fd`` names: ``fcntl(F_GETPATH)`` on macOS, ``/proc`` on Linux.

    Raises:
        OSError: If ``fd`` is not open.
    """
    get_path = getattr(fcntl, "F_GETPATH", None)
    if get_path is not None:
        return os.fsdecode(fcntl.fcntl(fd, get_path, bytes(1024)).split(b"\0", 1)[0])
    return os.readlink(f"/proc/self/fd/{fd}")


def open_descriptors(paths: Iterable[str | Path]) -> list[str]:
    """The files among ``paths`` this process holds an open descriptor on."""
    wanted = {os.path.realpath(path) for path in paths}
    held = []
    for name in os.listdir("/dev/fd"):
        try:
            target = os.path.realpath(descriptor_path(int(name)))
        except OSError:  # the descriptor closed while listed, such as the listing's own
            continue
        if target in wanted:
            held.append(target)
    return held


def mapped(paths: Iterable[str | Path]) -> list[str]:
    """The files among ``paths`` this process holds a memory mapping of (Linux: ``/proc``)."""
    wanted = {os.path.realpath(path) for path in paths}
    lines = Path("/proc/self/maps").read_text().splitlines()
    return [line.split()[-1] for line in lines if line.split() and line.split()[-1] in wanted]


def live(predicate: Callable[[object], bool]) -> int:
    """How many live objects satisfy ``predicate``."""
    return sum(1 for item in gc.get_objects() if predicate(item))


def released(check: Callable[[], bool], *, seconds: float = 5.0) -> bool:
    """Whether ``check`` holds after collecting, polled while a run's read thread exits."""
    for _ in range(int(seconds / 0.05)):
        gc.collect()
        if check():
            return True
        threading.Event().wait(0.05)
    return False
