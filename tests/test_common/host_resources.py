"""What a process still holds of files a dropped source read: objects, descriptors, mappings.

A source's host resources (a reader, an open file, a memory map) must go when the source and its
pipelines go, whatever a compiled-program cache keeps. These helpers look for them in the live
objects and in the process's ``/proc`` tables (Linux).
"""

from __future__ import annotations

import gc
import os
import threading
from collections.abc import Callable, Iterable
from pathlib import Path


def open_descriptors(paths: Iterable[str | Path]) -> list[str]:
    """The files among ``paths`` this process holds an open descriptor on."""
    wanted = {os.path.realpath(path) for path in paths}
    held = []
    for name in os.listdir("/proc/self/fd"):
        try:
            target = os.path.realpath(f"/proc/self/fd/{name}")
        except OSError:  # the descriptor closed while listed
            continue
        if target in wanted:
            held.append(target)
    return held


def mapped(paths: Iterable[str | Path]) -> list[str]:
    """The files among ``paths`` this process holds a memory mapping of."""
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
