"""Compiled programs shared across pipelines whose structures are equal.

A program traced for one pipeline serves every pipeline built the same way, so a loop that builds
pipelines (an evaluation per epoch, a sweep) compiles once per structure rather than once per
pipeline. The caches are lists matched by equality of their keys (graphs holding lists are
unhashable), held outside the modules: a graphdef stored as a module attribute would be embedded
in the next split's graphdef, making ``GraphDef.__eq__`` recurse into itself.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


MAX_COMPILED_PROGRAMS = 16
"""Programs one cache keeps; past it the least recently used is dropped."""


def cached_program[P](cache: list[tuple[Any, P]], key: Any, build: Callable[[], P]) -> P:
    """Return the compiled program cached under ``key``, building it on a miss.

    Keys are compared by equality, so structurally identical pipelines share one program. The
    most recently used entry moves last; the oldest is dropped past
    :data:`MAX_COMPILED_PROGRAMS`.

    Args:
        cache: The cache, a list of ``(key, program)`` entries.
        key: What the program was compiled for.
        build: Builds the program (or the programs) on a miss.

    Returns:
        What ``build`` built for ``key``.
    """
    for index, (cached_key, program) in enumerate(cache):
        if cached_key == key:
            cache.append(cache.pop(index))
            return program
    program = build()
    cache.append((key, program))
    if len(cache) > MAX_COMPILED_PROGRAMS:
        cache.pop(0)
    return program


__all__ = ["MAX_COMPILED_PROGRAMS", "cached_program"]
