"""Iteration-boundary caching for expensive pipelines.

Mirrors the semantics of ``grain.experimental.CacheIterDataset`` (reference:
grain ``_src/python/dataset/transformations/cache.py``) at datarax's JAX
iteration boundary: the first pass populates an in-memory cache from the wrapped
iterator, and re-iteration replays the cached elements, so an expensive upstream
pipeline runs its work once and later epochs reuse it.

This is an *iteration-level* wrapper, not a ``Pipeline.from_dag`` node: caching a
stage inside the jitted DAG would require data-dependent control flow within the
trace, which JAX cannot express. Caching at the iteration boundary (outside
``jit``) is the correct layer, matching grain's iterator-level design.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Generic, TypeVar


T = TypeVar("T")


class CachingIterator(Iterator[T], Generic[T]):
    """Caches a wrapped iterator's elements in memory for reuse across passes.

    Each ``__iter__`` resets the read position to the start. ``__next__`` serves
    already-cached positions from memory and computes-then-caches new ones from
    the wrapped iterator. This matches grain's ``CacheIterDataset`` core
    behaviour — cached positions come from memory, uncached ones are computed
    exactly once — without its checkpoint state machine, which the benchmark
    pipelines do not require. Suitable when the data fits in memory.
    """

    def __init__(self, source: Iterator[T]) -> None:
        """Wrap ``source``, caching each element the first time it is produced.

        Args:
            source: The iterator whose elements are cached on first read.
        """
        self._source = source
        self._cache: list[T] = []
        self._position = 0

    def __iter__(self) -> CachingIterator[T]:
        """Restart from the first cached element."""
        self._position = 0
        return self

    def close(self) -> None:
        """Forward close to the wrapped iterator, releasing its resources.

        The wrapped iterator may hold session state (for example a
        pipeline iteration session whose module write-back happens on
        close); forwarding keeps that contract intact.
        """
        close = getattr(self._source, "close", None)
        if callable(close):
            close()

    def __next__(self) -> T:
        """Return the next element, from cache when available else from the source.

        Returns:
            The element at the current position.

        Raises:
            StopIteration: When an uncached position exhausts the wrapped source.
        """
        if self._position < len(self._cache):
            value = self._cache[self._position]
        else:
            value = next(self._source)
            self._cache.append(value)
        self._position += 1
        return value
