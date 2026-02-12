"""Prefetcher implementation for Datarax.

This module provides a thread-based prefetcher that loads data in the background
while the main thread processes previously loaded data.

Design follows Grain's prefetch pattern:
- Warm start: background loading begins immediately on construction
- Sentinel-based StopIteration: uses _END sentinel, not isinstance(Exception)
- Clean shutdown: stop_event + thread join with timeout
"""

import queue
import threading
from typing import TypeVar
from collections.abc import Iterator


T = TypeVar("T")

# Sentinel object to signal end-of-iterator through the queue.
# Using a unique object avoids the Grain anti-pattern of putting Exception
# instances in the queue (which breaks if data items are Exception subclasses).
_END = object()

# Wrapper to propagate exceptions from the producer thread.
_ERROR = type("_ErrorWrapper", (), {"__init__": lambda self, e: setattr(self, "exc", e)})


class Prefetcher:
    """A prefetcher that uses threads to load data in the background.

    Starts loading immediately when prefetch() is called (warm start),
    not lazily on the first __next__() call.
    """

    def __init__(self, buffer_size: int = 2):
        """Initialize the prefetcher.

        Args:
            buffer_size: Number of items to prefetch ahead.
        """
        self.buffer_size = buffer_size

    def prefetch(self, iterator: Iterator[T]) -> Iterator[T]:
        """Prefetch items from the iterator in a background thread.

        The background thread starts loading immediately (warm start).
        Uses a sentinel value for clean end-of-iterator signaling.

        Args:
            iterator: Iterator to prefetch from.

        Returns:
            An iterator that yields prefetched items.
        """
        buf: queue.Queue = queue.Queue(maxsize=self.buffer_size)
        stop_event = threading.Event()

        def producer():
            try:
                for item in iterator:
                    if stop_event.is_set():
                        return
                    buf.put(item)
            except Exception as e:
                buf.put(_ERROR(e))
            finally:
                buf.put(_END)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        try:
            while True:
                try:
                    item = buf.get(timeout=0.5)
                except queue.Empty:
                    if not thread.is_alive():
                        break
                    continue

                if item is _END:
                    break
                if isinstance(item, _ERROR):
                    raise item.exc
                yield item
        finally:
            stop_event.set()
            thread.join(timeout=2.0)
