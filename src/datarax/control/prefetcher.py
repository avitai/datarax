"""Prefetcher implementation for Datarax.

This module provides prefetcher classes that can load data in the background
while the main thread processes previously loaded data.
"""

import queue
import threading
from typing import Iterator, TypeVar


T = TypeVar("T")


class Prefetcher:
    """A prefetcher that uses threads to load data in the background."""

    def __init__(self, buffer_size: int = 2):
        """Initialize the prefetcher.

        Args:
            buffer_size: Number of items to prefetch.
        """
        self.buffer_size = buffer_size

    def prefetch(self, iterator: Iterator[T]) -> Iterator[T]:
        """Prefetch items from the iterator in the background.

        Args:
            iterator: Iterator to prefetch from.

        Returns:
            An iterator that yields prefetched items.
        """
        # Create a queue to hold prefetched items
        buffer: queue.Queue[T] = queue.Queue(maxsize=self.buffer_size)
        # Flag to signal when the iterator is exhausted
        end_of_data = threading.Event()
        # Flag to signal when to stop the prefetching thread
        stop_event = threading.Event()

        # Function to populate the queue in a background thread
        def prefetch_fn():
            try:
                for item in iterator:
                    if stop_event.is_set():
                        break
                    buffer.put(item)
                # Signal that we've reached the end of the iterator
                end_of_data.set()
            except Exception as e:
                # If an exception occurs, put it in the queue
                buffer.put(e)
                end_of_data.set()

        # Start the prefetching thread
        prefetch_thread = threading.Thread(target=prefetch_fn, daemon=True)
        prefetch_thread.start()

        # Yield items from the queue until the iterator is exhausted
        try:
            while not end_of_data.is_set() or not buffer.empty():
                try:
                    item = buffer.get(timeout=0.1)
                    # If the item is an exception, raise it
                    if isinstance(item, Exception):
                        raise item
                    yield item
                except queue.Empty:
                    # If the queue is empty but the iterator isn't exhausted,
                    # try again
                    continue
        finally:
            # Signal the prefetching thread to stop
            stop_event.set()
            # Wait for the thread to finish
            prefetch_thread.join(timeout=1.0)
