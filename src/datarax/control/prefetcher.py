"""Prefetcher implementation for Datarax.

This module provides thread-based prefetchers that load data in the background
while the main thread processes previously loaded data.

Two-stage pipeline (P2.1):
    CPU prefetch buffer (size=4) → jax.device_put → device buffer (size=2)

Design follows Grain's prefetch pattern:
- Warm start: background loading begins immediately on construction
- Sentinel-based StopIteration: uses _END sentinel, not isinstance(Exception)
- Clean shutdown: stop_event + thread join with timeout
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast, TypeVar


logger = logging.getLogger(__name__)

T = TypeVar("T")

# Sentinel object to signal end-of-iterator through the queue.
# Using a unique object avoids the Grain anti-pattern of putting Exception
# instances in the queue (which breaks if data items are Exception subclasses).
_END = object()


@dataclass(frozen=True)
class _ErrorWrapper:
    """Container for forwarding producer exceptions to the consumer thread."""

    exc: Exception


class _PrefetchIterator(Iterator[T]):
    """Closeable iterator backed by a producer thread and bounded queue.

    Uses threading.Condition for zero-latency wake-up instead of polling
    timeouts. The producer signals immediately when data is available,
    and the consumer wakes within microseconds.
    """

    def __init__(self, iterator: Iterator[T], buffer_size: int) -> None:
        self._source_iterator = iterator
        self._buffer: queue.Queue[object] = queue.Queue(maxsize=buffer_size)
        self._stop_event = threading.Event()
        self._data_available = threading.Condition()
        self._is_closed = False
        self._thread = threading.Thread(target=self._producer, daemon=True)
        self._thread.start()

    def _put_with_stop_awareness(self, item: object) -> bool:
        """Try to enqueue item while reacting quickly to close requests."""
        while not self._stop_event.is_set():
            try:
                self._buffer.put(item, timeout=0.05)
                with self._data_available:
                    self._data_available.notify()
                return True
            except queue.Full:
                continue
        return False

    def _producer(self) -> None:
        try:
            for item in self._source_iterator:
                if self._stop_event.is_set():
                    return
                if not self._put_with_stop_awareness(item):
                    return
        except Exception as exc:  # noqa: BLE001 - propagate arbitrary iterator errors
            self._put_with_stop_awareness(_ErrorWrapper(exc))
        finally:
            self._put_with_stop_awareness(_END)

    def __iter__(self) -> _PrefetchIterator[T]:
        return self

    def __next__(self) -> T:
        if self._is_closed:
            raise StopIteration

        while True:
            try:
                item = self._buffer.get_nowait()
            except queue.Empty:
                if self._is_closed:
                    raise StopIteration from None
                if not self._thread.is_alive():
                    self.close()
                    raise StopIteration from None
                # Wait for producer to signal data availability
                with self._data_available:
                    self._data_available.wait(timeout=0.1)
                continue

            if item is _END:
                self.close()
                raise StopIteration
            if isinstance(item, _ErrorWrapper):
                self.close()
                raise item.exc
            return cast(T, item)

    def _drain_buffer(self) -> None:
        while True:
            try:
                self._buffer.get_nowait()
            except queue.Empty:
                return

    def close(self) -> None:
        """Stop producer thread and release resources deterministically."""
        if self._is_closed:
            return
        self._is_closed = True
        self._stop_event.set()

        close_source = getattr(self._source_iterator, "close", None)
        if callable(close_source):
            with contextlib.suppress(Exception):
                close_source()

        if self._thread.is_alive():
            self._thread.join(timeout=0.2)
            deadline = time.monotonic() + 0.3
            while self._thread.is_alive() and time.monotonic() < deadline:
                self._thread.join(timeout=0.01)

        self._drain_buffer()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()


class Prefetcher:
    """A prefetcher that uses threads to load data in the background.

    Starts loading immediately when prefetch() is called (warm start),
    not lazily on the first __next__() call.
    """

    def __init__(self, buffer_size: int = 2) -> None:
        """Initialize the prefetcher.

        Args:
            buffer_size: Number of items to prefetch ahead.
        """
        self.buffer_size = buffer_size

    def prefetch(self, iterator: Iterator[T]) -> _PrefetchIterator[T]:
        """Prefetch items from the iterator in a background thread.

        The background thread starts loading immediately (warm start).
        Uses a sentinel value for clean end-of-iterator signaling.

        Args:
            iterator: Iterator to prefetch from.

        Returns:
            A closeable iterator that yields prefetched items.
        """
        return _PrefetchIterator(iterator=iterator, buffer_size=self.buffer_size)


class _DevicePutIterator(Iterator[T]):
    """Background thread that transfers CPU data to device via jax.device_put.

    Stage 2 of the two-stage prefetch pipeline. Consumes CPU-side data from
    a source iterator and asynchronously transfers to the default JAX device.
    Uses the same sentinel/condition pattern as _PrefetchIterator.
    """

    def __init__(self, iterator: Iterator[T], buffer_size: int) -> None:
        self._source_iterator = iterator
        self._buffer: queue.Queue[object] = queue.Queue(maxsize=buffer_size)
        self._stop_event = threading.Event()
        self._data_available = threading.Condition()
        self._is_closed = False
        self._thread = threading.Thread(target=self._producer, daemon=True)
        self._thread.start()

    def _device_put(self, item: T) -> T:
        """Transfer a single item to device. Handles dicts, arrays, and pytrees."""
        import jax

        return jax.device_put(item)

    def _put_with_stop_awareness(self, item: object) -> bool:
        """Try to enqueue item while reacting quickly to close requests."""
        while not self._stop_event.is_set():
            try:
                self._buffer.put(item, timeout=0.05)
                with self._data_available:
                    self._data_available.notify()
                return True
            except queue.Full:
                continue
        return False

    def _producer(self) -> None:
        try:
            for item in self._source_iterator:
                if self._stop_event.is_set():
                    return
                device_item = self._device_put(item)
                if not self._put_with_stop_awareness(device_item):
                    return
        except Exception as exc:  # noqa: BLE001 - propagate arbitrary iterator errors
            self._put_with_stop_awareness(_ErrorWrapper(exc))
        finally:
            self._put_with_stop_awareness(_END)

    def __iter__(self) -> _DevicePutIterator[T]:
        return self

    def __next__(self) -> T:
        if self._is_closed:
            raise StopIteration

        while True:
            try:
                item = self._buffer.get_nowait()
            except queue.Empty:
                if self._is_closed:
                    raise StopIteration from None
                if not self._thread.is_alive():
                    self.close()
                    raise StopIteration from None
                with self._data_available:
                    self._data_available.wait(timeout=0.1)
                continue

            if item is _END:
                self.close()
                raise StopIteration
            if isinstance(item, _ErrorWrapper):
                self.close()
                raise item.exc
            return cast(T, item)

    def _drain_buffer(self) -> None:
        while True:
            try:
                self._buffer.get_nowait()
            except queue.Empty:
                return

    def close(self) -> None:
        """Stop producer thread and release resources."""
        if self._is_closed:
            return
        self._is_closed = True
        self._stop_event.set()

        close_source = getattr(self._source_iterator, "close", None)
        if callable(close_source):
            with contextlib.suppress(Exception):
                close_source()

        if self._thread.is_alive():
            self._thread.join(timeout=0.2)
            deadline = time.monotonic() + 0.3
            while self._thread.is_alive() and time.monotonic() < deadline:
                self._thread.join(timeout=0.01)

        self._drain_buffer()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()


class DevicePrefetcher:
    """Two-stage prefetcher: async host-to-device transfer via jax.device_put.

    Follows Grain's ``experimental/device_put/device_put.py`` pattern.
    Compose with ``Prefetcher`` for full two-stage pipeline::

        raw_iter → Prefetcher(buffer=4) → DevicePrefetcher(buffer=2) → consumer

    The consumer receives JAX arrays already on device, overlapping the
    H2D transfer with computation on the previous batch.
    """

    def __init__(self, buffer_size: int = 2) -> None:
        """Initialize the device prefetcher.

        Args:
            buffer_size: Number of device-side batches to buffer ahead.
        """
        self.buffer_size = buffer_size

    def prefetch(self, iterator: Iterator[T]) -> _DevicePutIterator[T]:
        """Begin async device transfer from the given iterator.

        Starts a background thread that calls ``jax.device_put`` on each
        item and buffers the results. The returned iterator yields
        device-resident JAX arrays.

        Args:
            iterator: Iterator of CPU-side data (numpy arrays, dicts, pytrees).

        Returns:
            A closeable iterator yielding device-resident data.
        """
        return _DevicePutIterator(iterator=iterator, buffer_size=self.buffer_size)

    def start_prefetch(self, iterator: Iterator[T]) -> _DevicePutIterator[T]:
        """Begin prefetching immediately (warm-start API, P2.3).

        Identical to ``prefetch()`` — the background thread starts on
        construction. Call this during checkpoint recovery or model compilation
        so the data pipeline warms up while other initialization completes.

        Args:
            iterator: Iterator of CPU-side data.

        Returns:
            A closeable iterator to consume when ready.
        """
        return self.prefetch(iterator)
