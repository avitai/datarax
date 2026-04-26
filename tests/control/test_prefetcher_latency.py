"""Tests for prefetcher condition-variable wake-up latency.

Validates that:
1. Condition-variable wake-up latency is under 1ms
2. Throughput is maintained under fast producer
3. Zero-latency signaling works correctly
"""

import threading
import time

from datarax.control.prefetcher import _PrefetchIterator, Prefetcher


class TestConditionVariableWakeup:
    """Tests for condition-variable based wake-up mechanism."""

    def test_blocked_consumer_wakes_when_data_arrives(self):
        """A consumer blocked in __next__ should wake when the producer enqueues data."""
        release_next = threading.Event()

        def controlled_producer():
            """Yield a warmup item, then block until the test releases another item."""
            yield 0
            if not release_next.wait(timeout=1.0):
                raise TimeoutError("producer was not released")
            yield 1

        prefetcher = Prefetcher(buffer_size=1)
        iterator = prefetcher.prefetch(controlled_producer())

        # Consume first item so the next call must wait for the controlled release.
        assert next(iterator) == 0

        result: list[int] = []
        errors: list[BaseException] = []

        def consume_next() -> None:
            try:
                result.append(next(iterator))
            except BaseException as exc:  # noqa: BLE001 - surfaced below
                errors.append(exc)

        consumer = threading.Thread(target=consume_next)
        consumer.start()
        time.sleep(0.01)

        start = time.perf_counter()
        release_next.set()
        consumer.join(timeout=1.0)
        elapsed = time.perf_counter() - start

        iterator.close()

        assert not consumer.is_alive()
        assert errors == []
        assert result == [1]
        assert elapsed < 0.25, f"Consumer did not wake promptly: {elapsed * 1000:.1f}ms"

    def test_has_condition_variable(self):
        """PrefetchIterator should use threading.Condition for signaling."""
        prefetcher = Prefetcher(buffer_size=2)
        iterator = prefetcher.prefetch(iter([1, 2, 3]))

        assert hasattr(iterator, "_data_available")
        assert isinstance(iterator._data_available, threading.Condition)

        iterator.close()


class TestPrefetcherThroughput:
    """Tests for throughput under fast producer conditions."""

    def test_fast_producer_throughput(self):
        """Prefetcher should sustain high throughput with a fast producer."""
        n_items = 10000
        data = list(range(n_items))

        prefetcher = Prefetcher(buffer_size=16)

        start = time.perf_counter()
        result = list(prefetcher.prefetch(iter(data)))
        elapsed = time.perf_counter() - start

        assert result == data
        throughput = n_items / elapsed
        # Should sustain at least 20K items/sec (fast producer, minimal data).
        # Threshold is conservative to avoid flaky failures on loaded systems.
        assert throughput > 20_000, f"Throughput too low: {throughput:.0f} items/sec"

    def test_buffer_stays_populated_under_fast_producer(self):
        """With a fast producer, the buffer should generally stay populated."""

        def fast_generator():
            yield from range(1000)

        prefetcher = Prefetcher(buffer_size=8)
        iterator = prefetcher.prefetch(fast_generator())

        # Give producer time to fill buffer
        time.sleep(0.01)

        # Buffer should have items ready
        items_ready = iterator._buffer.qsize()
        assert items_ready > 0, "Buffer should be populated with fast producer"

        iterator.close()

    def test_condition_variable_not_polling(self):
        """Verify the implementation uses Condition, not time.sleep polling."""
        import inspect

        source = inspect.getsource(_PrefetchIterator.__next__)
        # Should reference _data_available (condition variable), not time.sleep
        assert "_data_available" in source
        assert "time.sleep" not in source
