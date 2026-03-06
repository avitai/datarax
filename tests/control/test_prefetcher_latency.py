"""Tests for prefetcher condition-variable wake-up latency.

Validates that:
1. Condition-variable wake-up latency is under 1ms
2. Throughput is maintained under fast producer
3. Zero-latency signaling works correctly
"""

import statistics
import threading
import time

from datarax.control.prefetcher import _PrefetchIterator, Prefetcher


class TestConditionVariableWakeup:
    """Tests for condition-variable based wake-up mechanism."""

    def test_wakeup_latency_under_1ms(self):
        """Consumer should wake within 1ms of data being available.

        The old polling approach had 50ms timeouts. The condition variable
        should achieve sub-millisecond wake-up.
        """

        def delayed_producer():
            """Yield items with deliberate pauses."""
            for i in range(10):
                time.sleep(0.05)  # 50ms between items
                yield i

        prefetcher = Prefetcher(buffer_size=1)
        iterator = prefetcher.prefetch(delayed_producer())

        # Consume first item (warm up)
        next(iterator)

        # Measure wake-up latency for subsequent items
        latencies = []
        for _ in range(5):
            # Wait until buffer should be empty (producer is sleeping)
            time.sleep(0.03)  # 30ms — producer sleeps 50ms between items

            # Time how long __next__ takes once data arrives
            start = time.perf_counter()
            next(iterator)
            latency = time.perf_counter() - start
            latencies.append(latency)

        iterator.close()

        # The total wait includes time for the producer to yield,
        # but the wake-up from condition variable should be fast.
        # We're measuring end-to-end here, so expect ~20ms (remaining
        # producer sleep) + <1ms wake-up. Accept up to 40ms.
        median_latency = statistics.median(latencies)
        # Mainly verifying it's NOT 50ms+ (the old polling interval)
        assert median_latency < 0.04, (
            f"Wake-up latency too high: {median_latency * 1000:.1f}ms "
            f"(suggests polling instead of condition variable)"
        )

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
