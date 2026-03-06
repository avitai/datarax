"""Tests for DevicePrefetcher — two-stage CPU→device prefetch.

TDD: Tests written first per Section 6.3 P2.1 of the performance audit.
Validates that DevicePrefetcher asynchronously transfers data to device
while the consumer processes previous batches.
"""

import jax
import numpy as np
import pytest

from datarax.control.prefetcher import DevicePrefetcher, Prefetcher


class TestDevicePrefetcherBasic:
    """Basic lifecycle tests for DevicePrefetcher."""

    def test_yields_jax_arrays(self):
        """Output should be JAX arrays on device, not numpy."""
        data = [{"x": np.ones((4, 8), dtype=np.float32)} for _ in range(5)]
        dp = DevicePrefetcher(buffer_size=2)
        results = list(dp.prefetch(iter(data)))

        assert len(results) == 5
        assert isinstance(results[0]["x"], jax.Array)

    def test_preserves_dict_structure(self):
        """Dict keys and shapes must be preserved through transfer."""
        data = [
            {"image": np.zeros((2, 3), dtype=np.float32), "label": np.array([1, 0])}
            for _ in range(3)
        ]
        dp = DevicePrefetcher(buffer_size=2)
        results = list(dp.prefetch(iter(data)))

        assert set(results[0].keys()) == {"image", "label"}
        assert results[0]["image"].shape == (2, 3)
        assert results[0]["label"].shape == (2,)

    def test_empty_iterator(self):
        """Empty input should yield nothing."""
        dp = DevicePrefetcher(buffer_size=2)
        results = list(dp.prefetch(iter([])))
        assert results == []

    def test_single_item(self):
        """Single-item iterator should work."""
        data = [{"x": np.array([1.0, 2.0], dtype=np.float32)}]
        dp = DevicePrefetcher(buffer_size=1)
        results = list(dp.prefetch(iter(data)))
        assert len(results) == 1
        np.testing.assert_allclose(results[0]["x"], [1.0, 2.0])

    def test_non_dict_data(self):
        """Should handle plain numpy arrays (not just dicts)."""
        data = [np.ones((4,), dtype=np.float32) * i for i in range(5)]
        dp = DevicePrefetcher(buffer_size=2)
        results = list(dp.prefetch(iter(data)))

        assert len(results) == 5
        assert isinstance(results[0], jax.Array)
        np.testing.assert_allclose(results[2], np.ones(4) * 2)


class TestDevicePrefetcherComposition:
    """Test composing CPU Prefetcher → DevicePrefetcher (two-stage pipeline)."""

    def test_two_stage_pipeline(self):
        """CPU prefetch → device prefetch should work end-to-end."""
        data = [{"x": np.ones((4,), dtype=np.float32) * i} for i in range(10)]

        cpu_prefetcher = Prefetcher(buffer_size=4)
        device_prefetcher = DevicePrefetcher(buffer_size=2)

        stage1 = cpu_prefetcher.prefetch(iter(data))
        stage2 = device_prefetcher.prefetch(stage1)

        results = list(stage2)
        assert len(results) == 10
        assert isinstance(results[0]["x"], jax.Array)
        np.testing.assert_allclose(results[5]["x"], np.ones(4) * 5)

    def test_two_stage_with_large_batch(self):
        """Two-stage pipeline with realistic batch sizes."""
        batch = {"image": np.random.randint(0, 255, (32, 224, 224, 3), dtype=np.uint8)}
        data = [batch for _ in range(5)]

        cpu_pf = Prefetcher(buffer_size=4)
        dev_pf = DevicePrefetcher(buffer_size=2)

        results = list(dev_pf.prefetch(cpu_pf.prefetch(iter(data))))
        assert len(results) == 5
        assert results[0]["image"].shape == (32, 224, 224, 3)


class TestDevicePrefetcherCleanup:
    """Test resource cleanup and error propagation."""

    def test_close_stops_background_thread(self):
        """Closing should stop the background transfer thread."""

        def infinite_source():
            i = 0
            while True:
                yield {"x": np.ones((4,), dtype=np.float32) * i}
                i += 1

        dp = DevicePrefetcher(buffer_size=2)
        pf_iter = dp.prefetch(infinite_source())
        # Consume one item to start the pipeline
        next(pf_iter)
        pf_iter.close()
        # Should not hang

    def test_error_propagation(self):
        """Errors from source iterator should propagate to consumer."""

        def failing_source():
            yield {"x": np.ones((4,), dtype=np.float32)}
            raise ValueError("source error")

        dp = DevicePrefetcher(buffer_size=2)
        pf_iter = dp.prefetch(failing_source())
        next(pf_iter)  # First item should succeed
        with pytest.raises(ValueError, match="source error"):
            next(pf_iter)


class TestStartPrefetch:
    """Test warm-start API (P2.3)."""

    def test_start_prefetch_returns_closeable_iterator(self):
        """start_prefetch should return an iterator that can be consumed later."""
        data = [{"x": np.ones((4,), dtype=np.float32) * i} for i in range(5)]
        dp = DevicePrefetcher(buffer_size=2)
        handle = dp.start_prefetch(iter(data))

        # Pipeline is warming up in background...
        # Consumer reads when ready
        results = list(handle)
        assert len(results) == 5

    def test_start_prefetch_begins_immediately(self):
        """Background thread should start before consumer reads."""
        import time

        call_times: list[float] = []

        def timed_source():
            for i in range(3):
                call_times.append(time.monotonic())
                yield {"x": np.ones((2,), dtype=np.float32) * i}

        dp = DevicePrefetcher(buffer_size=3)
        handle = dp.start_prefetch(timed_source())

        # Wait a bit for background thread to prefetch
        time.sleep(0.1)
        start = time.monotonic()
        results = list(handle)

        assert len(results) == 3
        # Source should have been consumed before we started reading
        assert all(t < start for t in call_times)
