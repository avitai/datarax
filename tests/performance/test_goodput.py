"""Tests for goodput/badput telemetry (P2.4).

TDD: Tests written first per the performance audit Section 6.3 P2.4.
Validates pipeline-level visibility into time breakdown.
"""

import time

from datarax.performance.goodput import GoodputTracker


class TestGoodputTracker:
    """Test the GoodputTracker lifecycle and metrics."""

    def test_initial_state(self):
        tracker = GoodputTracker()
        metrics = tracker.summary()
        assert metrics.total_batches == 0
        assert metrics.wall_clock_sec == 0.0
        assert metrics.goodput_ratio == 0.0

    def test_record_source_time(self):
        tracker = GoodputTracker()
        tracker.start_batch()
        tracker.record_source(0.01)
        tracker.record_transform(0.02)
        tracker.record_transfer(0.005)
        tracker.end_batch()

        metrics = tracker.summary()
        assert metrics.total_batches == 1
        assert metrics.source_sec > 0
        assert metrics.transform_sec > 0
        assert metrics.transfer_sec > 0

    def test_goodput_ratio_computation(self):
        """Goodput = productive_time / wall_time, using real timing."""
        tracker = GoodputTracker()

        # Use context managers so wall clock matches productive time
        for _ in range(3):
            tracker.start_batch()
            with tracker.time_source():
                time.sleep(0.01)
            with tracker.time_transform():
                time.sleep(0.01)
            with tracker.time_transfer():
                time.sleep(0.005)
            tracker.end_batch()

        metrics = tracker.summary()
        assert metrics.total_batches == 3
        assert metrics.productive_sec > 0.05
        # Goodput ratio should be high (most time is productive)
        assert 0.5 < metrics.goodput_ratio <= 1.0

    def test_overhead_tracked(self):
        """Time not in source/transform/transfer is overhead."""
        tracker = GoodputTracker()
        tracker.start_batch()
        time.sleep(0.05)  # 50ms of "overhead" before recording
        tracker.record_source(0.001)
        tracker.record_transform(0.001)
        tracker.record_transfer(0.001)
        tracker.end_batch()

        metrics = tracker.summary()
        assert metrics.overhead_sec > 0.01  # Should capture the 50ms gap

    def test_per_batch_times(self):
        """Per-batch breakdown should be accessible."""
        tracker = GoodputTracker()
        for i in range(3):
            tracker.start_batch()
            tracker.record_source(0.01 * (i + 1))
            tracker.record_transform(0.02)
            tracker.record_transfer(0.005)
            tracker.end_batch()

        assert len(tracker.per_batch_source) == 3
        assert len(tracker.per_batch_transform) == 3
        assert len(tracker.per_batch_transfer) == 3
        # First batch source time < third batch
        assert tracker.per_batch_source[0] < tracker.per_batch_source[2]

    def test_context_manager_api(self):
        """Context managers for timing blocks."""
        tracker = GoodputTracker()
        tracker.start_batch()
        with tracker.time_source():
            time.sleep(0.01)
        with tracker.time_transform():
            time.sleep(0.01)
        with tracker.time_transfer():
            time.sleep(0.005)
        tracker.end_batch()

        metrics = tracker.summary()
        assert metrics.source_sec > 0.005
        assert metrics.transform_sec > 0.005

    def test_reset(self):
        """Reset should clear all accumulated metrics."""
        tracker = GoodputTracker()
        tracker.start_batch()
        tracker.record_source(0.01)
        tracker.end_batch()

        tracker.reset()
        metrics = tracker.summary()
        assert metrics.total_batches == 0
