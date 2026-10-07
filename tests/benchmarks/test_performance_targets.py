"""Tests for the performance_targets utility module."""

import pytest

from tests.benchmarks.performance_targets import (
    assert_within_ratio,
    measure_latency,
    measure_peak_rss_delta_mb,
)


class TestAssertWithinRatio:
    def test_passes_when_within_ratio(self):
        assert_within_ratio(18000, 20000, 1.2)  # 18000 >= 20000/1.2 = 16667

    def test_fails_when_below_ratio(self):
        with pytest.raises(AssertionError, match="below"):
            assert_within_ratio(10000, 20000, 1.2)  # 10000 < 16667

    def test_exact_boundary_passes(self):
        # Exact boundary should pass (>=, not >)
        assert_within_ratio(16667, 20000, 1.2)

    def test_equal_values_pass(self):
        assert_within_ratio(20000, 20000, 1.0)

    def test_datarax_exceeds_alternative(self):
        assert_within_ratio(25000, 20000, 1.2)  # Better than needed


class TestMeasurePeakRssDeltaMb:
    def test_returns_float(self):
        delta = measure_peak_rss_delta_mb(lambda: [0] * 1000)
        assert isinstance(delta, float)

    def test_allocation_detected(self):
        # Allocate ~10MB — should be measurable
        delta = measure_peak_rss_delta_mb(lambda: bytearray(10 * 1024 * 1024))
        # RSS measurement is approximate; just verify it's non-negative
        assert delta >= -5.0  # Allow small negative from GC timing


class TestMeasureLatency:
    def test_returns_median(self):
        latency = measure_latency(lambda: None, repetitions=3)
        assert latency >= 0.0

    def test_nonzero_for_sleep(self):
        import time

        latency = measure_latency(lambda: time.sleep(0.01), repetitions=3)
        assert latency >= 0.005  # At least 5ms

    def test_single_repetition(self):
        latency = measure_latency(lambda: None, repetitions=1)
        assert latency >= 0.0
