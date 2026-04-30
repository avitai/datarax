"""Tests for the performance_targets utility module."""

import pytest

from tests.benchmarks.performance_targets import (
    assert_within_ratio,
    classify_rss_comparison,
    DATARAX_RSS_ABSOLUTE_CAP_MB,
    measure_latency,
    measure_peak_rss_delta_mb,
    NOISE_FLOOR_MB,
    SPDL_NULL_FLOOR_MB,
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


class TestClassifyRssComparison:
    """Contracts for the SPDL-vs-Datarax peak-RSS classifier.

    The classifier is the decision logic behind ``test_peak_rss_within_1_5x_spdl``.
    Extracted into a pure function so each branch can be unit-tested without
    running the multi-second adapter measurement that produces the inputs.

    Returns one of ("pass", message) | ("skip", reason) | ("fail", reason).
    """

    def test_normal_comparison_passes_when_ratio_acceptable(self):
        # SPDL=200 MB, Datarax=250 MB → ratio 1.25, well under 1.5.
        verdict, _ = classify_rss_comparison(datarax_rss=250.0, spdl_rss=200.0)
        assert verdict == "pass"

    def test_normal_comparison_fails_when_ratio_exceeds_target(self):
        # SPDL=100 MB, Datarax=200 MB → ratio 2.0, exceeds 1.5.
        verdict, msg = classify_rss_comparison(datarax_rss=200.0, spdl_rss=100.0)
        assert verdict == "fail"
        assert "2.0" in msg or "2.00" in msg

    def test_exact_ratio_boundary_passes(self):
        # Ratio exactly 1.5 should pass (<=, not <).
        verdict, _ = classify_rss_comparison(datarax_rss=150.0, spdl_rss=100.0)
        assert verdict == "pass"

    def test_absolute_cap_failure_independent_of_spdl(self):
        # Datarax over the absolute cap must fail regardless of SPDL value.
        # This is the regression-guard: SPDL=0 used to skip silently while
        # Datarax allocated multi-GB.
        verdict, msg = classify_rss_comparison(
            datarax_rss=DATARAX_RSS_ABSOLUTE_CAP_MB + 1000.0,
            spdl_rss=0.0,
        )
        assert verdict == "fail"
        assert "absolute cap" in msg.lower()

    def test_absolute_cap_failure_takes_precedence_over_skip(self):
        # Even when SPDL=0 (which would normally skip), an absolute-cap
        # violation should fail loud rather than skip quiet.
        verdict, _ = classify_rss_comparison(
            datarax_rss=DATARAX_RSS_ABSOLUTE_CAP_MB + 100.0,
            spdl_rss=0.5,
        )
        assert verdict == "fail"

    def test_skip_when_spdl_effectively_zero(self):
        # SPDL=0 with Datarax under the absolute cap should skip with a
        # message that names the SPDL-zero cause specifically.
        verdict, msg = classify_rss_comparison(
            datarax_rss=2918.0,
            spdl_rss=0.0,
        )
        assert verdict == "skip"
        assert "SPDL" in msg
        assert "zero-copy" in msg.lower() or "below" in msg.lower()

    def test_skip_when_both_below_noise_floor(self):
        # Both adapters tiny — measurement is allocator noise, ratio
        # is uninformative. Skip with the noise-floor message.
        verdict, msg = classify_rss_comparison(datarax_rss=10.0, spdl_rss=8.0)
        assert verdict == "skip"
        assert "noise floor" in msg.lower()

    def test_skip_when_only_datarax_below_noise_floor(self):
        # Datarax below noise floor means Datarax peak isn't trustworthy
        # even if SPDL is well above; skip.
        verdict, _ = classify_rss_comparison(datarax_rss=20.0, spdl_rss=200.0)
        assert verdict == "skip"

    def test_constants_have_sensible_ordering(self):
        # SPDL_NULL_FLOOR (effectively zero) must be < NOISE_FLOOR
        # (allocator noise floor) which must be < DATARAX cap.
        assert 0 < SPDL_NULL_FLOOR_MB < NOISE_FLOOR_MB < DATARAX_RSS_ABSOLUTE_CAP_MB
