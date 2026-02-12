"""Tests for ResourceMonitor, ResourceSample, and ResourceSummary.

TDD tests written first per Section 6.2.3 of the benchmark report.
Verifies: context manager protocol, background thread sampling,
summary computation, GPU field handling, and daemon thread behavior.
"""

import time
from unittest.mock import MagicMock

import pytest

from datarax.benchmarking.resource_monitor import (
    ResourceMonitor,
    ResourceSample,
    ResourceSummary,
)


class TestResourceSample:
    """Tests for ResourceSample dataclass."""

    def test_creation_cpu_only(self):
        sample = ResourceSample(
            timestamp=1.0,
            cpu_percent=50.0,
            rss_mb=256.0,
            gpu_util=None,
            gpu_mem_mb=None,
        )
        assert sample.cpu_percent == 50.0
        assert sample.rss_mb == 256.0
        assert sample.gpu_util is None

    def test_creation_with_gpu(self):
        sample = ResourceSample(
            timestamp=1.0,
            cpu_percent=30.0,
            rss_mb=512.0,
            gpu_util=75.0,
            gpu_mem_mb=4096.0,
        )
        assert sample.gpu_util == 75.0
        assert sample.gpu_mem_mb == 4096.0


class TestResourceSummary:
    """Tests for ResourceSummary dataclass."""

    def test_creation(self):
        summary = ResourceSummary(
            peak_rss_mb=512.0,
            mean_rss_mb=400.0,
            peak_gpu_mem_mb=None,
            mean_gpu_util=None,
            memory_growth_mb=10.0,
            num_samples=50,
            duration_sec=5.0,
        )
        assert summary.peak_rss_mb == 512.0
        assert summary.num_samples == 50


class TestResourceMonitor:
    """Tests for ResourceMonitor context manager."""

    def test_context_manager_starts_and_stops_thread(self):
        """Context manager starts a background thread on enter, stops on exit."""
        mon = ResourceMonitor(sample_interval_sec=0.05)
        assert mon._thread is None

        with mon:
            assert mon._thread is not None
            assert mon._thread.is_alive()

        # After exit, thread should stop
        assert not mon._thread.is_alive()

    def test_samples_grow_during_monitoring(self):
        """Samples list should grow while monitor is active."""
        with ResourceMonitor(sample_interval_sec=0.05) as mon:
            time.sleep(0.3)

        assert len(mon.samples) > 0

    def test_summary_computes_resource_summary(self):
        """summary property returns a ResourceSummary."""
        with ResourceMonitor(sample_interval_sec=0.05) as mon:
            time.sleep(0.3)

        summary = mon.summary
        assert isinstance(summary, ResourceSummary)
        assert summary.num_samples > 0

    def test_peak_rss_at_least_mean_rss(self):
        """peak_rss_mb >= mean_rss_mb invariant."""
        with ResourceMonitor(sample_interval_sec=0.05) as mon:
            time.sleep(0.3)

        summary = mon.summary
        assert summary.peak_rss_mb >= summary.mean_rss_mb

    def test_memory_growth_is_last_minus_first(self):
        """memory_growth_mb = last RSS - first RSS."""
        with ResourceMonitor(sample_interval_sec=0.05) as mon:
            time.sleep(0.3)

        samples = mon.samples
        summary = mon.summary
        expected_growth = samples[-1].rss_mb - samples[0].rss_mb
        assert summary.memory_growth_mb == pytest.approx(expected_growth)

    def test_gpu_fields_none_without_gpu_profiler(self):
        """GPU fields are None when no GPUMemoryProfiler provided."""
        with ResourceMonitor(sample_interval_sec=0.05) as mon:
            time.sleep(0.2)

        summary = mon.summary
        assert summary.peak_gpu_mem_mb is None
        assert summary.mean_gpu_util is None

    def test_multiple_enter_exit_cycles(self):
        """Monitor can be reused across multiple context manager cycles."""
        mon = ResourceMonitor(sample_interval_sec=0.05)

        with mon:
            time.sleep(0.15)
        first_count = len(mon.samples)

        with mon:
            time.sleep(0.15)
        second_count = len(mon.samples)

        # Second cycle should have fresh samples (cleared on enter)
        assert first_count > 0
        assert second_count > 0

    def test_thread_is_daemon(self):
        """Background thread should be a daemon (won't block process exit)."""
        mon = ResourceMonitor(sample_interval_sec=0.05)
        with mon:
            assert mon._thread.daemon is True

    def test_empty_summary_when_no_samples(self):
        """Summary with no samples returns zeroed ResourceSummary."""
        mon = ResourceMonitor(sample_interval_sec=0.05)
        summary = mon.summary
        assert summary.num_samples == 0
        assert summary.peak_rss_mb == 0
        assert summary.duration_sec == 0

    def test_rss_values_positive(self):
        """RSS memory values should all be positive."""
        with ResourceMonitor(sample_interval_sec=0.05) as mon:
            time.sleep(0.2)

        for sample in mon.samples:
            assert sample.rss_mb > 0

    def test_duration_positive_with_multiple_samples(self):
        """Duration should be positive when multiple samples collected."""
        with ResourceMonitor(sample_interval_sec=0.05) as mon:
            time.sleep(0.3)

        summary = mon.summary
        if summary.num_samples > 1:
            assert summary.duration_sec > 0


class TestResourceMonitorGPU:
    """Cover GPU profiler integration paths (lines 114-117, 122-126)."""

    def test_gpu_profiler_returns_data(self):
        """GPU profiler providing utilization and memory -> samples have GPU data."""
        mock_gpu = MagicMock()
        mock_gpu.get_utilization.return_value = 65.0
        mock_gpu.get_memory_usage.return_value = {"gpu_memory_used_mb": 2048.0}

        with ResourceMonitor(sample_interval_sec=0.05, gpu_profiler=mock_gpu) as mon:
            time.sleep(0.25)

        # GPU fields should be populated
        gpu_samples = [s for s in mon.samples if s.gpu_util is not None]
        assert len(gpu_samples) > 0
        assert gpu_samples[0].gpu_util == 65.0
        assert gpu_samples[0].gpu_mem_mb == 2048.0

        summary = mon.summary
        assert summary.peak_gpu_mem_mb is not None
        assert summary.mean_gpu_util is not None
        assert summary.mean_gpu_util == pytest.approx(65.0)

    def test_gpu_profiler_utilization_raises(self):
        """GPU profiler raises on get_utilization -> gpu_util is None (lines 114-117)."""
        mock_gpu = MagicMock()
        mock_gpu.get_utilization.side_effect = RuntimeError("GPU error")
        mock_gpu.get_memory_usage.return_value = {"gpu_memory_used_mb": 1024.0}

        with ResourceMonitor(sample_interval_sec=0.05, gpu_profiler=mock_gpu) as mon:
            time.sleep(0.25)

        # Utilization should be None due to exception, memory still works
        for sample in mon.samples:
            assert sample.gpu_util is None
            assert sample.gpu_mem_mb == 1024.0

    def test_gpu_profiler_memory_raises(self):
        """GPU profiler raises on get_memory_usage -> gpu_mem_mb is None (lines 122-126)."""
        mock_gpu = MagicMock()
        mock_gpu.get_utilization.return_value = 50.0
        mock_gpu.get_memory_usage.side_effect = RuntimeError("GPU mem error")

        with ResourceMonitor(sample_interval_sec=0.05, gpu_profiler=mock_gpu) as mon:
            time.sleep(0.25)

        # Memory should be None due to exception, utilization still works
        for sample in mon.samples:
            assert sample.gpu_util == 50.0
            assert sample.gpu_mem_mb is None

    def test_gpu_profiler_both_raise(self):
        """Both GPU profiler methods raise -> all GPU fields None."""
        mock_gpu = MagicMock()
        mock_gpu.get_utilization.side_effect = RuntimeError("util error")
        mock_gpu.get_memory_usage.side_effect = RuntimeError("mem error")

        with ResourceMonitor(sample_interval_sec=0.05, gpu_profiler=mock_gpu) as mon:
            time.sleep(0.25)

        for sample in mon.samples:
            assert sample.gpu_util is None
            assert sample.gpu_mem_mb is None

        summary = mon.summary
        assert summary.peak_gpu_mem_mb is None
        assert summary.mean_gpu_util is None
