"""Unit tests for CalibraX monitoring integration used by benchmark tooling."""

from __future__ import annotations

import threading
import time
from collections import deque
from unittest.mock import Mock, patch

from calibrax.monitoring import (
    AdvancedMonitor,
    Alert,
    AlertManager,
    AlertSeverity,
    ProductionMonitor,
)
from calibrax.profiling import GPUMemoryProfiler, ResourceMonitor


class TestAlertSeverity:
    """Test AlertSeverity enum values."""

    def test_alert_severity_values(self):
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ERROR.value == "error"
        assert AlertSeverity.CRITICAL.value == "critical"


class TestAlert:
    """Test Alert model behavior."""

    def test_alert_to_dict(self):
        timestamp = time.time()
        alert = Alert(
            message="test",
            severity=AlertSeverity.WARNING,
            metric_name="metric",
            metric_value=2.0,
            threshold=1.0,
            timestamp=timestamp,
            metadata={"k": "v"},
        )

        payload = alert.to_dict()
        assert payload["message"] == "test"
        assert payload["severity"] == "warning"
        assert payload["metric_name"] == "metric"
        assert payload["metric_value"] == 2.0
        assert payload["threshold"] == 1.0
        assert payload["timestamp"] == timestamp
        assert payload["metadata"] == {"k": "v"}


class TestAlertManager:
    """Test AlertManager behavior."""

    def test_max_alert_cap_and_recent_order(self):
        manager = AlertManager(max_alerts=3)

        for i in range(5):
            manager.trigger_alert(
                message=f"Alert {i}",
                severity=AlertSeverity.INFO,
                metric_name="m",
                metric_value=float(i),
                threshold=0.0,
            )

        recent = manager.get_recent_alerts(10)
        assert [a.message for a in recent] == ["Alert 4", "Alert 3", "Alert 2"]

    def test_trigger_alert_calls_handlers(self):
        manager = AlertManager()
        handler = Mock()
        manager.add_alert_handler(handler)

        manager.trigger_alert(
            message="High memory",
            severity=AlertSeverity.WARNING,
            metric_name="memory",
            metric_value=85.0,
            threshold=80.0,
        )

        handler.assert_called_once()
        recent = manager.get_recent_alerts(1)
        assert len(recent) == 1
        assert recent[0].message == "High memory"

    def test_failing_handler_isolated(self):
        manager = AlertManager()
        bad = Mock(side_effect=Exception("boom"))
        ok = Mock()
        manager.add_alert_handler(bad)
        manager.add_alert_handler(ok)

        with patch("calibrax.monitoring.monitor.logger.exception") as mock_log:
            manager.trigger_alert(
                message="x",
                severity=AlertSeverity.INFO,
                metric_name="m",
                metric_value=1.0,
                threshold=0.5,
            )

        mock_log.assert_called_once()
        ok.assert_called_once()
        assert len(manager.get_recent_alerts(10)) == 1

    def test_filter_and_clear(self):
        manager = AlertManager()
        manager.trigger_alert(
            message="info",
            severity=AlertSeverity.INFO,
            metric_name="m",
            metric_value=1.0,
            threshold=0.0,
        )
        manager.trigger_alert(
            message="err",
            severity=AlertSeverity.ERROR,
            metric_name="m",
            metric_value=2.0,
            threshold=1.0,
        )
        assert len(manager.get_alerts_by_severity(AlertSeverity.INFO)) == 1
        assert len(manager.get_alerts_by_severity(AlertSeverity.ERROR)) == 1
        manager.clear_alerts()
        assert manager.get_recent_alerts(10) == []


class TestAdvancedMonitor:
    """Test AdvancedMonitor behavior."""

    def test_initialization_defaults(self):
        monitor = AdvancedMonitor()
        summary = monitor.get_monitoring_summary()

        assert isinstance(monitor.alert_manager, AlertManager)
        assert summary["is_monitoring"] is False
        assert summary["thresholds"] == {}
        assert summary["alert_count"] == 0

    def test_initialization_custom_components(self):
        alert_manager = AlertManager()
        gpu_profiler = Mock(spec=GPUMemoryProfiler)
        resource_monitor = Mock(spec=ResourceMonitor)

        monitor = AdvancedMonitor(
            alert_manager=alert_manager,
            gpu_profiler=gpu_profiler,
            resource_monitor=resource_monitor,
        )

        assert monitor.alert_manager is alert_manager
        assert monitor._gpu_profiler is gpu_profiler
        assert monitor._resource_monitor is resource_monitor

    def test_set_threshold_and_check_thresholds(self):
        monitor = AdvancedMonitor()
        handler = Mock()
        monitor.alert_manager.add_alert_handler(handler)

        monitor.set_threshold("memory_rss_mb", 100.0)
        monitor._check_thresholds({"memory_rss_mb": 130.0})

        handler.assert_called_once()
        alert = handler.call_args[0][0]
        assert alert.metric_name == "memory_rss_mb"
        assert alert.threshold == 100.0
        assert alert.severity == AlertSeverity.WARNING

    def test_determine_alert_severity(self):
        monitor = AdvancedMonitor()

        assert monitor._determine_alert_severity(1.1, 1.0) == AlertSeverity.INFO
        assert monitor._determine_alert_severity(1.3, 1.0) == AlertSeverity.WARNING
        assert monitor._determine_alert_severity(1.6, 1.0) == AlertSeverity.ERROR
        assert monitor._determine_alert_severity(2.1, 1.0) == AlertSeverity.CRITICAL

    def test_analyze_trend(self):
        monitor = AdvancedMonitor()

        assert monitor._analyze_trend("missing") == "stable"

        monitor._metric_history["inc"] = deque([1.0, 2.0, 3.0, 4.0], maxlen=100)
        assert monitor._analyze_trend("inc") == "increasing"

        monitor._metric_history["dec"] = deque([4.0, 3.0, 2.0, 1.0], maxlen=100)
        assert monitor._analyze_trend("dec") == "decreasing"

        monitor._metric_history["flat"] = deque([1.0, 1.0, 1.0, 1.0], maxlen=100)
        assert monitor._analyze_trend("flat") == "stable"

    def test_collect_metrics_with_gpu_profiler(self):
        fake_proc = Mock()
        fake_proc.cpu_percent.return_value = 42.0
        fake_proc.memory_info.return_value = Mock(rss=1024 * 1024 * 256)  # 256 MB

        gpu = Mock(spec=GPUMemoryProfiler)
        gpu.get_utilization.return_value = 77.0
        gpu.get_memory_usage.return_value = {"gpu_memory_used_mb": 512.0}

        with patch("calibrax.monitoring.monitor.psutil.Process", return_value=fake_proc):
            monitor = AdvancedMonitor(gpu_profiler=gpu)
            metrics = monitor._collect_metrics()

        assert metrics["cpu_percent"] == 42.0
        assert metrics["memory_rss_mb"] == 256.0
        assert metrics["gpu_utilization"] == 77.0
        assert metrics["gpu_memory_mb"] == 512.0

    def test_start_stop_monitoring_is_idempotent(self):
        monitor = AdvancedMonitor()

        monitor.start_monitoring(interval=0.05)
        first_thread = monitor._thread
        assert first_thread is not None
        assert first_thread.is_alive()

        monitor.start_monitoring(interval=0.05)
        assert monitor._thread is first_thread

        monitor.stop_monitoring()
        assert monitor._thread is None


class TestProductionMonitor:
    """Test ProductionMonitor behavior."""

    def test_set_performance_baseline(self):
        monitor = ProductionMonitor()
        monitor.set_performance_baseline("pipeline_a", 100.0)

        report = monitor.get_pipeline_health_report()
        assert report["baselines"]["pipeline_a"] == 100.0

    def test_record_pipeline_execution_and_report(self):
        monitor = ProductionMonitor()

        monitor.record_pipeline_execution("pipe", execution_time=10.0, success=True)
        monitor.record_pipeline_execution("pipe", execution_time=12.0, success=False)

        report = monitor.get_pipeline_health_report()
        stats = report["pipelines"]["pipe"]
        assert stats["total_executions"] == 2
        assert stats["success_rate"] == 0.5
        assert stats["error_rate"] == 0.5

    def test_performance_degradation_alert(self):
        monitor = ProductionMonitor()
        handler = Mock()
        monitor.alert_manager.add_alert_handler(handler)

        monitor.set_performance_baseline("pipe", 100.0)
        monitor.record_pipeline_execution("pipe", execution_time=110.0, success=True)
        handler.assert_not_called()

        monitor.record_pipeline_execution("pipe", execution_time=130.0, success=True)
        handler.assert_called_once()
        alert = handler.call_args[0][0]
        assert alert.severity == AlertSeverity.WARNING
        assert "degraded" in alert.message

    def test_error_rate_alert(self):
        monitor = ProductionMonitor()
        handler = Mock()
        monitor.alert_manager.add_alert_handler(handler)

        monitor.record_pipeline_execution("pipe", execution_time=10.0, success=False)
        monitor.record_pipeline_execution("pipe", execution_time=10.0, success=False)
        monitor.record_pipeline_execution("pipe", execution_time=10.0, success=False)

        handler.assert_called_once()
        alert = handler.call_args[0][0]
        assert alert.severity == AlertSeverity.ERROR
        assert "high error rate" in alert.message.lower()

    def test_health_report_and_overall_status(self):
        monitor = ProductionMonitor()

        for _ in range(5):
            monitor.record_pipeline_execution("healthy", execution_time=1.0, success=True)
        for _ in range(6):
            monitor.record_pipeline_execution("bad", execution_time=2.0, success=False)

        report = monitor.get_pipeline_health_report()
        assert report["pipelines"]["healthy"]["health"] == "healthy"
        assert report["pipelines"]["bad"]["health"] == "critical"
        assert report["overall_health"] == "degraded"

    def test_monitoring_with_exception_handling(self):
        monitor = ProductionMonitor()
        monitor._collect_metrics = Mock(side_effect=Exception("Collection error"))

        with patch("calibrax.monitoring.monitor.logger.exception") as mock_exception:
            monitor.start_monitoring(interval=0.05)
            time.sleep(0.15)
            monitor.stop_monitoring()

        assert mock_exception.called

    def test_concurrent_pipeline_recording(self):
        monitor = ProductionMonitor()
        num_threads = 4
        per_thread = 10

        def record(thread_id: int) -> None:
            for i in range(per_thread):
                monitor.record_pipeline_execution(
                    pipeline_name=f"pipe_{thread_id}",
                    execution_time=float(i),
                    success=(i % 2 == 0),
                )

        threads = [threading.Thread(target=record, args=(i,)) for i in range(num_threads)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        report = monitor.get_pipeline_health_report()
        assert report["total_executions"] == num_threads * per_thread
        for i in range(num_threads):
            assert report["pipelines"][f"pipe_{i}"]["total_executions"] == per_thread
