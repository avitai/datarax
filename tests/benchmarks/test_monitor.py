"""Unit tests for advanced monitoring and alerting system."""

import threading
import time
import warnings
from unittest.mock import Mock


from datarax.benchmarking.monitor import (
    AdvancedMonitor,
    Alert,
    AlertManager,
    AlertSeverity,
    ProductionMonitor,
)
from datarax.benchmarking.profiler import GPUMemoryProfiler
from datarax.monitoring.metrics import MetricsCollector


class TestAlertSeverity:
    """Test AlertSeverity enum."""

    def test_alert_severity_values(self):
        """Test that AlertSeverity has the expected values."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ERROR.value == "error"
        assert AlertSeverity.CRITICAL.value == "critical"

    def test_alert_severity_members(self):
        """Test that all severity levels are present."""
        severity_names = {s.name for s in AlertSeverity}
        expected_names = {"INFO", "WARNING", "ERROR", "CRITICAL"}
        assert severity_names == expected_names


class TestAlert:
    """Test Alert dataclass."""

    def test_alert_creation(self):
        """Test creating an Alert instance."""
        alert = Alert(
            message="Test alert",
            severity=AlertSeverity.WARNING,
            metric_name="test_metric",
            metric_value=10.5,
            threshold=5.0,
        )

        assert alert.message == "Test alert"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.metric_name == "test_metric"
        assert alert.metric_value == 10.5
        assert alert.threshold == 5.0
        assert isinstance(alert.timestamp, float)
        assert alert.metadata == {}

    def test_alert_with_metadata(self):
        """Test creating an Alert with metadata."""
        metadata = {"component": "test", "iteration": 5}
        alert = Alert(
            message="Test alert",
            severity=AlertSeverity.ERROR,
            metric_name="test_metric",
            metric_value=20.0,
            threshold=10.0,
            metadata=metadata,
        )

        assert alert.metadata == metadata

    def test_alert_to_dict(self):
        """Test converting Alert to dictionary."""
        timestamp = time.time()
        metadata = {"key": "value"}
        alert = Alert(
            message="Test message",
            severity=AlertSeverity.CRITICAL,
            metric_name="cpu_usage",
            metric_value=95.5,
            threshold=80.0,
            timestamp=timestamp,
            metadata=metadata,
        )

        alert_dict = alert.to_dict()

        assert alert_dict["message"] == "Test message"
        assert alert_dict["severity"] == "critical"
        assert alert_dict["metric_name"] == "cpu_usage"
        assert alert_dict["metric_value"] == 95.5
        assert alert_dict["threshold"] == 80.0
        assert alert_dict["timestamp"] == timestamp
        assert alert_dict["metadata"] == metadata


class TestAlertManager:
    """Test AlertManager class."""

    def test_alert_manager_initialization(self):
        """Test AlertManager initialization."""
        manager = AlertManager(max_alerts=100)
        assert manager.max_alerts == 100
        assert len(manager.alerts) == 0
        assert len(manager.alert_handlers) == 0

    def test_add_alert_handler(self):
        """Test adding alert handlers."""
        manager = AlertManager()
        handler1 = Mock()
        handler2 = Mock()

        manager.add_alert_handler(handler1)
        manager.add_alert_handler(handler2)

        assert len(manager.alert_handlers) == 2
        assert handler1 in manager.alert_handlers
        assert handler2 in manager.alert_handlers

    def test_trigger_alert(self):
        """Test triggering alerts."""
        manager = AlertManager()
        handler = Mock()
        manager.add_alert_handler(handler)

        manager.trigger_alert(
            message="High memory usage",
            severity=AlertSeverity.WARNING,
            metric_name="memory_usage",
            metric_value=85.0,
            threshold=80.0,
            metadata={"node": "gpu-0"},
        )

        # Check alert was stored
        assert len(manager.alerts) == 1
        alert = manager.alerts[0]
        assert alert.message == "High memory usage"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.metric_value == 85.0

        # Check handler was called
        handler.assert_called_once()
        called_alert = handler.call_args[0][0]
        assert called_alert.message == "High memory usage"

    def test_trigger_alert_with_failing_handler(self):
        """Test that failing handlers don't break alert triggering."""
        manager = AlertManager()

        # Add a failing handler and a working handler
        failing_handler = Mock(side_effect=Exception("Handler error"))
        working_handler = Mock()

        manager.add_alert_handler(failing_handler)
        manager.add_alert_handler(working_handler)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            manager.trigger_alert(
                message="Test alert",
                severity=AlertSeverity.INFO,
                metric_name="test",
                metric_value=1.0,
                threshold=0.5,
            )

            # Check warning was issued
            assert len(w) == 1
            assert "Alert handler failed" in str(w[0].message)

        # Alert should still be stored
        assert len(manager.alerts) == 1

        # Working handler should still be called
        working_handler.assert_called_once()

    def test_get_recent_alerts(self):
        """Test getting recent alerts."""
        manager = AlertManager()

        # Add multiple alerts
        for i in range(20):
            manager.trigger_alert(
                message=f"Alert {i}",
                severity=AlertSeverity.INFO,
                metric_name=f"metric_{i}",
                metric_value=float(i),
                threshold=0.0,
            )

        # Get recent 5 alerts
        recent = manager.get_recent_alerts(5)
        assert len(recent) == 5
        assert recent[0].message == "Alert 15"
        assert recent[-1].message == "Alert 19"

        # Get all alerts
        all_recent = manager.get_recent_alerts(30)
        assert len(all_recent) == 20

    def test_get_alerts_by_severity(self):
        """Test filtering alerts by severity."""
        manager = AlertManager()

        # Add alerts with different severities
        severities = [
            AlertSeverity.INFO,
            AlertSeverity.WARNING,
            AlertSeverity.ERROR,
            AlertSeverity.CRITICAL,
            AlertSeverity.WARNING,
            AlertSeverity.INFO,
        ]

        for i, severity in enumerate(severities):
            manager.trigger_alert(
                message=f"Alert {i}",
                severity=severity,
                metric_name=f"metric_{i}",
                metric_value=float(i),
                threshold=0.0,
            )

        # Filter by severity
        info_alerts = manager.get_alerts_by_severity(AlertSeverity.INFO)
        assert len(info_alerts) == 2

        warning_alerts = manager.get_alerts_by_severity(AlertSeverity.WARNING)
        assert len(warning_alerts) == 2

        error_alerts = manager.get_alerts_by_severity(AlertSeverity.ERROR)
        assert len(error_alerts) == 1

        critical_alerts = manager.get_alerts_by_severity(AlertSeverity.CRITICAL)
        assert len(critical_alerts) == 1

    def test_clear_alerts(self):
        """Test clearing all alerts."""
        manager = AlertManager()

        # Add some alerts
        for i in range(5):
            manager.trigger_alert(
                message=f"Alert {i}",
                severity=AlertSeverity.INFO,
                metric_name=f"metric_{i}",
                metric_value=float(i),
                threshold=0.0,
            )

        assert len(manager.alerts) == 5

        # Clear alerts
        manager.clear_alerts()
        assert len(manager.alerts) == 0

    def test_max_alerts_limit(self):
        """Test that max_alerts limit is respected."""
        manager = AlertManager(max_alerts=5)

        # Add more alerts than the limit
        for i in range(10):
            manager.trigger_alert(
                message=f"Alert {i}",
                severity=AlertSeverity.INFO,
                metric_name=f"metric_{i}",
                metric_value=float(i),
                threshold=0.0,
            )

        # Should only keep the last 5 alerts
        assert len(manager.alerts) == 5
        assert manager.alerts[0].message == "Alert 5"
        assert manager.alerts[-1].message == "Alert 9"

    def test_thread_safety(self):
        """Test thread safety of AlertManager."""
        manager = AlertManager()
        num_threads = 5
        alerts_per_thread = 20

        def add_alerts(thread_id):
            for i in range(alerts_per_thread):
                manager.trigger_alert(
                    message=f"Thread {thread_id} Alert {i}",
                    severity=AlertSeverity.INFO,
                    metric_name=f"metric_{thread_id}_{i}",
                    metric_value=float(i),
                    threshold=0.0,
                )

        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=add_alerts, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All alerts should be added
        assert len(manager.alerts) == num_threads * alerts_per_thread


class TestAdvancedMonitor:
    """Test AdvancedMonitor class."""

    def test_initialization_with_defaults(self):
        """Test AdvancedMonitor initialization with default components."""
        monitor = AdvancedMonitor()

        assert isinstance(monitor.alert_manager, AlertManager)
        assert isinstance(monitor.gpu_profiler, GPUMemoryProfiler)
        assert isinstance(monitor.metrics_collector, MetricsCollector)
        assert monitor._monitoring_active is False
        assert monitor._monitor_thread is None
        assert len(monitor.thresholds) > 0

    def test_initialization_with_custom_components(self):
        """Test AdvancedMonitor initialization with custom components."""
        alert_manager = AlertManager(max_alerts=50)
        gpu_profiler = GPUMemoryProfiler()
        metrics_collector = MetricsCollector(enabled=False)

        monitor = AdvancedMonitor(
            alert_manager=alert_manager,
            gpu_profiler=gpu_profiler,
            metrics_collector=metrics_collector,
        )

        assert monitor.alert_manager is alert_manager
        assert monitor.gpu_profiler is gpu_profiler
        assert monitor.metrics_collector is metrics_collector

    def test_set_threshold(self):
        """Test setting custom thresholds."""
        monitor = AdvancedMonitor()

        monitor.set_threshold("custom_metric", 100.0)
        assert monitor.thresholds["custom_metric"] == 100.0

        # Update existing threshold
        monitor.set_threshold("gpu_memory_utilization", 0.95)
        assert monitor.thresholds["gpu_memory_utilization"] == 0.95

    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        monitor = AdvancedMonitor()

        # Start monitoring
        monitor.start_monitoring(interval=0.1)
        assert monitor._monitoring_active is True
        assert monitor._monitor_thread is not None
        assert monitor._monitor_thread.is_alive()

        # Give it time to run
        time.sleep(0.2)

        # Stop monitoring
        monitor.stop_monitoring()
        assert monitor._monitoring_active is False
        time.sleep(0.1)  # Give thread time to stop
        assert monitor._monitor_thread is None

    def test_start_monitoring_when_already_active(self):
        """Test starting monitoring when already active."""
        monitor = AdvancedMonitor()
        monitor.start_monitoring(interval=0.1)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            monitor.start_monitoring(interval=0.1)
            assert len(w) == 1
            assert "Monitoring is already active" in str(w[0].message)

        monitor.stop_monitoring()

    def test_stop_monitoring_when_not_active(self):
        """Test stopping monitoring when not active."""
        monitor = AdvancedMonitor()
        # Should not raise an error
        monitor.stop_monitoring()

    def test_update_history(self):
        """Test updating metrics history."""
        monitor = AdvancedMonitor()

        # Update history for new metric
        monitor._update_history("test_metric", 10.0)
        assert "test_metric" in monitor.metrics_history
        assert list(monitor.metrics_history["test_metric"]) == [10.0]

        # Update existing metric
        monitor._update_history("test_metric", 20.0)
        assert list(monitor.metrics_history["test_metric"]) == [10.0, 20.0]

        # Test max history size
        for i in range(monitor.history_size + 5):
            monitor._update_history("overflow_metric", float(i))

        history = monitor.metrics_history["overflow_metric"]
        assert len(history) == monitor.history_size
        assert history[0] == 5.0  # First 5 values should be dropped

    def test_determine_alert_severity(self):
        """Test alert severity determination."""
        monitor = AdvancedMonitor()

        # Just over threshold
        severity = monitor._determine_alert_severity("test", 1.1, 1.0)
        assert severity == AlertSeverity.INFO

        # 1.2x threshold
        severity = monitor._determine_alert_severity("test", 1.3, 1.0)
        assert severity == AlertSeverity.WARNING

        # 1.5x threshold
        severity = monitor._determine_alert_severity("test", 1.6, 1.0)
        assert severity == AlertSeverity.ERROR

        # 2x threshold
        severity = monitor._determine_alert_severity("test", 2.1, 1.0)
        assert severity == AlertSeverity.CRITICAL

    def test_analyze_trend(self):
        """Test trend analysis."""
        monitor = AdvancedMonitor()

        # Unknown metric
        trend = monitor._analyze_trend("unknown_metric")
        assert trend == "unknown"

        # Insufficient data
        monitor._update_history("new_metric", 1.0)
        monitor._update_history("new_metric", 2.0)
        trend = monitor._analyze_trend("new_metric")
        assert trend == "insufficient_data"

        # Increasing trend
        for i in range(10):
            monitor._update_history("increasing", float(i * 2))
        trend = monitor._analyze_trend("increasing")
        assert trend == "increasing"

        # Decreasing trend
        for i in range(10):
            monitor._update_history("decreasing", float(20 - i * 2))
        trend = monitor._analyze_trend("decreasing")
        assert trend == "decreasing"

        # Stable trend
        for i in range(10):
            monitor._update_history("stable", 5.0 + (i % 2) * 0.01)
        trend = monitor._analyze_trend("stable")
        assert trend == "stable"

    def test_check_thresholds(self):
        """Test threshold checking and alert triggering."""
        monitor = AdvancedMonitor()
        alert_handler = Mock()
        monitor.alert_manager.add_alert_handler(alert_handler)

        # Set threshold
        monitor.set_threshold("test_metric", 10.0)

        # Add values below threshold
        monitor._update_history("test_metric", 5.0)
        monitor._update_history("test_metric", 8.0)
        monitor._check_thresholds()
        alert_handler.assert_not_called()

        # Add value above threshold
        monitor._update_history("test_metric", 15.0)
        monitor._check_thresholds()
        alert_handler.assert_called_once()

        alert = alert_handler.call_args[0][0]
        assert alert.metric_name == "test_metric"
        assert alert.metric_value == 15.0
        assert alert.threshold == 10.0
        assert "trend" in alert.metadata

    def test_collect_metrics_with_psutil(self):
        """Test metrics collection with psutil available."""
        import sys
        from unittest.mock import MagicMock

        # Create a mock psutil module
        mock_psutil = MagicMock()
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(rss=1024 * 1024 * 512)  # 512 MB
        mock_process.cpu_percent.return_value = 45.5
        mock_psutil.Process.return_value = mock_process

        # Temporarily add mock psutil to sys.modules
        original_psutil = sys.modules.get("psutil")
        sys.modules["psutil"] = mock_psutil

        try:
            monitor = AdvancedMonitor()

            # Mock GPU profiler
            monitor.gpu_profiler.has_gpu = True
            monitor.gpu_profiler.get_memory_usage = Mock(
                return_value={"gpu_memory_utilization": 0.75, "gpu_memory_used_mb": 1024}
            )

            monitor._collect_metrics()

            # Check GPU metrics were collected
            assert "gpu_memory_utilization" in monitor.metrics_history
            assert monitor.metrics_history["gpu_memory_utilization"][-1] == 0.75

            # Check system metrics were collected
            assert "memory_usage_mb" in monitor.metrics_history
            assert monitor.metrics_history["memory_usage_mb"][-1] == 512.0
            assert "cpu_usage_percent" in monitor.metrics_history
            assert monitor.metrics_history["cpu_usage_percent"][-1] == 45.5
        finally:
            # Restore original psutil module state
            if original_psutil is None:
                sys.modules.pop("psutil", None)
            else:
                sys.modules["psutil"] = original_psutil

    def test_collect_metrics_without_psutil(self):
        """Test metrics collection when psutil is not available."""
        monitor = AdvancedMonitor()

        # Mock GPU profiler
        monitor.gpu_profiler.has_gpu = False

        # This should not raise an error even without psutil
        monitor._collect_metrics()

    def test_get_monitoring_summary(self):
        """Test getting monitoring summary."""
        monitor = AdvancedMonitor()

        # Add some alerts and metrics
        monitor.alert_manager.trigger_alert(
            message="Test alert",
            severity=AlertSeverity.CRITICAL,
            metric_name="test",
            metric_value=1.0,
            threshold=0.5,
        )

        monitor._update_history("metric1", 10.0)
        monitor._update_history("metric1", 15.0)
        monitor._update_history("metric2", 5.0)

        summary = monitor.get_monitoring_summary()

        assert summary["monitoring_active"] is False
        assert summary["total_alerts"] == 1
        assert summary["critical_alerts"] == 1
        assert "metric1" in summary["metrics_tracked"]
        assert "metric2" in summary["metrics_tracked"]

        # Check current metrics
        assert "metric1" in summary["current_metrics"]
        assert summary["current_metrics"]["metric1"]["current"] == 15.0
        assert summary["current_metrics"]["metric1"]["samples"] == 2


class TestProductionMonitor:
    """Test ProductionMonitor class."""

    def test_initialization(self):
        """Test ProductionMonitor initialization."""
        monitor = ProductionMonitor()

        # Check enhanced thresholds
        assert monitor.thresholds["gpu_memory_utilization"] == 0.85
        assert monitor.thresholds["iteration_time_ms"] == 500
        assert monitor.thresholds["memory_usage_mb"] == 8000
        assert monitor.thresholds["error_rate"] == 0.05
        assert monitor.thresholds["throughput_degradation"] == 0.2

        # Check production-specific attributes
        assert len(monitor.performance_baselines) == 0
        assert len(monitor.error_counts) == 0
        assert len(monitor.request_counts) == 0

    def test_set_performance_baseline(self):
        """Test setting performance baselines."""
        monitor = ProductionMonitor()

        monitor.set_performance_baseline("pipeline_execution_time", 100.0)
        assert monitor.performance_baselines["pipeline_execution_time"] == 100.0

    def test_record_pipeline_execution_success(self):
        """Test recording successful pipeline execution."""
        monitor = ProductionMonitor()

        monitor.record_pipeline_execution(
            pipeline_name="test_pipeline",
            execution_time=50.0,
            success=True,
            metadata={"batch_size": 32},
        )

        assert monitor.request_counts["test_pipeline"] == 1
        assert monitor.error_counts["test_pipeline"] == 0

    def test_record_pipeline_execution_failure(self):
        """Test recording failed pipeline execution."""
        monitor = ProductionMonitor()

        monitor.record_pipeline_execution(
            pipeline_name="test_pipeline",
            execution_time=50.0,
            success=False,
        )

        assert monitor.request_counts["test_pipeline"] == 1
        assert monitor.error_counts["test_pipeline"] == 1

    def test_check_performance_degradation(self):
        """Test performance degradation detection."""
        monitor = ProductionMonitor()
        alert_handler = Mock()
        monitor.alert_manager.add_alert_handler(alert_handler)

        # Set baseline
        monitor.set_performance_baseline("test_pipeline_execution_time", 100.0)

        # Record execution within acceptable range
        monitor.record_pipeline_execution(
            pipeline_name="test_pipeline",
            execution_time=110.0,
            success=True,
        )
        alert_handler.assert_not_called()

        # Record execution with degradation
        monitor.record_pipeline_execution(
            pipeline_name="test_pipeline",
            execution_time=125.0,  # 25% slower
            success=True,
        )
        alert_handler.assert_called_once()

        alert = alert_handler.call_args[0][0]
        assert "Performance degradation" in alert.message
        assert alert.severity == AlertSeverity.WARNING

    def test_check_error_rate(self):
        """Test error rate checking."""
        monitor = ProductionMonitor()
        alert_handler = Mock()
        monitor.alert_manager.add_alert_handler(alert_handler)

        # Need at least 10 requests for error rate check
        for i in range(9):
            monitor.record_pipeline_execution(
                pipeline_name="test_pipeline",
                execution_time=50.0,
                success=(i % 2 == 0),  # 50% error rate
            )
        alert_handler.assert_not_called()  # Not enough requests

        # 10th request triggers check
        monitor.record_pipeline_execution(
            pipeline_name="test_pipeline",
            execution_time=50.0,
            success=False,
        )
        alert_handler.assert_called_once()

        alert = alert_handler.call_args[0][0]
        assert "High error rate" in alert.message
        assert alert.severity == AlertSeverity.ERROR

    def test_get_pipeline_health_report(self):
        """Test getting comprehensive pipeline health report."""
        monitor = ProductionMonitor()

        # Record some pipeline executions
        for i in range(20):
            monitor.record_pipeline_execution(
                pipeline_name="pipeline1",
                execution_time=100.0 + i,
                success=(i < 18),  # 10% error rate
            )

        for i in range(10):
            monitor.record_pipeline_execution(
                pipeline_name="pipeline2",
                execution_time=50.0,
                success=True,  # 0% error rate
            )

        # Add some metrics
        monitor._update_history("gpu_memory_utilization", 0.7)
        monitor._update_history("gpu_memory_utilization", 0.8)
        monitor._update_history("memory_usage_mb", 4000)
        monitor._update_history("memory_usage_mb", 4500)

        report = monitor.get_pipeline_health_report()

        assert "timestamp" in report
        assert "monitoring_summary" in report
        assert "pipeline_stats" in report
        assert "system_health" in report

        # Check pipeline stats
        assert "pipeline1" in report["pipeline_stats"]
        stats1 = report["pipeline_stats"]["pipeline1"]
        assert stats1["total_requests"] == 20
        assert stats1["error_count"] == 2
        assert stats1["error_rate"] == 0.1
        assert stats1["success_rate"] == 0.9

        assert "pipeline2" in report["pipeline_stats"]
        stats2 = report["pipeline_stats"]["pipeline2"]
        assert stats2["total_requests"] == 10
        assert stats2["error_count"] == 0
        assert stats2["error_rate"] == 0.0
        assert stats2["success_rate"] == 1.0

        # Check system health
        assert "gpu_memory" in report["system_health"]
        gpu_health = report["system_health"]["gpu_memory"]
        assert gpu_health["current"] == 0.8
        assert gpu_health["average"] == 0.75
        assert gpu_health["max"] == 0.8

        assert "memory" in report["system_health"]
        memory_health = report["system_health"]["memory"]
        assert memory_health["current_mb"] == 4500
        assert memory_health["average_mb"] == 4250
        assert memory_health["max_mb"] == 4500

    def test_monitoring_with_exception_handling(self):
        """Test that monitoring handles exceptions gracefully."""
        monitor = ProductionMonitor()

        # Mock _collect_metrics to raise an exception
        monitor._collect_metrics = Mock(side_effect=Exception("Collection error"))

        # Start monitoring
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            monitor.start_monitoring(interval=0.05)
            time.sleep(0.1)  # Let it run a bit
            monitor.stop_monitoring()

            # Should have warnings about monitoring errors
            assert any("Monitoring error" in str(warning.message) for warning in w)

    def test_concurrent_pipeline_recording(self):
        """Test thread-safe pipeline recording."""
        monitor = ProductionMonitor()
        num_threads = 5
        executions_per_thread = 20

        def record_executions(thread_id):
            for i in range(executions_per_thread):
                monitor.record_pipeline_execution(
                    pipeline_name=f"pipeline_{thread_id}",
                    execution_time=100.0 + i,
                    success=(i % 3 != 0),  # ~33% error rate
                )

        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=record_executions, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Check all executions were recorded
        for i in range(num_threads):
            pipeline_name = f"pipeline_{i}"
            assert monitor.request_counts[pipeline_name] == executions_per_thread
            # Approximately 1/3 should be errors
            expected_errors = executions_per_thread // 3
            assert abs(monitor.error_counts[pipeline_name] - expected_errors) <= 1
