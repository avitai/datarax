"""Advanced monitoring and alerting system for Datarax.

This module provides production-ready monitoring capabilities including
real-time performance tracking, alerting, and distributed monitoring.
"""

import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from datarax.benchmarking.profiler import GPUMemoryProfiler
from datarax.monitoring.metrics import MetricsCollector


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Represents a monitoring alert.

    Attributes:
        message: Alert message
        severity: Alert severity level
        metric_name: Name of the metric that triggered the alert
        metric_value: Current value of the metric
        threshold: Threshold that was exceeded
        timestamp: When the alert was triggered
        metadata: Additional alert metadata
    """

    message: str
    severity: AlertSeverity
    metric_name: str
    metric_value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "message": self.message,
            "severity": self.severity.value,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class AlertManager:
    """Manages alerts and notifications for monitoring system."""

    def __init__(self, max_alerts: int = 1000):
        """Initialize alert manager.

        Args:
            max_alerts: Maximum number of alerts to keep in memory
        """
        self.max_alerts = max_alerts
        self.alerts: deque[Alert] = deque(maxlen=max_alerts)
        self.alert_handlers: list[Callable[[Alert], None]] = []
        self._lock = threading.Lock()

    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add an alert handler function.

        Args:
            handler: Function that will be called when alerts are triggered
        """
        with self._lock:
            self.alert_handlers.append(handler)

    def trigger_alert(
        self,
        message: str,
        severity: AlertSeverity,
        metric_name: str,
        metric_value: float,
        threshold: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Trigger a new alert.

        Args:
            message: Alert message
            severity: Alert severity
            metric_name: Name of the triggering metric
            metric_value: Current metric value
            threshold: Threshold that was exceeded
            metadata: Additional alert metadata
        """
        alert = Alert(
            message=message,
            severity=severity,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
            metadata=metadata or {},
        )

        with self._lock:
            self.alerts.append(alert)

            # Notify all handlers
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    warnings.warn(f"Alert handler failed: {e}")

    def get_recent_alerts(self, count: int = 10) -> list[Alert]:
        """Get recent alerts.

        Args:
            count: Number of recent alerts to return

        Returns:
            List of recent alerts
        """
        with self._lock:
            return list(self.alerts)[-count:]

    def get_alerts_by_severity(self, severity: AlertSeverity) -> list[Alert]:
        """Get alerts filtered by severity.

        Args:
            severity: Severity level to filter by

        Returns:
            List of alerts with the specified severity
        """
        with self._lock:
            return [alert for alert in self.alerts if alert.severity == severity]

    def clear_alerts(self) -> None:
        """Clear all stored alerts."""
        with self._lock:
            self.alerts.clear()


class AdvancedMonitor:
    """Advanced monitoring system for Datarax pipelines."""

    def __init__(
        self,
        alert_manager: AlertManager | None = None,
        gpu_profiler: GPUMemoryProfiler | None = None,
        metrics_collector: MetricsCollector | None = None,
    ):
        """Initialize advanced monitor.

        Args:
            alert_manager: Alert manager for notifications
            gpu_profiler: GPU memory profiler
            metrics_collector: Metrics collector
        """
        self.alert_manager = alert_manager or AlertManager()
        self.gpu_profiler = gpu_profiler or GPUMemoryProfiler()
        self.metrics_collector = metrics_collector or MetricsCollector()

        # Monitoring state
        self._monitoring_active = False
        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Performance thresholds
        self.thresholds = {
            "gpu_memory_utilization": 0.9,  # 90% GPU memory
            "iteration_time_ms": 1000,  # 1 second per iteration
            "memory_usage_mb": 4000,  # 4GB memory usage
            "error_rate": 0.1,  # 10% error rate
        }

        # Metrics history for trend analysis
        self.metrics_history: dict[str, deque[float]] = {}
        self.history_size = 100

    def set_threshold(self, metric_name: str, threshold: float) -> None:
        """Set alert threshold for a metric.

        Args:
            metric_name: Name of the metric
            threshold: Threshold value
        """
        self.thresholds[metric_name] = threshold

    def start_monitoring(self, interval: float = 5.0) -> None:
        """Start continuous monitoring.

        Args:
            interval: Monitoring interval in seconds
        """
        if self._monitoring_active:
            warnings.warn("Monitoring is already active")
            return

        self._monitoring_active = True
        self._stop_event.clear()

        def monitor_loop():
            """Main monitoring loop."""
            while not self._stop_event.wait(interval):
                try:
                    self._collect_metrics()
                    self._check_thresholds()
                except Exception as e:
                    warnings.warn(f"Monitoring error: {e}")

        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        if not self._monitoring_active:
            return

        self._monitoring_active = False
        self._stop_event.set()

        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
            self._monitor_thread = None

    def _collect_metrics(self) -> None:
        """Collect current metrics."""
        timestamp = time.time()

        # Collect GPU metrics
        if self.gpu_profiler.has_gpu:
            gpu_metrics = self.gpu_profiler.get_memory_usage()
            for metric_name, value in gpu_metrics.items():
                self.metrics_collector.record_metric(
                    metric_name, value, component="gpu_monitor", metadata={"timestamp": timestamp}
                )
                self._update_history(metric_name, value)

        # Collect system metrics (if available)
        try:
            import psutil

            process = psutil.Process()

            # Memory usage
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.metrics_collector.record_metric(
                "memory_usage_mb", memory_mb, component="system_monitor"
            )
            self._update_history("memory_usage_mb", memory_mb)

            # CPU usage
            cpu_percent = process.cpu_percent()
            self.metrics_collector.record_metric(
                "cpu_usage_percent", cpu_percent, component="system_monitor"
            )
            self._update_history("cpu_usage_percent", cpu_percent)

        except ImportError:
            pass  # psutil not available

    def _update_history(self, metric_name: str, value: float) -> None:
        """Update metrics history for trend analysis."""
        if metric_name not in self.metrics_history:
            self.metrics_history[metric_name] = deque(maxlen=self.history_size)

        self.metrics_history[metric_name].append(value)

    def _check_thresholds(self) -> None:
        """Check metrics against thresholds and trigger alerts."""
        for metric_name, threshold in self.thresholds.items():
            if metric_name in self.metrics_history:
                history = self.metrics_history[metric_name]
                if not history:
                    continue

                current_value = history[-1]

                # Check if threshold is exceeded
                if current_value > threshold:
                    severity = self._determine_alert_severity(metric_name, current_value, threshold)

                    self.alert_manager.trigger_alert(
                        message=f"{metric_name} exceeded threshold: {current_value:.2f} > "
                        f"{threshold:.2f}",
                        severity=severity,
                        metric_name=metric_name,
                        metric_value=current_value,
                        threshold=threshold,
                        metadata={
                            "trend": self._analyze_trend(metric_name),
                            "history_length": len(history),
                        },
                    )

    def _determine_alert_severity(
        self, metric_name: str, value: float, threshold: float
    ) -> AlertSeverity:
        """Determine alert severity based on how much threshold is exceeded."""
        ratio = value / threshold

        if ratio > 2.0:  # More than 2x threshold
            return AlertSeverity.CRITICAL
        elif ratio > 1.5:  # More than 1.5x threshold
            return AlertSeverity.ERROR
        elif ratio > 1.2:  # More than 1.2x threshold
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO

    def _analyze_trend(self, metric_name: str) -> str:
        """Analyze trend for a metric."""
        if metric_name not in self.metrics_history:
            return "unknown"

        history = list(self.metrics_history[metric_name])
        if len(history) < 3:
            return "insufficient_data"

        # Simple trend analysis using linear regression slope
        n = len(history)
        x = list(range(n))
        y = history

        # Calculate slope
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] * x[i] for i in range(n))

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"

    def get_monitoring_summary(self) -> dict[str, Any]:
        """Get summary of current monitoring state."""
        summary = {
            "monitoring_active": self._monitoring_active,
            "total_alerts": len(self.alert_manager.alerts),
            "recent_alerts": len(self.alert_manager.get_recent_alerts(10)),
            "critical_alerts": len(
                self.alert_manager.get_alerts_by_severity(AlertSeverity.CRITICAL)
            ),
            "thresholds": self.thresholds.copy(),
            "metrics_tracked": list(self.metrics_history.keys()),
        }

        # Add current metric values
        current_metrics: dict[str, Any] = {}
        for metric_name, history in self.metrics_history.items():
            if history:
                current_metrics[metric_name] = {
                    "current": history[-1],
                    "trend": self._analyze_trend(metric_name),
                    "samples": len(history),
                }

        summary["current_metrics"] = current_metrics

        return summary


class ProductionMonitor(AdvancedMonitor):
    """Production-ready monitoring system with enhanced features."""

    def __init__(self, **kwargs: Any):
        """Initialize production monitor."""
        super().__init__(**kwargs)

        # Enhanced thresholds for production
        self.thresholds.update(
            {
                "gpu_memory_utilization": 0.85,  # 85% for production safety
                "iteration_time_ms": 500,  # 500ms for production performance
                "memory_usage_mb": 8000,  # 8GB memory limit
                "error_rate": 0.05,  # 5% error rate
                "throughput_degradation": 0.2,  # 20% throughput degradation
            }
        )

        # Production-specific metrics
        self.performance_baselines: dict[str, float] = {}
        self.error_counts: dict[str, int] = {}
        self.request_counts: dict[str, int] = {}

    def set_performance_baseline(self, metric_name: str, baseline_value: float) -> None:
        """Set performance baseline for comparison.

        Args:
            metric_name: Name of the metric
            baseline_value: Baseline value for comparison
        """
        self.performance_baselines[metric_name] = baseline_value

    def record_pipeline_execution(
        self,
        pipeline_name: str,
        execution_time: float,
        success: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a pipeline execution for monitoring.

        Args:
            pipeline_name: Name of the pipeline
            execution_time: Time taken for execution
            success: Whether execution was successful
            metadata: Additional execution metadata
        """
        # Record metrics
        self.metrics_collector.record_metric(
            f"{pipeline_name}_execution_time",
            execution_time,
            component="pipeline_monitor",
            metadata=metadata or {},
        )

        # Update counters
        if pipeline_name not in self.request_counts:
            self.request_counts[pipeline_name] = 0
        if pipeline_name not in self.error_counts:
            self.error_counts[pipeline_name] = 0

        self.request_counts[pipeline_name] += 1
        if not success:
            self.error_counts[pipeline_name] += 1

        # Check for performance degradation
        self._check_performance_degradation(pipeline_name, execution_time)

        # Check error rates
        self._check_error_rate(pipeline_name)

    def _check_performance_degradation(self, pipeline_name: str, execution_time: float) -> None:
        """Check for performance degradation against baseline."""
        baseline_key = f"{pipeline_name}_execution_time"

        if baseline_key in self.performance_baselines:
            baseline = self.performance_baselines[baseline_key]
            degradation = (execution_time - baseline) / baseline

            if degradation > self.thresholds.get("throughput_degradation", 0.2):
                self.alert_manager.trigger_alert(
                    message=f"Performance degradation detected for {pipeline_name}: "
                    f"{degradation:.1%} slower than baseline",
                    severity=AlertSeverity.WARNING,
                    metric_name=f"{pipeline_name}_performance_degradation",
                    metric_value=degradation,
                    threshold=self.thresholds["throughput_degradation"],
                    metadata={
                        "baseline_time": baseline,
                        "current_time": execution_time,
                        "pipeline": pipeline_name,
                    },
                )

    def _check_error_rate(self, pipeline_name: str) -> None:
        """Check error rate for a pipeline."""
        if pipeline_name not in self.request_counts or self.request_counts[pipeline_name] < 10:
            return  # Need at least 10 requests for meaningful error rate

        error_rate = self.error_counts[pipeline_name] / self.request_counts[pipeline_name]

        if error_rate > self.thresholds.get("error_rate", 0.05):
            self.alert_manager.trigger_alert(
                message=f"High error rate for {pipeline_name}: {error_rate:.1%}",
                severity=AlertSeverity.ERROR,
                metric_name=f"{pipeline_name}_error_rate",
                metric_value=error_rate,
                threshold=self.thresholds["error_rate"],
                metadata={
                    "total_requests": self.request_counts[pipeline_name],
                    "error_count": self.error_counts[pipeline_name],
                    "pipeline": pipeline_name,
                },
            )

    def get_pipeline_health_report(self) -> dict[str, Any]:
        """Get pipeline health report."""
        report = {
            "timestamp": time.time(),
            "monitoring_summary": self.get_monitoring_summary(),
            "pipeline_stats": {},
            "system_health": {},
        }

        # Pipeline statistics
        for pipeline_name in self.request_counts:
            total_requests = self.request_counts[pipeline_name]
            error_count = self.error_counts.get(pipeline_name, 0)
            error_rate = error_count / total_requests if total_requests > 0 else 0

            report["pipeline_stats"][pipeline_name] = {
                "total_requests": total_requests,
                "error_count": error_count,
                "error_rate": error_rate,
                "success_rate": 1.0 - error_rate,
            }

        # System health indicators
        if "gpu_memory_utilization" in self.metrics_history:
            gpu_history = list(self.metrics_history["gpu_memory_utilization"])
            if gpu_history:
                report["system_health"]["gpu_memory"] = {
                    "current": gpu_history[-1],
                    "average": sum(gpu_history) / len(gpu_history),
                    "max": max(gpu_history),
                }

        if "memory_usage_mb" in self.metrics_history:
            memory_history = list(self.metrics_history["memory_usage_mb"])
            if memory_history:
                report["system_health"]["memory"] = {
                    "current_mb": memory_history[-1],
                    "average_mb": sum(memory_history) / len(memory_history),
                    "max_mb": max(memory_history),
                }

        return report
