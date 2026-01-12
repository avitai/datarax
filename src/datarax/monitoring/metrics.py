"""Metrics collection and tracking utilities for Datarax.

This module provides classes for collecting, tracking, and aggregating metrics
during pipeline operation. These utilities enable monitoring of pipeline
performance, throughput, and resource usage.
"""

from dataclasses import dataclass
from time import time
from typing import Any


@dataclass
class MetricRecord:
    """A single metric measurement.

    Attributes:
        name: Name of the metric.
        value: Measured value.
        timestamp: Time when the metric was recorded.
        component: Component that generated the metric.
        metadata: Optional additional metadata about the metric.
    """

    name: str
    value: float
    timestamp: float
    component: str
    metadata: dict[str, Any] | None = None


class MetricsCollector:
    """Collects and aggregates metrics during pipeline operations.

    This class provides methods for tracking time-based metrics and recording
    arbitrary metrics during pipeline execution.

    Attributes:
        enabled: Whether metrics collection is enabled.
    """

    def __init__(self, enabled: bool = True):
        """Initialize a new MetricsCollector.

        Args:
            enabled: Whether metrics collection is enabled.
        """
        self.enabled = enabled
        self._metrics: list[MetricRecord] = []
        self._start_times: dict[str, float] = {}

    def start_timer(self, name: str, component: str = "pipeline"):
        """Start timing an operation.

        Args:
            name: Name of the timer.
            component: Component being timed.
        """
        if not self.enabled:
            return
        self._start_times[(name, component)] = time()

    def stop_timer(
        self, name: str, component: str = "pipeline", metadata: dict[str, Any] | None = None
    ):
        """Stop timing and record elapsed time.

        Args:
            name: Name of the timer.
            component: Component being timed.
            metadata: Optional additional metadata about the timing.
        """
        if not self.enabled:
            return

        key = (name, component)
        if key not in self._start_times:
            return

        elapsed = time() - self._start_times[key]
        self.record_metric(f"{name}_time", elapsed, component, metadata)
        del self._start_times[key]

    def record_metric(
        self,
        name: str,
        value: float,
        component: str = "pipeline",
        metadata: dict[str, Any] | None = None,
    ):
        """Record a metric value.

        Args:
            name: Name of the metric.
            value: Value to record.
            component: Component generating the metric.
            metadata: Optional additional metadata about the metric.
        """
        if not self.enabled:
            return

        self._metrics.append(
            MetricRecord(
                name=name,
                value=value,
                timestamp=time(),
                component=component,
                metadata=metadata or {},
            )
        )

    def get_metrics(self) -> list[MetricRecord]:
        """Get all collected metrics.

        Returns:
            List of collected metric records.
        """
        return self._metrics

    def clear(self):
        """Clear all metrics and timers."""
        self._metrics = []
        self._start_times = {}


class AggregatedMetrics:
    """Aggregates metrics over time.

    This class provides methods for calculating aggregate statistics over
    collected metrics.
    """

    def __init__(self):
        """Initialize a new AggregatedMetrics instance."""
        self._values: dict[str, list[float]] = {}
        self._timestamps: dict[str, list[float]] = {}
        self._components: dict[str, str] = {}

    def add_metrics(self, metrics: list[MetricRecord]):
        """Add metrics to the aggregator.

        Args:
            metrics: List of metric records to add.
        """
        for metric in metrics:
            key = f"{metric.component}.{metric.name}"
            if key not in self._values:
                self._values[key] = []
                self._timestamps[key] = []
                self._components[key] = metric.component

            self._values[key].append(metric.value)
            self._timestamps[key].append(metric.timestamp)

    def get_average(self, name: str, component: str = "pipeline") -> float | None:
        """Get the average value of a metric.

        Args:
            name: Name of the metric.
            component: Component that generated the metric.

        Returns:
            Average value, or None if no metrics were collected.
        """
        key = f"{component}.{name}"
        values = self._values.get(key, [])
        if not values:
            return None
        return sum(values) / len(values)

    def get_min(self, name: str, component: str = "pipeline") -> float | None:
        """Get the minimum value of a metric.

        Args:
            name: Name of the metric.
            component: Component that generated the metric.

        Returns:
            Minimum value, or None if no metrics were collected.
        """
        key = f"{component}.{name}"
        values = self._values.get(key, [])
        if not values:
            return None
        return min(values)

    def get_max(self, name: str, component: str = "pipeline") -> float | None:
        """Get the maximum value of a metric.

        Args:
            name: Name of the metric.
            component: Component that generated the metric.

        Returns:
            Maximum value, or None if no metrics were collected.
        """
        key = f"{component}.{name}"
        values = self._values.get(key, [])
        if not values:
            return None
        return max(values)

    def get_sum(self, name: str, component: str = "pipeline") -> float | None:
        """Get the sum of values for a metric.

        Args:
            name: Name of the metric.
            component: Component that generated the metric.

        Returns:
            Sum of values, or None if no metrics were collected.
        """
        key = f"{component}.{name}"
        values = self._values.get(key, [])
        if not values:
            return None
        return sum(values)

    def get_count(self, name: str, component: str = "pipeline") -> int:
        """Get the number of measurements for a metric.

        Args:
            name: Name of the metric.
            component: Component that generated the metric.

        Returns:
            Number of measurements.
        """
        key = f"{component}.{name}"
        return len(self._values.get(key, []))

    def get_rate(self, name: str, component: str = "pipeline") -> float | None:
        """Get the rate of a metric (count / time period).

        Args:
            name: Name of the metric.
            component: Component that generated the metric.

        Returns:
            Rate (occurrences per second), or None if less than two measurements
            exist.
        """
        key = f"{component}.{name}"
        timestamps = self._timestamps.get(key, [])

        if len(timestamps) < 2:
            return None

        time_period = max(timestamps) - min(timestamps)
        if time_period <= 0:
            return None

        return len(timestamps) / time_period

    def clear(self):
        """Clear all aggregated metrics."""
        self._values = {}
        self._timestamps = {}
        self._components = {}
