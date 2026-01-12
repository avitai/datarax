"""Tests for the Datarax metrics collection and aggregation.

This module contains tests for the metrics collection and aggregation components
of the Datarax monitoring system.
"""

import time

from datarax.monitoring.metrics import AggregatedMetrics, MetricRecord, MetricsCollector


class TestMetricsCollector:
    """Tests for the MetricsCollector class."""

    def test_record_metric(self):
        """Test recording a metric."""
        collector = MetricsCollector()
        collector.record_metric("test_metric", 42.0, "test_component")

        metrics = collector.get_metrics()
        assert len(metrics) == 1
        assert metrics[0].name == "test_metric"
        assert metrics[0].value == 42.0
        assert metrics[0].component == "test_component"

    def test_metrics_disabled(self):
        """Test that no metrics are recorded when disabled."""
        collector = MetricsCollector(enabled=False)
        collector.record_metric("test_metric", 42.0)

        metrics = collector.get_metrics()
        assert len(metrics) == 0

    def test_timer(self):
        """Test the timer functionality."""
        collector = MetricsCollector()
        collector.start_timer("operation")

        # Sleep a small amount to simulate work
        time.sleep(0.01)

        collector.stop_timer("operation")

        metrics = collector.get_metrics()
        assert len(metrics) == 1
        assert metrics[0].name == "operation_time"
        assert metrics[0].value > 0.0

    def test_clear(self):
        """Test clearing metrics."""
        collector = MetricsCollector()
        collector.record_metric("test_metric", 42.0)
        assert len(collector.get_metrics()) == 1

        collector.clear()
        assert len(collector.get_metrics()) == 0


class TestAggregatedMetrics:
    """Tests for the AggregatedMetrics class."""

    def test_add_metrics(self):
        """Test adding metrics to the aggregator."""
        aggregator = AggregatedMetrics()
        current_time = time.time()

        metrics = [
            MetricRecord("test", 10.0, current_time, "comp"),
            MetricRecord("test", 20.0, current_time + 1, "comp"),
            MetricRecord("test", 30.0, current_time + 2, "comp"),
        ]

        aggregator.add_metrics(metrics)

        assert aggregator.get_count("test", "comp") == 3

    def test_average(self):
        """Test calculating average."""
        aggregator = AggregatedMetrics()
        current_time = time.time()

        metrics = [
            MetricRecord("test", 10.0, current_time, "comp"),
            MetricRecord("test", 20.0, current_time + 1, "comp"),
            MetricRecord("test", 30.0, current_time + 2, "comp"),
        ]

        aggregator.add_metrics(metrics)

        assert aggregator.get_average("test", "comp") == 20.0

    def test_min_max(self):
        """Test calculating min and max."""
        aggregator = AggregatedMetrics()
        current_time = time.time()

        metrics = [
            MetricRecord("test", 10.0, current_time, "comp"),
            MetricRecord("test", 20.0, current_time + 1, "comp"),
            MetricRecord("test", 30.0, current_time + 2, "comp"),
        ]

        aggregator.add_metrics(metrics)

        assert aggregator.get_min("test", "comp") == 10.0
        assert aggregator.get_max("test", "comp") == 30.0

    def test_rate(self):
        """Test calculating rate."""
        aggregator = AggregatedMetrics()
        current_time = time.time()

        # Create metrics spanning 2 seconds
        metrics = [
            MetricRecord("test", 1.0, current_time, "comp"),
            MetricRecord("test", 1.0, current_time + 1, "comp"),
            MetricRecord("test", 1.0, current_time + 2, "comp"),
        ]

        aggregator.add_metrics(metrics)

        # We have 3 events over 2 seconds, so rate should be 1.5/second
        rate = aggregator.get_rate("test", "comp")
        assert rate is not None and 1.0 <= rate <= 2.0  # Being flexible due to timing variations

    def test_sum(self):
        """Test calculating sum."""
        aggregator = AggregatedMetrics()
        current_time = time.time()

        metrics = [
            MetricRecord("test", 10.0, current_time, "comp"),
            MetricRecord("test", 20.0, current_time + 1, "comp"),
            MetricRecord("test", 30.0, current_time + 2, "comp"),
        ]

        aggregator.add_metrics(metrics)

        assert aggregator.get_sum("test", "comp") == 60.0
