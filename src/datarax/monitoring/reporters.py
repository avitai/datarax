"""Reporters for Datarax monitoring.

This module provides implementations of MetricsObservers that report metrics
to various outputs like console, files, or external systems.
"""

import sys
import time
from pathlib import Path
from typing import Any, TextIO, cast

from datarax.monitoring.callbacks import MetricsObserver
from datarax.monitoring.metrics import AggregatedMetrics, MetricRecord


class ConsoleReporter(MetricsObserver):
    """Reports metrics to the console.

    This reporter displays metrics in a formatted way to the console,
    with optional filtering and aggregation.
    """

    def __init__(
        self,
        report_interval: float = 5.0,
        show_components: bool = True,
        filter_components: set[str] | None = None,
        filter_metrics: set[str] | None = None,
        output: TextIO = sys.stdout,
    ):
        """Initialize a new ConsoleReporter.

        Args:
            report_interval: Minimum time in seconds between reports.
            show_components: Whether to show component names in reports.
            filter_components: Set of component names to include, or None for all.
            filter_metrics: Set of metric names to include, or None for all.
            output: Text stream to write reports to.
        """
        self.report_interval = report_interval
        self.show_components = show_components
        self.filter_components = filter_components
        self.filter_metrics = filter_metrics
        self.output = output
        self.last_report_time = 0.0
        self.aggregated = AggregatedMetrics()

    def should_include(self, metric: MetricRecord) -> bool:
        """Check if a metric should be included in reports.

        Args:
            metric: Metric record to check.

        Returns:
            True if the metric should be included, False otherwise.
        """
        if self.filter_components and metric.component not in self.filter_components:
            return False
        if self.filter_metrics and metric.name not in self.filter_metrics:
            return False
        return True

    def update(self, metrics: list[MetricRecord]):
        """Handle updated metrics.

        Args:
            metrics: List of new metric records.
        """
        # Filter metrics
        filtered_metrics = [m for m in metrics if self.should_include(m)]
        if not filtered_metrics:
            return

        # Add to aggregated metrics
        self.aggregated.add_metrics(filtered_metrics)

        # Check if it's time to report
        current_time = time.time()
        if current_time - self.last_report_time < self.report_interval:
            return

        # Generate and print report
        report = self._generate_report()
        self.output.write(report)
        self.output.flush()

        # Update last report time
        self.last_report_time = current_time

    def _generate_report(self) -> str:
        """Generate a formatted report of current metrics.

        Returns:
            Formatted report string.
        """
        lines = ["\n=== Datarax Metrics Report ==="]

        # Get all unique metric keys
        all_metrics = set()
        all_components = set()

        for key in self.aggregated._values.keys():
            component, metric = key.split(".", 1)
            all_metrics.add(metric)
            all_components.add(component)

        # Sort components and metrics for consistent output
        sorted_components = sorted(all_components)
        sorted_metrics = sorted(all_metrics)

        # Generate the report
        for component in sorted_components:
            if self.show_components:
                lines.append(f"\n== {component} ==")

            for metric in sorted_metrics:
                # Check if this component has this metric
                key = f"{component}.{metric}"
                if key not in self.aggregated._values:
                    continue

                # Get statistics
                count = self.aggregated.get_count(metric, component)
                avg = self.aggregated.get_average(metric, component)
                min_val = self.aggregated.get_min(metric, component)
                max_val = self.aggregated.get_max(metric, component)

                # Format the metric line
                if self.show_components:
                    metric_line = f"{metric}: "
                else:
                    metric_line = f"{component}.{metric}: "

                if "_time" in metric and avg is not None:
                    # Format time metrics in milliseconds
                    # Ensure avg, min_val, and max_val are not None
                    avg_ms = 0.0 if avg is None else avg * 1000
                    min_ms = 0.0 if min_val is None else min_val * 1000
                    max_ms = 0.0 if max_val is None else max_val * 1000

                    metric_line += (
                        f"avg={avg_ms:.2f}ms, min={min_ms:.2f}ms, max={max_ms:.2f}ms, count={count}"
                    )
                else:
                    # Format regular metrics
                    # Ensure avg, min_val, and max_val are not None
                    avg_val = 0.0 if avg is None else avg
                    min_v = 0.0 if min_val is None else min_val
                    max_v = 0.0 if max_val is None else max_val

                    metric_line += (
                        f"avg={avg_val:.4f}, min={min_v:.4f}, max={max_v:.4f}, count={count}"
                    )

                lines.append(metric_line)

        lines.append("\n")
        return "\n".join(lines)

    def clear(self):
        """Clear all aggregated metrics."""
        self.aggregated.clear()
        self.last_report_time = 0.0


class FileReporter(ConsoleReporter):
    """Reports metrics to a file.

    This extends ConsoleReporter to write metrics to a file instead of stdout.
    By default, metrics are written to the temp/monitoring directory.
    """

    def __init__(
        self,
        filename: str,
        mode: str = "a",
        report_interval: float = 60.0,
        **kwargs: Any,
    ):
        """Initialize a new FileReporter.

        Args:
            filename: Path to the file to write reports to. If not an absolute path,
                it will be placed in temp/monitoring/ directory by default.
            mode: File open mode, typically "a" for append or "w" for write.
            report_interval: Minimum time in seconds between reports.
            **kwargs: Additional arguments to pass to ConsoleReporter.
        """
        # Ensure the file is written to the temp/monitoring directory by default
        filepath = Path(filename)
        if not filepath.is_absolute():
            # Create the temp/monitoring directory if it doesn't exist
            Path("temp/monitoring").mkdir(parents=True, exist_ok=True)
            # Prepend the temp/monitoring path if filename is relative
            if not filename.startswith("temp/"):
                filepath = Path("temp/monitoring") / filepath.name
                filename = str(filepath)

        # Ensure the directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        self.file = Path(filename).open(mode)
        # Cast file to TextIO to satisfy type checker
        file_output = cast(TextIO, self.file)
        super().__init__(
            report_interval=report_interval,
            output=file_output,
            **kwargs,
        )

    def close(self):
        """Close the file handle. Idempotent."""
        if hasattr(self, "file") and self.file and not self.file.closed:
            self.file.close()

    def __enter__(self) -> "FileReporter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self):
        """Safety net — prefer using as context manager."""
        self.close()
