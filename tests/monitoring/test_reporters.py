"""Tests for the Datarax monitoring reporters.

This module contains tests for the various reporters in the Datarax
monitoring system, such as console and file reporters.
"""

import io
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from datarax.monitoring.metrics import MetricRecord
from datarax.monitoring.reporters import ConsoleReporter, FileReporter


class TestConsoleReporter:
    """Tests for the ConsoleReporter class."""

    def test_update_report_interval(self):
        """Test that updates only trigger reports after the interval."""
        # Use a StringIO object as the output
        output = io.StringIO()
        with patch("time.time") as mock_time:
            # First call to time.time() during initialization
            mock_time.return_value = 100.0

            reporter = ConsoleReporter(
                report_interval=10.0,  # 10 second interval
                output=output,
            )

            # Manually reset the last_report_time to ensure clean state
            reporter.last_report_time = 100.0

            # Create a metric
            metrics = [MetricRecord("test", 42.0, 100.0, "test")]

            # First update - shouldn't report (same time)
            mock_time.return_value = 100.0
            reporter.update(metrics)
            assert output.getvalue() == ""

            # Second update - still shouldn't report (5 seconds later)
            mock_time.return_value = 105.0
            reporter.update(metrics)
            assert output.getvalue() == ""

            # Third update - should report (11 seconds later)
            mock_time.return_value = 111.0
            reporter.update(metrics)
            assert len(output.getvalue()) > 0

    def test_report_format(self):
        """Test the format of the report."""
        # Use a StringIO object as the output
        output = io.StringIO()
        reporter = ConsoleReporter(
            report_interval=0.0,  # Always report
            output=output,
        )

        current_time = time.time()
        metrics = [
            MetricRecord("latency", 0.1, current_time, "pipeline"),
            MetricRecord("throughput", 100.0, current_time, "pipeline"),
        ]

        reporter.update(metrics)

        # Verify something was printed
        report = output.getvalue()
        assert "Datarax Metrics Report" in report
        assert "latency" in report
        assert "throughput" in report


def test_file_reporter_context_manager():
    """Test that FileReporter works as a context manager and closes file on exit."""
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        path = tf.name
    try:
        with FileReporter(filename=path, report_interval=0.0) as reporter:
            reporter.update([MetricRecord("test", 1.0, time.time(), "test")])
        # File should be closed after context exit
        assert reporter.file.closed
    finally:
        Path(path).unlink()


def test_file_reporter_context_manager_on_exception():
    """Test that FileReporter closes file even when exception occurs."""
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        path = tf.name
    try:
        with pytest.raises(RuntimeError):
            with FileReporter(filename=path, report_interval=0.0) as reporter:
                reporter.update([MetricRecord("test", 1.0, time.time(), "test")])
                raise RuntimeError("test error")
        assert reporter.file.closed
    finally:
        Path(path).unlink()
