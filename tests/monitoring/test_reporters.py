"""Tests for the Datarax monitoring reporters.

This module contains tests for the various reporters in the Datarax
monitoring system, such as console and file reporters.
"""

import io
import os
import tempfile
import time
from unittest.mock import patch

import pytest

import numpy as np
import flax.nnx as nnx

from datarax.monitoring.metrics import MetricRecord
from datarax.monitoring.pipeline import MonitoredPipeline
from datarax.monitoring.reporters import ConsoleReporter, FileReporter
from datarax.sources import MemorySource
from datarax.sources.memory_source import MemorySourceConfig


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


def test_file_reporter():
    """Test the FileReporter."""
    # Create test data
    data = {"value": np.arange(50)}
    source = MemorySource(MemorySourceConfig(), data, rngs=nnx.Rngs(0))

    # Create a monitored pipeline
    pipeline = MonitoredPipeline(source).batch(1)

    # Create a temporary file for the reporter
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_filename = temp_file.name

        try:
            # Create and register the file reporter
            reporter = FileReporter(
                filename=temp_filename,
                report_interval=0.0,  # Report immediately
            )
            pipeline.callbacks.register(reporter)

            # Process all data
            list(pipeline)

            # Verify the report file was created and contains data
            with open(temp_filename, "r") as f:
                content = f.read()
                assert "Datarax Metrics Report" in content
                assert "pipeline" in content

        finally:
            # Clean up
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)


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
        os.unlink(path)


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
        os.unlink(path)


def test_end_to_end_monitoring_with_console_reporter():
    """Test end-to-end monitoring with console reporting."""
    # Create test data
    data = {"value": np.arange(100)}
    source = MemorySource(MemorySourceConfig(), data, rngs=nnx.Rngs(0))

    # Create a monitored pipeline
    pipeline = MonitoredPipeline(source).batch(1)

    # Create a string buffer to capture console output
    import io

    output = io.StringIO()

    # Create and register the console reporter
    reporter = ConsoleReporter(
        report_interval=0.0,  # Report immediately
        output=output,
    )
    pipeline.callbacks.register(reporter)

    # Process data directly from the monitored pipeline
    list(pipeline)

    # Verify that the console report was generated
    console_output = output.getvalue()
    assert "Datarax Metrics Report" in console_output
    assert "pipeline" in console_output
