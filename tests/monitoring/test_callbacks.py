"""Tests for the Datarax monitoring callbacks.

This module contains tests for the callbacks and observer components
of the Datarax monitoring system.
"""

import time
from unittest.mock import MagicMock

from datarax.monitoring.callbacks import CallbackRegistry, MetricsObserver
from datarax.monitoring.metrics import MetricRecord


class TestCallbackRegistry:
    """Tests for the CallbackRegistry class."""

    def test_register_and_notify(self):
        """Test registering an observer and notifying it."""
        registry = CallbackRegistry()
        observer = MagicMock(spec=MetricsObserver)

        registry.register(observer)

        metrics = [MetricRecord("test", 42.0, time.time(), "test")]
        registry.notify(metrics)

        observer.update.assert_called_once_with(metrics)

    def test_multiple_observers(self):
        """Test notifying multiple observers."""
        registry = CallbackRegistry()
        observer1 = MagicMock(spec=MetricsObserver)
        observer2 = MagicMock(spec=MetricsObserver)

        registry.register(observer1)
        registry.register(observer2)

        metrics = [MetricRecord("test", 42.0, time.time(), "test")]
        registry.notify(metrics)

        observer1.update.assert_called_once_with(metrics)
        observer2.update.assert_called_once_with(metrics)
