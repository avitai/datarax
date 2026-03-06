"""Callback system for Datarax monitoring.

This module implements the observer pattern for metrics collection and event
handling in Datarax pipelines.
"""

import logging

from datarax.monitoring.metrics import MetricRecord


logger = logging.getLogger(__name__)


class MetricsObserver:
    """Base class for metrics observers.

    Metrics observers receive notifications about metrics collected during
    pipeline operation.
    """

    def update(self, metrics: list[MetricRecord]) -> None:
        """Handle updated metrics.

        Args:
            metrics: List of new metric records.
        """
        raise NotImplementedError("Subclasses must implement update()")


class CallbackRegistry:
    """Registry for callbacks during pipeline operation.

    This class manages a collection of observers that receive notifications
    about metrics and events.
    """

    def __init__(self) -> None:
        """Initialize a new CallbackRegistry."""
        self._observers: list[MetricsObserver] = []

    def register(self, observer: MetricsObserver) -> None:
        """Register an observer.

        Args:
            observer: Observer to register.
        """
        if observer not in self._observers:
            self._observers.append(observer)

    def unregister(self, observer: MetricsObserver) -> bool:
        """Unregister an observer.

        Args:
            observer: Observer to unregister.

        Returns:
            True if the observer was found and removed, False otherwise.
        """
        if observer in self._observers:
            self._observers.remove(observer)
            return True
        return False

    def notify(self, metrics: list[MetricRecord]) -> None:
        """Notify all observers of new metrics.

        Args:
            metrics: List of new metric records.
        """
        for observer in self._observers:
            observer.update(metrics)

    def clear(self) -> None:
        """Remove all registered observers."""
        self._observers = []
