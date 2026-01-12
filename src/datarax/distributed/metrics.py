"""Distributed metrics collection utilities for Datarax.

This module provides a unified implementation for collecting and aggregating
metrics across multiple devices in distributed training settings.
Supports both static method usage and NNX module instantiation.
"""

from typing import Any, Union

import flax.nnx as nnx
import jax
from jax import lax


class DistributedMetrics(nnx.Module):
    """Unified distributed metrics collection utilities for Datarax.

    This class provides methods for aggregating metrics across multiple devices
    in a distributed training setting, including mean, sum, and custom reduction
    operations.

    Supports both static method usage (stateless) and instance method usage (stateful).

    Usage:
        # Static method usage (stateless)
        metrics = DistributedMetrics.reduce_mean_static(metrics)

        # Instance method usage (stateful)
        dm = DistributedMetrics()
        metrics = dm.reduce_mean(metrics)
    """

    def __init__(self):
        """Initialize DistributedMetrics module."""
        super().__init__()

    def all_gather(self, metrics: dict[str, Any], axis_name: str = "batch") -> dict[str, Any]:
        """Gather metrics from all devices.

        Args:
            metrics: The metrics to gather.
            axis_name: The name of the axis to gather across.

        Returns:
            A dictionary of gathered metrics.
        """

        def maybe_gather(x):
            if isinstance(x, jax.Array | jax.Array):
                return jax.lax.all_gather(x, axis_name=axis_name)
            return x

        return jax.tree.map(maybe_gather, metrics)

    @staticmethod
    def all_gather_static(metrics: dict[str, Any], axis_name: str = "batch") -> dict[str, Any]:
        """Static version of all_gather."""
        return jax.tree.map(
            lambda x: jax.lax.all_gather(x, axis_name=axis_name)
            if isinstance(x, jax.Array | jax.Array)
            else x,
            metrics,
        )

    def reduce_mean(self, metrics: dict[str, Any], axis_name: str = "batch") -> dict[str, Any]:
        """Compute the mean of metrics across devices.

        Args:
            metrics: The metrics to reduce.
            axis_name: The name of the axis to reduce across.

        Returns:
            A dictionary of mean metrics.
        """

        def maybe_mean(x):
            if isinstance(x, jax.Array | jax.Array):
                return lax.pmean(x, axis_name=axis_name)
            return x

        return jax.tree.map(maybe_mean, metrics)

    @staticmethod
    def reduce_mean_static(metrics: dict[str, Any], axis_name: str = "batch") -> dict[str, Any]:
        """Static version of reduce_mean."""
        return jax.tree.map(
            lambda x: lax.pmean(x, axis_name=axis_name)
            if isinstance(x, jax.Array | jax.Array)
            else x,
            metrics,
        )

    def reduce_sum(self, metrics: dict[str, Any], axis_name: str = "batch") -> dict[str, Any]:
        """Compute the sum of metrics across devices.

        Args:
            metrics: The metrics to reduce.
            axis_name: The name of the axis to reduce across.

        Returns:
            A dictionary of summed metrics.
        """

        def maybe_sum(x):
            if isinstance(x, jax.Array | jax.Array):
                return lax.psum(x, axis_name=axis_name)
            return x

        return jax.tree.map(maybe_sum, metrics)

    @staticmethod
    def reduce_sum_static(metrics: dict[str, Any], axis_name: str = "batch") -> dict[str, Any]:
        """Static version of reduce_sum."""
        return jax.tree.map(
            lambda x: lax.psum(x, axis_name=axis_name)
            if isinstance(x, jax.Array | jax.Array)
            else x,
            metrics,
        )

    def reduce_max(self, metrics: dict[str, Any], axis_name: str = "batch") -> dict[str, Any]:
        """Compute the maximum of metrics across devices.

        Args:
            metrics: The metrics to reduce.
            axis_name: The name of the axis to reduce across.

        Returns:
            A dictionary of maximum metrics.
        """

        def maybe_max(x):
            if isinstance(x, jax.Array | jax.Array):
                return lax.pmax(x, axis_name=axis_name)
            return x

        return jax.tree.map(maybe_max, metrics)

    @staticmethod
    def reduce_max_static(metrics: dict[str, Any], axis_name: str = "batch") -> dict[str, Any]:
        """Static version of reduce_max."""
        return jax.tree.map(
            lambda x: lax.pmax(x, axis_name=axis_name)
            if isinstance(x, jax.Array | jax.Array)
            else x,
            metrics,
        )

    def reduce_min(self, metrics: dict[str, Any], axis_name: str = "batch") -> dict[str, Any]:
        """Compute the minimum of metrics across devices.

        Args:
            metrics: The metrics to reduce.
            axis_name: The name of the axis to reduce across.

        Returns:
            A dictionary of minimum metrics.
        """

        def maybe_min(x):
            if isinstance(x, jax.Array | jax.Array):
                return lax.pmin(x, axis_name=axis_name)
            return x

        return jax.tree.map(maybe_min, metrics)

    @staticmethod
    def reduce_min_static(metrics: dict[str, Any], axis_name: str = "batch") -> dict[str, Any]:
        """Static version of reduce_min."""
        return jax.tree.map(
            lambda x: lax.pmin(x, axis_name=axis_name)
            if isinstance(x, jax.Array | jax.Array)
            else x,
            metrics,
        )

    def reduce_custom(
        self,
        metrics: dict[str, Any],
        reduce_fn: dict[str, str | None] | None = None,
        axis_name: str = "batch",
    ) -> dict[str, Any]:
        """Apply custom reduction operations to metrics.

        Args:
            metrics: The metrics to reduce.
            reduce_fn: A dictionary mapping metric names to reduction operations.
                Each operation should be one of {"mean", "sum", "max", "min"}.
                If None, defaults to "mean" for all metrics.
            axis_name: The name of the axis to reduce across.

        Returns:
            A dictionary of reduced metrics.
        """
        if reduce_fn is None:
            # Default to mean reduction for all metrics
            return self.reduce_mean(metrics, axis_name)

        result = {}
        for key, value in metrics.items():
            operation = reduce_fn.get(key, "mean")
            is_array = isinstance(value, jax.Array | jax.Array)

            if operation == "mean" and is_array:
                result[key] = lax.pmean(value, axis_name=axis_name)
            elif operation == "sum" and is_array:
                result[key] = lax.psum(value, axis_name=axis_name)
            elif operation == "max" and is_array:
                result[key] = lax.pmax(value, axis_name=axis_name)
            elif operation == "min" and is_array:
                result[key] = lax.pmin(value, axis_name=axis_name)
            else:
                # If unknown operation or not an array, keep the value as is
                result[key] = value

        return result

    @staticmethod
    def reduce_custom_static(
        metrics: dict[str, Any],
        reduce_fn: dict[str, str | None] | None = None,
        axis_name: str = "batch",
    ) -> dict[str, Any]:
        """Static version of reduce_custom."""
        if reduce_fn is None:
            # Default to mean reduction for all metrics
            return DistributedMetrics.reduce_mean_static(metrics, axis_name)

        result = {}
        for key, value in metrics.items():
            operation = reduce_fn.get(key, "mean")

            if operation == "mean":
                result[key] = (
                    lax.pmean(value, axis_name=axis_name)
                    if isinstance(value, jax.Array | jax.Array)
                    else value
                )
            elif operation == "sum":
                result[key] = (
                    lax.psum(value, axis_name=axis_name)
                    if isinstance(value, jax.Array | jax.Array)
                    else value
                )
            elif operation == "max":
                result[key] = (
                    lax.pmax(value, axis_name=axis_name)
                    if isinstance(value, jax.Array | jax.Array)
                    else value
                )
            elif operation == "min":
                result[key] = (
                    lax.pmin(value, axis_name=axis_name)
                    if isinstance(value, jax.Array | jax.Array)
                    else value
                )
            else:
                # If unknown operation, just keep the value as is
                result[key] = value

        return result

    def collect_from_devices(self, metrics: dict[str, Any]) -> dict[str, Union[list[Any], Any]]:
        """Collect metrics from all devices.

        This function should be called outside of a pmapped function to collect
        metrics from all devices.

        Args:
            metrics: The metrics from all devices, with the first dimension
                corresponding to the device axis.

        Returns:
            A dictionary of metrics, with each value being a list of the values
            from each device.
        """
        result: dict[str, Union[list[Any], Any]] = {}
        for key, value in metrics.items():
            is_array = isinstance(value, jax.Array | jax.Array)
            has_dim = is_array and value.ndim > 0

            if has_dim:
                # If array, collect the values from each device
                device_values = [value[i] for i in range(value.shape[0])]
                result[key] = device_values
            else:
                # If not array or scalar, just keep the value
                result[key] = value

        return result

    @staticmethod
    def collect_from_devices_static(metrics: dict[str, Any]) -> dict[str, Union[list[Any], Any]]:
        """Static version of collect_from_devices."""
        result: dict[str, Union[list[Any], Any]] = {}
        for key, value in metrics.items():
            if isinstance(value, jax.Array | jax.Array) and value.ndim > 0:
                # If array, collect the values from each device
                device_values = [value[i] for i in range(value.shape[0])]
                result[key] = device_values
            else:
                # If not array or scalar, just keep the value
                result[key] = value

        return result
