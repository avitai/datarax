"""Distributed metrics collection utilities for Datarax.

This module provides functions for collecting and aggregating metrics
across multiple devices in distributed training settings.

Two API variants are provided:

- **Default functions** (reduce_mean, reduce_sum, etc.): Use standard JAX
  operations (jnp.mean, jnp.sum) on global arrays. Work in SPMD contexts
  with nnx.jit + mesh.

- **Collective functions** (reduce_mean_collective, etc.): Use JAX collective
  operations (lax.pmean, lax.psum). Only valid inside pmap or shard_map contexts.
"""

from typing import Any

import jax
import jax.numpy as jnp
from jax import lax


# ---------------------------------------------------------------------------
# SPMD-compatible reductions (work with global arrays in nnx.jit)
# ---------------------------------------------------------------------------


def reduce_mean(metrics: dict[str, Any]) -> dict[str, Any]:
    """Compute the mean of metrics using standard JAX operations.

    Works with global arrays in SPMD contexts (nnx.jit + mesh).

    Args:
        metrics: The metrics to reduce.

    Returns:
        A dictionary of mean-reduced metrics.
    """

    def maybe_mean(x: Any) -> Any:
        if isinstance(x, jax.Array) and x.ndim > 0:
            return jnp.mean(x)
        return x

    return jax.tree.map(maybe_mean, metrics)


def reduce_sum(metrics: dict[str, Any]) -> dict[str, Any]:
    """Compute the sum of metrics using standard JAX operations.

    Works with global arrays in SPMD contexts (nnx.jit + mesh).

    Args:
        metrics: The metrics to reduce.

    Returns:
        A dictionary of summed metrics.
    """

    def maybe_sum(x: Any) -> Any:
        if isinstance(x, jax.Array) and x.ndim > 0:
            return jnp.sum(x)
        return x

    return jax.tree.map(maybe_sum, metrics)


def reduce_max(metrics: dict[str, Any]) -> dict[str, Any]:
    """Compute the maximum of metrics using standard JAX operations.

    Works with global arrays in SPMD contexts (nnx.jit + mesh).

    Args:
        metrics: The metrics to reduce.

    Returns:
        A dictionary of maximum metrics.
    """

    def maybe_max(x: Any) -> Any:
        if isinstance(x, jax.Array) and x.ndim > 0:
            return jnp.max(x)
        return x

    return jax.tree.map(maybe_max, metrics)


def reduce_min(metrics: dict[str, Any]) -> dict[str, Any]:
    """Compute the minimum of metrics using standard JAX operations.

    Works with global arrays in SPMD contexts (nnx.jit + mesh).

    Args:
        metrics: The metrics to reduce.

    Returns:
        A dictionary of minimum metrics.
    """

    def maybe_min(x: Any) -> Any:
        if isinstance(x, jax.Array) and x.ndim > 0:
            return jnp.min(x)
        return x

    return jax.tree.map(maybe_min, metrics)


_SPMD_REDUCTION_OPS: dict[str, Any] = {
    "mean": jnp.mean,
    "sum": jnp.sum,
    "max": jnp.max,
    "min": jnp.min,
}


def reduce_custom(
    metrics: dict[str, Any],
    reduce_fn: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """Apply custom reduction operations to metrics.

    Uses standard JAX operations. Works in SPMD contexts.

    Args:
        metrics: The metrics to reduce.
        reduce_fn: A dictionary mapping metric names to reduction operations.
            Each operation should be one of {"mean", "sum", "max", "min"}.
            If None, defaults to "mean" for all metrics.

    Returns:
        A dictionary of reduced metrics.
    """
    if reduce_fn is None:
        return reduce_mean(metrics)

    result = {}
    for key, value in metrics.items():
        operation = reduce_fn.get(key, "mean")
        op_fn = _SPMD_REDUCTION_OPS.get(operation) if operation else None

        if op_fn is not None and isinstance(value, jax.Array) and value.ndim > 0:
            result[key] = op_fn(value)
        else:
            result[key] = value

    return result


# ---------------------------------------------------------------------------
# Collective reductions (only valid inside pmap or shard_map)
# ---------------------------------------------------------------------------


def reduce_mean_collective(
    metrics: dict[str, Any],
    axis_name: str = "batch",
) -> dict[str, Any]:
    """Compute the mean of metrics using collective operations.

    Only valid inside a pmap or shard_map context.

    Args:
        metrics: The metrics to reduce.
        axis_name: The name of the axis to reduce across.

    Returns:
        A dictionary of mean metrics.
    """

    def maybe_mean(x: Any) -> Any:
        if isinstance(x, jax.Array):
            return lax.pmean(x, axis_name=axis_name)
        return x

    return jax.tree.map(maybe_mean, metrics)


def reduce_sum_collective(
    metrics: dict[str, Any],
    axis_name: str = "batch",
) -> dict[str, Any]:
    """Compute the sum of metrics using collective operations.

    Only valid inside a pmap or shard_map context.

    Args:
        metrics: The metrics to reduce.
        axis_name: The name of the axis to reduce across.

    Returns:
        A dictionary of summed metrics.
    """

    def maybe_sum(x: Any) -> Any:
        if isinstance(x, jax.Array):
            return lax.psum(x, axis_name=axis_name)
        return x

    return jax.tree.map(maybe_sum, metrics)


def all_gather(
    metrics: dict[str, Any],
    axis_name: str = "batch",
) -> dict[str, Any]:
    """Gather metrics from all devices.

    Only valid inside a pmap or shard_map context.

    Args:
        metrics: The metrics to gather.
        axis_name: The name of the axis to gather across.

    Returns:
        A dictionary of gathered metrics.
    """

    def maybe_gather(x: Any) -> Any:
        if isinstance(x, jax.Array):
            return lax.all_gather(x, axis_name=axis_name)
        return x

    return jax.tree.map(maybe_gather, metrics)


def collect_from_devices(metrics: dict[str, Any]) -> dict[str, list[Any] | Any]:
    """Collect metrics from all devices.

    Call outside of a pmapped function to split per-device values
    from the leading device axis.

    Args:
        metrics: The metrics from all devices, with the first dimension
            corresponding to the device axis.

    Returns:
        A dictionary of metrics, with array values split into per-device lists.
    """
    result: dict[str, list[Any] | Any] = {}
    for key, value in metrics.items():
        is_array = isinstance(value, jax.Array)
        if is_array and value.ndim > 0:
            result[key] = [value[i] for i in range(value.shape[0])]
        else:
            result[key] = value

    return result
