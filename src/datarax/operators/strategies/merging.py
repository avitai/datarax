"""Merging utilities for parallel and ensemble strategies."""

import logging
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import PyTree


logger = logging.getLogger(__name__)


def merge_outputs(
    outputs: list[PyTree],
    merge_strategy: str | None,
    merge_axis: int = 0,
    merge_fn: Callable | None = None,
) -> PyTree:
    """Merge parallel outputs based on strategy.

    Args:
        outputs: List of outputs from parallel operators
        merge_strategy: How to merge ("concat", "stack", "sum", "mean", "dict")
        merge_axis: Axis for stack/concat operations
        merge_fn: Custom merge function (overrides merge_strategy)

    Returns:
        Merged output
    """
    if not outputs:
        raise ValueError("outputs must not be empty")

    if merge_fn is not None:
        # Use custom merge function
        return merge_fn(outputs)

    if merge_strategy is None or merge_strategy == "concat":
        # Concatenate along merge_axis
        return jax.tree.map(
            lambda *args: jnp.concatenate(args, axis=merge_axis),
            *outputs,
        )
    elif merge_strategy == "stack":
        # Stack along merge_axis
        return jax.tree.map(lambda *args: jnp.stack(args, axis=merge_axis), *outputs)
    elif merge_strategy == "sum":
        # Element-wise sum with pure JAX reduction (trace/JIT friendly)
        return jax.tree.map(lambda *args: jnp.sum(jnp.stack(args, axis=0), axis=0), *outputs)
    elif merge_strategy == "mean":
        # Element-wise mean
        return jax.tree.map(lambda *args: jnp.mean(jnp.stack(args), axis=0), *outputs)
    elif merge_strategy == "dict":
        # Return dict with operator outputs separated by key
        # CRITICAL for vmap compatibility: Preserve input PyTree structure
        # Implementation: Use jax.tree.map to transform each leaf into operator dict

        def make_operator_dict(*values: Any) -> dict[str, Any]:
            """Create dict mapping operator_i to each operator's output value."""
            return {f"operator_{i}": val for i, val in enumerate(values)}

        # Apply to all leaves in the PyTree, preserving structure
        return jax.tree.map(make_operator_dict, *outputs)
    else:
        raise ValueError(f"Unknown merge_strategy: {merge_strategy}")


def merge_outputs_conditional(
    outputs: list[PyTree],
    conditions: list[jax.Array],
    merge_strategy: str | None,
    merge_axis: int = 0,
    merge_fn: Callable | None = None,
) -> PyTree:
    """Merge outputs with conditional masking (vmap-compatible).

    Args:
        outputs: List of ALL operator outputs (identity for False conditions)
        conditions: Boolean arrays indicating which operators executed
        merge_strategy: How to merge
        merge_axis: Axis used for merge
        merge_fn: Custom merge function

    Returns:
        Merged output with only True-condition outputs
    """
    if not outputs:
        raise ValueError("outputs must not be empty")
    if len(conditions) != len(outputs):
        raise ValueError(
            f"conditions length ({len(conditions)}) must match outputs length ({len(outputs)})"
        )

    # Convert conditions to boolean array
    cond_mask = jnp.stack(conditions)  # shape: (num_ops,) - use stack for 0-d arrays

    if merge_fn is not None:
        return merge_fn(outputs)

    if merge_strategy is None or merge_strategy == "concat":
        return jax.tree.map(lambda *args: jnp.concatenate(args, axis=merge_axis), *outputs)

    if merge_strategy == "stack":
        return jax.tree.map(lambda *args: jnp.stack(args, axis=merge_axis), *outputs)

    if merge_strategy == "sum":
        return jax.tree.map(lambda *args: _sum_masked(args, cond_mask), *outputs)

    if merge_strategy == "mean":
        return jax.tree.map(lambda *args: _mean_masked(args, cond_mask), *outputs)

    if merge_strategy == "dict":
        # Dict: include all operators (even False conditions)
        return merge_outputs(outputs, merge_strategy, merge_axis, merge_fn)

    raise ValueError(f"Unknown merge_strategy: {merge_strategy}")


def _masked_stack(values: tuple[jax.Array, ...], cond_mask: jax.Array) -> jax.Array:
    """Stack operator values and apply a broadcasted condition mask."""
    stacked = jnp.stack(values, axis=0)
    mask_shape = [len(cond_mask)] + [1] * (stacked.ndim - 1)
    return stacked * cond_mask.reshape(mask_shape)


def _sum_masked(values: tuple[jax.Array, ...], cond_mask: jax.Array) -> jax.Array:
    """Sum values from operators with True conditions."""
    masked = _masked_stack(values, cond_mask)
    return jnp.sum(masked, axis=0)


def _mean_masked(values: tuple[jax.Array, ...], cond_mask: jax.Array) -> jax.Array:
    """Average values from operators with True conditions."""
    total = _sum_masked(values, cond_mask)
    count = jnp.sum(cond_mask)
    return total / jnp.maximum(count, 1.0)
