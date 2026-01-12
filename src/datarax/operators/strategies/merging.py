"""Merging utilities for parallel and ensemble strategies."""

from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import PyTree


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
        # Element-wise sum
        return jax.tree.map(lambda *args: sum(args), *outputs)
    elif merge_strategy == "mean":
        # Element-wise mean
        return jax.tree.map(lambda *args: jnp.mean(jnp.stack(args), axis=0), *outputs)
    elif merge_strategy == "dict":
        # Return dict with operator outputs separated by key
        # CRITICAL for vmap compatibility: Preserve input PyTree structure
        # Implementation: Use jax.tree.map to transform each leaf into operator dict

        def make_operator_dict(*values):
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
    # Convert conditions to boolean array
    cond_mask = jnp.stack(conditions)  # shape: (num_ops,) - use stack for 0-d arrays

    if merge_fn is not None:
        return merge_fn(outputs)

    if merge_strategy is None or merge_strategy == "concat":
        # Concat: concatenate ALL outputs (identity or transformed)
        # This gives fixed shapes required for vmap compatibility
        def concat_all(*args):
            return jnp.concatenate(args, axis=merge_axis)

        return jax.tree.map(concat_all, *outputs)

    elif merge_strategy == "stack":
        # Stack: stack ALL outputs
        def stack_all(*args):
            return jnp.stack(args, axis=merge_axis)

        return jax.tree.map(stack_all, *outputs)

    elif merge_strategy == "sum":
        # Sum: zero-mask False outputs using JAX ops
        def sum_masked(*args):
            stacked = jnp.stack(args, axis=0)  # (num_ops, ...)
            # Broadcast mask to match stacked shape
            mask_shape = [len(cond_mask)] + [1] * (stacked.ndim - 1)
            mask_broadcast = cond_mask.reshape(mask_shape)
            masked = stacked * mask_broadcast
            return jnp.sum(masked, axis=0)

        return jax.tree.map(sum_masked, *outputs)

    elif merge_strategy == "mean":
        # Mean: zero-mask False outputs, divide by True count
        def mean_masked(*args):
            stacked = jnp.stack(args, axis=0)  # (num_ops, ...)
            mask_shape = [len(cond_mask)] + [1] * (stacked.ndim - 1)
            mask_broadcast = cond_mask.reshape(mask_shape)
            masked = stacked * mask_broadcast
            total = jnp.sum(masked, axis=0)
            count = jnp.sum(cond_mask)
            return total / jnp.maximum(count, 1.0)

        return jax.tree.map(mean_masked, *outputs)

    elif merge_strategy == "dict":
        # Dict: include all operators (even False conditions)
        return merge_outputs(outputs, merge_strategy, merge_axis, merge_fn)

    else:
        raise ValueError(f"Unknown merge_strategy: {merge_strategy}")
