"""Utility functions for working with JAX PyTrees and Batches.

This module provides a collection of helper functions to inspect, manipulate,
and validate JAX PyTrees and Datarax Batch objects. It encompasses:
- Type checking utilities (is_array, is_container, is_jax_array)
- Batch-specific leaf predicates (is_batch_leaf, is_non_jax_leaf)
- Batch manipulation helpers (split_batch, concatenate_batches)
- Dimensionality transformations (add/remove batch dimensions)
- Structure introspection and consistency validation
"""

from typing import Any, Callable

import jax
import numpy as np
from jaxtyping import Array

from datarax.core.element_batch import Batch, BatchOps, Element


# ============================================================================
# Core type-checking helpers (extracted from batching for DRY reuse)
# ============================================================================


def is_array(x: Any) -> bool:
    """Check if x is a JAX or numpy array.

    Args:
        x: Value to check

    Returns:
        True if x is a JAX or numpy array, False otherwise
    """
    return isinstance(x, jax.Array | np.ndarray)


def is_container(x: Any) -> bool:
    """Check if x is a pytree container type.

    Args:
        x: Value to check

    Returns:
        True if x is dict, list, or tuple, False otherwise
    """
    return isinstance(x, dict | list | tuple)


def is_jax_array(x: Any) -> bool:
    """Check if x is a JAX array or compatible numeric type.

    Args:
        x: Value to check

    Returns:
        True if x is JAX-compatible, False otherwise
    """
    return isinstance(x, jax.Array | int | float | bool | complex)


def is_batch_leaf(x: Any) -> bool:
    """Check if x should be treated as a leaf for batching operations.

    Arrays and non-containers are leaves. Containers (dict, list, tuple)
    are traversed as pytree structure.

    Args:
        x: Value to check

    Returns:
        True if x is a leaf, False otherwise
    """
    return is_array(x) or not is_container(x)


def is_non_jax_leaf(x: Any) -> bool:
    """Check if x should be treated as a leaf for rebatching operations.

    For rebatching, Python lists and tuples are treated as atomic data
    payloads (e.g., [0, 1, 2] as batch labels) rather than containers.
    Only dicts are traversed as pytree structure.

    Args:
        x: Value to check

    Returns:
        True if x should be treated as a leaf, False otherwise

    Examples:
        batch = {"data": jnp.array([1, 2]), "labels": [0, 1]}
        jax.tree.map(fn, batch, is_leaf=is_non_jax_leaf)
        # labels list is treated as a leaf, not traversed
    """
    # Arrays are leaves (stop traversal at data)
    if is_array(x):
        return True
    # Python lists/tuples are batch data payloads - treat as leaves
    if isinstance(x, list | tuple):
        return True
    return False


def get_batch_size(batch: Batch | dict) -> int | None:
    """Extract batch size from a Batch object or plain dict.

    Supports both proper Batch objects (with .batch_size property) and
    plain dicts containing JAX arrays (infers size from first axis).

    Args:
        batch: Batch object or dict containing JAX arrays

    Returns:
        Batch size, or None if batch size cannot be determined
    """
    # Handle proper Batch objects
    if hasattr(batch, "batch_size"):
        return batch.batch_size

    # Handle plain dicts (common in tests and simple pipelines)
    if isinstance(batch, dict):
        # Find first array and get its first dimension
        for value in batch.values():
            if hasattr(value, "shape") and len(value.shape) > 0:
                return value.shape[0]

    return None


def is_single_element(data: Element | Batch) -> bool:
    """Determine if data is a single element or a batch.

    Args:
        data: Element or Batch object to check

    Returns:
        True if single element, False if batch
    """
    return isinstance(data, Element)


def add_batch_dimension(element: Element) -> Batch:
    """Add batch dimension to single element by creating a Batch.

    Args:
        element: Single Element

    Returns:
        Batch containing the single element
    """
    return Batch([element])


def remove_batch_dimension(batch: Batch) -> Element:
    """Remove batch dimension from a batch of size 1.

    Args:
        batch: Batch with size 1

    Returns:
        Single element extracted from batch

    Raises:
        ValueError: If batch size is not 1
    """
    if batch.batch_size != 1:
        raise ValueError(f"Cannot remove batch dimension from batch of size {batch.batch_size}")
    return batch.get_element(0)


def split_batch(batch: Batch, num_splits: int) -> list[Batch]:
    """Split a batch into multiple smaller batches.

    Delegates to Batch.split_for_devices() for consistent implementation.

    Args:
        batch: Batch to split
        num_splits: Number of splits to create

    Returns:
        List of smaller batches

    Raises:
        ValueError: If batch size is not divisible by num_splits
    """
    # Delegate to existing Batch method (DRY principle)
    return batch.split_for_devices(num_splits)


def concatenate_batches(batches: list[Batch]) -> Batch:
    """Concatenate multiple batches into a single batch.

    Delegates to BatchOps.concatenate_batches() for consistent implementation.

    Args:
        batches: List of Batch objects with same structure

    Returns:
        Single concatenated batch

    Raises:
        ValueError: If batches list is empty
    """
    if not batches:
        raise ValueError("Cannot concatenate empty list of batches")

    # Delegate to existing BatchOps method (DRY principle)
    return BatchOps.concatenate_batches(batches)


def apply_to_batch_dimension(
    batch: Batch, fn: Callable[..., Array], axis: int = 0, keepdims: bool = False
) -> dict[str, Array]:
    """Apply a reduction function along the batch dimension.

    Uses jax.tree.map for idiomatic PyTree traversal.

    Args:
        batch: Batch object
        fn: Reduction function to apply (e.g., jnp.mean, jnp.std, jnp.sum)
        axis: Axis to apply function along (default 0 for batch)
        keepdims: Whether to keep the reduced dimension

    Returns:
        Dictionary with function applied along batch dimension to each data field
    """

    def apply_fn(array: Any) -> Any:
        """Apply reduction if array has shape, otherwise pass through."""
        if hasattr(array, "shape") and len(array.shape) > 0:
            return fn(array, axis=axis, keepdims=keepdims)
        return array

    # Use jax.tree.map for idiomatic PyTree traversal
    return jax.tree.map(apply_fn, batch.data.get_value())


def validate_batch_consistency(batch: Batch) -> bool:
    """Validate that all arrays in a batch have consistent batch dimensions.

    Args:
        batch: Batch object to validate

    Returns:
        True if batch is consistent, False otherwise
    """
    try:
        batch._validate()
        return True
    except ValueError:
        return False


def get_pytree_structure_info(data: Element | Batch) -> dict[str, Any]:
    """Get information about Element or Batch structure for debugging.

    Args:
        data: Element or Batch to analyze

    Returns:
        Dictionary with structure information
    """
    if isinstance(data, Element):
        element = data
        leaf_shapes = []
        leaf_dtypes = []

        # Collect info from data arrays using jax.tree utilities
        for key, array in element.data.items():
            if hasattr(array, "shape"):
                leaf_shapes.append((key, array.shape))
                if hasattr(array, "dtype"):
                    leaf_dtypes.append((key, str(array.dtype)))

        return {
            "type": "Element",
            "num_data_fields": len(element.data),
            "data_fields": list(element.data.keys()),
            "leaf_shapes": leaf_shapes,
            "leaf_dtypes": leaf_dtypes,
            "batch_size": None,
            "is_single_element": True,
            "is_batch_consistent": True,
            "has_state": bool(element.state),
            "has_metadata": element.metadata is not None,
        }
    else:
        batch = data
        leaf_shapes = []
        leaf_dtypes = []

        # Collect info from batch data arrays
        data_val = batch.data.get_value()
        for key, array in data_val.items():
            if hasattr(array, "shape"):
                leaf_shapes.append((key, array.shape))
                if hasattr(array, "dtype"):
                    leaf_dtypes.append((key, str(array.dtype)))

        states_val = batch.states.get_value()
        return {
            "type": "Batch",
            "batch_size": batch.batch_size,
            "num_data_fields": len(data_val),
            "data_fields": list(data_val.keys()),
            "leaf_shapes": leaf_shapes,
            "leaf_dtypes": leaf_dtypes,
            "is_single_element": False,
            "is_batch_consistent": validate_batch_consistency(batch),
            "num_elements": len(states_val),
        }
