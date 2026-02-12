"""Zero-copy data conversion utilities optimized for performance.

This module provides DLPack-based zero-copy conversions from TensorFlow and
HuggingFace data to JAX arrays, with numpy fallbacks for incompatible types.

Performance: DLPack is ~10x faster than numpy intermediate for large tensors.

Key Functions:
    - tf_to_jax: Convert TensorFlow tensor to JAX array (DLPack when possible)
    - hf_to_jax: Convert HuggingFace element (PIL, numpy, etc.) to JAX array
    - convert_batch_to_jax: Convert entire batch using specified converter
"""

from __future__ import annotations

from typing import Any
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np


# DLPack-compatible dtypes for TensorFlow->JAX zero-copy transfer
# These dtypes support DLPack protocol without data type conversion
_TF_DLPACK_COMPATIBLE_DTYPES: frozenset[str] = frozenset(
    [
        "float16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "bool",
        "bfloat16",
    ]
)


def tf_to_jax(tf_tensor: Any) -> jax.Array:
    """Convert TensorFlow tensor to JAX array using DLPack when possible.

    This function attempts zero-copy conversion via DLPack for compatible dtypes,
    falling back to numpy intermediate conversion otherwise.

    Performance: DLPack is ~10x faster than numpy intermediate for large tensors
    because it avoids memory copy between frameworks.

    Args:
        tf_tensor: TensorFlow tensor to convert.

    Returns:
        JAX array containing the data.

    Note:
        DLPack zero-copy requires:
        - Compatible dtype (float32, int32, uint8, etc.)
        - Contiguous memory layout
        - TensorFlow 2.x with dlpack support

    Example:
        >>> import tensorflow as tf
        >>> tf_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        >>> jax_array = tf_to_jax(tf_tensor)
        >>> print(jax_array.shape)
        (2, 2)
    """
    try:
        import tensorflow as tf

        # Check if tensor dtype is DLPack-compatible
        dtype_name = tf_tensor.dtype.name
        if dtype_name in _TF_DLPACK_COMPATIBLE_DTYPES:
            # DLPack zero-copy path
            # to_dlpack creates a DLPack capsule without copying data
            dlpack = tf.experimental.dlpack.to_dlpack(tf_tensor)
            return jax.dlpack.from_dlpack(dlpack)
    except (ImportError, AttributeError, RuntimeError):
        # TensorFlow not available, dlpack not supported, or conversion failed
        pass
    except Exception:
        # Any other DLPack error - fall back to numpy
        pass

    # Fallback: numpy intermediate (slower but always works)
    # This path is taken for:
    # - String tensors
    # - Complex numbers
    # - Sparse tensors
    # - Non-contiguous tensors
    if hasattr(tf_tensor, "numpy"):
        return jnp.array(tf_tensor.numpy())
    return jnp.array(tf_tensor)


def hf_to_jax(value: Any) -> jax.Array:
    """Convert HuggingFace element to JAX array.

    This function handles common HuggingFace data types:
    - PIL Images: Converted via numpy
    - NumPy arrays: Direct conversion
    - Lists/tuples: Converted via jnp.array
    - Scalars: Wrapped in array

    Args:
        value: HuggingFace dataset element (PIL Image, numpy array, list, scalar).

    Returns:
        JAX array containing the data.

    Example:
        >>> from PIL import Image
        >>> img = Image.new('RGB', (28, 28))
        >>> jax_array = hf_to_jax(img)
        >>> print(jax_array.shape)
        (28, 28, 3)
    """
    # Handle PIL Images (common in HuggingFace image datasets)
    # PIL Images have a 'mode' attribute (e.g., 'RGB', 'L')
    if hasattr(value, "mode"):
        # Convert PIL Image to numpy, then to JAX
        return jnp.array(np.array(value))

    # Handle numpy arrays
    if isinstance(value, np.ndarray):
        return jnp.array(value)

    # Handle objects with __array__ protocol (e.g., torch tensors)
    if hasattr(value, "__array__"):
        return jnp.array(value)

    # Handle sequences (lists, tuples)
    if isinstance(value, list | tuple):
        try:
            return jnp.array(value)
        except (ValueError, TypeError):
            # Can't convert to array (e.g., list of strings)
            # Return as-is, caller must handle
            return value  # type: ignore[return-value]

    # Handle scalars and everything else
    return jnp.array(value)


def convert_batch_to_jax(
    batch: dict[str, Any],
    converter: Callable[[Any], jax.Array],
) -> dict[str, jax.Array]:
    """Convert entire batch dictionary using specified converter function.

    This utility applies a converter function (tf_to_jax or hf_to_jax)
    to all values in a batch dictionary.

    Args:
        batch: Dictionary mapping keys to data values.
        converter: Function to convert each value to JAX array.

    Returns:
        Dictionary with all values converted to JAX arrays.

    Example:
        >>> import tensorflow as tf
        >>> batch = {"image": tf.constant([1, 2, 3]), "label": tf.constant(5)}
        >>> jax_batch = convert_batch_to_jax(batch, tf_to_jax)
    """
    return {k: converter(v) for k, v in batch.items()}


def stack_batches(batches: list[dict[str, jax.Array]]) -> dict[str, jax.Array]:
    """Stack a list of batch dictionaries into a single batch.

    This utility concatenates batches along the first (batch) dimension,
    useful for collecting all data from an iterator into a single array.

    Args:
        batches: List of dictionaries, each with JAX arrays of shape (batch_size, ...).

    Returns:
        Single dictionary with concatenated arrays.

    Raises:
        ValueError: If batches list is empty.

    Example:
        >>> batch1 = {"image": jnp.ones((32, 28, 28))}
        >>> batch2 = {"image": jnp.ones((32, 28, 28))}
        >>> combined = stack_batches([batch1, batch2])
        >>> print(combined["image"].shape)
        (64, 28, 28)
    """
    if not batches:
        raise ValueError("Cannot stack empty list of batches")

    keys = batches[0].keys()
    result = {}

    for key in keys:
        arrays = [batch[key] for batch in batches]
        result[key] = jnp.concatenate(arrays, axis=0)

    return result
