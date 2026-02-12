"""Shared operations for eager and streaming data sources (composition helpers).

These standalone functions extract duplicated patterns from HFEagerSource,
TFDSEagerSource, HFStreamingSource, and TFDSStreamingSource. Sources
delegate to these helpers for:
- Shuffled index computation (Grain's Feistel cipher)
- Iteration with O(1) memory shuffling
- Batch retrieval (stateless and stateful)
- Reset logic (eager and streaming)
- Config validation
- Key filtering
- Element conversion and batch stacking (streaming)

Design: Functions accept nnx.Variable references as arguments so mutations
propagate correctly back to the caller (Flax NNX reference semantics).
Streaming helpers use callback parameters (convert_fn) to stay backend-agnostic.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Callable, Iterator

import jax
import jax.numpy as jnp


def get_shuffled_index(
    index: int,
    shuffle: bool,
    seed: int,
    epoch: int,
    length: int,
) -> int:
    """Get shuffled index using Grain's O(1) memory Feistel cipher.

    Args:
        index: Original sequential index
        shuffle: Whether shuffling is enabled
        seed: Base integer seed for Grain's index_shuffle
        epoch: Current epoch (used for per-epoch seeding)
        length: Total number of elements

    Returns:
        Shuffled index for current epoch, or original index if shuffle=False
    """
    if not shuffle:
        return index

    try:
        from grain._src.python.experimental.index_shuffle.python import (
            index_shuffle_module as index_shuffle,
        )

        per_epoch_seed = (seed + epoch) % (2**32)
        return index_shuffle.index_shuffle(
            index, max_index=length - 1, seed=per_epoch_seed, rounds=4
        )
    except ImportError:
        key = jax.random.key(seed + epoch)
        perm = jax.random.permutation(key, length)
        return int(perm[index])


def eager_iter(
    data: dict[str, Any],
    length: int,
    index_var: Any,
    epoch_var: Any,
    shuffle: bool,
    seed: int,
    build_element: Callable[[dict[str, Any], int], dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    """Shared iteration pattern for eager sources.

    Args:
        data: The data dictionary
        length: Total number of elements
        index_var: nnx.Variable for current index
        epoch_var: nnx.Variable for current epoch
        shuffle: Whether to shuffle
        seed: Base shuffle seed
        build_element: Callback to build element dict from data and index.
            Signature: (data, idx) -> dict. This handles type differences
            between HF (mixed types) and TFDS (all JAX arrays).

    Yields:
        Data elements in (optionally shuffled) order
    """
    index_var.set_value(0)
    epoch_var.set_value(epoch_var.get_value() + 1)
    epoch = epoch_var.get_value()

    for i in range(length):
        idx = get_shuffled_index(i, shuffle, seed, epoch, length)
        yield build_element(data, idx)


def eager_get_batch(
    data: dict[str, Any],
    length: int,
    index_var: Any,
    epoch_var: Any,
    shuffle: bool,
    seed: int,
    batch_size: int,
    key: jax.Array | None,
    gather_fn: Callable[[dict[str, Any], Any], dict[str, Any]],
) -> dict[str, Any]:
    """Shared batch retrieval for eager sources.

    Args:
        data: The data dictionary
        length: Total number of elements
        index_var: nnx.Variable for current index
        epoch_var: nnx.Variable for current epoch
        shuffle: Whether to shuffle
        seed: Base shuffle seed
        batch_size: Number of elements in the batch
        key: Optional RNG key for stateless mode
        gather_fn: Callback to gather batch from data given indices.
            Signature: (data, indices_array) -> dict

    Returns:
        Batch of data as dictionary
    """
    if key is not None:
        # Stateless mode
        if shuffle:
            indices = jax.random.permutation(key, length)[:batch_size]
        else:
            indices = jnp.arange(batch_size)
        return gather_fn(data, indices)

    # Stateful mode
    start = index_var.get_value()
    end = min(start + batch_size, length)
    epoch = epoch_var.get_value()

    shuffled_indices = [
        get_shuffled_index(i, shuffle, seed, epoch, length) for i in range(start, end)
    ]

    index_var.set_value(end % length)
    if end >= length:
        epoch_var.set_value(epoch + 1)

    return gather_fn(data, jnp.array(shuffled_indices))


def eager_reset(
    index_var: Any,
    epoch_var: Any,
    cache: Any | None,
) -> None:
    """Shared reset logic for eager sources.

    Args:
        index_var: nnx.Variable for current index
        epoch_var: nnx.Variable for current epoch
        cache: Optional cache to clear
    """
    index_var.set_value(0)
    epoch_var.set_value(0)
    if cache is not None:
        cache.clear()


def validate_eager_config(
    name: str | None,
    split: str | None,
    include_keys: set[str] | None,
    exclude_keys: set[str] | None,
    config_class_name: str,
    *,
    try_gcs: bool = False,
    data_dir: str | None = None,
) -> None:
    """Shared config validation for eager source configs.

    Args:
        name: Dataset name (required)
        split: Dataset split (required)
        include_keys: Optional include keys
        exclude_keys: Optional exclude keys
        config_class_name: Name of config class for error messages
        try_gcs: Whether to load from GCS (mutually exclusive with data_dir)
        data_dir: Custom data directory (mutually exclusive with try_gcs)

    Raises:
        ValueError: If validation fails
    """
    if name is None:
        raise ValueError(f"name is required for {config_class_name}")
    if split is None:
        raise ValueError(f"split is required for {config_class_name}")
    if include_keys is not None and exclude_keys is not None:
        raise ValueError("Cannot specify both include_keys and exclude_keys")
    if try_gcs and data_dir is not None:
        raise ValueError(
            f"Cannot specify both try_gcs=True and data_dir='{data_dir}' in {config_class_name}. "
            "try_gcs overrides data_dir to the public GCS bucket (gs://tfds-data/datasets/)."
        )


def filter_keys(
    element: dict[str, Any],
    include_keys: set[str] | None,
    exclude_keys: set[str] | None,
) -> dict[str, Any]:
    """Filter element keys based on include/exclude sets.

    Args:
        element: Dictionary to filter
        include_keys: If set, only include these keys
        exclude_keys: If set, exclude these keys

    Returns:
        Filtered dictionary
    """
    if include_keys is not None:
        return {k: v for k, v in element.items() if k in include_keys}
    if exclude_keys is not None:
        return {k: v for k, v in element.items() if k not in exclude_keys}
    return element


# =============================================================================
# Streaming source helpers
# =============================================================================


def convert_and_filter_element(
    element: dict[str, Any],
    include_keys: set[str] | None,
    exclude_keys: set[str] | None,
    convert_fn: Callable[[Any], Any],
) -> dict[str, Any]:
    """Convert and filter a raw element from a streaming source.

    Applies key filtering then converts each value via the backend-specific
    convert_fn (e.g., hf_to_jax or tf_to_jax). For HF sources, also promotes
    scalars to JAX arrays; for TFDS, convert_fn handles everything.

    Args:
        element: Raw element dict from the dataset iterator.
        include_keys: If set, only include these keys.
        exclude_keys: If set, exclude these keys.
        convert_fn: Backend-specific conversion (value -> JAX array or passthrough).

    Returns:
        Filtered and converted element dict.
    """
    result: dict[str, Any] = {}
    for k, v in element.items():
        if include_keys and k not in include_keys:
            continue
        if exclude_keys and k in exclude_keys:
            continue
        converted = convert_fn(v)
        if isinstance(converted, jax.Array):
            result[k] = converted
        elif isinstance(converted, int | float | bool):
            result[k] = jnp.array(converted)
        else:
            result[k] = converted
    return result


def batch_elements_to_dict(elements: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack a list of element dicts into a single batched dict.

    JAX arrays are stacked, numeric scalars are arrayed, and other types
    (strings, etc.) are collected as lists.

    Args:
        elements: List of element dicts with consistent keys.

    Returns:
        Batched dict where each value is stacked/collected across elements.
    """
    if not elements:
        return {}
    keys = elements[0].keys()
    batch: dict[str, Any] = {}
    for k in keys:
        values = [elem[k] for elem in elements]
        first_val = values[0]
        if isinstance(first_val, jax.Array):
            batch[k] = jnp.stack(values)
        elif isinstance(first_val, int | float | bool):
            batch[k] = jnp.array(values)
        else:
            batch[k] = values
    return batch


def reset_streaming_state(
    epoch_var: Any,
    cache: Any | None,
) -> None:
    """Reset streaming source state to initial values.

    Args:
        epoch_var: nnx.Variable for current epoch.
        cache: Optional cache to clear.
    """
    epoch_var.set_value(0)
    if cache is not None:
        cache.clear()
