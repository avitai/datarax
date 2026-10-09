"""Operations a data source is built from (public helpers for source authors).

The bundled sources (HFEagerSource, TFDSEagerSource, HFStreamingSource,
TFDSStreamingSource, MemorySource) and sources in other packages delegate to
these helpers for:
- Wrapped index resolution, in the order a key selects (``resolve_wrapped_indices``)
- A worker's share of the records (``partition_length``)
- A named-dataset source's repr (``format_source_repr``)
- Key filtering

In-memory sources read through ``datarax.sources.EagerSource``; streams build on
``datarax.sources.StreamingSourceBase``.
"""

from __future__ import annotations

from collections.abc import Set as AbstractSet
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from datarax.core import index_words
from datarax.core.index_shuffle import shuffle_positions


def partition_length(length: int, num_workers: int = 1, shard_id: int = 0) -> int:
    """Return how many records worker ``shard_id`` of ``num_workers`` serves.

    The worker serves positions ``[shard_id::num_workers]`` of the dataset order.

    Args:
        length: Total number of records in the dataset.
        num_workers: Number of workers the dataset is split across.
        shard_id: This worker's index.

    Returns:
        ``len(range(shard_id, length, num_workers))``, computed by integer division, which
        ``len`` cannot do past ``2**63 - 1`` positions.
    """
    return max(0, -(-(length - shard_id) // num_workers))


def resolve_wrapped_indices(
    start: int | ArrayLike,
    size: int,
    length: int,
    key: jax.Array | None,
    *,
    num_workers: int = 1,
    shard_id: int = 0,
) -> jax.Array:
    """Return the global record indices for a wrapped slice of a worker's record order.

    The dataset order is ``arange(length)``, or the keyed bijection
    :func:`~datarax.core.index_shuffle.shuffle_positions` of it when a ``key`` is given
    (the pipeline passes its epoch key exactly when it shuffles); each index costs O(1), so a
    slice costs O(size) at every
    dataset size and no order is stored. Worker ``shard_id`` of
    ``num_workers`` serves positions ``[shard_id::num_workers]`` of that order, and its
    logical positions ``start + arange(size)`` wrap at its own length. The same arguments
    always yield the same indices. Positions and indices are 64-bit, exact for every length up
    to ``2**64 - 1`` (:func:`~datarax.core.index_words.wrapped_positions`).

    Args:
        start: Starting logical position: a Python int of any size, its two uint32 words
            ``(hi, lo)`` (a position of the worker's order, below its length), or a traced
            int32 ``jax.Array``.
        size: Number of records to return (static Python int).
        length: Total number of records in the dataset.
        key: The key selecting the order, or ``None`` for the sequential order.
        num_workers: Number of workers the dataset is split across.
        shard_id: This worker's index.

    Returns:
        uint32 ``(size, 2)`` global record indices, each as its words ``(hi, lo)``.
    """
    worker_length = partition_length(length, num_workers, shard_id)
    positions = index_words.wrapped_positions(start, size, worker_length)
    if num_workers > 1 or shard_id:
        scaled = index_words.multiply_word(
            (positions[:, 0], positions[:, 1]), np.uint32(num_workers)
        )
        high, low = index_words.add(scaled, index_words.split_constant(shard_id))
        positions = jnp.stack([high, low], axis=-1)
    if key is not None:
        return shuffle_positions(positions, length, key)
    return positions


def format_source_repr(
    class_name: str,
    dataset_name: str | None,
    split_name: str | None,
    length: int | None,
    extra_fields: dict[str, Any] | None = None,
) -> str:
    """Format a stable source repr string with optional extra fields."""
    fields: list[tuple[str, Any]] = [
        ("dataset", f"{dataset_name}:{split_name}"),
        ("length", length),
    ]
    if extra_fields:
        fields.extend(extra_fields.items())
    serialized = ", ".join(f"{key}={value}" for key, value in fields)
    return f"{class_name}({serialized})"


def filter_keys(
    element: dict[str, Any],
    include_keys: AbstractSet[str] | None,
    exclude_keys: AbstractSet[str] | None,
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
