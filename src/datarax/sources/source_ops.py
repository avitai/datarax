"""Operations a data source is built from (public helpers for source authors).

The bundled sources (HFEagerSource, TFDSEagerSource, HFStreamingSource,
TFDSStreamingSource, MemorySource) and sources in other packages delegate to
these helpers for:
- Wrapped, optionally shuffled index resolution (``resolve_wrapped_indices``)
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

import logging
from collections.abc import Callable, Iterator
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from datarax.samplers.index_shuffle import index_shuffle


def partition_length(length: int, num_workers: int = 1, shard_id: int = 0) -> int:
    """Return how many records worker ``shard_id`` of ``num_workers`` serves.

    The worker serves positions ``[shard_id::num_workers]`` of the dataset order.

    Args:
        length: Total number of records in the dataset.
        num_workers: Number of workers the dataset is split across.
        shard_id: This worker's index.

    Returns:
        ``len(range(shard_id, length, num_workers))``.
    """
    return len(range(shard_id, length, num_workers))


def resolve_wrapped_indices(  # noqa: PLR0913 - the order, its partition and the slice are separate
    start: jax.Array | int,
    size: int,
    length: int,
    is_random_order: bool,
    key: jax.Array | None,
    *,
    num_workers: int = 1,
    shard_id: int = 0,
    order: jax.Array | None = None,
) -> jax.Array:
    """Return the global record indices for a wrapped slice of a worker's record order.

    The dataset order is ``arange(length)``, or a full-dataset permutation derived from
    ``key`` when ``is_random_order`` is set and a ``key`` is supplied. Worker ``shard_id`` of
    ``num_workers`` serves positions ``[shard_id::num_workers]`` of that order, and its
    logical positions ``start + arange(size)`` wrap at its own length. The same arguments
    always yield the same indices.

    Args:
        start: Starting logical position (concrete int or traced ``jax.Array``).
        size: Number of records to return (static Python int).
        length: Total number of records in the dataset.
        is_random_order: Whether the source serves records in shuffled order.
        key: PRNG key for shuffled mode; ignored when ``is_random_order`` is
            False or ``key`` is None.
        num_workers: Number of workers the dataset is split across.
        shard_id: This worker's index.
        order: The epoch's permutation of ``length`` when the caller already holds it
            (an :class:`EpochOrderCache`); ``None`` derives it from ``key`` here, which
            costs O(length) per call.

    Returns:
        Int32 ``jax.Array`` of shape ``(size,)`` with the global record indices.
    """
    worker_length = partition_length(length, num_workers, shard_id)
    start_arr = jnp.asarray(start, dtype=jnp.int32)
    offsets = jnp.arange(size, dtype=jnp.int32)
    positions = (start_arr + offsets) % jnp.int32(worker_length)
    global_positions = jnp.int32(shard_id) + positions * jnp.int32(num_workers)
    if is_random_order and key is not None:
        permutation = jax.random.permutation(key, length) if order is None else order
        return permutation[global_positions]
    return global_positions


class EpochOrderCache(nnx.Module):
    """The permutation one epoch serves, computed once per key and reused by every batch.

    A shuffled source is asked for records position by position, with the epoch's key
    each time; deriving the full permutation on every call is O(length) per batch. The
    cache holds the last key's permutation and recomputes only when the key changes
    (under ``lax.cond``, so the compiled step runs the permutation once per epoch). The
    stored order and key are NNX state, so a compiled iteration session carries them.
    """

    def __init__(self, length: int) -> None:
        """Start empty; the first ``order_for`` fills the cache.

        Args:
            length: Number of records the permutation covers.
        """
        self.length = length
        key_shape = jax.random.key_data(jax.random.key(0)).shape
        self._key_data = nnx.Variable(jnp.zeros(key_shape, dtype=jnp.uint32))
        self._order = nnx.Variable(jnp.arange(length, dtype=jnp.int32))
        self._filled = nnx.Variable(jnp.zeros((), dtype=jnp.bool_))

    def order_for(self, key: jax.Array) -> jax.Array:
        """The permutation ``jax.random.permutation(key, length)``, from the cache when it holds it.

        Args:
            key: The epoch's key.

        Returns:
            Int32 array of shape ``(length,)``.
        """
        key_data = jax.random.key_data(key)
        same = jnp.logical_and(self._filled[...], jnp.all(key_data == self._key_data[...]))
        order = jax.lax.cond(
            same,
            lambda: self._order[...],
            lambda: jax.random.permutation(key, self.length).astype(jnp.int32),
        )
        self._order[...] = order
        self._key_data[...] = key_data
        self._filled[...] = jnp.ones((), dtype=jnp.bool_)
        return order


logger = logging.getLogger(__name__)


def configure_stochastic_from_shuffle(
    config: Any,
    *,
    shuffle: bool,
    default_stream_name: str = "shuffle",
) -> None:
    """Set stochastic and stream_name based on shuffle flag."""
    if shuffle:
        object.__setattr__(config, "stochastic", True)
        if config.stream_name is None:
            object.__setattr__(config, "stream_name", default_stream_name)
    else:
        object.__setattr__(config, "stochastic", False)


def validate_required_name_split(
    name: str | None,
    split: str | None,
    config_class_name: str,
) -> None:
    """Validate required name/split fields."""
    if name is None:
        raise ValueError(f"name is required for {config_class_name}")
    if split is None:
        raise ValueError(f"split is required for {config_class_name}")


def validate_include_exclude_keys(
    include_keys: set[str] | None,
    exclude_keys: set[str] | None,
) -> None:
    """Validate include/exclude key filters are not both set."""
    if include_keys is not None and exclude_keys is not None:
        raise ValueError("Cannot specify both include_keys and exclude_keys")


def validate_seed_range(seed: int) -> None:
    """Validate shuffle seed range for Grain index_shuffle compatibility."""
    if seed < 0 or seed >= 2**32:
        raise ValueError("seed must be in [0, 2**32)")


def validate_positive_optional_int(value: int | None, field_name: str) -> None:
    """Validate an optional integer field is positive when provided."""
    if value is not None and value < 1:
        raise ValueError(f"{field_name} must be a positive integer")


def shuffled_index_for_position(
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

    per_epoch_seed = (seed + epoch) % (2**32)
    return index_shuffle(index=index, seed=per_epoch_seed, num_elements=length)


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
        idx = shuffled_index_for_position(i, shuffle, seed, epoch, length)
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
        shuffled_index_for_position(i, shuffle, seed, epoch, length) for i in range(start, end)
    ]

    index_var.set_value(end % length)
    if end >= length:
        epoch_var.set_value(epoch + 1)

    return gather_fn(data, jnp.array(shuffled_indices))


def eager_reset(
    index_var: Any,
    epoch_var: Any,
) -> None:
    """Shared reset logic for eager sources.

    Args:
        index_var: nnx.Variable for current index
        epoch_var: nnx.Variable for current epoch
    """
    index_var.set_value(0)
    epoch_var.set_value(0)


def build_eager_element(data: dict[str, Any], idx: int) -> dict[str, Any]:
    """Build one element from eager in-memory data for a given index."""
    return {k: v[idx] if isinstance(v, jax.Array) else v[idx] for k, v in data.items()}


def get_eager_item(data: dict[str, Any], length: int, index: int) -> dict[str, Any]:
    """Return one eager element with bounds checking and negative indexing."""
    resolved_index = index + length if index < 0 else index
    if resolved_index < 0 or resolved_index >= length:
        raise IndexError(f"Index {index} out of range for {length} elements")
    return build_eager_element(data, resolved_index)


def gather_eager_batch(data: dict[str, Any], indices: jax.Array) -> dict[str, Any]:
    """Gather an eager batch for JAX arrays and Python sequence leaves."""
    batch: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, jax.Array):
            batch[k] = v[indices]
        else:
            batch[k] = [v[int(i)] for i in indices]
    return batch


def eager_iter_default(
    data: dict[str, Any],
    length: int,
    index_var: Any,
    epoch_var: Any,
    shuffle: bool,
    seed: int,
) -> Iterator[dict[str, Any]]:
    """Iterate eager source data with the shared default element builder."""
    return eager_iter(data, length, index_var, epoch_var, shuffle, seed, build_eager_element)


def eager_get_batch_default(
    data: dict[str, Any],
    length: int,
    index_var: Any,
    epoch_var: Any,
    shuffle: bool,
    seed: int,
    batch_size: int,
    key: jax.Array | None,
) -> dict[str, Any]:
    """Get eager source batches with the shared default gather function."""
    return eager_get_batch(
        data,
        length,
        index_var,
        epoch_var,
        shuffle,
        seed,
        batch_size,
        key,
        gather_eager_batch,
    )


def format_source_repr(
    class_name: str,
    dataset_name: str | None,
    split_name: str | None,
    length: int | None,
    shuffle: bool,
    epoch: int,
    extra_fields: dict[str, Any] | None = None,
) -> str:
    """Format a stable source repr string with optional extra fields."""
    fields: list[tuple[str, Any]] = [
        ("dataset", f"{dataset_name}:{split_name}"),
        ("length", length),
        ("shuffle", shuffle),
    ]
    if extra_fields:
        fields.extend(extra_fields.items())
    fields.append(("epoch", epoch))
    serialized = ", ".join(f"{key}={value}" for key, value in fields)
    return f"{class_name}({serialized})"


def streaming_apply_batch(
    next_item: Callable[[], Any],
    batch_size: int,
) -> dict[str, Any]:
    """Collect up to batch_size elements from a streaming iterator."""
    elements = []
    for _ in range(batch_size):
        try:
            elements.append(next_item())
        except StopIteration:
            break
    return batch_elements_to_dict(elements)


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
    validate_required_name_split(name, split, config_class_name)
    validate_include_exclude_keys(include_keys, exclude_keys)
    if try_gcs and data_dir is not None:
        raise ValueError(
            f"Cannot specify both try_gcs=True and data_dir='{data_dir}' in {config_class_name}. "
            "try_gcs overrides data_dir to the public GCS bucket (gs://tfds-data/datasets/)."
        )


def finalize_eager_config_validation(
    *,
    config: Any,
    super_post_init: Callable[[], None],
    config_class_name: str,
    name: str | None,
    split: str | None,
    include_keys: set[str] | None,
    exclude_keys: set[str] | None,
    seed: int,
    try_gcs: bool = False,
    data_dir: str | None = None,
) -> None:
    """Run shared eager-config validation flow."""
    configure_stochastic_from_shuffle(config, shuffle=config.shuffle)
    if config.shuffle:
        validate_seed_range(seed)
    super_post_init()
    validate_eager_config(
        name,
        split,
        include_keys,
        exclude_keys,
        config_class_name,
        try_gcs=try_gcs,
        data_dir=data_dir,
    )


def finalize_streaming_config_validation(
    *,
    config: Any,
    super_post_init: Callable[[], None],
    config_class_name: str,
    name: str | None,
    split: str | None,
    include_keys: set[str] | None,
    exclude_keys: set[str] | None,
) -> None:
    """Run shared streaming-config validation flow."""
    configure_stochastic_from_shuffle(config, shuffle=config.shuffle)
    super_post_init()
    validate_required_name_split(name, split, config_class_name)
    validate_include_exclude_keys(include_keys, exclude_keys)


def _get_super_post_init(config: Any) -> Callable[[], None]:
    """Get the __post_init__ method from the parent class of the given config instance."""
    parent_post_init = getattr(super(type(config), config), "__post_init__", None)
    if parent_post_init is None:

        def noop() -> None:
            pass

        return noop
    return parent_post_init


def validate_eager_source_settings(
    config: Any,
    config_class_name: str,
    *,
    seed: int,
    try_gcs: bool = False,
    data_dir: str | None = None,
) -> None:
    """Validate a source eager config using standard dataclass fields."""
    finalize_eager_config_validation(
        config=config,
        super_post_init=_get_super_post_init(config),
        config_class_name=config_class_name,
        name=config.name,
        split=config.split,
        include_keys=config.include_keys,
        exclude_keys=config.exclude_keys,
        seed=seed,
        try_gcs=try_gcs,
        data_dir=data_dir,
    )


def validate_streaming_source_settings(
    config: Any,
    config_class_name: str,
) -> None:
    """Validate a source streaming config using standard dataclass fields."""
    finalize_streaming_config_validation(
        config=config,
        super_post_init=_get_super_post_init(config),
        config_class_name=config_class_name,
        name=config.name,
        split=config.split,
        include_keys=config.include_keys,
        exclude_keys=config.exclude_keys,
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


def converted_filtered_record(
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
) -> None:
    """Reset streaming source state to initial values.

    Args:
        epoch_var: nnx.Variable for current epoch.
    """
    epoch_var.set_value(0)
