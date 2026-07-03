"""Shared source base classes for eager and streaming backends."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from datarax.core.data_source import DataSourceModule
from datarax.sources._eager_source_ops import (
    eager_get_batch_default,
    eager_iter_default,
    eager_reset,
    format_source_repr,
    gather_eager_batch,
    get_eager_item,
    reset_streaming_state,
    streaming_apply_batch,
)
from datarax.sources._grain_bridge import records_from_batched_mapping, validate_index_batch


logger = logging.getLogger(__name__)


def resolve_wrapped_indices(
    start: jax.Array | int,
    size: int,
    length: int,
    is_random_order: bool,
    key: jax.Array | None,
) -> jax.Array:
    """Return the record indices for a wrapped, optionally shuffled slice.

    Computes ``(start + arange(size)) % length`` and, when ``is_random_order``
    is set and a ``key`` is supplied, gathers those positions through a
    deterministic full-dataset permutation derived from ``key``. Same
    ``(start, size, length, key)`` always yields the same indices.

    Args:
        start: Starting logical index (concrete int or traced ``jax.Array``).
        size: Number of records to return (static Python int).
        length: Total number of records in the dataset.
        is_random_order: Whether the source serves records in shuffled order.
        key: PRNG key for shuffled mode; ignored when ``is_random_order`` is
            False or ``key`` is None.

    Returns:
        Int32 ``jax.Array`` of shape ``(size,)`` with the resolved indices.
    """
    start_arr = jnp.asarray(start, dtype=jnp.int32)
    offsets = jnp.arange(size, dtype=jnp.int32)
    base_indices = (start_arr + offsets) % jnp.int32(length)
    if is_random_order and key is not None:
        permutation = jax.random.permutation(key, length)
        return permutation[base_indices]
    return base_indices


class EagerSourceBase(DataSourceModule):
    """Shared eager-source behavior for in-memory JAX-backed datasets.

    Subclasses must define the following attributes in their ``__init__``:

    - ``data`` (``dict[str, Any]``): The loaded dataset as a key→array mapping.
    - ``length`` (``int``): Total number of elements.
    - ``index`` (``nnx.Variable``): Current iteration index.
    - ``epoch`` (``nnx.Variable``): Current epoch counter.
    - ``_seed`` (``int``): Base integer seed for Grain index_shuffle.
    - ``_is_random_order`` (``bool``): Whether to randomize iteration order.
    - ``dataset_name`` (``str | None``): Human-readable dataset name.
    - ``split_name`` (``str | None``): Dataset split identifier.
    - ``_dataset_info`` (``Any``): Cached backend-specific dataset metadata.
    """

    # -- Abstract attribute declarations (set by concrete subclasses) --
    data: dict[str, Any]
    length: int
    index: nnx.Variable[int]  # pyright: ignore[reportGeneralTypeIssues]
    epoch: nnx.Variable[int]  # pyright: ignore[reportGeneralTypeIssues]
    _seed: int
    _is_random_order: bool
    dataset_name: str | None
    split_name: str | None
    _dataset_info: Any

    def __len__(self) -> int:
        """Return total number of elements."""
        return self.length

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate through eager data with optional deterministic shuffling."""
        return eager_iter_default(
            self.data,
            self.length,
            self.index,
            self.epoch,
            self.is_random_order,
            self._seed,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Retrieve one eager element by index."""
        return get_eager_item(self.data, self.length, index)

    def _getitems(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        """Retrieve multiple eager elements with vectorized array-leaf indexing."""
        resolved = validate_index_batch(indices, self.length)
        batch = gather_eager_batch(self.data, jnp.asarray(resolved))
        return records_from_batched_mapping(batch, len(resolved))

    def get_batch(self, batch_size: int, key: jax.Array | None = None) -> dict[str, Any]:
        """Get one eager batch in stateful or stateless mode.

        This follows the iterator ``next()`` idiom: the stateful mode both returns
        a batch and advances internal position, which is a deliberate, documented
        exception to command-query separation.

        Args:
            batch_size: Number of records to return.
            key: If provided, selects **stateless** mode — the batch is derived
                purely from ``key`` and no internal state is read or mutated. If
                ``None``, selects **stateful** mode: the batch starts at the current
                ``self.index`` and this call advances ``self.index`` (and rolls
                ``self.epoch`` at wrap-around), so successive calls stream forward.
                For side-effect-free indexed access, use :meth:`get_batch_at`.

        Returns:
            A batch dictionary of ``batch_size`` records.
        """
        return eager_get_batch_default(
            self.data,
            self.length,
            self.index,
            self.epoch,
            self.is_random_order,
            self._seed,
            batch_size,
            key,
        )

    def supports_indexed_access(self) -> bool:
        """Eager sources support random-access ``get_batch_at``."""
        return True

    def get_batch_at(
        self,
        start: int | jax.Array,
        size: int,
        key: jax.Array | None = None,
    ) -> dict[str, Any]:
        """Stateless indexed batch access; JIT-traceable for scan-based iteration.

        Returns ``size`` records starting at logical position ``start``.
        Does not advance ``self.index`` or any other internal state, so
        callers (typically ``Pipeline``) can drive iteration via their
        own position counter and trace ``get_batch_at`` under
        ``nnx.scan`` / ``nnx.jit``.

        Two modes:

        - **Sequential** (``self.is_random_order == False``): returns the
          contiguous slice ``data[start : start + size]`` with
          wrap-around at the end of the source.
        - **Shuffled** (``self.is_random_order == True``): applies a
          deterministic permutation derived from ``key`` and returns the
          slice of that permutation. Same ``(start, size, key)`` always
          returns the same output. The permutation is materialized via
          ``jax.random.permutation(key, length)`` per call — O(length)
          per batch.

        Args:
            start: Starting logical index; accepts concrete int or
                traced ``jax.Array``.
            size: Number of records to return (Python int — JAX shapes
                are static).
            key: PRNG key for shuffled mode. Required when
                ``is_random_order=True``; ignored otherwise.

        Returns:
            Dict mapping each data key to a JAX array with leading
            dimension ``size``.
        """
        indices = resolve_wrapped_indices(start, size, self.length, self.is_random_order, key)
        return {
            data_key: jnp.take(jnp.asarray(value), indices, axis=0, mode="wrap")
            for data_key, value in self.data.items()
        }

    def get_dataset_info(self) -> Any:
        """Return cached backend-specific dataset metadata."""
        return self._dataset_info

    def reset(self, seed: int | None = None) -> None:
        """Reset eager-source iteration state."""
        del seed
        eager_reset(self.index, self.epoch, self._cache)

    @property
    def is_random_order(self) -> bool:
        """Whether iteration order is randomized."""
        return self._is_random_order

    def set_random_order(self, enabled: bool) -> None:
        """Update runtime random-order behavior."""
        self._is_random_order = enabled

    def _repr_extra_fields(self) -> dict[str, Any]:
        """Optional additional repr fields for subclasses."""
        return {}

    def __repr__(self) -> str:
        """String representation."""
        return format_source_repr(
            type(self).__name__,
            self.dataset_name,
            self.split_name,
            self.length,
            self.is_random_order,
            self.epoch.get_value(),
            self._repr_extra_fields(),
        )

    def element_spec(self) -> Any:
        """Derive per-element spec from the eager dict-of-arrays storage.

        EagerSourceBase subclasses store data as a dict mapping keys to arrays
        whose leading axis is the dataset size. This default implementation
        strips that leading axis from every leaf to produce one
        ``jax.ShapeDtypeStruct`` per key.

        Subclasses with non-dict storage should override.

        Raises:
            ValueError: If the source is empty.
        """
        # Imported lazily to keep module import light (matches sibling sources).
        from datarax.core.spec import array_to_spec_strip_leading  # noqa: PLC0415

        if self.length == 0:
            raise ValueError(
                f"{type(self).__name__} has zero elements; element_spec() "
                "cannot be inferred from an empty dataset."
            )
        return {key: array_to_spec_strip_leading(value) for key, value in self.data.items()}


class StreamingSourceBase(DataSourceModule):
    """Shared streaming-source behavior for iterator-backed datasets.

    Subclasses must define the following attributes in their ``__init__``:

    - ``epoch`` (``nnx.Variable``): Current epoch counter.
    - ``_iterator`` (``Iterator | None``): The active backend iterator.
    - ``dataset_name`` (``str | None``): Human-readable dataset name.
    - ``split_name`` (``str | None``): Dataset split identifier.
    - ``length`` (``int | None``): Total number of elements (None if unknown).
    - ``_is_random_order`` (``bool``): Whether to randomize iteration order.
    - ``_dataset_info`` (``Any``): Cached backend-specific dataset metadata.
    """

    # -- Abstract attribute declarations (set by concrete subclasses) --
    epoch: nnx.Variable[int]  # pyright: ignore[reportGeneralTypeIssues]
    _iterator: Iterator[Any] | None
    dataset_name: str | None
    split_name: str | None
    length: int | None
    _is_random_order: bool
    _dataset_info: Any

    @property
    def is_random_order(self) -> bool:
        """Whether iteration order is randomized."""
        return self._is_random_order

    def get_dataset_info(self) -> Any:
        """Return cached backend-specific dataset metadata."""
        return self._dataset_info

    def reset(self, seed: int | None = None) -> None:
        """Reset streaming iterator state."""
        del seed
        self._iterator = None
        reset_streaming_state(self.epoch, self._cache)

    def get_batch(self, batch_size: int) -> dict[str, Any]:
        """Collect up to batch_size items from the streaming iterator."""
        return streaming_apply_batch(self.__next__, batch_size)

    def _repr_extra_fields(self) -> dict[str, Any]:
        """Optional additional repr fields for subclasses."""
        return {}

    def __repr__(self) -> str:
        """String representation."""
        return format_source_repr(
            type(self).__name__,
            self.dataset_name,
            self.split_name,
            self.length,
            self.is_random_order,
            self.epoch.get_value(),
            self._repr_extra_fields(),
        )
