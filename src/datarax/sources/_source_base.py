"""Shared source base classes for eager and streaming backends."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from datarax.core.data_source import DataSourceModule
from datarax.sources._grain_bridge import records_from_batched_mapping, validate_index_batch
from datarax.sources.source_ops import (
    eager_get_batch_default,
    eager_iter_default,
    eager_reset,
    format_source_repr,
    gather_eager_batch,
    get_eager_item,
    record_count,
    reset_streaming_state,
    resolve_wrapped_indices,
    streaming_apply_batch,
)


logger = logging.getLogger(__name__)


class EagerSourceBase(DataSourceModule):
    """Shared eager-source behavior for in-memory JAX-backed datasets.

    Subclasses must define the following attributes in their ``__init__``:

    - ``data`` (``dict[str, Any]``): The loaded dataset as a key→array mapping.
    - ``index`` (``nnx.Variable``): Current iteration index.
    - ``epoch`` (``nnx.Variable``): Current epoch counter.
    - ``_seed`` (``int``): Base integer seed of the shuffle.
    - ``_is_random_order`` (``bool``): Whether to randomize iteration order.
    - ``dataset_name`` (``str | None``): Human-readable dataset name.
    - ``split_name`` (``str | None``): Dataset split identifier.
    - ``_dataset_info`` (``Any``): Cached backend-specific dataset metadata.
    """

    # -- Abstract attribute declarations (set by concrete subclasses) --
    data: dict[str, Any]
    index: nnx.Variable[int]  # pyright: ignore[reportGeneralTypeIssues]
    epoch: nnx.Variable[int]  # pyright: ignore[reportGeneralTypeIssues]
    _seed: int
    _is_random_order: bool
    dataset_name: str | None
    split_name: str | None
    _dataset_info: Any

    @property
    def length(self) -> int:
        """Records the source's data holds now, read from ``data`` (see :func:`record_count`)."""
        return record_count(self.data)

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

    def record_indices_at(
        self,
        start: int | jax.Array,
        size: int,
        key: jax.Array | None = None,
    ) -> jax.Array:
        """Return the index of each record ``get_batch_at(start, size, key)`` returns.

        Args:
            start: Starting logical position; concrete int or traced ``jax.Array``.
            size: Number of records (Python int).
            key: PRNG key for shuffled mode.

        Returns:
            Int32 ``jax.Array`` of shape ``(size,)``.
        """
        return resolve_wrapped_indices(start, size, self.length, self.is_random_order, key)

    def get_records(self, indices: jax.Array) -> dict[str, Any]:
        """Gather the records at ``indices``; JIT-traceable for scan-based iteration.

        Args:
            indices: Int32 record indices in ``[0, len(self))``, as :meth:`record_indices_at`
                names them; concrete or traced.

        Returns:
            Dict mapping each data key to a JAX array with leading dimension ``len(indices)``.
        """
        return {
            data_key: jnp.take(jnp.asarray(value), indices, axis=0)
            for data_key, value in self.data.items()
        }

    def get_dataset_info(self) -> Any:
        """Return cached backend-specific dataset metadata."""
        return self._dataset_info

    def reset(self, seed: int | None = None) -> None:
        """Reset eager-source iteration state."""
        del seed
        eager_reset(self.index, self.epoch)

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
        """Derive the spec of emitted records from the eager dict-of-arrays storage.

        EagerSourceBase subclasses store data as a dict mapping keys to arrays
        whose leading axis is the dataset size, and ``get_batch_at`` gathers
        those arrays as JAX arrays. This default implementation strips the
        leading axis from every leaf and states the result as JAX arrays hold it
        (``device_spec``), reading only array metadata.

        Subclasses with non-dict storage should override.

        Returns:
            A dict mapping each key to the ``jax.ShapeDtypeStruct`` of one element.

        Raises:
            ValueError: If the source is empty.
        """
        # Imported lazily to keep module import light (matches sibling sources).
        from datarax.core.spec import array_to_spec_strip_leading, device_spec  # noqa: PLC0415

        if self.length == 0:
            raise ValueError(
                f"{type(self).__name__} has zero elements; element_spec() "
                "cannot be inferred from an empty dataset."
            )
        return device_spec(
            {key: array_to_spec_strip_leading(value) for key, value in self.data.items()}
        )


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
        reset_streaming_state(self.epoch)

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
