"""Shared source base classes for eager and streaming backends."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import flax.nnx as nnx
import jax

from datarax.core.data_source import DataSourceModule
from datarax.sources._eager_source_ops import (
    eager_get_batch_default,
    eager_iter_default,
    eager_reset,
    format_source_repr,
    get_eager_item,
    reset_streaming_state,
    streaming_apply_batch,
)


logger = logging.getLogger(__name__)


class EagerSourceBase(DataSourceModule):
    """Shared eager-source behavior for in-memory JAX-backed datasets.

    Subclasses must define the following attributes in their ``__init__``:

    - ``data`` (``dict[str, Any]``): The loaded dataset as a key→array mapping.
    - ``length`` (``int``): Total number of elements.
    - ``index`` (``nnx.Variable``): Current iteration index.
    - ``epoch`` (``nnx.Variable``): Current epoch counter.
    - ``_seed`` (``int``): Base integer seed for Grain index_shuffle.
    - ``shuffle`` (``bool``): Whether to shuffle during iteration.
    - ``dataset_name`` (``str | None``): Human-readable dataset name.
    - ``split_name`` (``str | None``): Dataset split identifier.
    - ``_dataset_info`` (``Any``): Cached backend-specific dataset metadata.
    """

    # -- Abstract attribute declarations (set by concrete subclasses) --
    data: dict[str, Any] = nnx.data()
    length: int
    index: nnx.Variable[int]  # pyright: ignore[reportGeneralTypeIssues]
    epoch: nnx.Variable[int]  # pyright: ignore[reportGeneralTypeIssues]
    _seed: int
    shuffle: bool
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
            self.shuffle,
            self._seed,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Retrieve one eager element by index."""
        return get_eager_item(self.data, self.length, index)

    def get_batch(self, batch_size: int, key: jax.Array | None = None) -> dict[str, Any]:
        """Get one eager batch in stateful or stateless mode."""
        return eager_get_batch_default(
            self.data,
            self.length,
            self.index,
            self.epoch,
            self.shuffle,
            self._seed,
            batch_size,
            key,
        )

    def get_dataset_info(self) -> Any:
        """Return cached backend-specific dataset metadata."""
        return self._dataset_info

    def reset(self, seed: int | None = None) -> None:
        """Reset eager-source iteration state."""
        del seed
        eager_reset(self.index, self.epoch, self._cache)

    def set_shuffle(self, shuffle: bool) -> None:
        """Update runtime shuffle behavior."""
        self.shuffle = shuffle

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
            self.shuffle,
            self.epoch.get_value(),
            self._repr_extra_fields(),
        )


class StreamingSourceBase(DataSourceModule):
    """Shared streaming-source behavior for iterator-backed datasets.

    Subclasses must define the following attributes in their ``__init__``:

    - ``epoch`` (``nnx.Variable``): Current epoch counter.
    - ``_iterator`` (``Iterator | None``): The active backend iterator.
    - ``dataset_name`` (``str | None``): Human-readable dataset name.
    - ``split_name`` (``str | None``): Dataset split identifier.
    - ``length`` (``int | None``): Total number of elements (None if unknown).
    - ``shuffle`` (``bool``): Whether to shuffle during iteration.
    - ``_dataset_info`` (``Any``): Cached backend-specific dataset metadata.
    """

    # -- Abstract attribute declarations (set by concrete subclasses) --
    epoch: nnx.Variable[int]  # pyright: ignore[reportGeneralTypeIssues]
    _iterator: Iterator[Any] | None
    dataset_name: str | None
    split_name: str | None
    length: int | None
    shuffle: bool
    _dataset_info: Any

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
            self.shuffle,
            self.epoch.get_value(),
            self._repr_extra_fields(),
        )
