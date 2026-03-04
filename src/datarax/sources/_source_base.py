"""Shared source base classes for eager and streaming backends."""

from __future__ import annotations

from typing import Any
from collections.abc import Iterator

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


class EagerSourceBase(DataSourceModule):
    """Shared eager-source behavior for in-memory JAX-backed datasets."""

    data: dict[str, Any] = nnx.data()

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
    """Shared streaming-source behavior for iterator-backed datasets."""

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
