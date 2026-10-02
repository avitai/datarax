"""Shared source pieces: the named-dataset mixin of the eager sources and the streaming base."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from flax import nnx

from datarax.core.data_source import DataSourceModule
from datarax.sources.eager_source import HostValue
from datarax.sources.source_ops import (
    format_source_repr,
    reset_streaming_state,
    streaming_apply_batch,
)


logger = logging.getLogger(__name__)


class DatasetSourceMixin:
    """What an eager source loaded from a named dataset reports: its name, split and info.

    A subclass sets ``dataset_name``, ``split_name`` and ``_dataset_info`` in its ``__init__``.
    The info is held in a :class:`~datarax.sources.eager_source.HostValue`: a backend's info
    object compares by identity (TFDS's ``DatasetInfo``), so held as an attribute it would make
    every source a graphdef of its own and every jitted step over a new source compile again.
    """

    dataset_name: str | None
    split_name: str | None
    _dataset_info: HostValue
    length: int

    def get_dataset_info(self) -> Any:
        """Return cached backend-specific dataset metadata."""
        return self._dataset_info.value

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
            {
                "shuffle": self.is_random_order,
                **self._repr_extra_fields(),
                "epoch": self.epoch.get_value(),
            },
        )
