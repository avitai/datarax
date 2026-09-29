"""Bridge utilities between Datarax sources and Grain loaders."""

from __future__ import annotations

from collections.abc import Sequence
from operator import index as to_index
from typing import Any, SupportsIndex


def validate_index_batch(indices: Sequence[int], length: int) -> list[int]:
    """Validate Grain-style non-negative batched random-access indices."""
    resolved = [int(index) for index in indices]
    for index in resolved:
        if index < 0 or index >= length:
            raise IndexError(f"Index {index} out of range for {length} records")
    return resolved


def records_from_batched_mapping(batch: dict[str, Any], count: int) -> list[dict[str, Any]]:
    """Convert a vectorized mapping gather into ordered individual records."""
    return [{key: value[row] for key, value in batch.items()} for row in range(count)]


class _DataraxRandomAccessBase:
    """Shared implementation for Grain random-access adapter protocols."""

    def __init__(self, source: Any) -> None:
        self.source = source
        self.length = len(source)

    def __len__(self) -> int:
        """Return the wrapped source length."""
        return self.length

    def _read_index(self, record_key: SupportsIndex) -> Any:
        """Return one record using Grain's non-negative index contract."""
        index = to_index(record_key)
        validate_index_batch([index], self.length)
        return self.source[index]

    def _getitems(self, indices: Sequence[int]) -> list[Any]:
        """Return records for Grain batched random-access reads."""
        return self.get_batch(indices)

    def get_batch(self, indices: Sequence[int]) -> list[Any]:
        """Return records for an ordered batch of random-access indices."""
        resolved = validate_index_batch(indices, self.length)
        getitems = getattr(self.source, "_getitems", None)
        if getitems is not None:
            return list(getitems(resolved))
        return [self.source[index] for index in resolved]

    def __repr__(self) -> str:
        """Deterministic representation required by Grain checkpointing."""
        return f"{type(self).__name__}(source={type(self.source).__name__}, length={self.length})"


class DataraxRandomAccessAdapter(_DataraxRandomAccessBase):
    """Adapter for Grain DataLoader random-access data sources."""

    def __getitem__(self, record_key: SupportsIndex) -> Any:
        """Return one record for Grain DataLoader."""
        return self._read_index(record_key)


class DataraxMapDatasetAdapter(_DataraxRandomAccessBase):
    """Adapter for Grain MapDataset random-access data sources."""

    def __getitem__(self, index: int) -> Any:
        """Return one record for Grain MapDataset."""
        return self._read_index(index)
