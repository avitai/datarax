"""The index check of Grain-style batched random-access reads."""

from __future__ import annotations

from collections.abc import Sequence


def validate_index_batch(indices: Sequence[int], length: int) -> list[int]:
    """Validate Grain-style non-negative batched random-access indices."""
    resolved = [int(index) for index in indices]
    for index in resolved:
        if index < 0 or index >= length:
            raise IndexError(f"Index {index} out of range for {length} records")
    return resolved
