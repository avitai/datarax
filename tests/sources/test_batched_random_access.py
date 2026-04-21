"""Tests for source-level batched random access."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from datarax.sources.memory_source import MemorySource, MemorySourceConfig


def test_memory_source_getitems_gathers_array_leaves_in_order() -> None:
    """Dict-of-array sources should support ordered batched reads."""
    source = MemorySource(
        MemorySourceConfig(),
        {"x": jnp.arange(6), "y": jnp.arange(12).reshape(6, 2)},
    )

    records = source._getitems([4, 1, 3])

    assert [int(record["x"]) for record in records] == [4, 1, 3]
    assert [record["y"].tolist() for record in records] == [[8, 9], [2, 3], [6, 7]]


def test_memory_source_getitems_gathers_python_sequence_leaves() -> None:
    """Python list and tuple leaves should use ordered indexed gathers."""
    source = MemorySource(
        MemorySourceConfig(),
        {"x": [10, 11, 12, 13], "label": ("a", "b", "c", "d")},
    )

    records = source._getitems([2, 0])

    assert records == [{"x": 12, "label": "c"}, {"x": 10, "label": "a"}]


def test_memory_source_getitems_rejects_invalid_indices() -> None:
    """Invalid batched indices should raise a precise bounds error."""
    source = MemorySource(MemorySourceConfig(), {"x": jnp.arange(2)})

    with pytest.raises(IndexError, match="out of range"):
        source._getitems([-1])

    with pytest.raises(IndexError, match="out of range"):
        source._getitems([0, 2])
