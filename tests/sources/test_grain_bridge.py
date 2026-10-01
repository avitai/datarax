"""Tests for the Datarax-to-Grain random-access bridge."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from grain import python as grain

from datarax.sources._grain_bridge import (
    DataraxMapDatasetAdapter,
    DataraxRandomAccessAdapter,
)
from datarax.sources.memory_source import MemorySource, MemorySourceConfig


def test_random_access_adapter_preserves_order_and_repr() -> None:
    """The adapter should expose deterministic Grain random access."""
    source = MemorySource(
        MemorySourceConfig(), {"x": jnp.arange(5), "y": ["a", "b", "c", "d", "e"]}
    )
    adapter = DataraxRandomAccessAdapter(source)

    assert len(adapter) == 5
    assert repr(adapter) == "DataraxRandomAccessAdapter(source=MemorySource, length=5)"
    assert adapter[2]["x"] == 2

    records = adapter.get_batch([3, 1, 4])
    assert [int(record["x"]) for record in records] == [3, 1, 4]
    # Text is the records' provenance, kept by the source and never served as a field.
    assert all(set(record) == {"x"} for record in records)


def test_map_dataset_adapter_preserves_grain_map_dataset_contract() -> None:
    """The MapDataset adapter should satisfy Grain's dataset source protocol."""
    source = MemorySource(MemorySourceConfig(), {"x": jnp.arange(4)})
    dataset = grain.MapDataset.source(DataraxMapDatasetAdapter(source))

    record = dataset[2]
    assert record is not None
    assert int(record["x"]) == 2
    records = [record for record in dataset._getitems([3, 0]) if record is not None]
    assert len(records) == 2
    assert [int(record["x"]) for record in records] == [3, 0]


def test_random_access_adapter_rejects_invalid_indices() -> None:
    """Batched reads should fail before Grain workers see invalid indices."""
    source = MemorySource(MemorySourceConfig(), {"x": jnp.arange(3)})
    adapter = DataraxRandomAccessAdapter(source)

    with pytest.raises(IndexError, match="out of range"):
        adapter._getitems([0, 3])

    with pytest.raises(IndexError, match="out of range"):
        adapter._getitems([-1])
