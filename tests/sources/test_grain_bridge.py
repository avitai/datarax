"""Tests for the Datarax-to-Grain random-access bridge."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from grain import python as grain

from datarax.core.element_batch import Batch, Element
from datarax.sources._grain_bridge import (
    DataraxMapDatasetAdapter,
    DataraxRandomAccessAdapter,
    record_to_element,
    records_to_batch,
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
    assert [record["y"] for record in records] == ["d", "b", "e"]


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


def test_records_to_batch_converts_grain_batched_dict_to_datarax_batch() -> None:
    """Grain batch outputs should become native Datarax Batch objects."""
    batch = records_to_batch({"x": jnp.arange(3), "y": jnp.ones((3, 2))})

    assert isinstance(batch, Batch)
    assert batch.batch_size == 3
    assert jnp.array_equal(batch.data.get_value()["x"], jnp.arange(3))


def test_records_to_batch_converts_record_sequences_to_datarax_batch() -> None:
    """Unbatched record sequences should become a Datarax Batch."""
    batch = records_to_batch([{"x": jnp.array(1)}, {"x": jnp.array(2)}])

    assert isinstance(batch, Batch)
    assert batch.batch_size == 2
    assert jnp.array_equal(batch.data.get_value()["x"], jnp.array([1, 2]))


def test_record_to_element_preserves_explicit_element_shape() -> None:
    """Records already in Element wire format should preserve state and metadata."""
    element = record_to_element(
        {"data": {"x": jnp.array(1)}, "state": {"seen": True}, "metadata": None}
    )

    assert isinstance(element, Element)
    assert element.state == {"seen": True}
    assert element.data["x"] == 1
