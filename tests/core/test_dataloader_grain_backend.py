"""Tests for Grain-backed and explicit Datarax DataLoader behavior."""

from __future__ import annotations

from dataclasses import dataclass

import grain
import jax.numpy as jnp
import pytest

from datarax.core.element_batch import Batch
from datarax.dag.nodes import BatchNode, DataLoader
from datarax.dag.nodes.loaders import DataLoaderRestoreError
from datarax.sources.memory_source import MemorySource, MemorySourceConfig


@dataclass(frozen=True)
class AddOffset(grain.transforms.Map):
    """Simple source-stage Grain transform used by tests."""

    offset: int

    def map(self, element):
        return {**element, "x": element["x"] + self.offset}


def test_auto_backend_uses_grain_for_random_access_source() -> None:
    """Auto mode should choose Grain for known-size random-access sources."""
    source = MemorySource(MemorySourceConfig(), {"x": jnp.arange(5)})
    loader = DataLoader(source, batch_size=2, backend="auto")

    assert loader.backend == "grain"
    batches = list(loader)

    assert [batch.batch_size for batch in batches] == [2, 2, 1]
    assert all(isinstance(batch, Batch) for batch in batches)
    assert jnp.array_equal(batches[0].data.get_value()["x"], jnp.array([0, 1]))
    assert jnp.array_equal(batches[-1].data.get_value()["x"], jnp.array([4]))


def test_grain_backend_applies_source_transforms_before_batch_construction() -> None:
    """Grain source transforms should run before Datarax Batch construction."""
    source = MemorySource(MemorySourceConfig(), {"x": jnp.arange(4)})
    loader = DataLoader(
        source,
        batch_size=2,
        backend="grain",
        source_transforms=(AddOffset(10),),
    )

    first_batch = next(iter(loader))

    assert isinstance(first_batch, Batch)
    assert jnp.array_equal(first_batch.data.get_value()["x"], jnp.array([10, 11]))


def test_explicit_datarax_backend_preserves_dag_batch_node_semantics() -> None:
    """Explicit Datarax backend should keep the existing DAG batching path."""
    source = MemorySource(MemorySourceConfig(), {"x": jnp.arange(5)})
    loader = DataLoader(source, batch_size=2, backend="datarax")

    assert loader.backend == "datarax"
    assert any(isinstance(node, BatchNode) for node in loader.nodes)
    batches = list(loader)

    assert [batch.batch_size for batch in batches] == [2, 2, 1]
    assert jnp.array_equal(batches[0].data.get_value()["x"], jnp.array([0, 1]))


def test_set_state_mismatch_raises_structured_restore_error() -> None:
    """DataLoader restore should not silently swallow incompatible state."""
    source = MemorySource(MemorySourceConfig(), {"x": jnp.arange(4)})
    loader = DataLoader(source, batch_size=2, backend="datarax")

    with pytest.raises(DataLoaderRestoreError) as exc_info:
        loader.set_state({"iteration_count": 0, "nodes": []})

    assert exc_info.value.failures
    assert exc_info.value.failures[0][0] == "nodes"
