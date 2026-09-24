"""An in-memory source's length is a fact about its data, read from the data it holds.

``MemorySource`` and the eager sources copied their length at construction, while their
``data`` stays a public attribute: after the data was replaced, ``len(source)`` kept the old
count and a pipeline silently skipped the new records. ``MixDataSourcesNode`` froze its total
and per-source offsets while it sampled records from each child's current length, so a child
that grew produced record indices colliding with another source's.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from datarax.core.config import StructuralConfig
from datarax.pipeline import Pipeline
from datarax.sources._source_base import EagerSourceBase
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from datarax.sources.mixed_source import MixDataSourcesConfig, MixDataSourcesNode


def _memory(data: dict[str, Any] | Sequence[Any]) -> MemorySource:
    return MemorySource(MemorySourceConfig(shuffle=False), data=data)


class TestMemorySource:
    def test_replaced_dict_data_is_served_in_full(self) -> None:
        source = _memory({"x": np.arange(8, dtype=np.float32)[:, None]})
        source.data = {"x": np.arange(12, dtype=np.float32)[:, None]}
        pipeline = Pipeline(source=source, stages=[], batch_size=4, rngs=nnx.Rngs(0))

        served = np.concatenate([np.asarray(batch["x"]).ravel() for batch in pipeline])

        assert len(source) == 12
        np.testing.assert_array_equal(served, np.arange(12.0))

    def test_replaced_list_data_is_counted(self) -> None:
        source = _memory(list(range(5)))
        source.data = list(range(9))
        assert len(source) == 9

    def test_replaced_data_of_unequal_lengths_is_refused(self) -> None:
        source = _memory({"x": np.zeros(4), "y": np.zeros(4)})
        source.data = {"x": np.zeros(4), "y": np.zeros(6)}
        with pytest.raises(ValueError, match="same length"):
            len(source)

    def test_data_of_unequal_lengths_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            _memory({"x": np.zeros(4), "y": np.zeros(6)})

    def test_a_column_that_is_a_mapping_is_refused_by_name(self) -> None:
        """A mapping is not a column: its key count is not a record count."""
        with pytest.raises(TypeError, match=r"column 'a' is a mapping.*flat"):
            _memory({"a": {"b": np.zeros((4, 2))}, "c": np.zeros(4)})

    def test_a_mapping_column_whose_key_count_equals_the_records_is_refused(self) -> None:
        """Counted by its keys, this mapping agreed with the records and passed unnoticed."""
        nested = {"b": np.zeros(2), "d": np.zeros(2)}
        with pytest.raises(TypeError, match="column 'a' is a mapping"):
            _memory({"a": nested, "c": np.zeros(2)})

    def test_list_records_holding_mappings_stay_host_records(self) -> None:
        records = [{"features": {"x": i}} for i in range(3)]
        assert len(_memory(records)) == 3


class _Eager(EagerSourceBase):
    """Minimal eager source: data and nothing about its length."""

    def __init__(self, data: dict) -> None:
        super().__init__(StructuralConfig())
        self.data = nnx.data(data)
        self.index = nnx.Variable(jnp.int32(0))
        self.epoch = nnx.Variable(jnp.int32(0))
        self._seed = 0
        self._is_random_order = False
        self.dataset_name = "eager"
        self.split_name = "all"
        self._dataset_info = None


class TestEagerSource:
    def test_the_length_follows_the_data(self) -> None:
        source = _Eager({"x": jnp.zeros((6, 2))})
        assert len(source) == 6
        source.data = {"x": jnp.zeros((10, 2))}
        assert len(source) == 10

    def test_a_source_without_records_has_length_zero(self) -> None:
        assert len(_Eager({})) == 0


class TestMixedSource:
    def test_the_total_and_offsets_follow_a_child_that_grew(self) -> None:
        grows = _memory({"x": jnp.arange(4, dtype=jnp.float32)})
        fixed = _memory({"x": jnp.arange(100, 104, dtype=jnp.float32)})
        mix = MixDataSourcesNode(
            MixDataSourcesConfig(num_sources=2, weights=(0.5, 0.5)), [grows, fixed]
        )
        grows.data = {"x": jnp.arange(8, dtype=jnp.float32)}
        key = jax.random.key(3)

        ids = np.asarray(mix.record_indices_at(start=0, size=256, key=key))
        values = np.asarray(mix.get_batch_at(start=0, size=256, key=key)["x"])

        assert len(mix) == 12
        assert set(ids.tolist()) <= set(range(12))
        # One index, one record: the grown source owns 0..7, the other 8..11.
        expected = np.where(ids < 8, ids.astype(np.float32), 100.0 + (ids - 8).astype(np.float32))
        np.testing.assert_array_equal(values, expected)
