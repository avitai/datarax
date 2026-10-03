"""An in-memory source's length is a fact about its data, read from the data it holds.

``MemorySource`` and the eager sources copied their length at construction, while their
``data`` stays a public attribute: after the data was replaced, ``len(source)`` kept the old
count and a pipeline silently skipped the new records. A ``MixDataSourcesNode``'s epoch length
and its children's offsets follow its children's current lengths, so a child that grows moves
the records of the children after it and lengthens the epoch by Grain's rule.
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
from datarax.core.index_words import from_words
from datarax.pipeline import Pipeline
from datarax.sources.eager_source import EagerSource
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from datarax.sources.mixed_source import MixDataSourcesConfig, MixDataSourcesNode


def _memory(data: dict[str, Any] | Sequence[Any]) -> MemorySource:
    return MemorySource(MemorySourceConfig(), data=data)


class TestMemorySource:
    def test_replaced_dict_data_is_served_in_full(self) -> None:
        source = _memory({"x": np.arange(8, dtype=np.float32)[:, None]})
        source.data = {"x": np.arange(12, dtype=np.float32)[:, None]}
        pipeline = Pipeline(source=source, stages=[], batch_size=4, rngs=nnx.Rngs(0))

        served = np.concatenate([np.asarray(batch["x"]).ravel() for batch in pipeline])

        assert len(source) == 12
        np.testing.assert_array_equal(served, np.arange(12.0))

    def test_replaced_columns_are_counted(self) -> None:
        source = _memory(list(range(5)))
        source.data = np.arange(9)
        assert len(source) == 9

    def test_replaced_columns_of_unequal_lengths_are_refused(self) -> None:
        source = _memory({"x": np.zeros(4), "y": np.zeros(4)})
        source.data = {"x": np.zeros(4), "y": np.zeros(6)}
        with pytest.raises(ValueError, match="one row per record"):
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


class _Eager(EagerSource):
    """Minimal eager source: data and nothing about its length."""

    def __init__(self, data: dict) -> None:
        super().__init__(StructuralConfig())
        self._store(data)


class TestEagerSource:
    def test_the_length_follows_the_data(self) -> None:
        source = _Eager({"x": jnp.zeros((6, 2))})
        assert len(source) == 6
        source.data = {"x": jnp.zeros((10, 2))}
        assert len(source) == 10

    def test_a_source_without_records_has_length_zero(self) -> None:
        assert len(_Eager({})) == 0


class TestMixedSource:
    def test_the_length_and_offsets_follow_a_child_that_grew(self) -> None:
        grows = _memory({"x": np.arange(4, dtype=np.float32)})
        fixed = _memory({"x": np.arange(100, 120, dtype=np.float32)})
        mix = MixDataSourcesNode(MixDataSourcesConfig(weights=(0.5, 0.5)), [grows, fixed])
        assert len(mix) == 8  # Grain's length: the shorter child at 1:1, twice over
        grows.data = {"x": np.arange(8, dtype=np.float32)}
        key = jax.random.key(3)

        ids = from_words(mix.record_indices_at(start=0, size=len(mix), key=key)).astype(np.int64)
        values = np.asarray(mix.get_records(mix.record_indices_at(0, len(mix), key))["x"])

        assert len(mix) == 16
        assert len(set(ids.tolist())) == 16
        # One index, one record: the grown source owns 0..7, the other 8..27.
        expected = np.where(ids < 8, ids.astype(np.float32), 100.0 + (ids - 8).astype(np.float32))
        np.testing.assert_array_equal(values, expected)
