"""A source's record names: one spelling names one value, across data, provenance and a mix.

A record's provenance is keyed by spelling, so a nested ``{"meta": {"id": ...}}`` and a literal
``"meta/id"`` name one provenance value, and the split must refuse them rather than keep the last
one written. A data field and a provenance value may not share a spelling either, and a mix's
children must agree on what each spelling names.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from flax import nnx

from datarax import Pipeline
from datarax.core.block_order import BlockOrder
from datarax.core.data_source import DataSourceModule
from datarax.core.index_words import to_words
from datarax.sources.eager_source import split_record
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from datarax.sources.mixed_source import MixDataSourcesConfig, MixDataSourcesNode
from tests.test_common.streams import RecordStream


class TestProvenanceSpellings:
    """Defect 4: the nested and the flat ``meta/id`` were merged, the last write winning."""

    @pytest.mark.parametrize(
        "record",
        [
            {"x": np.float32(1), "meta": {"id": "nested"}, "meta/id": "flat"},
            {"x": np.float32(1), "meta/id": "flat", "meta": {"id": "nested"}},
        ],
        ids=["nested-first", "flat-first"],
    )
    def test_a_nested_and_a_flat_provenance_value_with_one_spelling_are_refused(
        self, record: dict
    ) -> None:
        with pytest.raises(
            ValueError,
            match=r"provenance: 'meta/id' names two fields: \('meta', 'id'\) and \('meta/id',\)",
        ):
            split_record(record)

    def test_provenance_values_with_distinct_spellings_are_kept(self) -> None:
        part, provenance = split_record({"x": np.float32(1), "meta": {"id": "a"}, "meta/file": "b"})
        assert provenance == {"meta/id": "a", "meta/file": "b"}
        assert isinstance(part, dict) and set(part) == {"x"}


class TestDataAgainstProvenance:
    """Defect 5: a data field and a provenance value with one spelling went unchecked."""

    def test_an_eager_source_refuses_a_data_field_spelled_as_a_provenance_value(self) -> None:
        records = [{"meta": {"id": np.float32(i)}, "meta/id": f"text{i}"} for i in range(2)]
        with pytest.raises(
            ValueError,
            match=r"MemorySource: 'meta/id' names two fields: data field \('meta', 'id'\) and "
            r"provenance field \('meta/id',\)",
        ):
            MemorySource(MemorySourceConfig(), records)

    def test_held_columns_and_their_provenance_are_checked_together(self) -> None:
        columns = {"x": np.arange(3, dtype=np.float32)}
        with pytest.raises(ValueError, match=r"'x' names two fields: data field \('x',\)"):
            MemorySource.of_columns(columns, [{"x": f"name{i}"} for i in range(3)])

    def test_distinct_data_and_provenance_spellings_are_held(self) -> None:
        records = [{"meta": {"id": np.float32(i)}, "meta/file": f"f{i}"} for i in range(2)]
        source = MemorySource(MemorySourceConfig(), records)
        assert [dict(p) for p in source.provenance(to_words(np.arange(2)))] == [
            {"meta/file": "f0"},
            {"meta/file": "f1"},
        ]


class TestTheMix:
    """One spelling names one node across a mix's children (the census's case B)."""

    def test_children_spelling_one_name_two_ways_are_refused(self) -> None:
        nested = MemorySource.of_columns({"meta": {"id": np.arange(4, dtype=np.int32)}})
        flat = MemorySource(MemorySourceConfig(), {"meta/id": np.arange(4, dtype=np.int32)})
        with pytest.raises(ValueError) as caught:
            MixDataSourcesNode(MixDataSourcesConfig(weights=(1.0, 1.0)), [nested, flat])
        message = str(caught.value)
        assert "a mix's children disagree on field 'meta/id'" in message
        assert "child 0 holds ('meta', 'id'), child 1 holds ('meta/id',)" in message

    def test_a_childs_data_field_spelled_as_anothers_provenance_is_refused(self) -> None:
        data = MemorySource(MemorySourceConfig(), {"x": np.arange(4, dtype=np.float32)})
        named = MemorySource(
            MemorySourceConfig(), [{"y": np.float32(i), "x": f"text{i}"} for i in range(4)]
        )
        with pytest.raises(ValueError) as caught:
            MixDataSourcesNode(MixDataSourcesConfig(weights=(1.0, 1.0)), [data, named])
        message = str(caught.value)
        assert "a mix's children disagree on field 'x'" in message
        assert "child 0 holds data field ('x',), child 1 provenance field ('x',)" in message


class _DeclaresAnExoticKey(MemorySource):
    """A source whose declared record breaks N2, as a hand-written source's spec can."""

    def element_spec(self) -> dict:
        return {"a//b": jax.ShapeDtypeStruct((), np.float32)}


class _DeclaresNoRecord(MemorySource):
    """A source that leaves ``element_spec`` to the base, which every source must override."""

    element_spec = DataSourceModule.element_spec


class TestThePipelineConstructionCheck:
    def test_a_refused_configuration_is_refused_before_the_record_is_read(self) -> None:
        """Reading a stream's record opens a pass; a configuration no data can rescue needs none."""
        stream = RecordStream({"x": np.arange(6, dtype=np.float32)})
        order = BlockOrder(block_size=3, blocks_per_buffer=2)
        with pytest.raises(ValueError, match="INDEXED"):
            Pipeline(
                source=stream,
                stages=[],
                batch_size=2,
                rngs=nnx.Rngs(0),
                shuffle=True,
                block_order=order,
            )
        assert stream.opened == []

    def test_a_source_declaring_no_record_is_refused_when_the_pipeline_is_built(self) -> None:
        source = _DeclaresNoRecord(MemorySourceConfig(), {"x": np.zeros(4, np.float32)})
        refusal = "_DeclaresNoRecord must implement element_spec"
        with pytest.raises(NotImplementedError, match=refusal):
            Pipeline(source=source, stages=[], batch_size=2, rngs=nnx.Rngs(0))

    def test_a_source_whose_record_breaks_the_naming_rules_is_refused_when_the_pipeline_is_built(
        self,
    ) -> None:
        source = _DeclaresAnExoticKey(MemorySourceConfig(), {"x": np.zeros(4, np.float32)})
        with pytest.raises(
            ValueError,
            match=r"_DeclaresAnExoticKey declares a record in which the key 'a//b', which holds "
            r"'//'; a key may not start or end with '/' or hold '//'",
        ):
            Pipeline(source=source, stages=[], batch_size=2, rngs=nnx.Rngs(0))
