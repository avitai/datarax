"""The in-memory source contract: host columns, provenance beside them, one stateless host read.

``EagerSource`` (``MemorySource``, ``TFDSEagerSource`` and ``HFEagerSource`` build on it) holds a
record's array part as host NumPy columns, one row per record, and its non-array part (strings,
bytes, Python objects) as the record's provenance in a host holder that NNX keeps out of module
state and out of every trace. ``get_batch(indices, *, epochs=0)`` reads the named records with one
host gather and returns a ``Batch`` named with the given indices and epochs; it reads and changes
no state. A list of records is turned into columns once, at construction.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from substrax.testing.compiles import expect_compiles

from datarax.core.config import StructuralConfig
from datarax.core.element_batch import Batch, Element, PADDING_INDEX
from datarax.core.index_words import from_words, to_words
from datarax.sources import eager_source, EagerSource, MemorySource, MemorySourceConfig, source_ops
from tests.test_common.device_arrays import arrays_made_since


_N = 12


def _columns(n: int = _N) -> dict[str, np.ndarray]:
    return {
        "x": np.arange(n * 3, dtype=np.float32).reshape(n, 3),
        "y": np.arange(n, dtype=np.int32),
    }


class _Columns(EagerSource):
    """An eager source over given columns: a subclass storing through the base."""

    def __init__(
        self, columns: dict[str, Any], provenance: tuple[dict[str, Any], ...] = ()
    ) -> None:
        super().__init__(StructuralConfig())
        self._store(columns, provenance)


def _memory(data: Any = None) -> MemorySource:
    return MemorySource(MemorySourceConfig(), _columns() if data is None else data)


_SOURCES: dict[str, Callable[[], EagerSource]] = {
    "memory": _memory,
    "eager-subclass": lambda: _Columns(_columns()),
}


def _words(rows: list[int]) -> np.ndarray:
    return to_words(np.asarray(rows, np.uint64))


@pytest.mark.parametrize("name", sorted(_SOURCES))
class TestTheHostRead:
    """T2: ``get_batch(indices, epochs=...)`` is one stateless host gather returning a ``Batch``."""

    def test_it_returns_the_named_records_as_a_batch(self, name: str) -> None:
        source = _SOURCES[name]()
        rows = [7, 0, 11, 3]
        batch = source.get_batch(_words(rows), epochs=np.asarray([2, 2, 3, 3], np.int32))
        assert isinstance(batch, Batch)
        for field, column in _columns().items():
            assert isinstance(batch[field], np.ndarray)
            np.testing.assert_array_equal(batch[field], column[rows])
        np.testing.assert_array_equal(batch.indices, _words(rows))
        np.testing.assert_array_equal(batch.epochs, [2, 2, 3, 3])
        np.testing.assert_array_equal(batch.draws, np.zeros(4, np.int32))
        assert batch.indices.dtype == np.uint32
        assert batch.epochs.dtype == np.int32

    def test_one_epoch_is_every_row_s_epoch_and_one_record_is_a_batch(self, name: str) -> None:
        batch = _SOURCES[name]().get_batch(_words([5]), epochs=4)
        assert batch.batch_size == 1
        np.testing.assert_array_equal(batch.epochs, [4])
        np.testing.assert_array_equal(batch["y"], [5])

    def test_a_contiguous_read_is_a_view_and_any_other_a_copy(self, name: str) -> None:
        source = _SOURCES[name]()
        run = source.get_batch(_words([4, 5, 6, 7]), contiguous=True)
        shuffled = source.get_batch(_words([6, 4, 7, 5]))
        assert np.shares_memory(run["x"], source.data["x"])
        assert not np.shares_memory(shuffled["x"], source.data["x"])
        np.testing.assert_array_equal(run["x"], _columns()["x"][4:8])
        np.testing.assert_array_equal(from_words(run.indices), [4, 5, 6, 7])

    def test_a_read_declared_contiguous_must_be_a_run(self, name: str) -> None:
        with pytest.raises(ValueError, match="contiguous"):
            _SOURCES[name]().get_batch(_words([4, 5, 9]), contiguous=True)

    @pytest.mark.parametrize("rows", [[3, _N], [_N + 5]], ids=["at-length", "past-length"])
    def test_an_index_outside_the_source_is_refused(self, name: str, rows: list[int]) -> None:
        with pytest.raises(IndexError, match=rf"outside \[0, {_N}\)"):
            _SOURCES[name]().get_batch(_words(rows))

    def test_the_padding_index_is_refused_by_name(self, name: str) -> None:
        indices = np.stack([_words([1])[0], PADDING_INDEX])
        with pytest.raises(IndexError, match="padding"):
            _SOURCES[name]().get_batch(indices)

    def test_indices_must_be_uint32_words(self, name: str) -> None:
        with pytest.raises(ValueError, match="to_words"):
            _SOURCES[name]().get_batch(np.asarray([1, 2], np.int64))

    def test_two_reads_are_equal_and_change_no_state(self, name: str) -> None:
        source = _SOURCES[name]()
        before = jax.tree.map(np.asarray, nnx.to_pure_dict(nnx.state(source, nnx.Variable)))
        first = source.get_batch(_words([2, 9]))
        second = source.get_batch(_words([2, 9]))
        after = jax.tree.map(np.asarray, nnx.to_pure_dict(nnx.state(source, nnx.Variable)))
        np.testing.assert_array_equal(first["x"], second["x"])
        jax.tree.map(np.testing.assert_array_equal, before, after)

    def test_a_read_creates_no_device_array(self, name: str) -> None:
        source = _SOURCES[name]()
        indices = _words([3, 1, 8])
        before = jax.live_arrays()
        control = jnp.asarray(np.zeros(3))  # the instrument sees a device array made in between
        assert any(a is control for a in arrays_made_since(before)), "the check sees no new array"
        before = jax.live_arrays()
        batch = source.get_batch(indices, epochs=1)
        assert arrays_made_since(before) == []
        del control, batch

    def test_iteration_is_stateless_and_sequential(self, name: str) -> None:
        source = _SOURCES[name]()
        first = [record["y"] for record in source]
        second = [record["y"] for record in source]
        assert [int(y) for y in first] == list(range(_N))
        assert first == second

    def test_no_counter_or_reset_remains(self, name: str) -> None:
        source = _SOURCES[name]()
        for removed in ("index", "epoch", "reset", "get_batch_at", "_shuffle_seed"):
            assert not hasattr(source, removed), removed


class TestOneHostRead:
    """T8: one ``get_batch`` and one gather, shared; the eager helpers are gone."""

    def test_memory_source_reads_through_the_base(self) -> None:
        assert MemorySource.get_batch is EagerSource.get_batch
        assert MemorySource.get_records is EagerSource.get_records
        assert MemorySource.__getitem__ is EagerSource.__getitem__

    def test_the_eager_sources_are_eager_sources(self) -> None:
        from datarax.sources.hf_source import HFEagerSource
        from datarax.sources.tfds_source import TFDSEagerSource

        for source_class in (MemorySource, TFDSEagerSource, HFEagerSource):
            assert issubclass(source_class, EagerSource)

    def test_the_removed_helpers_are_gone(self) -> None:
        import datarax.sources as sources

        removed = (
            "eager_get_batch",
            "eager_iter",
            "eager_reset",
            "eager_get_batch_default",
            "eager_iter_default",
            "build_eager_element",
            "get_eager_item",
            "gather_eager_batch",
        )
        for name in removed:
            assert not hasattr(source_ops, name), name
            assert name not in sources.__all__, name
        assert "EagerSource" in sources.__all__
        assert "resolve_wrapped_indices" in sources.__all__

    def test_the_private_eager_base_is_gone(self) -> None:
        import importlib

        assert not hasattr(
            importlib.import_module("datarax.sources._source_base"), "EagerSourceBase"
        )

    def test_the_host_read_takes_indices_and_epochs(self) -> None:
        parameters = inspect.signature(EagerSource.get_batch).parameters
        assert list(parameters) == ["self", "indices", "epochs", "contiguous"]
        assert parameters["epochs"].kind is inspect.Parameter.KEYWORD_ONLY


class TestRecordsBecomeColumns:
    """T6: a list of records is stored as host columns and provenance, once (P1-P5)."""

    def test_numeric_leaves_become_numpy_columns_without_device_arrays(self) -> None:
        records = [
            {"a": i, "b": float(i) / 2, "c": np.float32(i), "d": [i, i + 1]} for i in range(6)
        ]
        before = jax.live_arrays()
        source = _memory(records)
        assert arrays_made_since(before) == []
        for column in jax.tree.leaves(source.data):
            assert isinstance(column, np.ndarray)
        np.testing.assert_array_equal(source.data["a"], np.arange(6))
        np.testing.assert_array_equal(source.data["d"], [[i, i + 1] for i in range(6)])
        assert len(source) == 6

    def test_nested_records_keep_their_structure(self) -> None:
        records = [{"features": {"x": i, "y": 2 * i}, "label": i % 3} for i in range(5)]
        source = _memory(records)
        record = source[4]
        assert int(record["features"]["y"]) == 8
        assert int(record["label"]) == 1
        batch = source.get_batch(_words([4, 0]))
        np.testing.assert_array_equal(batch["features"]["x"], [4, 0])

    def test_non_array_leaves_are_provenance_kept_aligned_and_never_served(self) -> None:
        marker = object()
        records = [
            {"x": np.full(2, i, np.float32), "name": f"r{i}", "raw": b"\x00" * i, "obj": marker}
            for i in range(4)
        ]
        source = _memory(records)
        provenance = source._provenance.value
        assert len(provenance) == len(source) == 4
        assert [p["name"] for p in provenance] == ["r0", "r1", "r2", "r3"]
        assert provenance[2]["raw"] == b"\x00\x00"
        assert provenance[1]["obj"] is marker
        served = [
            source[1],
            next(iter(source)),
            source._getitems([2])[0],
            source.get_batch(_words([3])).data,
        ]
        for record in served:
            assert set(record) == {"x"}

    def test_nested_provenance_is_keyed_by_its_path(self) -> None:
        records = [
            {"features": {"x": i}, "metadata": {"id": f"item_{i}", "t": i}} for i in range(3)
        ]
        source = _memory(records)
        assert source._provenance.value[2]["metadata/id"] == "item_2"
        assert int(source[2]["metadata"]["t"]) == 2
        assert "id" not in source[2]["metadata"]

    def test_unequal_shapes_are_refused_naming_the_field_shapes_padding_and_packing(self) -> None:
        records = [{"tokens": np.zeros(3)}, {"tokens": np.zeros(5)}]
        with pytest.raises(ValueError) as refused:
            _memory(records)
        message = str(refused.value)
        for part in ("tokens", "(3,)", "(5,)", "pad", "pack"):
            assert part in message, part

    def test_records_of_different_fields_are_refused(self) -> None:
        with pytest.raises(ValueError, match="record 1"):
            _memory([{"a": 1, "b": 2}, {"a": 1}])

    def test_a_refusal_names_a_nested_field_by_its_slash_path(self) -> None:
        records = [{"a": {"b": np.zeros(3)}}, {"a": {"b": np.zeros(5)}}]
        with pytest.raises(ValueError, match="field 'a/b' is"):
            _memory(records)

    def test_columns_of_unequal_lengths_are_refused_naming_each_path(self) -> None:
        """A dictionary key is named by its key and a list position by ``[i]``, joined by ``/``."""
        columns = {"a": {"b": np.zeros(2)}, "c": [np.zeros(3)]}
        with pytest.raises(ValueError, match=r"\{'a/b': 2, 'c/\[0\]': 3\}"):
            eager_source.column_length(columns)

    def test_elements_contribute_their_data(self) -> None:
        records = [Element({"x": np.full(2, i, np.float32), "id": f"e{i}"}) for i in range(3)]
        source = _memory(records)
        np.testing.assert_array_equal(source.data["x"][2], [2.0, 2.0])
        assert source._provenance.value[1]["id"] == "e1"

    def test_an_element_carrying_an_index_is_refused(self) -> None:
        records = [Element({"x": np.zeros(2)}, index=to_words([3])[0])]
        with pytest.raises(ValueError, match="index"):
            _memory(records)

    def test_an_element_carrying_state_is_refused(self) -> None:
        records = [Element({"x": np.zeros(2)}, state={"weight": np.float32(1.0)})]
        with pytest.raises(ValueError, match="state"):
            _memory(records)

    def test_records_with_nothing_numeric_are_refused(self) -> None:
        with pytest.raises(ValueError, match="numeric"):
            _memory([{"name": "a"}, {"name": "b"}])

    def test_scalar_records_are_one_column(self) -> None:
        source = _memory([3, 1, 4])
        assert int(source[2]) == 4
        np.testing.assert_array_equal(source.get_batch(_words([0, 2])).data, [3, 4])


class TestColumnsAreHostArrays:
    """T6, SQ3: every column is stored as NumPy; non-array columns are provenance."""

    def test_device_columns_are_stored_as_numpy(self) -> None:
        source = _memory({"x": jnp.arange(6.0), "y": jnp.ones((6, 2))})
        for column in source.data.values():
            assert isinstance(column, np.ndarray)
        np.testing.assert_array_equal(source.data["x"], np.arange(6.0))

    def test_a_string_column_is_provenance_and_the_arrays_are_served(self) -> None:
        source = _memory({"x": np.arange(3.0), "name": ["a", "b", "c"]})
        assert set(source.data) == {"x"}
        assert [p["name"] for p in source._provenance.value] == ["a", "b", "c"]
        assert set(source.get_batch(_words([1])).data) == {"x"}

    def test_a_numeric_list_column_is_a_column(self) -> None:
        source = _memory({"x": [1, 2, 3], "y": np.arange(3)})
        np.testing.assert_array_equal(source.data["x"], [1, 2, 3])

    def test_a_scalar_value_is_every_record_s(self) -> None:
        source = _memory({"x": np.arange(3), "scale": 2.5})
        np.testing.assert_array_equal(source.get_batch(_words([0, 2]))["scale"], [2.5, 2.5])

    def test_a_mapping_column_is_refused(self) -> None:
        with pytest.raises(TypeError, match="column 'a' is a mapping"):
            _memory({"a": {"b": np.zeros(2)}, "c": np.zeros(2)})


def _graphdef(source: EagerSource) -> Any:
    return nnx.split(source, graph=False)[0]


class TestSourcesUnderTransforms:
    """Section 7: the converted sources in tree mode and under jit."""

    @pytest.mark.parametrize("name", sorted(_SOURCES))
    def test_a_tree_mode_split_and_merge_round_trips(self, name: str) -> None:
        source = _SOURCES[name]()
        graphdef, state = nnx.split(source, graph=False)
        merged = nnx.merge(graphdef, state)
        words = jnp.asarray(_words([5, 2]))
        np.testing.assert_array_equal(
            merged.get_records(words)["x"], source.get_records(words)["x"]
        )

    @pytest.mark.parametrize("name", sorted(_SOURCES))
    def test_no_variable_holds_a_python_value(self, name: str) -> None:
        state = nnx.state(_SOURCES[name](), nnx.Variable)
        for leaf in jax.tree.leaves(state):
            assert isinstance(leaf, jax.Array | np.ndarray), type(leaf)

    def test_the_provenance_holder_is_not_module_state(self) -> None:
        source = _memory([{"x": i, "name": f"r{i}"} for i in range(3)])
        _, state = nnx.split(source, graph=False)
        for leaf in jax.tree.leaves(state):
            assert not isinstance(leaf, str | eager_source.HostProvenance)

    def test_sources_differing_only_in_provenance_share_one_graphdef_and_program(self) -> None:
        def source(prefix: str) -> MemorySource:
            return _memory([{"x": float(i), "name": f"{prefix}{i}"} for i in range(4)])

        a, b = source("a"), source("b")
        assert _graphdef(a) == _graphdef(b)
        assert _graphdef(_memory()) == _graphdef(_memory())
        gather = nnx.jit(lambda s, words: s.get_records(words)["x"])
        words = jnp.asarray(_words([1, 3]))
        with expect_compiles(1):
            jax.block_until_ready(gather(a, words))
            jax.block_until_ready(gather(b, words))
