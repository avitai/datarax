"""Ragged records: sources that hold a field whose records' lengths differ, and read it for packing.

A field is ragged when its records' arrays differ in length along their first axis and agree on
every other axis. ``MemorySource`` (a column of per-record arrays, or a list of records) and
``HFEagerSource`` (an Arrow list column) hold such a field as one values array and int64 offsets,
and serve every field through the ragged read: ``lengths(indices, field)`` and
``get_ragged(indices, field) -> (values, offsets)``. A field's length for a record is the length of
that record's array, a scalar counting 1, as Grain's packer counts it, so a fixed column is read the
same way. A batch's shapes are static, so a source holding a ragged field refuses every unpacked
read, naming the field, packing and padding. Sources that do not read ragged records refuse the
ragged read: ArrayRecord and TFDS sources name the open work, and a stream names why it cannot be
packed.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from substrax.testing.compiles import compiled_programs

from datarax.core.data_source import RecordIdentity
from datarax.core.element_batch import PADDING_INDEX
from datarax.core.index_words import to_words
from datarax.pipeline import Pipeline
from datarax.sources import MemorySource, MemorySourceConfig
from tests.test_common.streams import RecordStream


_RAGGED = "'tokens' is ragged.*pack records with segment ids.*pad the field"


def _words(rows: list[int]) -> np.ndarray:
    return to_words(np.asarray(rows, np.uint64))


def _tokens() -> list[np.ndarray]:
    return [
        np.arange(2, dtype=np.int32),
        np.arange(10, 13, dtype=np.int32),
        np.array([7], np.int32),
    ]


def _expected(records: list[np.ndarray], rows: list[int]) -> tuple[np.ndarray, np.ndarray]:
    chosen = [records[row] for row in rows]
    offsets = np.concatenate([[0], np.cumsum([len(r) for r in chosen])]).astype(np.int64)
    values = np.concatenate(chosen) if chosen else np.zeros((0,), records[0].dtype)
    return values, offsets


def _assert_read(source: MemorySource, field: str, records: list[np.ndarray], rows: list[int]):
    values, offsets = source.get_ragged(_words(rows), field)
    want_values, want_offsets = _expected(records, rows)
    assert offsets.dtype == np.int64
    np.testing.assert_array_equal(offsets, want_offsets)
    assert values.dtype == want_values.dtype
    np.testing.assert_array_equal(values, want_values)
    np.testing.assert_array_equal(
        source.lengths(_words(rows), field), [len(records[row]) for row in rows]
    )


class TestMemorySourceHoldsARaggedColumn:
    """A column of per-record arrays of different lengths is one ragged field."""

    @pytest.mark.parametrize("form", ["list", "tuple", "object array"])
    def test_a_column_of_per_record_arrays_is_ragged(self, form: str) -> None:
        records = _tokens()
        column: object = records
        if form == "tuple":
            column = tuple(records)
        elif form == "object array":
            column = np.empty(len(records), dtype=object)
            column[:] = records
        source = MemorySource(MemorySourceConfig(), {"tokens": column, "y": np.arange(3)})

        assert source.ragged_fields == ("tokens",)
        assert len(source) == 3
        for rows in ([0, 1, 2], [2, 0], [1, 1], []):
            _assert_read(source, "tokens", records, rows)

    def test_a_record_of_length_zero_holds_no_values(self) -> None:
        records = [np.arange(3, dtype=np.int32), np.zeros(0, np.int32), np.array([5], np.int32)]
        source = MemorySource(MemorySourceConfig(), {"tokens": records})

        np.testing.assert_array_equal(source.lengths(_words([1, 0]), "tokens"), [0, 3])
        _assert_read(source, "tokens", records, [0, 1, 2])

    def test_a_ragged_field_keeps_its_inner_shape(self) -> None:
        records = [np.full((n, 2), n, np.float32) for n in (3, 1, 2)]
        source = MemorySource(MemorySourceConfig(), {"points": records})

        values, offsets = source.get_ragged(_words([2, 0]), "points")
        assert values.shape == (5, 2)
        np.testing.assert_array_equal(offsets, [0, 2, 5])
        np.testing.assert_array_equal(values, np.concatenate([records[2], records[0]]))

    def test_records_of_a_list_are_ragged_where_their_lengths_differ(self) -> None:
        records = _tokens()
        source = MemorySource(
            MemorySourceConfig(),
            [{"tokens": t, "label": np.int32(k), "id": f"r{k}"} for k, t in enumerate(records)],
        )

        assert source.ragged_fields == ("tokens",)
        _assert_read(source, "tokens", records, [1, 2, 0])
        assert source.provenance(_words([1]))[0]["id"] == "r1"

    def test_a_nested_record_field_is_named_by_its_path(self) -> None:
        records = _tokens()
        source = MemorySource(MemorySourceConfig(), [{"text": {"tokens": t}} for t in records])

        assert source.ragged_fields == ("text/tokens",)
        _assert_read(source, "text/tokens", records, [0, 2])


class TestTheRaggedReadServesEveryField:
    """Every field is read the same way: a record's length is its array's, a scalar counts 1."""

    def test_a_fixed_column_is_read_as_records_of_its_row_length(self) -> None:
        fixed = np.arange(12, dtype=np.int32).reshape(3, 4)
        source = MemorySource(MemorySourceConfig(), {"tokens": _tokens(), "window": fixed})

        np.testing.assert_array_equal(source.lengths(_words([2, 0]), "window"), [4, 4])
        values, offsets = source.get_ragged(_words([2, 0]), "window")
        np.testing.assert_array_equal(values, np.concatenate([fixed[2], fixed[0]]))
        np.testing.assert_array_equal(offsets, [0, 4, 8])

    def test_a_scalar_field_counts_one_per_record(self) -> None:
        source = MemorySource(MemorySourceConfig(), {"tokens": _tokens(), "y": np.array([5, 6, 7])})

        np.testing.assert_array_equal(source.lengths(_words([1, 2]), "y"), [1, 1])
        values, offsets = source.get_ragged(_words([1, 2]), "y")
        np.testing.assert_array_equal(values, [6, 7])
        np.testing.assert_array_equal(offsets, [0, 1, 2])

    def test_a_source_without_ragged_fields_reads_its_columns_too(self) -> None:
        fixed = np.arange(6, dtype=np.int32).reshape(3, 2)
        source = MemorySource(MemorySourceConfig(), {"x": fixed})

        assert source.ragged_fields == ()
        values, offsets = source.get_ragged(_words([1]), "x")
        np.testing.assert_array_equal(values, fixed[1])
        np.testing.assert_array_equal(offsets, [0, 2])

    def test_an_unknown_field_is_refused_naming_the_fields(self) -> None:
        source = MemorySource(MemorySourceConfig(), {"tokens": _tokens(), "y": np.arange(3)})

        with pytest.raises(ValueError, match=r"'z'.*\['tokens', 'y'\]"):
            source.get_ragged(_words([0]), "z")
        with pytest.raises(ValueError, match=r"'z'.*\['tokens', 'y'\]"):
            source.lengths(_words([0]), "z")

    def test_indices_naming_no_record_are_refused(self) -> None:
        source = MemorySource(MemorySourceConfig(), {"tokens": _tokens()})
        padding = np.full((1, 2), PADDING_INDEX, np.uint32)

        with pytest.raises(IndexError, match="padding index"):
            source.get_ragged(padding, "tokens")
        with pytest.raises(IndexError, match="outside"):
            source.lengths(_words([3]), "tokens")


class TestEqualLengthsStayAColumn:
    """Per-record arrays of one shape are a fixed column, served by unpacked reads as today."""

    def test_a_list_of_equal_arrays_is_a_column(self) -> None:
        source = MemorySource(MemorySourceConfig(), {"x": [np.arange(3)] * 4})

        assert source.ragged_fields == ()
        assert source.get_batch(_words([0, 3])).data["x"].shape == (2, 3)

    def test_an_object_array_of_strings_stays_provenance(self) -> None:
        names = np.array(["a", "b", "c"], dtype=object)
        source = MemorySource(MemorySourceConfig(), {"name": names, "y": np.arange(3)})

        assert source.ragged_fields == ()
        assert source.provenance(_words([2]))[0]["name"] == "c"


class TestUnpackedReadsOfARaggedSourceAreRefused:
    """A batch's shapes are static: every unpacked read names the field, packing and padding."""

    @pytest.fixture
    def source(self) -> MemorySource:
        return MemorySource(MemorySourceConfig(), {"tokens": _tokens(), "y": np.arange(3)})

    def test_the_host_read_is_refused(self, source: MemorySource) -> None:
        with pytest.raises(ValueError, match=_RAGGED):
            source.get_batch(_words([0, 1]))

    def test_indexing_and_iteration_are_refused(self, source: MemorySource) -> None:
        with pytest.raises(ValueError, match=_RAGGED):
            source[0]
        with pytest.raises(ValueError, match=_RAGGED):
            list(source)
        with pytest.raises(ValueError, match=_RAGGED):
            source._getitems([0, 1])

    def test_the_traced_read_and_its_columns_are_refused(self, source: MemorySource) -> None:
        with pytest.raises(ValueError, match=_RAGGED):
            source.get_records(jnp.asarray(_words([0])))
        with pytest.raises(ValueError, match=_RAGGED):
            source.device_columns()

    def test_the_element_spec_is_refused(self, source: MemorySource) -> None:
        with pytest.raises(ValueError, match=_RAGGED):
            source.element_spec()


_NOT_RAGGED = "'points'.*more than their first axis.*pad the field to one shape"


class TestUnderTransforms:
    """A ragged source is refused while a program is traced, never compiled; a column source's
    traced read compiles once, so the refusal's check costs no program."""

    @pytest.fixture
    def ragged(self) -> MemorySource:
        return MemorySource(MemorySourceConfig(), {"tokens": _tokens(), "y": np.arange(3)})

    def test_jax_jit_of_the_traced_read_is_refused_while_tracing(
        self, ragged: MemorySource
    ) -> None:
        with pytest.raises(ValueError, match=_RAGGED):
            jax.jit(ragged.get_records)(jnp.asarray(_words([0, 1])))

    @pytest.mark.parametrize("graph", [True, False], ids=["graph mode", "tree mode"])
    def test_nnx_jit_over_the_source_is_refused_while_tracing(
        self, ragged: MemorySource, graph: bool
    ) -> None:
        @nnx.jit(graph=graph)
        def read(source: MemorySource, indices: jax.Array) -> object:
            return source.get_records(indices)

        with pytest.raises(ValueError, match=_RAGGED):
            read(ragged, jnp.asarray(_words([0, 1])))

    def test_a_pipeline_over_a_ragged_source_is_refused(self, ragged: MemorySource) -> None:
        with pytest.raises(ValueError, match=_RAGGED):
            pipe = Pipeline(source=ragged, stages=[], batch_size=2, rngs=nnx.Rngs(0))
            next(iter(pipe))
        with pytest.raises(ValueError, match=_RAGGED):
            Pipeline(source=ragged, stages=[], batch_size=2, rngs=nnx.Rngs(0)).step()

    def test_a_column_source_s_traced_read_compiles_once(self) -> None:
        source = MemorySource(MemorySourceConfig(), {"x": np.arange(12.0).reshape(6, 2)})
        columns = source.device_columns()
        read = jax.jit(lambda indices, columns: source.get_records(indices, columns=columns))

        jax.clear_caches()
        # Inputs are made outside the measured window: placing them compiles programs of its own.
        indices = [jnp.asarray(_words(rows)) for rows in ([0, 1], [2, 3], [4, 5])]
        with compiled_programs() as programs:
            for batch in indices:
                read(batch, columns)
        assert [str(program) for program in programs] == ["jit(<lambda>)"]


class TestShapesThatAreNotRaggedAreRefused:
    """Packing joins records along their first axis only: records differing past it are refused.

    The refusal advises padding alone ("pack records" is the packing advice other refusals give),
    since no packing of such a field exists.
    """

    def test_records_differing_past_their_first_axis_are_refused(self) -> None:
        records = [np.zeros((2, 3), np.float32), np.zeros((2, 4), np.float32)]

        with pytest.raises(ValueError, match=_NOT_RAGGED) as refused:
            MemorySource(MemorySourceConfig(), {"points": records})
        assert "pack records" not in str(refused.value)

    def test_records_differing_in_rank_are_refused(self) -> None:
        records = [np.zeros((2,), np.float32), np.zeros((2, 1), np.float32)]

        with pytest.raises(ValueError, match=_NOT_RAGGED) as refused:
            MemorySource(MemorySourceConfig(), [{"points": r} for r in records])
        assert "pack records" not in str(refused.value)


class TestSourcesThatDoNotReadRaggedRecordsRefuse:
    """The ragged read is refused, naming what packing reads from or why a stream cannot be."""

    def test_an_array_record_source_names_the_open_work(self, tmp_path: Path) -> None:
        from array_record.python.array_record_module import ArrayRecordWriter  # noqa: PLC0415

        from datarax.sources.array_record_source import (  # noqa: PLC0415
            ArrayRecordSourceConfig,
            ArrayRecordSourceModule,
        )

        path = str(tmp_path / "records.array_record")
        writer = ArrayRecordWriter(path, "group_size:1")
        for k in range(3):
            writer.write(np.int32(k).tobytes())
        writer.close()
        source = ArrayRecordSourceModule(
            ArrayRecordSourceConfig(),
            [path],
            decode=lambda records: [{"x": np.frombuffer(r, np.int32)} for r in records],
        )

        assert source.ragged_fields == ()
        with pytest.raises(NotImplementedError, match="ArrayRecordSourceModule.*MemorySource"):
            source.get_ragged(_words([0]), "x")
        with pytest.raises(NotImplementedError, match="ArrayRecordSourceModule.*MemorySource"):
            source.lengths(_words([0]), "x")

    @pytest.mark.parametrize("kind", [RecordIdentity.STREAM_IDS, RecordIdentity.ARRIVAL])
    def test_a_stream_names_why_it_cannot_be_packed(self, kind: RecordIdentity) -> None:
        stream = RecordStream({"x": np.arange(4)}, kind=kind)

        assert stream.ragged_fields == ()
        with pytest.raises(TypeError, match="RecordStream.*stream.*indexed"):
            stream.get_ragged(_words([0]), "x")
        with pytest.raises(TypeError, match="RecordStream.*stream.*indexed"):
            stream.lengths(_words([0]), "x")


datasets = pytest.importorskip("datasets")

from datarax.sources.hf_source import HFEagerConfig, HFEagerSource  # noqa: E402


def _hf_source(monkeypatch: pytest.MonkeyPatch, dataset: object) -> HFEagerSource:
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: dataset)
    return HFEagerSource(HFEagerConfig(name="fake", split="train"))


class TestHFEagerSourceReadsArrowListColumns:
    """A list column whose records' lengths differ is ragged, read through Arrow's offsets."""

    def test_a_list_column_of_different_lengths_is_ragged_with_its_declared_dtype(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        records = [np.arange(3, dtype=np.uint8), np.zeros(0, np.uint8), np.array([9, 8], np.uint8)]
        table = datasets.Dataset.from_dict(
            {"tokens": records, "label": [0, 1, 2], "text": ["a", "b", "c"]}
        )
        source = _hf_source(monkeypatch, table)

        assert source.ragged_fields == ("tokens",)
        values, offsets = source.get_ragged(_words([2, 0, 1]), "tokens")
        assert values.dtype == np.uint8
        np.testing.assert_array_equal(values, [9, 8, 0, 1, 2])
        np.testing.assert_array_equal(offsets, [0, 2, 5, 5])
        np.testing.assert_array_equal(source.lengths(_words([0, 1, 2]), "tokens"), [3, 0, 2])
        assert source.provenance(_words([2]))[0]["text"] == "c"

    def test_a_selected_dataset_reads_its_selected_rows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        records = [[1, 2, 3], [4], [5, 6], []]
        table = datasets.Dataset.from_dict({"tokens": records}).select([3, 1, 2])
        source = _hf_source(monkeypatch, table)

        values, offsets = source.get_ragged(_words([0, 1, 2]), "tokens")
        np.testing.assert_array_equal(values, [4, 5, 6])
        np.testing.assert_array_equal(offsets, [0, 0, 1, 3])

    def test_a_nested_list_column_keeps_its_inner_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        records = [np.full((n, 2), n, np.float32) for n in (3, 1)]
        source = _hf_source(monkeypatch, datasets.Dataset.from_dict({"points": records}))

        values, offsets = source.get_ragged(_words([1, 0]), "points")
        assert values.dtype == np.float32
        assert values.shape == (4, 2)
        np.testing.assert_array_equal(offsets, [0, 1, 4])
        np.testing.assert_array_equal(values, np.concatenate([records[1], records[0]]))

    def test_a_list_column_of_equal_lengths_stays_a_column(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        table = datasets.Dataset.from_dict({"feature": [np.zeros(3, np.float32)] * 4})
        source = _hf_source(monkeypatch, table)

        assert source.ragged_fields == ()
        assert source.get_batch(_words([0, 1])).data["feature"].shape == (2, 3)

    def test_an_unpacked_read_of_a_ragged_column_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        table = datasets.Dataset.from_dict({"tokens": [[1, 2], [3]]})
        source = _hf_source(monkeypatch, table)

        with pytest.raises(ValueError, match=_RAGGED):
            source.get_batch(_words([0]))
