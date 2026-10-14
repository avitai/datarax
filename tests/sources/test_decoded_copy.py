"""The decoded copy's format and reader: one per-unit writer, the manifest, and every refusal.

A decoded copy holds a source's records as one ``.npy`` file per column and their provenance in
one ArrayRecord file, with a manifest written last. It is written unit by unit through
:func:`~datarax.sources.decoded_copy._units.write_unit`, the one writer, and read back by
:func:`~datarax.sources.decoded_copy.open_decoded_copy` as a ``StreamingDiskSource`` serving
what the source serves, record for record.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle  # nosec B403 - a test pickles the copy's own source
import shutil
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest
from flax import nnx

from datarax.core.data_source import RecordIdentity
from datarax.core.index_words import to_words
from datarax.sources import open_decoded_copy, StreamingDiskSource
from datarax.sources._disk_provenance import decode_provenance, encode_provenance
from datarax.sources.decoded_copy._files import begin_copy
from datarax.sources.decoded_copy._layout import ColumnLayout, COPY_FORMAT
from datarax.sources.decoded_copy._units import (
    build_in_memory,
    source_layout,
    unit_digest,
    write_unit,
)
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from tests.test_common.decoded_copies import every_record, IDENTITY, write_copy
from tests.test_common.worker_reads import logged, png_source, PngDecoder, write_png_records


_RECORDS = 37
_UNIT = 8


def _memory_source() -> MemorySource:
    """Records with an int64 label, a float64 value, a nested float32 field and a text name."""
    rng = np.random.default_rng(0)
    return MemorySource(
        MemorySourceConfig(),
        [
            {
                "label": np.int64(i),
                "value": np.float64(i) / 3,
                "meta": {"score": rng.standard_normal(2).astype(np.float32)},
                "name": f"record-{i}",
                "raw": bytes([i, 255 - i]),
            }
            for i in range(_RECORDS)
        ],
    )


def _same(left: Any, right: Any) -> None:
    """Assert two pytrees of host arrays equal leaf for leaf: structure, dtype and bytes."""
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        a, b = np.asarray(a), np.asarray(b)
        assert a.dtype == b.dtype
        assert a.shape == b.shape
        assert a.tobytes() == b.tobytes()


@pytest.fixture
def png(tmp_path: Path) -> Any:
    return png_source(write_png_records(tmp_path / "png", _RECORDS, shards=3))


class TestTheLayout:
    def test_names_fields_by_path_and_keeps_the_stored_dtypes(self) -> None:
        layout = source_layout(_memory_source())

        assert jax.config.jax_enable_x64 is False
        found = {column.name: (column.dtype, column.shape) for column in layout.columns}
        assert found == {
            "label": ("<i8", ()),
            "value": ("<f8", ()),
            "meta/score": ("<f4", (2,)),
        }
        assert layout.length == _RECORDS
        assert layout.nbytes == _RECORDS * (8 + 8 + 2 * 4)

    def test_a_source_without_a_record_has_no_layout(self) -> None:
        empty = MemorySource(MemorySourceConfig(), {"x": np.zeros((0, 3), np.float32)})

        with pytest.raises(ValueError, match="no record"):
            source_layout(empty)


class TestTheWriter:
    def test_fills_the_given_rows_as_the_source_reads_them(self) -> None:
        source = _memory_source()
        layout = source_layout(source)
        out = {c.name: np.zeros((5, *c.shape), c.dtype) for c in layout.columns}

        provenance = write_unit(source, range(10, 15), out, layout=layout, with_provenance=True)

        batch = source.get_batch(to_words(np.arange(10, 15, dtype=np.uint64)))
        _same(layout.nested(out), batch.data)
        assert provenance is not None
        assert provenance == tuple(
            {"name": f"record-{i}", "raw": bytes([i, 255 - i])} for i in range(10, 15)
        )
        assert all(type(record) is dict for record in provenance)

    def test_returns_no_provenance_unless_asked(self) -> None:
        source = _memory_source()
        layout = source_layout(source)
        out = {c.name: np.zeros((2, *c.shape), c.dtype) for c in layout.columns}

        assert write_unit(source, range(0, 2), out, layout=layout, with_provenance=False) is None

    @pytest.mark.parametrize(
        ("change", "field"),
        [
            ({"dtype": "<f4"}, "value"),
            ({"shape": (3,)}, "meta/score"),
        ],
        ids=["dtype", "shape"],
    )
    def test_refuses_a_leaf_unlike_its_layout_column_naming_the_field(
        self, change: dict[str, Any], field: str
    ) -> None:
        source = _memory_source()
        layout = source_layout(source)
        columns = tuple(
            type(c)(**{"name": c.name, "dtype": c.dtype, "shape": c.shape, **change})
            if c.name == field
            else c
            for c in layout.columns
        )
        other = ColumnLayout(columns=columns, length=layout.length)
        out = {c.name: np.zeros((2, *c.shape), c.dtype) for c in other.columns}

        with pytest.raises(ValueError, match=field):
            write_unit(source, range(0, 2), out, layout=other, with_provenance=False)

    def test_refuses_a_field_the_layout_does_not_hold(self) -> None:
        source = _memory_source()
        layout = source_layout(source)
        fewer = ColumnLayout(
            columns=tuple(c for c in layout.columns if c.name != "value"), length=layout.length
        )
        out = {c.name: np.zeros((2, *c.shape), c.dtype) for c in fewer.columns}

        with pytest.raises(ValueError, match="value"):
            write_unit(source, range(0, 2), out, layout=fewer, with_provenance=False)


def test_a_build_decodes_every_record_once(tmp_path: Path) -> None:
    """The layout comes from unit 0's read, which is written as unit 0: no record twice."""
    log = tmp_path / "decoded.log"
    source = png_source(
        write_png_records(tmp_path / "png", _RECORDS, shards=2), PngDecoder(log=str(log))
    )

    build_in_memory(source, unit_records=_UNIT)

    assert sorted(label for _, label in logged(log)) == list(range(_RECORDS))


def test_a_unit_digest_is_the_sha256_of_its_columns_in_layout_order() -> None:
    out = {"a": np.arange(6, dtype=np.int16), "b/c": np.ones((6, 2), np.float32)}

    expected = hashlib.sha256(out["a"].tobytes() + out["b/c"].tobytes()).digest()

    assert unit_digest(out) == expected


class TestProvenanceEncoding:
    def test_text_bytes_numbers_and_none_round_trip(self) -> None:
        record = {"name": "é-1", "raw": b"\x00\xff", "count": 3, "share": 0.5, "flag": True}
        record["none"] = None

        assert decode_provenance(encode_provenance(record)) == record

    @pytest.mark.parametrize("value", [["a", "b"], ("a",), {"x": 1}, object()])
    def test_a_value_json_cannot_hold_exactly_is_refused_naming_the_field(self, value: Any) -> None:
        with pytest.raises(TypeError, match="tags"):
            encode_provenance({"tags": value})


class TestWritingACopy:
    def test_partial_files_and_no_manifest_until_it_finishes(self, tmp_path: Path) -> None:
        source = _memory_source()
        layout = source_layout(source)
        directory = tmp_path / "copy"

        with begin_copy(directory, layout, unit_records=_UNIT) as files:
            names = sorted(p.name for p in directory.iterdir())
            assert files.units == -(-_RECORDS // _UNIT)
        assert names == [f"{f}.partial" for f in sorted(files.column_files())]
        assert not (directory / "manifest.json").exists()
        assert not list(directory.glob("*.partial")), "an unfinished copy's files are unlinked"

    def test_a_finished_copy_holds_the_columns_the_provenance_and_the_manifest(
        self, tmp_path: Path
    ) -> None:
        directory = tmp_path / "copy"
        write_copy(_memory_source(), directory, identity=IDENTITY, unit_records=_UNIT)

        names = {p.name for p in directory.iterdir()}
        manifest = json.loads((directory / "manifest.json").read_text())
        assert not {name for name in names if name.endswith(".partial")}
        assert manifest["format"] == COPY_FORMAT
        assert manifest["length"] == _RECORDS
        assert manifest["unit_records"] == _UNIT
        assert manifest["source"] == IDENTITY["source"]
        assert manifest["decoder"] == IDENTITY["decoder"]
        assert manifest["provenance"]["file"] in names
        assert {c["file"] for c in manifest["columns"]} <= names
        assert manifest["content"]["algorithm"] == "sha256-of-unit-sha256"


class TestReadingACopy:
    @pytest.mark.parametrize("kind", ["memory", "png"])
    def test_serves_every_record_and_its_provenance_as_the_source_does(
        self, tmp_path: Path, png: Any, kind: str
    ) -> None:
        source = _memory_source() if kind == "memory" else png
        directory = tmp_path / "copy"
        write_copy(source, directory, identity=IDENTITY, unit_records=_UNIT)

        copy = open_decoded_copy(directory)

        assert isinstance(copy, StreamingDiskSource)
        assert copy.record_identity is RecordIdentity.INDEXED
        assert len(copy) == len(source)
        data, provenance = every_record(source)
        copied, copied_provenance = every_record(copy)
        _same(copied, data)
        assert [dict(p) for p in copied_provenance] == [dict(p) for p in provenance]
        rows = to_words(np.array([30, 2, 17, 2], dtype=np.uint64))
        _same(copy.get_batch(rows).data, source.get_batch(rows).data)

    def test_holds_its_maps_and_provenance_out_of_nnx_state_and_pickles_by_place(
        self, tmp_path: Path
    ) -> None:
        directory = tmp_path / "copy"
        write_copy(_memory_source(), directory, identity=IDENTITY, unit_records=_UNIT)
        copy = open_decoded_copy(directory)
        copy.provenance(to_words(np.arange(3, dtype=np.uint64)))  # opens the provenance file

        shapes = [np.shape(leaf) for leaf in jax.tree.leaves(nnx.state(copy))]
        assert all(len(shape) == 0 or shape[0] < _RECORDS for shape in shapes), shapes
        assert len(pickle.dumps(copy)) < 64 << 10
        clone = pickle.loads(pickle.dumps(copy))  # nosec B301 - the test's own object
        _same(every_record(clone)[0], every_record(copy)[0])

    def test_opens_without_its_source_once_moved(self, tmp_path: Path, png: Any) -> None:
        directory = tmp_path / "copy"
        write_copy(png, directory, identity=png.source_identity(), unit_records=_UNIT)
        expected = every_record(png)[0]
        shutil.rmtree(tmp_path / "png")
        moved = tmp_path / "elsewhere" / "copy"
        moved.parent.mkdir()
        directory.rename(moved)

        _same(every_record(open_decoded_copy(moved))[0], expected)

    def test_verify_recomputes_the_content_digest(self, tmp_path: Path) -> None:
        directory = tmp_path / "copy"
        write_copy(_memory_source(), directory, identity=IDENTITY, unit_records=_UNIT)

        open_decoded_copy(directory, verify=True)

    def test_positive_control_one_flipped_byte_fails_verification(self, tmp_path: Path) -> None:
        directory = tmp_path / "copy"
        write_copy(_memory_source(), directory, identity=IDENTITY, unit_records=_UNIT)
        manifest = json.loads((directory / "manifest.json").read_text())
        column = directory / manifest["columns"][0]["file"]
        data = bytearray(column.read_bytes())
        data[-1] ^= 0xFF
        column.write_bytes(bytes(data))

        open_decoded_copy(directory)  # sizes match: only a full read sees it
        with pytest.raises(ValueError, match="content digest"):
            open_decoded_copy(directory, verify=True)


class TestARefusedCopy:
    def _copy(self, tmp_path: Path) -> Path:
        directory = tmp_path / "copy"
        write_copy(_memory_source(), directory, identity=IDENTITY, unit_records=_UNIT)
        return directory

    def test_without_a_manifest_it_is_incomplete(self, tmp_path: Path) -> None:
        directory = self._copy(tmp_path)
        (directory / "manifest.json").unlink()

        with pytest.raises(FileNotFoundError, match=r"no manifest\.json.*incomplete"):
            open_decoded_copy(directory)

    def test_another_format_is_refused_naming_both(self, tmp_path: Path) -> None:
        directory = self._copy(tmp_path)
        path = directory / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["format"] = COPY_FORMAT + 1
        path.write_text(json.dumps(manifest))

        with pytest.raises(ValueError, match=rf"format {COPY_FORMAT + 1}.*format {COPY_FORMAT}"):
            open_decoded_copy(directory)

    @pytest.mark.parametrize("change", ["missing", "shorter"])
    def test_a_column_file_unlike_the_manifest_is_refused(
        self, tmp_path: Path, change: str
    ) -> None:
        directory = self._copy(tmp_path)
        manifest = json.loads((directory / "manifest.json").read_text())
        column = directory / manifest["columns"][1]["file"]
        if change == "missing":
            column.unlink()
        else:
            os.truncate(column, column.stat().st_size - 1)

        with pytest.raises((FileNotFoundError, ValueError), match=manifest["columns"][1]["name"]):
            open_decoded_copy(directory)

    def test_an_identity_that_differs_is_refused_naming_the_field(self, tmp_path: Path) -> None:
        directory = self._copy(tmp_path)
        other = {"source": {**IDENTITY["source"], "records": 1}, "decoder": {"lib": "2"}}

        with pytest.raises(ValueError, match=r"decoder/lib.*source/records"):
            open_decoded_copy(directory, identity=other)

    def test_the_identity_it_was_written_with_matches(self, tmp_path: Path) -> None:
        open_decoded_copy(self._copy(tmp_path), identity=IDENTITY)
