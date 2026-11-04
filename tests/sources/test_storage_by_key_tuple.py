"""Stored fields are named by key tuple, so every storage path serves a record's fields as given.

A field whose key holds ``/`` (``("a/b",)``) and a nested one (``("meta", "score")``) are different
fields with one spelling's worth of characters, so storage names neither by spelling. RAM
materialize, the per-unit in-memory build, a decoded copy written and reopened, and the disk source
it opens as all serve the source's key tuples and values: COCO's field beside a literal key and
group beside a literal key included (defect 6). A record two of whose fields share a spelling is
refused once, at the layout, before any file is reserved. A ``Maybe`` comes back from RAM as a
``Maybe`` (defect 7), and a decoded copy, which stores arrays, refuses one through the field-node
helper. A copy's identity is compared by key tuple and its refusals print each field by label.
"""

from __future__ import annotations

import json
import logging
import pickle  # nosec B403 - the test pickles its own copy
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import cloudpickle
import jax
import numpy as np
import pytest
from flax import nnx

from datarax.core.data_source import DataSourceModule
from datarax.core.index_words import to_words
from datarax.core.maybe import Maybe
from datarax.pipeline import materialize, Pipeline
from datarax.pipeline.read_plan import HostReadPath
from datarax.sources import open_decoded_copy, StreamingDiskSource
from datarax.sources.decoded_copy._manifest import identity_differences
from datarax.sources.decoded_copy._units import build_in_memory, source_layout, write_unit
from datarax.sources.memory_source import MemorySource
from datarax.sources.mixed_source import MixDataSourcesConfig, MixDataSourcesNode
from tests.test_common.worker_reads import (
    png_source,
    resources,
    served,
    stream_of,
    WORKER_COUNTS,
    write_png_records,
)


_LENGTH = 12
_UNIT = 5


class _Identified(MemorySource):
    """An in-memory source that declares an identity, so a decoded copy of it can be written."""

    def source_identity(self) -> dict[str, Any]:
        """A fixed identity: the test's records are its own."""
        return {"source": {"kind": "test-identified", "records": len(self)}, "decoder": {}}


def _nested_and_literal(length: int = _LENGTH) -> dict[str, Any]:
    rows = np.arange(length)
    return {
        "meta": {"score": np.stack([rows, -rows], axis=1).astype(np.float32)},
        "a/b": (rows[:, None] * np.ones(3)).astype(np.float32) / 10,
        "label": rows.astype(np.int64) % 4,
    }


def _coco() -> dict[str, Any]:
    """COCO's shape: a field beside a literal key, and a group beside a literal key."""
    rows = np.arange(_LENGTH)
    return {
        "image": np.broadcast_to(rows[:, None, None], (_LENGTH, 4, 3)).astype(np.uint8),
        "image/id": rows.astype(np.int64) + 1000,
        "objects": {"label": np.stack([rows, rows + 1], axis=1).astype(np.int64)},
        "objects/bbox": np.broadcast_to(
            rows[:, None, None].astype(np.float32), (_LENGTH, 2, 4)
        ).copy(),
    }


def _with_a_maybe() -> dict[str, Any]:
    rows = np.arange(_LENGTH)
    return {
        "x": rows.astype(np.float32),
        "y": Maybe(rows.astype(np.float32) * 2, rows % 3 != 0),
    }


def _source(columns: Mapping[str, Any]) -> _Identified:
    return _Identified.of_columns(columns)


def _words(rows: list[int]) -> np.ndarray:
    return to_words(np.asarray(rows, dtype=np.uint64))


_ROWS = [7, 0, 11, 3, 3]


def _same_record(found: DataSourceModule, source: DataSourceModule) -> None:
    """``found`` serves ``source``'s fields by jax key path, value for value, and its spec.

    A jax key path prints ``['a/b']`` and ``['a']['b']`` apart, so a literal key never passes for
    a nested one.
    """
    want = source.get_batch(_words(_ROWS))  # type: ignore[attr-defined]
    got = found.get_batch(_words(_ROWS))  # type: ignore[attr-defined]
    assert served(got)[3] == served(want)[3]
    assert jax.tree.structure(found.element_spec()) == jax.tree.structure(source.element_spec())


def _ram(source: DataSourceModule, tmp_path: Path) -> DataSourceModule:
    del tmp_path
    return materialize(source, unit_records=_UNIT)


def _in_memory(source: DataSourceModule, tmp_path: Path) -> DataSourceModule:
    del tmp_path
    columns, provenance = build_in_memory(source, unit_records=_UNIT)
    return MemorySource.of_columns(columns, provenance)


def _copy(source: DataSourceModule, tmp_path: Path) -> DataSourceModule:
    directory = tmp_path / "copy"
    materialize(source, directory=directory, unit_records=_UNIT)
    return open_decoded_copy(directory, verify=True)


_PATHS: dict[str, Callable[[DataSourceModule, Path], DataSourceModule]] = {
    "ram": _ram,
    "build_in_memory": _in_memory,
    "copy": _copy,
}


class TestEveryPathServesTheKeyTuples:
    """The brief's first red test: a nested and a literal field round-trip through storage."""

    @pytest.mark.parametrize("path", list(_PATHS))
    def test_a_nested_field_beside_a_literal_key(self, path: str, tmp_path: Path) -> None:
        source = _source(_nested_and_literal())

        found = _PATHS[path](source, tmp_path)

        _same_record(found, source)
        data = found.get_batch(_words([4])).data  # type: ignore[attr-defined]
        assert set(data) == {"meta", "a/b", "label"}
        assert set(data["meta"]) == {"score"}

    @pytest.mark.parametrize("path", list(_PATHS))
    def test_coco_s_field_and_group_beside_literal_keys(self, path: str, tmp_path: Path) -> None:
        """Defect 6: ``image`` beside ``image/id``, ``objects`` beside ``objects/bbox``."""
        source = _source(_coco())

        _same_record(_PATHS[path](source, tmp_path), source)

    def test_the_copy_is_a_disk_source_whose_columns_hold_the_key_tuples(
        self, tmp_path: Path
    ) -> None:
        copy = _copy(_source(_nested_and_literal()), tmp_path)

        assert isinstance(copy, StreamingDiskSource)
        assert {column.keys for column in copy.config.columns} == {
            ("meta", "score"),
            ("a/b",),
            ("label",),
        }
        manifest = json.loads((tmp_path / "copy" / "manifest.json").read_text())
        assert manifest["format"] == 2
        assert sorted(column["keys"] for column in manifest["columns"]) == [
            ["a/b"],
            ["label"],
            ["meta", "score"],
        ]
        assert all("name" not in column for column in manifest["columns"])

    def test_the_ram_result_holds_its_columns_out_of_nnx_state(self, tmp_path: Path) -> None:
        """OQ-T24: the RAM result's columns, the literal key's among them, are no state."""
        ram = _ram(_source(_nested_and_literal()), tmp_path)

        for leaf in jax.tree.leaves(nnx.state(ram)):
            assert np.ndim(leaf) == 0 or np.shape(leaf)[0] != _LENGTH, np.shape(leaf)

    @pytest.mark.parametrize("dumps", [pickle.dumps, cloudpickle.dumps], ids=["pickle", "cloud"])
    def test_a_copy_s_key_tuples_cross_a_pickle(
        self, tmp_path: Path, dumps: Callable[[Any], bytes]
    ) -> None:
        source = _source(_nested_and_literal())
        copy = _copy(source, tmp_path)

        clone = pickle.loads(dumps(copy))  # nosec B301 - the test's own object

        _same_record(clone, source)


class TestAMixReadInWorkerProcesses:
    """A literal-key copy beside a GIL-bound child: the mix is read in Grain's worker processes."""

    def test_serves_the_thread_run_s_stream(self, tmp_path: Path) -> None:
        workers = WORKER_COUNTS[min(1, len(WORKER_COUNTS) - 1)]  # derived from available_cpus()
        paths = write_png_records(tmp_path / "png", _LENGTH, shards=2)
        record = _nested_and_literal()
        copy = _copy(_source({"meta": record["meta"], "a/b": record["a/b"]}), tmp_path)

        def run(host_workers: int) -> tuple[Any, Pipeline]:
            mix = MixDataSourcesNode(
                MixDataSourcesConfig(weights=(1.0, 1.0)), [png_source(paths), copy]
            )
            pipe = Pipeline(
                source=mix,
                stages=[],
                batch_size=4,
                rngs=nnx.Rngs(1),
                num_epochs=1,
                host_resources=resources(host_workers) if host_workers else None,
            )
            return stream_of(pipe.raw_batches()), pipe

        reference, threads = run(0)
        found, processes = run(workers)

        assert processes.host_plan.path is HostReadPath.PROCESSES
        assert found == reference
        keys = {key for unit in found for key, _ in unit[3]}
        assert "['a/b'].value" in keys and "['meta']['score'].value" in keys
        threads.close()
        processes.close()


class TestACopyOfAnotherFormat:
    def test_materialize_rebuilds_it_and_logs_why(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A current copy mutated to format 1 is rebuilt, as every other format is."""
        directory = tmp_path / "copy"
        record = _nested_and_literal()
        source = _source({"meta": record["meta"], "label": record["label"]})
        materialize(source, directory=directory, unit_records=_UNIT)
        path = directory / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["format"] = 1
        path.write_text(json.dumps(manifest))

        with caplog.at_level(logging.INFO, logger="datarax.pipeline.materialization"):
            rebuilt = materialize(source, directory=directory, unit_records=_UNIT)

        assert "format 1, and this datarax reads format 2" in caplog.text
        assert json.loads(path.read_text())["format"] == 2
        _same_record(rebuilt, source)


class TestASharedSpellingIsRefusedAtTheLayout:
    """``("objects", "bbox")`` beside ``("objects/bbox",)``: no Pipeline accepts the record."""

    def _source(self) -> _Identified:
        return _source(
            {
                "objects": {"bbox": np.zeros((_LENGTH, 4), np.float32)},
                "objects/bbox": np.ones((_LENGTH, 4), np.float32),
            }
        )

    _REFUSAL = r"'objects/bbox' names two fields: \('objects', 'bbox'\) and \('objects/bbox',\)"

    def test_by_ram_materialize_and_the_in_memory_build(self, tmp_path: Path) -> None:
        for path in ("ram", "build_in_memory"):
            with pytest.raises(ValueError, match=self._REFUSAL):
                _PATHS[path](self._source(), tmp_path)

    def test_by_the_copy_writer_before_any_file_is_reserved(self, tmp_path: Path) -> None:
        directory = tmp_path / "copy"
        with pytest.raises(ValueError, match=self._REFUSAL):
            materialize(self._source(), directory=directory, unit_records=_UNIT)

        assert {p.name for p in directory.iterdir()} <= {".lock"}

    def test_as_a_pipeline_over_the_source_refuses_it(self) -> None:
        with pytest.raises(ValueError, match=self._REFUSAL):
            Pipeline(source=self._source(), stages=[], batch_size=4, rngs=nnx.Rngs(0))


class TestAMaybe:
    """Defect 7: RAM keeps a field node; a decoded copy, which stores arrays, refuses one."""

    @pytest.mark.parametrize("path", ["ram", "build_in_memory"])
    def test_comes_back_from_ram_as_a_maybe(self, path: str, tmp_path: Path) -> None:
        source = _source(_with_a_maybe())

        found = _PATHS[path](source, tmp_path)

        data = found.get_batch(_words(_ROWS)).data  # type: ignore[attr-defined]
        assert isinstance(data["y"], Maybe)
        np.testing.assert_array_equal(data["y"].present, np.asarray(_ROWS) % 3 != 0)
        np.testing.assert_array_equal(data["y"].value, np.asarray(_ROWS, np.float32) * 2)
        assert isinstance(found.element_spec()["y"], Maybe)
        _same_record(found, source)

    def test_the_mix_s_maybe_comes_back_from_ram_as_a_maybe(self, tmp_path: Path) -> None:
        mix = MixDataSourcesNode(
            MixDataSourcesConfig(weights=(1.0, 1.0)),
            [
                MemorySource.of_columns({"x": np.zeros(8, np.float32)}),
                MemorySource.of_columns(
                    {"x": np.ones(8, np.float32), "y": np.arange(8, dtype=np.float32)}
                ),
            ],
        )

        found = _ram(mix, tmp_path)

        _same_record(found, mix)
        assert isinstance(found.element_spec()["y"], Maybe)
        for leaf in jax.tree.leaves(nnx.state(found)):
            assert np.ndim(leaf) == 0 or np.shape(leaf)[0] != len(mix), np.shape(leaf)

    def test_a_copy_refuses_it_through_the_field_node_helper(self, tmp_path: Path) -> None:
        directory = tmp_path / "copy"

        with pytest.raises(
            TypeError, match=r"a decoded copy stores its fields as arrays, and 'y' is a Maybe"
        ):
            materialize(_source(_with_a_maybe()), directory=directory, unit_records=_UNIT)

        assert {p.name for p in directory.iterdir()} <= {".lock"}


class TestTheIdentity:
    def test_a_literal_key_and_a_nested_one_are_compared_apart(self) -> None:
        found = {"source": {"a/b": 1, "a": {"b": 2}}, "decoder": {}}
        expected = {"source": {"a/b": 1, "a": {"b": 3}}, "decoder": {}}

        assert identity_differences(found, expected) == [("source", "a", "b")]
        assert identity_differences(found, found) == []

    def test_its_refusal_prints_each_field_by_label(self, tmp_path: Path) -> None:
        directory = tmp_path / "copy"
        materialize(_source(_nested_and_literal()), directory=directory, unit_records=_UNIT)
        other = {"source": {"kind": "test-identified", "records": 1, "a/b": 0}, "decoder": {}}

        with pytest.raises(
            ValueError,
            match=r"differs in 'source/a/b' \(key tuple \('source', 'a/b'\)\), 'source/records';",
        ):
            open_decoded_copy(directory, identity=other)


class TestRefusalTexts:
    """Each prints a field by label and lists fields in name order."""

    def test_a_unit_whose_fields_differ_lists_both_in_name_order(self) -> None:
        source = _source(_nested_and_literal())
        layout = source_layout(source)
        fewer = type(layout)(
            columns=tuple(c for c in layout.columns if c.keys != ("label",)),
            length=layout.length,
        )
        out = {c.keys: np.zeros((2, *c.shape), c.dtype) for c in fewer.columns}

        with pytest.raises(
            ValueError,
            match=(
                r"hold the fields 'a/b' \(key tuple \('a/b',\)\), 'label', 'meta/score' where "
                r"the copy's columns are 'a/b' \(key tuple \('a/b',\)\), 'meta/score': "
                r"'label' differ"
            ),
        ):
            write_unit(source, range(0, 2), out, layout=fewer, with_provenance=False)

    def test_a_unit_field_unlike_its_column_is_named_by_label(self) -> None:
        source = _source(_nested_and_literal())
        layout = source_layout(source)
        columns = tuple(
            type(c)(keys=c.keys, dtype="<f8", shape=c.shape) if c.keys == ("a/b",) else c
            for c in layout.columns
        )
        other = type(layout)(columns=columns, length=layout.length)
        out = {c.keys: np.zeros((2, *c.shape), c.dtype) for c in other.columns}

        with pytest.raises(ValueError, match=r"field 'a/b' \(key tuple \('a/b',\)\) of records"):
            write_unit(source, range(0, 2), out, layout=other, with_provenance=False)

    def test_a_reopened_copy_s_missing_column_is_named_by_label(self, tmp_path: Path) -> None:
        directory = tmp_path / "copy"
        materialize(_source(_nested_and_literal()), directory=directory, unit_records=_UNIT)
        manifest = json.loads((directory / "manifest.json").read_text())
        entry = next(c for c in manifest["columns"] if c["keys"] == ["a/b"])
        (directory / entry["file"]).unlink()

        with pytest.raises(FileNotFoundError, match=r"column 'a/b' \(key tuple \('a/b',\)\)'s"):
            open_decoded_copy(directory)

    def test_the_disk_source_s_repr_prints_each_column_by_label(self, tmp_path: Path) -> None:
        copy = _copy(_source(_nested_and_literal()), tmp_path)

        text = repr(copy)

        assert "'meta/score': " in text
        assert "'a/b' (key tuple ('a/b',)): " in text

    def test_materialize_logs_the_differing_fields_by_label(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        directory = tmp_path / "copy"
        materialize(_source(_nested_and_literal()), directory=directory, unit_records=_UNIT)
        longer = _source(_nested_and_literal(_LENGTH + 1))  # its identity's record count differs

        with caplog.at_level(logging.INFO, logger="datarax.pipeline.materialization"):
            materialize(longer, directory=directory, unit_records=_UNIT)

        assert "(it differs in 'source/records')" in caplog.text, "by label, not a list's repr"
