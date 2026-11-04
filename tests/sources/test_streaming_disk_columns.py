"""``StreamingDiskSource`` over a column set: arrays stored raw in files, each at an offset.

A column is ``(file, offset, dtype, shape)``: a ``.npy`` file is the case whose offset is its
header's length, an uncompressed contiguous HDF5 dataset another. Each column is a memory map
held on the host, out of NNX state, pickling as its place on disk. A column's name is its field:
a key tuple (``("meta", "score")``) nests, and a string is one key, ``/`` and all, since a source
receives no record a spelling could resolve against (C3).
"""

from __future__ import annotations

import pickle  # nosec B403 - the pickles are this test's own
from pathlib import Path

import cloudpickle
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from datarax.core.data_source import HostRead, RecordIdentity
from datarax.core.index_words import to_words
from datarax.pipeline import Pipeline
from datarax.sources.streaming_disk_source import (
    DiskColumn,
    StreamingDiskSource,
    StreamingDiskSourceConfig,
)


_ROWS = 12


def _images() -> np.ndarray:
    return np.arange(_ROWS * 4 * 3, dtype=np.uint8).reshape(_ROWS, 4, 3)


def _labels() -> np.ndarray:
    return np.arange(_ROWS, dtype=np.int64) % 5


def _scores() -> np.ndarray:
    return np.linspace(-1.0, 1.0, _ROWS * 2, dtype=np.float32).reshape(_ROWS, 2)


def _npy(directory: Path, name: str, array: np.ndarray) -> Path:
    path = directory / f"{name}.npy"
    np.save(path, array)
    return path


@pytest.fixture
def columns(tmp_path: Path) -> tuple[DiskColumn, ...]:
    return (
        DiskColumn.from_npy("image", _npy(tmp_path, "image", _images())),
        DiskColumn.from_npy("label", _npy(tmp_path, "label", _labels())),
        DiskColumn.from_npy(("meta", "score"), _npy(tmp_path, "score", _scores())),
    )


def _source(columns: tuple[DiskColumn, ...]) -> StreamingDiskSource:
    return StreamingDiskSource(StreamingDiskSourceConfig(columns=columns))


def _words(rows: list[int]) -> np.ndarray:
    return to_words(np.asarray(rows, dtype=np.uint64))


class TestAColumnFromANpyFile:
    def test_its_offset_is_the_header_length(self, tmp_path: Path) -> None:
        path = _npy(tmp_path, "a", _scores())
        column = DiskColumn.from_npy("a", path)
        assert column.file == str(path)
        assert column.shape == (_ROWS, 2)
        assert np.dtype(column.dtype) == np.float32
        expected = path.stat().st_size - _scores().nbytes
        assert column.offset == expected > 0

    def test_a_fortran_ordered_file_is_refused(self, tmp_path: Path) -> None:
        path = _npy(tmp_path, "f", np.asfortranarray(_scores()))
        with pytest.raises(ValueError, match="Fortran order"):
            DiskColumn.from_npy("f", path)

    def test_a_non_numeric_file_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "text.npy"
        np.save(path, np.asarray(["a", "b"]))
        with pytest.raises(ValueError, match="numeric"):
            DiskColumn.from_npy("text", path)


class TestTheSourceOverColumns:
    def test_records_keep_their_nesting_and_stored_dtypes(
        self, columns: tuple[DiskColumn, ...]
    ) -> None:
        source = _source(columns)
        assert source.record_identity is RecordIdentity.INDEXED
        assert source.host_read is HostRead.GIL_FREE
        assert len(source) == _ROWS
        spec = source.element_spec()
        assert set(spec) == {"image", "label", "meta"}
        assert spec["image"].shape == (4, 3)
        assert spec["image"].dtype == np.uint8
        assert spec["meta"]["score"].shape == (2,)

    def test_the_host_read_gathers_every_column(self, columns: tuple[DiskColumn, ...]) -> None:
        rows = [7, 0, 11, 3]
        batch = _source(columns).get_batch(_words(rows))
        np.testing.assert_array_equal(batch["image"], _images()[rows])
        np.testing.assert_array_equal(batch["label"], _labels()[rows])
        np.testing.assert_array_equal(batch.data["meta"]["score"], _scores()[rows])
        assert batch["label"].dtype == np.int64

    def test_the_traced_read_gathers_every_column(self, columns: tuple[DiskColumn, ...]) -> None:
        rows = [2, 9]
        records = _source(columns).get_records(jnp.asarray(_words(rows)))
        np.testing.assert_array_equal(np.asarray(records["image"]), _images()[rows])
        np.testing.assert_array_equal(np.asarray(records["meta"]["score"]), _scores()[rows])

    def test_a_pipeline_serves_every_record_in_order(self, columns: tuple[DiskColumn, ...]) -> None:
        pipe = Pipeline(source=_source(columns), stages=[], batch_size=4, rngs=nnx.Rngs(0))
        served = list(pipe.raw_batches())
        np.testing.assert_array_equal(np.concatenate([b["image"] for b in served]), _images())
        np.testing.assert_array_equal(
            np.concatenate([b.data["meta"]["score"] for b in served]), _scores()
        )
        pipe.close()


class TestAColumnAtARawOffset:
    def test_raw_bytes_after_a_header_read_as_the_npy_column(self, tmp_path: Path) -> None:
        header = b"any header of a file format, 37 bytes"
        raw = tmp_path / "raw.bin"
        raw.write_bytes(header + _scores().tobytes())
        at_offset = DiskColumn(
            name="score", file=str(raw), offset=len(header), dtype="float32", shape=(_ROWS, 2)
        )
        npy = DiskColumn.from_npy("score", _npy(tmp_path, "score", _scores()))
        rows = [11, 4, 4, 0]
        np.testing.assert_array_equal(
            _source((at_offset,)).get_batch(_words(rows))["score"],
            _source((npy,)).get_batch(_words(rows))["score"],
        )

    def test_a_contiguous_hdf5_dataset_reads_at_its_offset(self, tmp_path: Path) -> None:
        path = tmp_path / "data.h5"
        with h5py.File(path, "w") as file:
            dataset = file.create_dataset("score", data=_scores())  # contiguous, uncompressed
            offset = dataset.id.get_offset()
        column = DiskColumn(
            name="score", file=str(path), offset=offset, dtype="float32", shape=(_ROWS, 2)
        )
        rows = [3, 10, 1]
        np.testing.assert_array_equal(
            _source((column,)).get_batch(_words(rows))["score"], _scores()[rows]
        )

    def test_a_file_shorter_than_its_column_is_refused(self, tmp_path: Path) -> None:
        raw = tmp_path / "short.bin"
        raw.write_bytes(_scores().tobytes()[:-4])
        column = DiskColumn(name="score", file=str(raw), offset=0, dtype="float32", shape=(12, 2))
        with pytest.raises(ValueError, match=r"score.*holds 92 bytes.*96"):
            _source((column,))


class TestRefusals:
    def test_columns_of_different_lengths_are_refused(self, tmp_path: Path) -> None:
        long = DiskColumn.from_npy("a", _npy(tmp_path, "a", _scores()))
        short = DiskColumn.from_npy("b", _npy(tmp_path, "b", _labels()[:5]))
        with pytest.raises(ValueError, match=r"one row per record.*'a': 12.*'b': 5"):
            StreamingDiskSourceConfig(columns=(long, short))

    def test_no_column_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one column"):
            StreamingDiskSourceConfig(columns=())

    def test_a_name_twice_is_refused(self, tmp_path: Path) -> None:
        column = DiskColumn.from_npy("a", _npy(tmp_path, "a", _scores()))
        with pytest.raises(ValueError, match="'a' names two columns"):
            StreamingDiskSourceConfig(columns=(column, column))

    @pytest.mark.parametrize("name", ["", "a//b", "/a", "a/"])
    def test_an_empty_name_or_level_is_refused(self, name: str, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="empty"):
            DiskColumn(name=name, file=str(tmp_path / "x"), offset=0, dtype="uint8", shape=(1,))

    def test_a_negative_offset_or_a_scalar_column_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="offset"):
            DiskColumn(name="a", file=str(tmp_path / "x"), offset=-1, dtype="uint8", shape=(1,))
        with pytest.raises(ValueError, match="record axis"):
            DiskColumn(name="a", file=str(tmp_path / "x"), offset=0, dtype="uint8", shape=())

    def test_a_missing_file_is_refused(self, tmp_path: Path) -> None:
        column = DiskColumn(
            name="a", file=str(tmp_path / "missing.bin"), offset=0, dtype="uint8", shape=(1,)
        )
        with pytest.raises(FileNotFoundError, match="missing.bin"):
            _source((column,))


class TestTheMapsStayOnTheHost:
    def test_no_column_is_module_state(self, columns: tuple[DiskColumn, ...]) -> None:
        source = _source(columns)
        leaves = jax.tree.leaves(nnx.state(source))
        assert all(np.shape(leaf)[:1] != (_ROWS,) for leaf in leaves)

    def test_a_tree_mode_round_trip_reads_the_same_rows(
        self, columns: tuple[DiskColumn, ...]
    ) -> None:
        graphdef, state = nnx.split(_source(columns), graph=False)
        merged = nnx.merge(graphdef, state)
        np.testing.assert_array_equal(merged.get_batch(_words([5, 2]))["image"], _images()[[5, 2]])

    def test_every_column_pickles_by_its_place_on_disk(self, tmp_path: Path) -> None:
        big = np.zeros((200_000, 8), np.float32)  # 6.1 MiB of rows
        source = _source((DiskColumn.from_npy("x", _npy(tmp_path, "big", big)),))
        source.get_batch(_words([0]))  # the map is open
        for dumps in (pickle.dumps, cloudpickle.dumps):
            payload = dumps(source)
            assert len(payload) < 64 * 1024, len(payload)
            copy = pickle.loads(payload)  # nosec B301 - this test's own pickle
            np.testing.assert_array_equal(copy.get_batch(_words([7]))["x"], big[[7]])


def _column(tmp_path: Path, name: object, file: str) -> DiskColumn:
    return DiskColumn.from_npy(name, _npy(tmp_path, file, _scores()))  # type: ignore[arg-type]


def _both_orders(first: DiskColumn, second: DiskColumn) -> list[tuple[DiskColumn, ...]]:
    return [(first, second), (second, first)]


class TestAColumnNamedByKeyTuple:
    def test_a_key_tuple_nests_and_a_string_is_one_key(self, tmp_path: Path) -> None:
        nested = _column(tmp_path, ("meta", "score"), "a")
        literal = _column(tmp_path, "a/b", "b")

        spec = _source((nested, literal)).element_spec()

        assert set(spec) == {"meta", "a/b"}
        assert spec["meta"]["score"].shape == (2,)
        np.testing.assert_array_equal(
            _source((nested, literal)).get_batch(_words([3]))["a/b"], _scores()[[3]]
        )

    def test_a_list_and_a_tuple_build_one_config_and_a_string_another(self, tmp_path: Path) -> None:
        path = _npy(tmp_path, "a", _scores())

        as_list = DiskColumn.from_npy(["meta", "score"], path)  # type: ignore[arg-type]
        as_tuple = DiskColumn.from_npy(("meta", "score"), path)
        as_string = DiskColumn.from_npy("x", path)
        as_one_key = DiskColumn.from_npy(("x",), path)

        assert as_list == as_tuple
        assert hash(as_list) == hash(as_tuple)
        assert as_list.name == ("meta", "score")
        assert as_string.name == "x"
        assert as_string != as_one_key, "a string and its one-key tuple are two configs"
        assert as_string.keys == as_one_key.keys == ("x",)

    @pytest.mark.parametrize("name", [5, [], ["a", 1]])
    def test_a_name_that_is_neither_a_string_nor_keys_is_refused(
        self, name: object, tmp_path: Path
    ) -> None:
        file = str(tmp_path / "x")
        with pytest.raises(TypeError, match="names one field"):
            DiskColumn(
                name=name,  # type: ignore[arg-type]  # the refused name
                file=file,
                offset=0,
                dtype="uint8",
                shape=(1,),
            )

    @pytest.mark.parametrize("keys", [("a/", "b"), ("a", ""), ("a", "b//c")])
    def test_an_empty_or_exotic_key_in_a_tuple_is_refused(
        self, keys: tuple[str, ...], tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="is empty or starts or ends with '/' or holds '//'"):
            DiskColumn(name=keys, file=str(tmp_path / "x"), offset=0, dtype="uint8", shape=(1,))


class TestOneDeclaration:
    """The source's columns are one declaration: checked once over their union, in any order."""

    def test_a_spelling_beside_a_field_is_refused_offering_its_key_tuple(
        self, tmp_path: Path
    ) -> None:
        for columns in _both_orders(
            _column(tmp_path, "meta", "a"), _column(tmp_path, "meta/score", "b")
        ):
            with pytest.raises(
                ValueError, match=r"'meta/score'.*'meta' is a field.*name=\('meta/score',\)"
            ):
                StreamingDiskSourceConfig(columns=columns)

    def test_the_key_tuple_of_that_literal_key_is_accepted(self, tmp_path: Path) -> None:
        for columns in _both_orders(
            _column(tmp_path, "meta", "a"), _column(tmp_path, ("meta/score",), "b")
        ):
            spec = _source(columns).element_spec()
            assert set(spec) == {"meta", "meta/score"}

    def test_a_key_tuple_through_a_column_is_refused(self, tmp_path: Path) -> None:
        for columns in _both_orders(
            _column(tmp_path, "meta", "a"), _column(tmp_path, ("meta", "score"), "b")
        ):
            with pytest.raises(ValueError, match=r"passes through 'meta'"):
                StreamingDiskSourceConfig(columns=columns)

    def test_two_columns_of_one_spelling_are_refused_naming_both(self, tmp_path: Path) -> None:
        for columns in _both_orders(
            _column(tmp_path, ("a", "b"), "a"), _column(tmp_path, ("a/b",), "b")
        ):
            both = r"'a/b' names two fields: \('a', 'b'\) and \('a/b',\)"
            with pytest.raises(ValueError, match=both):
                StreamingDiskSourceConfig(columns=columns)

    def test_a_spelling_beside_a_group_stays_a_top_level_literal_key(self, tmp_path: Path) -> None:
        """The source receives no record, so ``meta/score`` resolves against nothing (N3a)."""
        for columns in _both_orders(
            _column(tmp_path, ("meta", "x"), "a"), _column(tmp_path, "meta/score", "b")
        ):
            spec = _source(columns).element_spec()
            assert set(spec) == {"meta", "meta/score"}
            assert set(spec["meta"]) == {"x"}

    def test_a_reopened_copy_of_a_field_beside_a_literal_key_opens(self, tmp_path: Path) -> None:
        """COCO's shape, every name a key tuple as a copy's manifest holds them."""
        columns = (
            _column(tmp_path, ("image",), "a"),
            _column(tmp_path, ("image/id",), "b"),
            _column(tmp_path, ("objects", "label"), "c"),
            _column(tmp_path, ("objects/bbox",), "d"),
        )

        spec = _source(columns).element_spec()

        assert set(spec) == {"image", "image/id", "objects", "objects/bbox"}
        assert set(spec["objects"]) == {"label"}
