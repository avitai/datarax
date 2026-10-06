"""Tests for the TFDS sources.

``TFDSEagerSource`` reads a split TFDS prepared as ArrayRecord, through TFDS's random-access
reader, into host NumPy columns and per-record provenance; it never prepares a dataset and never
imports TensorFlow, and it refuses a copy that is not prepared or is prepared in another format,
naming the call that prepares it. The stream over a TFRecord copy is ``test_tfds_stream``.
The tests marked ``tfds`` read the offline fixture (``tests.test_common.tfds_fixture``), which is
prepared with TensorFlow; CI runs them in the job that installs it.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest
from flax import nnx

from datarax.core.index_words import to_words
from datarax.sources import from_tfds, TFDSEagerConfig, TFDSEagerSource
from datarax.sources.array_record_source import ArrayRecordSourceModule
from tests.test_common.tfds_fixture import (
    FIXTURE,
    IMAGE_SHAPE,
    RAGGED,
    TEST_RECORDS,
    TFDSFixture,
    TRAIN_RECORDS,
)


_PREPARATION_FIELDS = ("try_gcs", "download_and_prepare_kwargs", "beam_num_workers")


def _eager(fixture: TFDSFixture, split: str = "train", **kwargs: object) -> TFDSEagerSource:
    config = TFDSEagerConfig(
        name=FIXTURE,
        split=split,
        data_dir=str(fixture.array_record),
        **kwargs,  # type: ignore[arg-type]
    )
    return TFDSEagerSource(config)


# =============================================================================
# The eager configuration and factory
# =============================================================================


def test_eager_config_requires_a_name_and_a_split() -> None:
    with pytest.raises(ValueError, match="name is required"):
        TFDSEagerConfig(split="train")
    with pytest.raises(ValueError, match="split is required"):
        TFDSEagerConfig(name="mnist")


def test_eager_config_refuses_both_key_filters() -> None:
    with pytest.raises(ValueError, match="Cannot specify both"):
        TFDSEagerConfig(name="mnist", split="test", include_keys={"image"}, exclude_keys={"label"})


@pytest.mark.parametrize("field", [*_PREPARATION_FIELDS, "local_files_only"])
def test_eager_config_takes_no_preparation_field(field: str) -> None:
    """The eager source never prepares a dataset, so it takes nothing that configures preparing."""
    with pytest.raises(TypeError, match=field):
        TFDSEagerConfig(name="mnist", split="train", **{field: True})  # type: ignore[arg-type]


def test_from_tfds_takes_no_preparation_argument() -> None:
    """Neither TFDS source prepares a dataset."""
    parameters = inspect.signature(from_tfds).parameters
    assert [name for name in _PREPARATION_FIELDS if name in parameters] == []


# =============================================================================
# The eager source over the offline fixture
# =============================================================================


@pytest.mark.tfds
class TestTheEagerSourceReadsArrayRecord:
    def test_every_record_of_the_split_is_a_host_row(self, tfds_fixture: TFDSFixture) -> None:
        source = _eager(tfds_fixture)

        assert len(source) == TRAIN_RECORDS
        assert set(source.data) == {"image", "label", "meta"}
        image, label, score = (
            source.data["image"],
            source.data["label"],
            source.data["meta"]["score"],
        )
        for column in (image, label, score):
            assert isinstance(column, np.ndarray)
        assert (image.dtype, image.shape) == (np.uint8, (TRAIN_RECORDS, *IMAGE_SHAPE))
        # TFDS stores a class label as int64; the host column keeps the stored dtype.
        assert (label.dtype, label.shape) == (np.int64, (TRAIN_RECORDS,))
        assert (score.dtype, score.shape) == (np.float32, (TRAIN_RECORDS, 2))

    def test_each_row_is_the_record_tfds_reads_at_that_position(
        self, tfds_fixture: TFDSFixture
    ) -> None:
        import tensorflow_datasets as tfds

        source = _eager(tfds_fixture)
        records: Any = tfds.data_source(
            FIXTURE, split="train", data_dir=str(tfds_fixture.array_record), download=False
        )
        for row in (0, 7, TRAIN_RECORDS - 1):
            np.testing.assert_array_equal(source.data["image"][row], records[row]["image"])
            assert source.data["label"][row] == records[row]["label"]
            assert source._provenance.value[row]["name"] == records[row]["name"]

    def test_a_text_feature_is_provenance_aligned_with_the_rows(
        self, tfds_fixture: TFDSFixture
    ) -> None:
        source = _eager(tfds_fixture)
        names = [record["name"] for record in source._provenance.value]

        assert len(names) == TRAIN_RECORDS
        assert all(isinstance(name, bytes) and name.startswith(b"record_0_") for name in names)
        # Record i was generated with score [i, -i]; its name ends in i.
        for row, name in enumerate(names):
            assert int(name.rsplit(b"_", 1)[1]) == int(source.data["meta"]["score"][row, 0])

    def test_a_text_feature_is_never_in_a_batch_or_the_module_state(
        self, tfds_fixture: TFDSFixture
    ) -> None:
        source = _eager(tfds_fixture)
        batch = source.get_batch(to_words(np.arange(4, dtype=np.uint64)))

        assert set(batch.data) == {"image", "label", "meta"}
        for leaf in jax.tree.leaves(batch.data):
            assert np.asarray(leaf).dtype.kind in "biuf"
        _, state = nnx.split(source, graph=False)
        for leaf in jax.tree.leaves(state):
            assert isinstance(leaf, np.ndarray | jax.Array), type(leaf)

    def test_iteration_and_the_host_read_serve_the_columns(self, tfds_fixture: TFDSFixture) -> None:
        source = _eager(tfds_fixture)
        rows = np.asarray([19, 3, 3, 0, 11], dtype=np.uint64)

        first = next(iter(source))
        batch = source.get_batch(to_words(rows), epochs=2)

        np.testing.assert_array_equal(first["image"], source.data["image"][0])
        np.testing.assert_array_equal(batch["image"], source.data["image"][rows.astype(int)])
        np.testing.assert_array_equal(batch["label"], source.data["label"][rows.astype(int)])
        np.testing.assert_array_equal(np.asarray(batch.epochs), np.full(5, 2, np.int32))

    def test_a_split_slice_indexes_like_the_full_split(self, tfds_fixture: TFDSFixture) -> None:
        full = _eager(tfds_fixture)
        part = _eager(tfds_fixture, "train[5:10]")

        assert len(part) == 5
        np.testing.assert_array_equal(part.data["image"], full.data["image"][5:10])
        assert part._provenance.value == full._provenance.value[5:10]

    def test_another_split_is_read(self, tfds_fixture: TFDSFixture) -> None:
        source = _eager(tfds_fixture, "test")

        assert len(source) == TEST_RECORDS
        assert source._provenance.value[0]["name"].startswith(b"record_1_")

    def test_include_keys_keeps_only_those_features(self, tfds_fixture: TFDSFixture) -> None:
        source = _eager(tfds_fixture, include_keys={"image"})

        assert set(source.data) == {"image"}
        assert source._provenance.value == ()

    def test_exclude_keys_drops_those_features(self, tfds_fixture: TFDSFixture) -> None:
        source = _eager(tfds_fixture, exclude_keys={"label", "name"})

        assert set(source.data) == {"image", "meta"}
        assert source._provenance.value == ()

    def test_as_supervised_keeps_the_supervised_features_under_their_names(
        self, tfds_fixture: TFDSFixture
    ) -> None:
        source = _eager(tfds_fixture, as_supervised=True)

        assert source.get_dataset_info().supervised_keys == ("image", "label")
        assert set(source.data) == {"image", "label"}
        assert source._provenance.value == ()

    def test_a_feature_whose_shape_varies_is_refused_naming_padding_or_packing(
        self, tfds_fixture: TFDSFixture
    ) -> None:
        config = TFDSEagerConfig(name=RAGGED, split="train", data_dir=str(tfds_fixture.ragged))

        with pytest.raises(ValueError, match="'image'.*pad the field.*pack records"):
            TFDSEagerSource(config)

    def test_the_dataset_info_and_repr_name_the_dataset(self, tfds_fixture: TFDSFixture) -> None:
        source = _eager(tfds_fixture)

        assert source.get_dataset_info().splits["train"].num_examples == TRAIN_RECORDS
        assert FIXTURE in repr(source)
        assert "TFDSEagerSource" in repr(source)

    def test_from_tfds_builds_the_eager_source_for_a_small_split(
        self, tfds_fixture: TFDSFixture
    ) -> None:
        source = from_tfds(FIXTURE, "train", data_dir=str(tfds_fixture.array_record))

        assert isinstance(source, TFDSEagerSource)
        assert len(source) == TRAIN_RECORDS

    @pytest.mark.parametrize(
        ("split", "options"),
        [
            ("train", {}),
            ("train[3:11]", {}),
            ("train", {"as_supervised": True}),
            ("test", {"exclude_keys": {"score"}}),
        ],
    )
    def test_from_tfds_not_in_memory_reads_per_batch_what_the_eager_source_holds(
        self, tfds_fixture: TFDSFixture, split: str, options: dict[str, Any]
    ) -> None:
        data_dir = str(tfds_fixture.array_record)
        per_batch = from_tfds(FIXTURE, split, data_dir=data_dir, in_memory=False, **options)
        eager = from_tfds(FIXTURE, split, data_dir=data_dir, **options)
        assert isinstance(per_batch, ArrayRecordSourceModule)
        assert isinstance(eager, TFDSEagerSource)
        words = to_words(np.arange(len(eager), dtype=np.uint64)[::-1])

        got, want = per_batch.get_batch(words), eager.get_batch(words)

        assert len(per_batch) == len(eager)
        assert per_batch.element_spec() == eager.element_spec()
        assert jax.tree.structure(got.data) == jax.tree.structure(want.data)
        for value, expected in zip(
            jax.tree.leaves(got.data), jax.tree.leaves(want.data), strict=True
        ):
            assert value.dtype == expected.dtype
            np.testing.assert_array_equal(value, expected)
        assert per_batch.provenance(words) == eager.provenance(words)


# =============================================================================
# The refusal
# =============================================================================


def _refusal_pattern(name: str, data_dir: Path, found: str) -> str:
    """The parts a refusal names: the dataset, the directory, what was found and the fix."""
    directory = re.escape(str(data_dir))
    return (
        rf"(?s){name}.*{directory}.*{found}.*tfds\.builder\('{name}', data_dir='{directory}', "
        r"file_format='array_record'\)\.download_and_prepare\(\)"
    )


@pytest.mark.tfds
class TestTheEagerSourceRefusesAnUnreadableCopy:
    def test_a_registered_dataset_that_is_not_prepared(self, tmp_path: Path) -> None:
        config = TFDSEagerConfig(name="mnist", split="train", data_dir=str(tmp_path))

        with pytest.raises(
            FileNotFoundError, match=_refusal_pattern("mnist", tmp_path, "not prepared")
        ):
            TFDSEagerSource(config)

    def test_a_dataset_nothing_in_the_directory_holds(self, tmp_path: Path) -> None:
        config = TFDSEagerConfig(name=FIXTURE, split="train", data_dir=str(tmp_path))

        with pytest.raises(
            FileNotFoundError, match=_refusal_pattern(FIXTURE, tmp_path, "not prepared")
        ):
            TFDSEagerSource(config)

    def test_a_tfrecord_copy(self, tfds_fixture: TFDSFixture) -> None:
        directory = tfds_fixture.tfrecord
        config = TFDSEagerConfig(name=FIXTURE, split="train", data_dir=str(directory))

        with pytest.raises(
            FileNotFoundError, match=_refusal_pattern(FIXTURE, directory, "tfrecord")
        ):
            TFDSEagerSource(config)
