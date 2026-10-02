"""Tests for the TFDS sources.

``TFDSEagerSource`` reads a split TFDS prepared as ArrayRecord, through TFDS's random-access
reader, into host NumPy columns and per-record provenance; it never prepares a dataset and never
imports TensorFlow, and it refuses a copy that is not prepared or is prepared in another format,
naming the call that prepares it. ``TFDSStreamingSource`` reads a TFRecord copy through tf.data.
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
from datarax.sources import (
    from_tfds,
    TFDSEagerConfig,
    TFDSEagerSource,
    TFDSStreamingConfig,
    TFDSStreamingSource,
)
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
    """A stream that needs one is built with ``TFDSStreamingConfig``."""
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


# =============================================================================
# The streaming source over the TFRecord copy (tf.data)
# =============================================================================


def _streaming(fixture: TFDSFixture, **kwargs: object) -> TFDSStreamingSource:
    """The stream over the fixture's TFRecord copy, without the text and nested features.

    The stream converts each feature with ``tf_to_jax``, which takes neither text nor a nested
    feature; that conversion is the stream's own (C5b).
    """
    config = TFDSStreamingConfig(
        name=FIXTURE,
        split="train",
        data_dir=str(fixture.tfrecord),
        local_files_only=True,
        exclude_keys={"name", "meta"},
        **kwargs,  # type: ignore[arg-type]
    )
    return TFDSStreamingSource(config, rngs=nnx.Rngs(0))


@pytest.mark.tfds
class TestTheStreamingSource:
    def test_it_yields_jax_arrays(self, tfds_fixture: TFDSFixture) -> None:
        record = next(iter(_streaming(tfds_fixture)))

        assert isinstance(record["image"], jax.Array)
        assert record["image"].shape == IMAGE_SHAPE
        assert isinstance(record["label"], jax.Array)

    def test_it_streams_every_record_with_a_shuffle_buffer(self, tfds_fixture: TFDSFixture) -> None:
        source = _streaming(tfds_fixture, shuffle=True, shuffle_buffer_size=TRAIN_RECORDS)

        assert len(list(source)) == TRAIN_RECORDS
        assert len(source) == TRAIN_RECORDS

    def test_from_tfds_builds_the_stream_when_asked(self, tfds_fixture: TFDSFixture) -> None:
        source = from_tfds(FIXTURE, "train", eager=False, data_dir=str(tfds_fixture.tfrecord))

        assert isinstance(source, TFDSStreamingSource)


def test_streaming_config_requires_a_name_and_a_split_and_one_key_filter() -> None:
    with pytest.raises(ValueError, match="name is required"):
        TFDSStreamingConfig(split="train")
    with pytest.raises(ValueError, match="split is required"):
        TFDSStreamingConfig(name="mnist")
    with pytest.raises(ValueError, match="Cannot specify both"):
        TFDSStreamingConfig(
            name="mnist", split="test", include_keys={"image"}, exclude_keys={"label"}
        )


class TestTFDSStreamingConfigTryGcs:
    """Tests for try_gcs field on TFDSStreamingConfig."""

    def test_try_gcs_defaults_to_false(self) -> None:
        config = TFDSStreamingConfig(name="mnist", split="train")
        assert config.try_gcs is False

    def test_try_gcs_true_accepted(self) -> None:
        config = TFDSStreamingConfig(name="mnist", split="train", try_gcs=True)
        assert config.try_gcs is True

    def test_try_gcs_true_with_data_dir_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot specify both try_gcs=True and data_dir"):
            TFDSStreamingConfig(name="mnist", split="train", try_gcs=True, data_dir="/tmp/data")


class TestTFDSStreamingConfigBeamWorkers:
    """Tests for beam_num_workers field on TFDSStreamingConfig."""

    def test_beam_num_workers_defaults_to_none(self) -> None:
        config = TFDSStreamingConfig(name="mnist", split="train")
        assert config.beam_num_workers is None

    def test_beam_num_workers_positive_accepted(self) -> None:
        config = TFDSStreamingConfig(name="mnist", split="train", beam_num_workers=8)
        assert config.beam_num_workers == 8

    def test_beam_num_workers_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="beam_num_workers must be a positive integer"):
            TFDSStreamingConfig(name="mnist", split="train", beam_num_workers=0)


# =============================================================================
# _prepare_tfds_builder (the streaming source's), mocked
# =============================================================================


class TestPrepareTfdsBuilderBeamWorkers:
    """Tests that _prepare_tfds_builder constructs beam options from beam_num_workers."""

    def test_beam_options_constructed_when_workers_set(self) -> None:
        """beam_num_workers should produce PipelineOptions in download_config."""
        pytest.importorskip("apache_beam")
        from unittest.mock import MagicMock, patch

        from datarax.sources.tfds_source import _prepare_tfds_builder

        mock_builder = MagicMock()

        with (
            patch("tensorflow_datasets.builder", return_value=mock_builder),
            patch("datarax.sources.tfds_source._is_read_only_tfds_source", return_value=False),
            patch("tensorflow_datasets.download.DownloadConfig") as mock_dl_config,
        ):
            _prepare_tfds_builder("nsynth", None, False, None, beam_num_workers=4)

        assert mock_dl_config.called
        beam_opts = mock_dl_config.call_args.kwargs.get("beam_options")
        assert beam_opts is not None
        dl_call_kwargs = mock_builder.download_and_prepare.call_args.kwargs
        assert "download_config" in dl_call_kwargs

    def test_no_beam_options_when_workers_none(self) -> None:
        """No beam_options should be constructed when beam_num_workers is None."""
        from unittest.mock import MagicMock, patch

        from datarax.sources.tfds_source import _prepare_tfds_builder

        mock_builder = MagicMock()

        with (
            patch("tensorflow_datasets.builder", return_value=mock_builder),
            patch("datarax.sources.tfds_source._is_read_only_tfds_source", return_value=False),
        ):
            _prepare_tfds_builder("mnist", None, False, None, beam_num_workers=None)

        mock_builder.download_and_prepare.assert_called_once_with()

    def test_beam_options_not_constructed_for_read_only_builder(self) -> None:
        """ReadOnlyBuilder should skip download entirely, no beam options."""
        from unittest.mock import MagicMock, patch

        from datarax.sources.tfds_source import _prepare_tfds_builder

        mock_builder = MagicMock()

        with (
            patch("tensorflow_datasets.builder", return_value=mock_builder),
            patch("datarax.sources.tfds_source._is_read_only_tfds_source", return_value=True),
        ):
            _prepare_tfds_builder("nsynth", None, True, None, beam_num_workers=8)

        mock_builder.download_and_prepare.assert_not_called()


class TestPrepareTfdsBuilder:
    """Tests for the _prepare_tfds_builder helper function."""

    def test_regular_builder_calls_download_and_prepare(self) -> None:
        """Non-ReadOnlyBuilder should have download_and_prepare called."""
        from unittest.mock import MagicMock, patch

        from datarax.sources.tfds_source import _prepare_tfds_builder

        mock_builder = MagicMock()
        mock_builder.__class__ = type("RegularBuilder", (), {})  # type: ignore[reportAttributeAccessIssue]

        with patch("tensorflow_datasets.builder", return_value=mock_builder) as mock_tfds_builder:
            with patch("datarax.sources.tfds_source._is_read_only_tfds_source", return_value=False):
                result = _prepare_tfds_builder("mnist", None, False, None)

        mock_tfds_builder.assert_called_once_with("mnist", data_dir=None, try_gcs=False)
        mock_builder.download_and_prepare.assert_called_once_with()
        assert result is mock_builder

    def test_read_only_builder_skips_download_and_prepare(self) -> None:
        """ReadOnlyBuilder (from try_gcs) should NOT call download_and_prepare."""
        from unittest.mock import MagicMock, patch

        from datarax.sources.tfds_source import _prepare_tfds_builder

        mock_builder = MagicMock()

        with patch("tensorflow_datasets.builder", return_value=mock_builder) as mock_tfds_builder:
            with patch("datarax.sources.tfds_source._is_read_only_tfds_source", return_value=True):
                result = _prepare_tfds_builder("nsynth", None, True, None)

        mock_tfds_builder.assert_called_once_with("nsynth", data_dir=None, try_gcs=True)
        mock_builder.download_and_prepare.assert_not_called()
        assert result is mock_builder

    def test_download_kwargs_passed_through(self) -> None:
        """download_and_prepare_kwargs should be unpacked to download_and_prepare."""
        from unittest.mock import MagicMock, patch

        from datarax.sources.tfds_source import _prepare_tfds_builder

        mock_builder = MagicMock()
        kwargs = {"download_dir": "/tmp", "max_examples_per_split": 100}

        with patch("tensorflow_datasets.builder", return_value=mock_builder):
            with patch("datarax.sources.tfds_source._is_read_only_tfds_source", return_value=False):
                _prepare_tfds_builder("mnist", None, False, kwargs)

        mock_builder.download_and_prepare.assert_called_once_with(
            download_dir="/tmp", max_examples_per_split=100
        )


class TestStreamingSourceTryGcsPassthrough:
    """Tests that TFDSStreamingSource passes try_gcs through to _prepare_tfds_builder."""

    def test_try_gcs_passed_to_prepare_builder(self) -> None:
        """try_gcs should be forwarded to _prepare_tfds_builder."""
        from unittest.mock import MagicMock, patch

        mock_builder = MagicMock()
        mock_builder.info.splits = {"train": MagicMock(num_examples=100)}

        mock_tf_dataset = MagicMock()
        mock_tf_dataset.prefetch.return_value = mock_tf_dataset
        mock_builder.as_dataset.return_value = mock_tf_dataset

        with patch(
            "datarax.sources.tfds_source._prepare_tfds_builder", return_value=mock_builder
        ) as mock_prepare:
            config = TFDSStreamingConfig(name="mnist", split="train", try_gcs=True)
            TFDSStreamingSource(config)

            mock_prepare.assert_called_once_with(
                "mnist", None, True, None, beam_num_workers=None, local_files_only=False
            )
