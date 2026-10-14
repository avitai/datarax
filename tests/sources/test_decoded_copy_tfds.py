"""A TFDS split's decoded copy: what identifies it, and ``from_tfds(..., decoded_dir=...)``.

A copy is checked against what its source declares (``source_identity()``), never where the
source's files are: an ArrayRecord source declares its files' base names and sizes, its record
count and its decode's identity; TFDS's decode declares the dataset, version, config, split, kept
features, ``features.json``'s digest and the versions of the libraries that decode it, so a
decoder upgrade invalidates a copy. A copy carried to another directory or host still matches; an
edited source or a different decoder does not.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest
import tensorflow_datasets as tfds

from datarax.core.data_source import SourceIdentified
from datarax.core.index_words import to_words
from datarax.sources import (
    _tfds_identity,
    from_tfds,
    open_decoded_copy,
    StreamingDiskSource,
    tfds_source,
)
from datarax.sources._tfds_identity import SplitOrigin
from datarax.sources.decoded_copy._manifest import identity_differences
from datarax.sources.tfds_source import _per_batch_split, TFDSEagerConfig, TFDSEagerSource
from tests.test_common.decoded_copies import every_record, write_copy
from tests.test_common.tfds_fixture import FIXTURE, RAGGED, TFDSFixture
from tests.test_common.worker_reads import png_source, PngDecoder, write_png_records


def _same(left: Any, right: Any) -> None:
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        assert np.asarray(a).dtype == np.asarray(b).dtype
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


class TestAnArrayRecordSourcesIdentity:
    def _files(self, tmp_path: Path) -> list[str]:
        return write_png_records(tmp_path / "png", 12, shards=2)

    def test_is_its_files_names_and_sizes_records_and_decode(self, tmp_path: Path) -> None:
        source = png_source(self._files(tmp_path))

        identity = source.source_identity()

        assert isinstance(source, SourceIdentified)
        assert identity["source"]["records"] == 12
        assert [f["name"] for f in identity["source"]["files"]] == [
            "png-00000.array_record",
            "png-00001.array_record",
        ]
        assert all(f["bytes"] > 0 for f in identity["source"]["files"])
        assert identity["source"]["decode"] == {
            "callable": f"{PngDecoder.__module__}.{PngDecoder.__qualname__}"
        }
        assert str(tmp_path) not in repr(identity), "no path is part of an identity"

    def test_files_moved_elsewhere_or_touched_still_match(self, tmp_path: Path) -> None:
        files = self._files(tmp_path)
        before = png_source(files).source_identity()
        moved = tmp_path / "moved"
        shutil.copytree(tmp_path / "png", moved)
        os.utime(moved / "png-00000.array_record", (1, 1))

        after = png_source(sorted(str(p) for p in moved.iterdir())).source_identity()

        assert identity_differences(after, before) == []

    @pytest.mark.parametrize("change", ["renamed", "resized"])
    def test_a_renamed_or_resized_file_differs_naming_the_field(
        self, tmp_path: Path, change: str
    ) -> None:
        files = self._files(tmp_path)
        before = png_source(files).source_identity()
        if change == "renamed":
            renamed = str(Path(files[1]).with_name("other.array_record"))
            os.rename(files[1], renamed)
            files = [files[0], renamed]
        else:
            files = write_png_records(tmp_path / "png", 2000, shards=2)

        assert "source/files" in identity_differences(png_source(files).source_identity(), before)


@pytest.mark.tfds
class TestATFDSSplitsIdentity:
    def _split(self, data_dir: Path, **options: Any) -> Any:
        return _per_batch_split(
            FIXTURE, options.pop("split", "train"), data_dir=str(data_dir), **options
        )

    def test_names_the_dataset_split_features_and_decoder_versions(
        self, tfds_fixture: TFDSFixture
    ) -> None:
        identity = self._split(tfds_fixture.array_record).source_identity()

        decode = identity["source"]["decode"]
        assert decode["dataset"] == FIXTURE
        assert decode["version"] == "1.0.0"
        assert decode["split"] == "train"
        assert len(decode["features"]) == 64, "features.json's sha256"
        assert set(identity["decoder"]) >= {"tensorflow-datasets", "protobuf", "numpy", "pillow"}
        assert "opencv" in identity["decoder"]

    def test_the_data_directory_moved_still_matches(
        self, tfds_fixture: TFDSFixture, tmp_path: Path
    ) -> None:
        moved = tmp_path / "data"
        shutil.copytree(tfds_fixture.array_record, moved)

        assert (
            identity_differences(
                self._split(moved).source_identity(),
                self._split(tfds_fixture.array_record).source_identity(),
            )
            == []
        )

    def test_another_split_or_filter_differs(self, tfds_fixture: TFDSFixture) -> None:
        train = self._split(tfds_fixture.array_record).source_identity()
        sliced = self._split(tfds_fixture.array_record, split="train[2:9]").source_identity()
        kept = self._split(tfds_fixture.array_record, exclude_keys={"name"}).source_identity()

        assert "source/decode/split" in identity_differences(sliced, train)
        assert "source/decode/exclude_keys" in identity_differences(kept, train)

    def test_an_edited_features_json_differs(
        self, tfds_fixture: TFDSFixture, tmp_path: Path
    ) -> None:
        moved = tmp_path / "data"
        shutil.copytree(tfds_fixture.array_record, moved)
        features = next(moved.glob(f"{FIXTURE}/*/features.json"))
        features.write_text(features.read_text().replace("}", " }", 1))

        assert "source/decode/features" in identity_differences(
            self._split(moved).source_identity(),
            self._split(tfds_fixture.array_record).source_identity(),
        )

    def test_a_decoder_library_upgrade_differs_naming_it(
        self, tfds_fixture: TFDSFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        before = self._split(tfds_fixture.array_record).source_identity()
        versions = _tfds_identity.library_versions

        def upgraded(names: Any) -> dict[str, str]:
            return {**versions(names), "pillow": "99.0"}

        monkeypatch.setattr(_tfds_identity, "library_versions", upgraded)

        after = self._split(tfds_fixture.array_record).source_identity()
        assert identity_differences(after, before) == ["decoder/pillow"]

    def test_a_video_feature_has_no_copy(self) -> None:
        video = object.__new__(tfds.features.Video)  # constructing one would import TensorFlow
        decoder = tfds_source._ExampleDecoder(  # noqa: SLF001 - the decode a split declares
            {"clip": video, "label": object()},
            None,
            None,
            None,
            origin=SplitOrigin(
                dataset="clips", version="1.0.0", config=None, split="train", features="0" * 64
            ),
        )

        with pytest.raises(ValueError, match="clip.*Video"):
            decoder.source_identity()


@pytest.mark.tfds
class TestFromTfdsDecodedDir:
    def _copy(self, fixture: TFDSFixture, directory: Path, split: str = "train") -> None:
        write_copy(_per_batch_split(FIXTURE, split, data_dir=str(fixture.array_record)), directory)

    def test_opens_the_copy_of_the_split_and_serves_what_the_split_serves(
        self, tfds_fixture: TFDSFixture, tmp_path: Path
    ) -> None:
        directory = tmp_path / "copy"
        self._copy(tfds_fixture, directory)

        copy = from_tfds(
            FIXTURE,
            "train",
            data_dir=str(tfds_fixture.array_record),
            in_memory=False,
            decoded_dir=str(directory),
        )

        assert isinstance(copy, StreamingDiskSource)
        eager = TFDSEagerSource(
            TFDSEagerConfig(name=FIXTURE, split="train", data_dir=str(tfds_fixture.array_record))
        )
        data, provenance = every_record(eager)
        copied, copied_provenance = every_record(copy)
        _same(copied, data)
        assert copied_provenance == provenance

    def test_a_copy_of_another_split_is_refused_naming_the_field(
        self, tfds_fixture: TFDSFixture, tmp_path: Path
    ) -> None:
        directory = tmp_path / "copy"
        self._copy(tfds_fixture, directory, split="test")

        with pytest.raises(ValueError, match="source/decode/split"):
            from_tfds(
                FIXTURE,
                "train",
                data_dir=str(tfds_fixture.array_record),
                in_memory=False,
                decoded_dir=str(directory),
            )

    def test_with_in_memory_it_is_refused_naming_both(
        self, tfds_fixture: TFDSFixture, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match=r"decoded_dir.*in_memory=False"):
            from_tfds(
                FIXTURE,
                "train",
                data_dir=str(tfds_fixture.array_record),
                decoded_dir=str(tmp_path / "copy"),
            )

    def test_a_ragged_split_has_no_copy(self, tfds_fixture: TFDSFixture, tmp_path: Path) -> None:
        ragged = _per_batch_split(RAGGED, "train", data_dir=str(tfds_fixture.ragged))

        with pytest.raises(ValueError, match="image"):
            write_copy(ragged, tmp_path / "copy", unit_records=2)
        assert not list((tmp_path / "copy").glob("*.partial"))

    def test_without_its_source_the_copy_opens_on_its_manifest(
        self, tfds_fixture: TFDSFixture, tmp_path: Path
    ) -> None:
        data = tmp_path / "data"
        shutil.copytree(tfds_fixture.array_record, data)
        directory = tmp_path / "copy"
        write_copy(_per_batch_split(FIXTURE, "train", data_dir=str(data)), directory)
        expected = every_record(open_decoded_copy(directory))[0]
        shutil.rmtree(data)

        _same(every_record(open_decoded_copy(directory))[0], expected)
        words = to_words(np.arange(3, dtype=np.uint64))
        assert len(open_decoded_copy(directory).provenance(words)) == 3
