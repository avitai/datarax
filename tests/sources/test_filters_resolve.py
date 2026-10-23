"""``include_keys`` and ``exclude_keys`` hold field names, resolved once at construction.

A name resolves against the dataset's data and provenance names, as any field name does (N3), so
an unknown name is refused, listing the fields, rather than dropped without a word, and a nested
name such as ``meta/score`` filters that nested feature.
"""

from __future__ import annotations

from typing import Any

import datasets
import numpy as np
import pytest
import tensorflow_datasets as tfds

from datarax.core.index_words import to_words
from datarax.sources import (
    from_tfds,
    HFEagerConfig,
    HFEagerSource,
    tfds_source,
    TFDSEagerConfig,
    TFDSEagerSource,
)
from datarax.sources._tfds_identity import split_identity, SplitOrigin
from datarax.sources._tfds_prepared import decoded_names, feature_tree
from datarax.sources.hf_source import HFStreamingConfig, HFStreamingSource
from datarax.sources.source_ops import KeyFilter
from datarax.sources.tfds_streaming_source import TFDSStreamingConfig, TFDSStreamingSource
from tests.test_common.tfds_fixture import FIXTURE, TFDSFixture


# In name order: a loaded TFDS FeaturesDict has no stable order of its own.
TFDS_FIELDS = "its fields are 'image', 'label', 'meta/score', 'name'"


def test_the_refusal_lists_the_fields_in_name_order_whatever_the_dataset_s_order() -> None:
    with pytest.raises(ValueError) as caught:
        KeyFilter.resolved("Config", {"name": None, "meta": {"score": None}}, {"nope"}, None)
    assert str(caught.value) == (
        "Config names 'nope' in include_keys, which the dataset does not have; "
        "its fields are 'meta/score', 'name'"
    )


@pytest.mark.tfds
class TestTFDS:
    @staticmethod
    def _eager(fixture: TFDSFixture, **filters: Any) -> TFDSEagerSource:
        return TFDSEagerSource(
            TFDSEagerConfig(
                name=FIXTURE, split="train", data_dir=str(fixture.array_record), **filters
            )
        )

    @pytest.mark.parametrize("parameter", ["include_keys", "exclude_keys"])
    def test_an_unknown_name_is_refused_listing_the_fields(
        self, tfds_fixture: TFDSFixture, parameter: str
    ) -> None:
        with pytest.raises(ValueError) as caught:
            self._eager(tfds_fixture, **{parameter: {"absent"}})
        assert str(caught.value) == (
            f"TFDSEagerConfig names 'absent' in {parameter}, which the dataset does not have; "
            f"{TFDS_FIELDS}"
        )

    def test_a_known_name_beside_an_unknown_one_is_still_refused(
        self, tfds_fixture: TFDSFixture
    ) -> None:
        with pytest.raises(ValueError, match="names 'absent' in include_keys"):
            self._eager(tfds_fixture, include_keys={"label", "absent"})

    def test_the_stream_refuses_an_unknown_name_when_it_is_built(
        self, tfds_fixture: TFDSFixture
    ) -> None:
        with pytest.raises(ValueError) as caught:
            TFDSStreamingSource(
                TFDSStreamingConfig(
                    name=FIXTURE,
                    split="train",
                    data_dir=str(tfds_fixture.tfrecord),
                    include_keys={"absent"},
                )
            )
        assert str(caught.value) == (
            "TFDSStreamingConfig names 'absent' in include_keys, which the dataset does not have; "
            f"{TFDS_FIELDS}"
        )

    def test_the_per_batch_read_refuses_an_unknown_name_when_it_is_built(
        self, tfds_fixture: TFDSFixture
    ) -> None:
        with pytest.raises(ValueError, match=f"names 'absent' in include_keys.*{TFDS_FIELDS}"):
            from_tfds(
                FIXTURE,
                "train",
                data_dir=str(tfds_fixture.array_record),
                in_memory=False,
                include_keys={"absent"},
            )

    def test_a_nested_name_includes_only_that_nested_feature(
        self, tfds_fixture: TFDSFixture
    ) -> None:
        source = self._eager(tfds_fixture, include_keys={"meta/score"})
        assert set(source.element_spec()) == {"meta"}
        assert set(source.element_spec()["meta"]) == {"score"}

    def test_a_nested_name_excludes_that_nested_feature(self, tfds_fixture: TFDSFixture) -> None:
        source = self._eager(tfds_fixture, exclude_keys={"meta/score"})
        assert set(source.element_spec()) == {"image", "label"}

    def test_a_filter_may_name_provenance(self, tfds_fixture: TFDSFixture) -> None:
        source = self._eager(tfds_fixture, include_keys={"image", "name"})
        assert set(source.element_spec()) == {"image"}
        assert set(source.provenance(to_words(np.arange(1)))[0]) == {"name"}


def _coco_like() -> tfds.features.FeaturesDict:
    """COCO's shape: a sequence of dicts (one group per record), a sequence of arrays, a group."""
    return tfds.features.FeaturesDict(
        {
            "objects": tfds.features.Sequence(
                {
                    "bbox": tfds.features.Tensor(shape=(4,), dtype=np.float32),
                    "label": tfds.features.ClassLabel(num_classes=3),
                }
            ),
            "clip": tfds.features.Sequence(tfds.features.Tensor(shape=(2,), dtype=np.float32)),
            "meta": {"score": tfds.features.Tensor(shape=(), dtype=np.float32)},
        }
    )


class TestTFDSFeatureStructure:
    """A filter names the fields TFDS decodes; the decode itself keeps whole features."""

    def test_names_follow_the_decoded_record(self) -> None:
        """A sequence of dicts decodes to a group of arrays; a sequence of arrays to one array."""
        assert decoded_names(_coco_like()) == {
            "objects": {"bbox": None, "label": None},
            "clip": None,
            "meta": {"score": None},
        }

    def test_a_key_holding_a_slash_stays_one_field(self) -> None:
        """TFDS COCO holds a literal ``image/id`` beside ``image``; neither is a level."""
        features = tfds.features.FeaturesDict(
            {
                "image": tfds.features.Tensor(shape=(2,), dtype=np.uint8),
                "image/id": tfds.features.Tensor(shape=(), dtype=np.int64),
            }
        )
        assert decoded_names(features) == {"image": None, "image/id": None}

    def test_a_field_inside_a_sequence_of_dicts_resolves_and_an_unknown_one_lists_them(
        self,
    ) -> None:
        names = decoded_names(_coco_like())
        assert KeyFilter.resolved("Config", names, {"objects/bbox"}, None).include == {
            ("objects", "bbox")
        }
        with pytest.raises(ValueError) as caught:
            KeyFilter.resolved("Config", names, {"objects/area"}, None)
        assert str(caught.value).endswith(
            "its fields are 'clip', 'meta/score', 'objects/bbox', 'objects/label'"
        )

    def test_the_decode_keeps_a_sequence_whole_and_the_record_keeps_its_named_field(
        self,
    ) -> None:
        features = _coco_like()
        key_filter = KeyFilter.resolved(
            "Config", decoded_names(features), {"objects/bbox", "meta/score"}, None
        )
        kept = key_filter.keep(feature_tree(features))
        assert kept["objects"] is features["objects"]
        assert set(kept) == {"objects", "meta"} and set(kept["meta"]) == {"score"}
        record = {"objects": {"bbox": np.zeros((2, 4)), "label": np.zeros(2)}, "clip": np.zeros(3)}
        assert key_filter.keep(record) == {"objects": {"bbox": record["objects"]["bbox"]}}

    def test_selection_keeps_a_sequence_whole_without_reading_it(self) -> None:
        """Only a ``FeaturesDict`` opens; an unbuilt ``Video`` recurses on any attribute read."""
        video = object.__new__(tfds.features.Video)
        assert KeyFilter().keep(feature_tree({"clip": video}))["clip"] is video


class TestTFDSDecodedSize:
    """The bytes a record's kept numeric features decode to, from TFDS's own tensor infos."""

    def test_a_fixed_length_sequence_of_dicts_counts_every_element(self) -> None:
        objects = tfds.features.Sequence(
            {
                "bbox": tfds.features.Tensor(shape=(4,), dtype=np.float32),
                "name": tfds.features.Text(),
            },
            length=3,
        )
        meta = {"score": tfds.features.Tensor(shape=(2,), dtype=np.float64)}
        size = tfds_source._decoded_bytes({"objects": objects, "meta": meta})  # noqa: SLF001
        assert size == 3 * 4 * 4 + 2 * 8  # text is provenance, not counted

    def test_a_variable_length_sequence_of_dicts_has_no_size_before_reading(self) -> None:
        with pytest.raises(ValueError, match=r"feature 'objects/bbox' has shape \(None, 4\)"):
            tfds_source._decoded_bytes(feature_tree(_coco_like()))  # noqa: SLF001


class TestTFDSVideoInAGroup:
    def test_a_video_inside_a_group_has_no_decoded_copy(self) -> None:
        video = object.__new__(tfds.features.Video)
        origin = SplitOrigin(
            dataset="clips", version="1.0.0", config=None, split="train", features="0" * 64
        )
        with pytest.raises(ValueError, match="feature 'media/clip' is a Video"):
            split_identity(origin, {"label": object(), "media": {"clip": video}}, {})


def _hf_dataset() -> datasets.Dataset:
    return datasets.Dataset.from_dict(
        {
            "text": [f"text {i}" for i in range(6)],
            "label": list(range(6)),
            "feature": [np.full(3, i, np.float32) for i in range(6)],
        }
    )


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve a local dataset in place of the Hub, map-style or streaming."""
    dataset = _hf_dataset()

    def load_dataset(name: str, split: str | None = None, **kwargs: Any) -> Any:
        del name, split
        return dataset.to_iterable_dataset(num_shards=2) if kwargs.get("streaming") else dataset

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)


HF_FIELDS = "its fields are 'feature', 'label', 'text'"


@pytest.mark.usefixtures("hub")
class TestHF:
    @staticmethod
    def _eager(**filters: Any) -> HFEagerSource:
        return HFEagerSource(HFEagerConfig(name="local", split="train", **filters))

    @pytest.mark.parametrize("parameter", ["include_keys", "exclude_keys"])
    def test_an_unknown_name_is_refused_listing_the_columns(self, parameter: str) -> None:
        with pytest.raises(ValueError) as caught:
            self._eager(**{parameter: {"nope"}})
        assert str(caught.value) == (
            f"HFEagerConfig names 'nope' in {parameter}, which the dataset does not have; "
            f"{HF_FIELDS}"
        )

    def test_the_stream_refuses_an_unknown_name_when_it_is_built(self) -> None:
        with pytest.raises(ValueError) as caught:
            HFStreamingSource(HFStreamingConfig(name="local", split="train", include_keys={"nope"}))
        assert str(caught.value) == (
            "HFStreamingConfig names 'nope' in include_keys, which the dataset does not have; "
            f"{HF_FIELDS}"
        )

    def test_known_names_still_filter(self) -> None:
        source = self._eager(include_keys={"label"})
        assert set(source.element_spec()) == {"label"}
