"""The in-memory TFDS build under a caller's RAM budget: a split over it is refused before decoding.

With ``host_resources`` the eager build computes the split's decoded size from TFDS's fixed-shape
features before reading a record (each kept numeric feature's values at its stored dtype, times
the split's records) and refuses a size above ``ram_budget_bytes``, naming the size, the budget and
the per-batch read, ``in_memory=False``. Without a budget nothing is checked. A per-batch read takes
its budget from the pipeline that reads it.
"""

from __future__ import annotations

import math
import re
from typing import Any

import pytest
import tensorflow_datasets as tfds

from datarax.core.host_resources import HostResources
from datarax.sources import from_tfds, TFDSEagerConfig, TFDSEagerSource
from tests.test_common.tfds_fixture import (
    FIXTURE,
    IMAGE_SHAPE,
    RAGGED,
    TFDSFixture,
    TRAIN_RECORDS,
)


pytestmark = pytest.mark.tfds

_IMAGE = math.prod(IMAGE_SHAPE)  # uint8
_LABEL = 8  # ClassLabel, stored int64
_SCORE = 2 * 4  # meta/score, float32 (2,)


def _budget(ram_budget_bytes: int) -> HostResources:
    return HostResources(ram_budget_bytes=ram_budget_bytes, max_workers=1)


@pytest.fixture
def decodes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Counts every record TFDS decodes in this process."""
    calls: list[int] = []
    decode = tfds.features.FeaturesDict.deserialize_example_np

    def counting(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return decode(self, *args, **kwargs)

    monkeypatch.setattr(tfds.features.FeaturesDict, "deserialize_example_np", counting)
    return calls


def _eager(fixture: TFDSFixture, budget: HostResources | None, **config: Any) -> TFDSEagerSource:
    options: dict[str, Any] = {
        "name": FIXTURE,
        "split": "train",
        "data_dir": str(fixture.array_record),
        **config,
    }
    return TFDSEagerSource(TFDSEagerConfig(**options), host_resources=budget)


@pytest.mark.parametrize(
    ("config", "decoded"),
    [
        ({}, TRAIN_RECORDS * (_IMAGE + _LABEL + _SCORE)),
        ({"split": "train[3:11]"}, 8 * (_IMAGE + _LABEL + _SCORE)),
        ({"as_supervised": True}, TRAIN_RECORDS * (_IMAGE + _LABEL)),
        ({"exclude_keys": {"image"}}, TRAIN_RECORDS * (_LABEL + _SCORE)),
    ],
    ids=["whole", "slice", "supervised", "filtered"],
)
class TestASplitOverTheBudget:
    def test_is_refused_before_any_record_is_decoded(
        self,
        tfds_fixture: TFDSFixture,
        decodes: list[int],
        config: dict[str, Any],
        decoded: int,
    ) -> None:
        with pytest.raises(ValueError) as refusal:
            _eager(tfds_fixture, _budget(decoded - 1), **config)
        message = str(refusal.value)
        assert re.search(rf"decodes to {decoded} bytes", message), message
        assert f"ram_budget_bytes={decoded - 1}" in message
        assert "in_memory=False" in message
        assert decodes == []


def test_without_a_budget_nothing_is_checked(tfds_fixture: TFDSFixture) -> None:
    assert len(_eager(tfds_fixture, None)) == TRAIN_RECORDS


def test_a_feature_without_a_fixed_shape_is_refused_under_a_budget(
    tfds_fixture: TFDSFixture, decodes: list[int]
) -> None:
    config = TFDSEagerConfig(name=RAGGED, split="train", data_dir=str(tfds_fixture.ragged))
    with pytest.raises(ValueError, match=r"'image'.*\(None, None, 3\).*fixed shape"):
        TFDSEagerSource(config, host_resources=_budget(1 << 30))
    assert decodes == []


class TestFromTfds:
    def test_passes_the_budget_to_the_eager_build(
        self, tfds_fixture: TFDSFixture, decodes: list[int]
    ) -> None:
        with pytest.raises(ValueError, match="in_memory=False"):
            from_tfds(
                FIXTURE,
                "train",
                data_dir=str(tfds_fixture.array_record),
                host_resources=_budget(100),
            )
        assert decodes == []

    def test_a_per_batch_read_refuses_a_budget_naming_the_pipeline(
        self, tfds_fixture: TFDSFixture
    ) -> None:
        with pytest.raises(ValueError, match=re.escape("Pipeline(host_resources=...)")):
            from_tfds(
                FIXTURE,
                "train",
                data_dir=str(tfds_fixture.array_record),
                in_memory=False,
                host_resources=_budget(1 << 30),
            )
