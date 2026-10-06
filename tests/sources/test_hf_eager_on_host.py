"""T7: ``HFEagerSource`` keeps its records on the host: NumPy columns and provenance.

The loader turns each array column into one host NumPy column and hands every non-array column
(text, objects) to the source's provenance, so building the source places nothing on a device and
refuses no column for being a string. The dataset is a small in-memory ``datasets.Dataset``
substituted for ``load_dataset``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from datarax.core.index_words import to_words


datasets = pytest.importorskip("datasets")
PIL = pytest.importorskip("PIL.Image")

from datarax.sources.hf_source import HFEagerConfig, HFEagerSource  # noqa: E402
from tests.test_common.identity import check_identity_reaches_the_stages  # noqa: E402


_N = 6


def _dataset() -> object:
    images = [PIL.fromarray(np.full((4, 4, 3), i, np.uint8)) for i in range(_N)]
    return datasets.Dataset.from_dict(
        {
            "image": images,
            "label": list(range(_N)),
            "feature": [np.full(3, i, np.float32) for i in range(_N)],
            "text": [f"review {i}" for i in range(_N)],
        }
    )


@pytest.fixture
def source(monkeypatch: pytest.MonkeyPatch) -> HFEagerSource:
    dataset = _dataset()
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: dataset)
    return HFEagerSource(HFEagerConfig(name="fake", split="train"))


def _dataset_sized_device_arrays() -> set[int]:
    return {id(array) for array in jax.live_arrays() if array.ndim and array.shape[0] == _N}


def test_building_the_source_leaves_no_dataset_sized_device_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset()
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: dataset)
    before = _dataset_sized_device_arrays()
    control = jnp.zeros((_N, 2))  # the instrument finds a dataset-sized device array
    assert id(control) in _dataset_sized_device_arrays() - before
    before = _dataset_sized_device_arrays()
    source = HFEagerSource(HFEagerConfig(name="fake", split="train"))
    assert _dataset_sized_device_arrays() - before == set()
    assert len(source) == _N


def test_the_array_columns_are_numpy(source: HFEagerSource) -> None:
    assert set(source.data) == {"image", "label", "feature"}
    for column in source.data.values():
        assert isinstance(column, np.ndarray)
        assert column.shape[0] == _N
    np.testing.assert_array_equal(source.data["image"][3], np.full((4, 4, 3), 3, np.uint8))
    np.testing.assert_array_equal(source.data["label"], np.arange(_N))


def test_a_text_column_is_provenance_not_refused_and_not_served(source: HFEagerSource) -> None:
    assert [record["text"] for record in source._provenance.value] == [
        f"review {i}" for i in range(_N)
    ]
    batch = source.get_batch(to_words(np.asarray([4, 1], np.uint64)))
    assert set(batch.data) == {"image", "label", "feature"}
    assert "text" not in source[2]


def test_the_identity_of_each_record_reaches_the_stages_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset()
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: dataset)
    assert check_identity_reaches_the_stages(
        lambda: HFEagerSource(HFEagerConfig(name="fake", split="train")), batch_size=2
    )
