"""What the HuggingFace sources pass to ``datasets.load_dataset``.

``data_dir`` is ``load_dataset``'s own: a folder inside the dataset's repository on the Hub
(``datasets/load.py``: it selects data files under that folder), not where the data is stored.
The storage location is ``cache_dir``. Both reach ``load_dataset`` unchanged, from the eager and
the streaming source.
"""

from __future__ import annotations

from typing import Any

import datasets
import numpy as np
import pytest
from flax import nnx

from datarax.sources.hf_source import (
    HFEagerConfig,
    HFEagerSource,
    HFStreamingConfig,
    HFStreamingSource,
)


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Serve a small numeric dataset from ``load_dataset``, recording every call's arguments."""
    dataset = datasets.Dataset.from_dict(
        {"label": list(range(4)), "feature": [np.zeros(3, np.float32)] * 4}
    )
    recorded: list[dict[str, Any]] = []

    def load_dataset(name: str, **kwargs: Any) -> datasets.Dataset:
        recorded.append({"name": name, **kwargs})
        return dataset

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)
    return recorded


def build(kind: str, **fields: Any) -> None:
    if kind == "eager":
        HFEagerSource(HFEagerConfig(name="org/dataset", split="train", **fields), rngs=nnx.Rngs(0))
    else:
        HFStreamingSource(
            HFStreamingConfig(name="org/dataset", split="train", **fields), rngs=nnx.Rngs(0)
        )


@pytest.mark.parametrize("kind", ["eager", "streaming"])
def test_data_dir_and_cache_dir_reach_load_dataset_unchanged(
    kind: str, calls: list[dict[str, Any]]
) -> None:
    build(kind, data_dir="data/en", cache_dir="/tmp/hf-cache")

    assert calls
    for call in calls:
        assert call["name"] == "org/dataset"
        assert call["data_dir"] == "data/en"
        assert call["cache_dir"] == "/tmp/hf-cache"


@pytest.mark.parametrize("kind", ["eager", "streaming"])
def test_neither_is_set_by_default(kind: str, calls: list[dict[str, Any]]) -> None:
    build(kind)

    assert calls
    for call in calls:
        assert call.get("data_dir") is None
        assert call.get("cache_dir") is None
