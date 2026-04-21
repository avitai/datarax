"""Grain streaming composition contracts for mixed and streaming sources."""

import grain
import numpy as np
import pytest

from datarax.sources._grain_streaming import (
    ensure_iter_dataset,
    interleave_streaming_sources,
    limit_streaming_records,
    mix_streaming_sources,
    repeat_streaming_records,
)


def test_ensure_iter_dataset_accepts_sequences_without_materializing_iterators() -> None:
    dataset = ensure_iter_dataset([{"x": 0}, {"x": 1}])

    assert isinstance(dataset, grain.IterDataset)
    assert list(dataset) == [{"x": 0}, {"x": 1}]


def test_ensure_iter_dataset_rejects_one_shot_iterators() -> None:
    with pytest.raises(TypeError, match="Grain IterDataset or Sequence"):
        ensure_iter_dataset(iter([{"x": 0}]))


def test_mix_iter_datasets_delegates_to_grain_mix() -> None:
    left = ensure_iter_dataset([{"source": "left"}, {"source": "left"}])
    right = ensure_iter_dataset([{"source": "right"}, {"source": "right"}])

    mixed = mix_streaming_sources([left, right], weights=[0.5, 0.5])

    assert isinstance(mixed, grain.IterDataset)
    assert {item["source"] for item in mixed} == {"left", "right"}


def test_interleave_iter_datasets_delegates_to_grain_interleave() -> None:
    left = ensure_iter_dataset([{"x": "a0"}, {"x": "a1"}])
    right = ensure_iter_dataset([{"x": "b0"}, {"x": "b1"}])

    interleaved = interleave_streaming_sources([left, right])

    assert [item["x"] for item in interleaved] == ["a0", "b0", "a1", "b1"]


def test_repeat_and_limit_use_grain_streaming_transforms() -> None:
    dataset = ensure_iter_dataset([{"x": np.array(1)}, {"x": np.array(2)}])

    limited = limit_streaming_records(repeat_streaming_records(dataset, num_epochs=2), count=4)

    assert [int(item["x"]) for item in limited] == [1, 2, 1, 2]
