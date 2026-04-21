"""Grain-native streaming dataset composition helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import grain

from datarax.core.data_source import DataSourceModule
from datarax.sources._grain_bridge import DataraxMapDatasetAdapter


def ensure_iter_dataset(source: Any) -> grain.IterDataset:
    """Return a Grain IterDataset for explicit Grain or sequence inputs."""
    if isinstance(source, grain.IterDataset):
        return source
    if isinstance(source, grain.MapDataset):
        return source.to_iter_dataset()
    if isinstance(source, Sequence) and not isinstance(source, str | bytes | bytearray):
        return grain.MapDataset.source(source).to_iter_dataset()
    raise TypeError(
        "Expected a Grain IterDataset or Sequence. One-shot Python iterators are "
        "not accepted because they cannot provide Grain iterator state."
    )


def data_source_to_iter_dataset(source: DataSourceModule) -> grain.IterDataset:
    """Convert a finite random-access Datarax source to a Grain IterDataset."""
    try:
        length = len(source)
    except NotImplementedError as exc:
        raise TypeError("Only finite random-access sources can be converted to Grain") from exc
    if length < 0:
        raise ValueError("Source length must be non-negative")
    return grain.MapDataset.source(DataraxMapDatasetAdapter(source)).to_iter_dataset()


def mix_streaming_sources(
    datasets: Sequence[Any],
    *,
    weights: Sequence[float] | None = None,
) -> grain.IterDataset:
    """Mix datasets with Grain's weighted streaming mixer."""
    if not datasets:
        raise ValueError("At least one dataset is required")
    return grain.IterDataset.mix(
        [ensure_iter_dataset(dataset) for dataset in datasets],
        weights=weights,
    )


def interleave_streaming_sources(
    datasets: Sequence[Any],
    *,
    cycle_length: int | None = None,
    num_make_iter_threads: int = 1,
    make_iter_buffer_size: int = 1,
    iter_buffer_size: int = 1,
) -> grain.IterDataset:
    """Interleave datasets through Grain's streaming interleave primitive."""
    if not datasets:
        raise ValueError("At least one dataset is required")
    iter_datasets = [ensure_iter_dataset(dataset) for dataset in datasets]
    return grain.experimental.InterleaveIterDataset(
        iter_datasets,
        cycle_length=cycle_length or len(iter_datasets),
        num_make_iter_threads=num_make_iter_threads,
        make_iter_buffer_size=make_iter_buffer_size,
        iter_buffer_size=iter_buffer_size,
    )


def repeat_streaming_records(dataset: Any, *, num_epochs: int | None = None) -> grain.IterDataset:
    """Repeat a streaming dataset through Grain."""
    return grain.experimental.RepeatIterDataset(
        ensure_iter_dataset(dataset),
        num_epochs=num_epochs,
    )


def limit_streaming_records(dataset: Any, *, count: int) -> grain.IterDataset:
    """Limit a streaming dataset through Grain."""
    if count < 0:
        raise ValueError("count must be non-negative")
    return grain.experimental.LimitIterDataset(ensure_iter_dataset(dataset), count=count)
