"""Tests for Grain-aligned shuffle primitives."""

from __future__ import annotations

import inspect
from typing import Any, cast

import flax.nnx as nnx
import grain
import jax.numpy as jnp
import pytest

from datarax.core.element_batch import Batch
from datarax.dag.nodes.data_source import ShuffleNode
from datarax.samplers.index_shuffle import index_shuffle
from datarax.samplers.shuffle_sampler import ShuffleSampler, ShuffleSamplerConfig


def _grain_index_shuffle_order(size: int, seed: int) -> list[int]:
    return [
        grain.experimental.index_shuffle(index=index, max_index=size - 1, seed=seed, rounds=4)
        for index in range(size)
    ]


def _grain_sampler_order(size: int, seed: int) -> list[int]:
    sampler = grain.samplers.IndexSampler(
        num_records=size,
        shard_options=grain.sharding.NoSharding(),
        shuffle=True,
        seed=seed,
        num_epochs=1,
    )
    order = []
    for metadata in sampler:
        if metadata.record_key is None:
            raise ValueError("Grain IndexSampler emitted metadata without a record key")
        order.append(int(metadata.record_key))
    return order


def test_index_shuffle_matches_grain_experimental() -> None:
    """Datarax index_shuffle should be a thin wrapper around Grain."""
    for size in (1, 2, 5, 10, 97):
        for seed in (0, 42, 2**32 - 1):
            actual = [index_shuffle(index, seed, size) for index in range(size)]
            assert actual == _grain_index_shuffle_order(size, seed)


def test_shuffle_sampler_uses_grain_index_sampler_order() -> None:
    """Known-size shuffle sampling should align with Grain IndexSampler."""
    config = ShuffleSamplerConfig(dataset_size=10, seed=123)
    sampler = ShuffleSampler(config, rngs=nnx.Rngs(0))

    assert list(sampler) == _grain_sampler_order(10, 123)


def test_shuffle_sampler_replays_remaining_indices_after_checkpoint() -> None:
    """Restoring a partial sampler state should replay the remaining Grain order."""
    config = ShuffleSamplerConfig(dataset_size=13, seed=999)
    sampler = ShuffleSampler(config, rngs=nnx.Rngs(0))
    iterator = iter(sampler)
    prefix = [next(iterator) for _ in range(5)]
    state = sampler.get_state()
    suffix = list(iterator)

    restored = ShuffleSampler(
        ShuffleSamplerConfig(dataset_size=1, seed=1),
        rngs=nnx.Rngs(1),
    )
    restored.set_state(state)

    assert list(restored) == suffix
    assert prefix + suffix == _grain_sampler_order(13, 999)


def test_shuffle_sampler_static_iterator_avoids_front_pop_and_matches_grain() -> None:
    """The static convenience path should use Grain order without front-pop buffers."""
    order = list(ShuffleSampler.create_static_iterator(9, seed=321))

    assert ".pop(0)" not in inspect.getsource(ShuffleSampler.create_static_iterator)
    assert order == _grain_sampler_order(9, 321)


def test_shuffle_node_accepts_batches_only() -> None:
    """DAG ShuffleNode is batch-level; source-element shuffling belongs to Grain."""
    node = ShuffleNode(buffer_size=1, seed=0)

    with pytest.raises(TypeError, match="Batch"):
        node(cast(Any, {"x": jnp.array(1)}))

    batch = Batch.from_parts(data={"x": jnp.arange(2)}, states={}, validate=False)

    assert node(batch) is None
    assert node.flush() is batch
