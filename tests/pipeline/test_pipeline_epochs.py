"""Epoch contracts of ``Pipeline``: a shuffled epoch is a permutation, and epochs reset.

The pipeline owns the iteration position and the per-epoch shuffle key. Every batch
of an epoch is a slice of one permutation, so an epoch visits each record exactly
once; ``reset()`` starts the next epoch at position 0 with a new permutation; the
iterator state carries the epoch so a resumed session reproduces the same slices.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax.pipeline import Pipeline
from datarax.sources.memory_source import MemorySource, MemorySourceConfig


_N = 64
_BATCH = 16


def _pipeline(*, shuffle: bool = True, seed: int = 0) -> Pipeline:
    source = MemorySource(
        MemorySourceConfig(shuffle=shuffle),
        data={"id": jnp.arange(_N, dtype=jnp.int32)},
        rngs=nnx.Rngs(seed),
    )
    return Pipeline(source=source, stages=[], batch_size=_BATCH, rngs=nnx.Rngs(seed))


def _epoch_ids(pipeline: Pipeline) -> list[int]:
    ids: list[int] = []
    for batch in pipeline:
        ids.extend(np.asarray(batch["id"]).tolist())
    return ids


def test_shuffled_epoch_visits_every_record_once() -> None:
    ids = _epoch_ids(_pipeline())

    assert len(ids) == _N
    assert sorted(ids) == list(range(_N))
    assert ids != list(range(_N))


def test_reset_starts_the_next_epoch_with_a_new_permutation() -> None:
    pipeline = _pipeline()
    first = _epoch_ids(pipeline)
    assert _epoch_ids(pipeline) == []  # exhausted until reset

    pipeline.reset()

    assert int(pipeline._position[...]) == 0
    assert int(pipeline._epoch[...]) == 1
    second = _epoch_ids(pipeline)
    assert sorted(second) == list(range(_N))
    assert second != first


def test_reset_keeps_a_sequential_source_in_order() -> None:
    pipeline = _pipeline(shuffle=False)
    first = _epoch_ids(pipeline)
    pipeline.reset()

    assert first == list(range(_N))
    assert _epoch_ids(pipeline) == first


def test_epochs_are_reproducible_from_the_seed() -> None:
    a, b = _pipeline(seed=7), _pipeline(seed=7)
    for _ in range(2):
        assert _epoch_ids(a) == _epoch_ids(b)
        a.reset()
        b.reset()


def test_iterator_state_carries_the_epoch_and_resumes_exactly() -> None:
    reference = _pipeline()
    _epoch_ids(reference)
    reference.reset()
    session = iter(reference)
    next(session)
    state = session.get_state()  # type: ignore[union-attr]
    expected = [np.asarray(next(session)["id"]) for _ in range(2)]

    assert state["epoch"] == 1
    resumed = iter(_pipeline())
    resumed.set_state(state)  # type: ignore[union-attr]
    got = [np.asarray(next(resumed)["id"]) for _ in range(2)]

    for g, e in zip(got, expected, strict=True):
        np.testing.assert_array_equal(g, e)


def test_scan_epoch_visits_every_record_once() -> None:
    pipeline = _pipeline()

    stacked = pipeline.scan(lambda batch: batch["id"], length=_N // _BATCH)

    ids = np.asarray(stacked).reshape(-1).tolist()
    assert sorted(ids) == list(range(_N))
