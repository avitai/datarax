"""End-to-end per-record RNG determinism for stochastic pipelines.

A stochastic operator keys each record's randomness on the record itself and on the epoch:
``fold_in(fold_in(base_key, epoch), record_id)``, where ``record_id`` is the stable index of the
record the source served. So within an epoch a record is augmented identically regardless of
batch size, shuffle order, how the records are split across workers, or where a run resumed,
and every epoch draws fresh augmentation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax import Pipeline
from datarax.core.config import ElementOperatorConfig
from datarax.operators import ElementOperator
from datarax.pipeline import PipelineIterator
from datarax.sources.memory_source import MemorySource, MemorySourceConfig


N_RECORDS = 24
BATCH_SIZE = 4


def _augment(element, key):
    """Add per-record Gaussian noise keyed on the record's PRNG key."""
    noise = jax.random.normal(key, element.data["value"].shape) * 0.5
    return element.update_data({"value": element.data["value"] + noise})


def _operator() -> ElementOperator:
    return ElementOperator(
        ElementOperatorConfig(stochastic=True, stream_name="aug"),
        fn=_augment,
        rngs=nnx.Rngs(aug=7),
    )


def _build_pipeline(batch_size: int) -> Pipeline:
    data = {"value": jnp.arange(24, dtype=jnp.float32).reshape(24, 1)}
    source = MemorySource(MemorySourceConfig(shuffle=False), data)
    return Pipeline(source=source, stages=[_operator()], batch_size=batch_size, rngs=nnx.Rngs(0))


def _source(*, shuffle: bool = False, num_workers: int = 1, shard_id: int | None = None):
    """Records whose ``value`` starts at zero, so the output is the augmentation itself."""
    data = {
        "value": jnp.zeros((N_RECORDS, 1), dtype=jnp.float32),
        "id": jnp.arange(N_RECORDS, dtype=jnp.int32),
    }
    config = MemorySourceConfig(shuffle=shuffle, num_workers=num_workers, shard_id=shard_id)
    return MemorySource(config, data, rngs=nnx.Rngs(3))


def _pipeline(source: MemorySource) -> Pipeline:
    return Pipeline(source=source, stages=[_operator()], batch_size=BATCH_SIZE, rngs=nnx.Rngs(0))


def _session(pipeline: Pipeline) -> PipelineIterator:
    session = iter(pipeline)
    assert isinstance(session, PipelineIterator)
    return session


def _epoch(pipeline: Pipeline) -> tuple[list[int], dict[int, float]]:
    """Run one epoch; return the record ids in serving order and each record's augmentation."""
    order: list[int] = []
    augmentation: dict[int, float] = {}
    for batch in _session(pipeline):
        for record_id, value in zip(np.asarray(batch["id"]), np.asarray(batch["value"])[:, 0]):
            order.append(int(record_id))
            augmentation[int(record_id)] = float(value)
    return order, augmentation


def _collect(pipeline: Pipeline, n_records: int) -> jax.Array:
    """Collect the first ``n_records`` augmented values across successive steps."""
    collected: list[jax.Array] = []
    total = 0
    while total < n_records:
        part = pipeline.step()["value"]
        collected.append(part)
        total += int(part.shape[0])
    return jnp.concatenate(collected, axis=0)[:n_records]


def test_augmentation_is_per_record_deterministic_across_batch_size():
    """The same global record is augmented identically under different batch sizes.

    batch_size=4 groups records [0-3][4-7][8-11]; batch_size=6 groups them
    [0-5][6-11]. Because randomness keys on the record's global index (not its
    slot in a batch), the first 12 records must come out identical either way.
    """
    n_records = 12
    out_bs4 = _collect(_build_pipeline(4), n_records)
    out_bs6 = _collect(_build_pipeline(6), n_records)

    # A stochastic op genuinely perturbed the data (not a no-op)...
    baseline = jnp.arange(n_records, dtype=jnp.float32).reshape(n_records, 1)
    assert not jnp.allclose(out_bs4, baseline)
    # ...yet each record's augmentation is invariant to the batching.
    assert jnp.allclose(out_bs4, out_bs6)


def test_same_pipeline_reproduces_across_runs():
    """Two identically-seeded pipelines produce identical augmented streams."""
    out_a = _collect(_build_pipeline(4), 12)
    out_b = _collect(_build_pipeline(4), 12)
    assert jnp.array_equal(out_a, out_b)


def test_a_record_keeps_its_augmentation_when_the_order_is_shuffled():
    ordered_ids, ordered = _epoch(_pipeline(_source(shuffle=False)))
    shuffled_ids, shuffled = _epoch(_pipeline(_source(shuffle=True)))

    assert ordered_ids == list(range(N_RECORDS))
    assert sorted(shuffled_ids) == ordered_ids
    assert shuffled_ids != ordered_ids
    assert shuffled == ordered


def test_every_epoch_draws_fresh_augmentation():
    pipeline = _pipeline(_source())
    _, first = _epoch(pipeline)
    pipeline.reset()
    _, second = _epoch(pipeline)

    assert first.keys() == second.keys()
    assert all(first[record_id] != second[record_id] for record_id in first)


def test_resuming_mid_second_epoch_reproduces_the_remaining_batches():
    reference = _pipeline(_source(shuffle=True))
    _epoch(reference)
    reference.reset()
    session = _session(reference)
    for _ in range(2):
        next(session)
    state = session.get_state()
    expected = [np.asarray(batch["value"]) for batch in session]

    resumed = _session(_pipeline(_source(shuffle=True)))
    resumed.set_state(state)
    got = [np.asarray(batch["value"]) for batch in resumed]

    assert len(expected) == N_RECORDS // BATCH_SIZE - 2
    assert len(got) == len(expected)
    assert all(np.array_equal(g, e) for g, e in zip(got, expected, strict=True))


def test_workers_split_the_records_and_each_record_keeps_its_augmentation():
    _, whole = _epoch(_pipeline(_source()))
    shards = [_epoch(_pipeline(_source(num_workers=2, shard_id=k)))[1] for k in (0, 1)]

    assert shards[0].keys().isdisjoint(shards[1].keys())
    assert shards[0].keys() | shards[1].keys() == whole.keys()
    assert all(shard[record_id] == whole[record_id] for shard in shards for record_id in shard)
