"""End-to-end per-record RNG determinism for stochastic pipelines.

Datarax keys each record's augmentation on its stable global index
(``fold_in(base_key, index)``), so a given record is augmented identically
regardless of how records are grouped into batches — the invariant that makes
augmentation reproducible across batch size, host/shard count, and resume point.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from datarax import Pipeline
from datarax.core.config import ElementOperatorConfig
from datarax.operators import ElementOperator
from datarax.sources.memory_source import MemorySource, MemorySourceConfig


def _augment(element, key):
    """Add per-record Gaussian noise keyed on the record's PRNG key."""
    noise = jax.random.normal(key, element.data["value"].shape) * 0.5
    return element.update_data({"value": element.data["value"] + noise})


def _build_pipeline(batch_size: int) -> Pipeline:
    data = {"value": jnp.arange(24, dtype=jnp.float32).reshape(24, 1)}
    source = MemorySource(MemorySourceConfig(shuffle=False), data)
    operator = ElementOperator(
        ElementOperatorConfig(stochastic=True, stream_name="aug"),
        fn=_augment,
        rngs=nnx.Rngs(aug=7),
    )
    return Pipeline(source=source, stages=[operator], batch_size=batch_size, rngs=nnx.Rngs(0))


def _collect(pipeline: Pipeline, n_records: int) -> jax.Array:
    """Collect the first ``n_records`` augmented values across successive steps."""
    collected: list[jax.Array] = []
    total = 0
    while total < n_records:
        part = pipeline.step()["value"]  # type: ignore[reportCallIssue]
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
