"""Streaming iteration keeps module-graph traversal out of the per-batch path.

Flax's performance guide names the Python graph traversal inside ``nnx.jit`` as
its per-call overhead and recommends splitting the module once and driving a
plain ``jax.jit`` step. Streaming iteration follows that pattern for the stage
DAG, so a batch costs a host pull, a check of shapes and dtypes, and one compiled
dispatch. The guard compares a streaming pass with applying the same DAG through
``nnx.jit`` on every batch the stream serves, the pattern the guide describes as slow.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from flax import nnx

from datarax.core.data_source import RecordIdentity
from datarax.core.element_batch import Batch
from datarax.pipeline.pipeline import Pipeline
from tests.benchmarks.performance_targets import measure_latency
from tests.test_common.streams import RecordStream


_BATCHES = 200
_BATCH_SIZE = 64
_FEATURES = 32
# Streaming must cost at most this fraction of per-batch nnx.jit. Traversing the
# graph per batch puts the ratio near 1; hoisting it out measured about 0.1 on CPU.
_MAX_RATIO = 0.5


class _Noise(nnx.Module):
    def __init__(self) -> None:
        self.rngs = nnx.Rngs(noise=0)

    def __call__(self, batch: Batch) -> Batch:
        return batch.replace(
            data={
                **batch.data,
                "x": batch["x"] + jax.random.normal(self.rngs.noise(), batch["x"].shape),
            }
        )


def _stream() -> RecordStream:
    records = _BATCHES * _BATCH_SIZE
    return RecordStream(
        {
            "x": np.ones((records, _FEATURES), np.float32),
            "y": np.zeros((records,), np.int32),
        },
        kind=RecordIdentity.ARRIVAL,
        chunk=_BATCH_SIZE,
    )


def _pipeline() -> Pipeline:
    return Pipeline(source=_stream(), stages=[_Noise()], batch_size=_BATCH_SIZE, rngs=nnx.Rngs(0))


_apply_per_batch = nnx.jit(lambda dag, batch: dag(batch))


def _stream_pass() -> None:
    jax.block_until_ready(list(_pipeline()))


def _nnx_jit_pass() -> None:
    source = _stream()
    pipeline = Pipeline(source=source, stages=[_Noise()], batch_size=_BATCH_SIZE, rngs=nnx.Rngs(0))
    outputs = []
    while (batch := source.get_batch(_BATCH_SIZE)).batch_size:
        outputs.append(_apply_per_batch(pipeline.dag, batch))
    jax.block_until_ready(outputs)


@pytest.mark.benchmark
def test_streaming_costs_at_most_half_of_applying_the_dag_through_nnx_jit_per_batch() -> None:
    """Paired interleaved rounds; the median ratio is the guard."""
    _stream_pass()
    _nnx_jit_pass()
    ratios = []
    for _ in range(7):
        streaming = measure_latency(_stream_pass, repetitions=1)
        per_batch = measure_latency(_nnx_jit_pass, repetitions=1)
        ratios.append(streaming / per_batch)
    ratio = float(np.median(ratios))
    assert ratio <= _MAX_RATIO, (
        f"A streaming pass takes {ratio:.2f}x the per-batch nnx.jit pass over "
        f"{len(ratios)} paired rounds (limit {_MAX_RATIO}x): module-graph traversal "
        "is back in the per-batch path."
    )
