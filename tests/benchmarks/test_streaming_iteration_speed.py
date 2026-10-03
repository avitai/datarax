"""Streaming iteration keeps module-graph traversal out of the per-batch path.

Flax's performance guide names the Python graph traversal inside ``nnx.jit`` as
its per-call overhead and recommends splitting the module once and driving a
plain ``jax.jit`` step. Streaming iteration follows that pattern for the stage
DAG: the host stage reads each batch on its producer thread (a pull and a check of
shapes and dtypes), places it as it is taken, and the DAG runs in one compiled
dispatch. The guard compares a streaming pass with the same host-stage batches
(``raw_batches()``) put through ``nnx.jit`` on every batch, the pattern the guide
describes as slow, so both sides pay the host stage and differ by the traversal.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import numpy as np
import pytest
from flax import nnx
from flax.nnx import graphlib

from datarax.core.data_source import RecordIdentity
from datarax.core.element_batch import Batch
from datarax.pipeline.pipeline import Pipeline
from tests.benchmarks.performance_targets import measure_latency
from tests.test_common.streams import RecordStream


_BATCHES = 200
_BATCH_SIZE = 64
_FEATURES = 32
# Streaming must cost at most this fraction of per-batch nnx.jit over the same host-stage
# batches. Both sides include the host stage's per-batch thread hand-off and placement, so the
# statistic is higher than when the guard compared against pulls made in the consumer thread
# (0.29 there, 2026-10-02). Measured 2026-10-03 on an Intel i7-13700 (CPU backend), median of 10
# runs of this test's statistic: 0.57 (0.50 to 0.68) on 24 threads and 0.53 (0.51 to 0.56)
# pinned to 4; with the graph traversed per batch the statistic is 1 by construction.
_MAX_RATIO = 0.8


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
    pipeline = _pipeline()
    outputs = [_apply_per_batch(pipeline.dag, batch) for batch in pipeline.raw_batches()]
    jax.block_until_ready(outputs)


@pytest.mark.benchmark
def test_streaming_costs_less_than_applying_the_dag_through_nnx_jit_per_batch() -> None:
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


def _graph_flattens(monkeypatch: pytest.MonkeyPatch, run: Callable[[], object]) -> int:
    """The module-graph flattens ``run`` makes: NNX's per-call traversal goes through one each."""
    calls = 0
    flatten = graphlib.flatten

    def counted(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return flatten(*args, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(graphlib, "flatten", counted)
        jax.block_until_ready(run())
    return calls


def test_a_streaming_pass_traverses_the_module_graph_once_not_per_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard's invariant counted rather than timed, so no machine moves it."""
    _stream_pass()

    streamed = _graph_flattens(monkeypatch, lambda: list(_pipeline()))
    per_batch = _graph_flattens(monkeypatch, _nnx_jit_pass)

    assert streamed == 1
    assert per_batch >= _BATCHES  # the control: nnx.jit traverses on every call
