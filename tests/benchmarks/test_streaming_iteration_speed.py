"""Streaming iteration keeps module-graph traversal out of the per-batch path.

Flax's performance guide names the Python graph traversal inside ``nnx.jit`` as
its per-call overhead and recommends splitting the module once and driving a
plain ``jax.jit`` step. Streaming iteration follows that pattern for the stage
DAG, so a batch costs a host pull, a check of shapes and dtypes, and one compiled
dispatch. The guard compares a streaming pass with applying the same DAG through
``nnx.jit`` on every batch, the pattern the guide describes as slow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule
from datarax.pipeline.pipeline import Pipeline
from tests.benchmarks.performance_targets import measure_latency


_BATCHES = 200
_BATCH_SIZE = 64
_FEATURES = 32
# Streaming must cost at most this fraction of per-batch nnx.jit. Traversing the
# graph per batch puts the ratio near 1; hoisting it out measured about 0.1 on CPU.
_MAX_RATIO = 0.5


@dataclass(frozen=True)
class _Config(StructuralConfig):
    pass


class _Stream(DataSourceModule):
    """Forward-only source replaying one prepared batch."""

    def __init__(self, batch: dict[str, jax.Array]) -> None:
        super().__init__(_Config())
        self._batch = nnx.data(batch)
        self.cursor = nnx.Variable(0)

    def supports_indexed_access(self) -> bool:
        return False

    def element_spec(self) -> dict[str, jax.ShapeDtypeStruct]:
        return {
            "x": jax.ShapeDtypeStruct((_FEATURES,), jnp.float32),
            "y": jax.ShapeDtypeStruct((), jnp.int32),
        }

    def get_batch(self, batch_size: int) -> dict[str, Any]:
        del batch_size
        if int(self.cursor.get_value()) >= _BATCHES:
            return {}
        self.cursor.set_value(int(self.cursor.get_value()) + 1)
        return dict(self._batch)


class _Noise(nnx.Module):
    def __init__(self) -> None:
        self.rngs = nnx.Rngs(noise=0)

    def __call__(self, batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
        return {**batch, "x": batch["x"] + jax.random.normal(self.rngs.noise(), batch["x"].shape)}


def _pipeline() -> tuple[Pipeline, _Stream]:
    batch = {
        "x": jnp.ones((_BATCH_SIZE, _FEATURES), jnp.float32),
        "y": jnp.zeros((_BATCH_SIZE,), jnp.int32),
    }
    source = _Stream(batch)
    pipeline = Pipeline(source=source, stages=[_Noise()], batch_size=_BATCH_SIZE, rngs=nnx.Rngs(0))
    return pipeline, source


_apply_per_batch = nnx.jit(Pipeline.__call__)


def _stream_pass() -> None:
    pipeline, _ = _pipeline()
    jax.block_until_ready(list(pipeline))


def _nnx_jit_pass() -> None:
    pipeline, source = _pipeline()
    outputs = []
    while batch := source.get_batch(_BATCH_SIZE):
        outputs.append(_apply_per_batch(pipeline, batch))
        pipeline._position[...] = pipeline._position[...] + jnp.int32(_BATCH_SIZE)
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
