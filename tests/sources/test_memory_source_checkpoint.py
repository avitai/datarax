"""An in-memory source holds no checkpoint state, and a pipeline's checkpoint holds no records.

The records a ``MemorySource`` was built with belong to its construction, and the position and
epoch of iteration belong to the pipeline, so the source has nothing to save; a pipeline's
checkpoint holds its position, epoch and keys and none of the source's records.
"""

from __future__ import annotations

import jax
import numpy as np
from flax import nnx

from datarax.pipeline.pipeline import Pipeline
from datarax.sources.memory_source import MemorySource, MemorySourceConfig


_N = 10_000


def _source() -> MemorySource:
    data = {"x": np.arange(_N * 4, dtype=np.float32).reshape(_N, 4)}
    return MemorySource(MemorySourceConfig(), data=data)


def test_an_in_memory_source_has_no_checkpoint_state() -> None:
    assert jax.tree.leaves(_source().get_state()) == []


def test_a_pipeline_checkpoint_holds_no_record_data() -> None:
    pipeline = Pipeline(source=_source(), stages=[], batch_size=8, rngs=nnx.Rngs(0), shuffle=True)
    pipeline.step()
    sizes = [int(np.size(leaf)) for leaf in jax.tree.leaves(pipeline.get_state())]
    assert sizes, "the checkpoint holds nothing"
    assert max(sizes) < _N


def test_a_pipeline_round_trip_resumes_the_same_records() -> None:
    def pipeline() -> Pipeline:
        return Pipeline(source=_source(), stages=[], batch_size=8, rngs=nnx.Rngs(0), shuffle=True)

    running = pipeline()
    running.step()
    state = running.get_state()
    expected = np.asarray(running.step()["x"])

    resumed = pipeline()
    resumed.set_state(state)
    np.testing.assert_array_equal(np.asarray(resumed.step()["x"]), expected)
