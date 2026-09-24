"""A source's checkpoint holds the state that changes, not the records it was built with.

``get_state`` used to capture every leaf ``nnx.state`` sees, including a ``MemorySource``'s data
arrays, so each checkpoint carried the whole dataset and ``set_state`` could not restore it.
"""

from __future__ import annotations

import jax
import numpy as np
from flax import nnx

from datarax.sources.memory_source import MemorySource, MemorySourceConfig


_N = 10_000


def _source() -> MemorySource:
    data = {"x": np.arange(_N * 4, dtype=np.float32).reshape(_N, 4)}
    return MemorySource(MemorySourceConfig(shuffle=True), data=data, rngs=nnx.Rngs(0, shuffle=1))


def test_the_checkpoint_holds_no_record_data() -> None:
    sizes = [int(np.size(leaf)) for leaf in jax.tree.leaves(_source().get_state())]
    assert sizes, "the checkpoint holds nothing"
    assert max(sizes) < _N


def test_a_round_trip_resumes_the_same_records() -> None:
    source = _source()
    source.get_batch(8)
    state = source.get_state()
    expected = np.asarray(source.get_batch(8)["x"])

    resumed = _source()
    resumed.set_state(state)
    np.testing.assert_array_equal(np.asarray(resumed.get_batch(8)["x"]), expected)
