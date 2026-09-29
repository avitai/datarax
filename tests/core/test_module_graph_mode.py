"""A module sharing Variables keeps its checkpoint and clone API when tree mode is the default.

Flax plans to make tree mode the default for ``nnx.split``, ``nnx.state`` and the transforms,
and tree mode rejects shared Variables. A source that hands one ``Rngs`` to its metadata
manager shares Variables, as a pipeline built from one ``Rngs`` does, so datarax's own graph
calls name graph mode.
"""

from __future__ import annotations

import numpy as np
import pytest
from flax import nnx

from datarax.sources.memory_source import MemorySource, MemorySourceConfig


class _SharingSource(MemorySource):
    """A source holding its ``Rngs`` under a second name: one RNG count on two graph paths."""

    def __init__(self) -> None:
        config = MemorySourceConfig(shuffle=True)
        data = {"x": np.arange(16, dtype=np.float32)}
        super().__init__(config, data=data, rngs=nnx.Rngs(0, shuffle=1))
        self.shared_rngs = self.rngs


def _sharing_source() -> MemorySource:
    return _SharingSource()


def test_the_fixture_shares_variables() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        nnx.split(_sharing_source(), graph=False)


def test_get_and_set_state_run_with_tree_mode_as_the_default() -> None:
    source, reference = _sharing_source(), _sharing_source()
    with nnx.set_graph_mode(False), nnx.set_graph_updates(False):
        state = source.get_state()
        source.set_state(state)
    assert state.keys() == reference.get_state().keys()


def test_clone_runs_with_tree_mode_as_the_default() -> None:
    source = _sharing_source()
    with nnx.set_graph_mode(False), nnx.set_graph_updates(False):
        clone = source.clone()
    assert clone is not source
    assert clone.get_state().keys() == source.get_state().keys()
