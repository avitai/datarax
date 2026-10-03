"""States saved in an earlier layout are refused by the pipeline, naming both versions.

The fixture holds a ``Pipeline.get_state()`` and an iterator ``get_state()`` as datarax
``611bf97`` saved them, from a ``MemorySource`` that held an ``nnx.Rngs``, a drawn shuffle seed
and its own position and epoch; its provenance is stored in the file. A pipeline's state is now
its host stage's cursor, version 3: the saved module state (no version) and the saved session
state (version 2) are both refused by ``Pipeline.set_state``, naming the version each is and the
one the pipeline reads. The compiled session still refuses the old iterator state itself. No code
converts an old layout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import numpy as np
import pytest
from flax import nnx

from datarax.pipeline.pipeline import Pipeline
from datarax.sources.memory_source import MemorySource, MemorySourceConfig


_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "pipeline"
    / "pipeline_state_with_source_state.npz"
)
_KEY_PREFIX = "key:"


def _saved() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """The saved pipeline state (nested as ``get_state`` returns it), iterator state, provenance."""
    state: dict[str, Any] = {}
    with np.load(_FIXTURE) as saved:
        iterator_state = json.loads(str(saved["iterator_state"]))
        provenance = json.loads(str(saved["provenance"]))
        for name in saved.files:
            if name in ("iterator_state", "provenance"):
                continue
            value: Any = saved[name]
            if name.startswith(_KEY_PREFIX):
                name = name[len(_KEY_PREFIX) :]
                value = jax.random.wrap_key_data(value)
            node = state
            *parents, leaf = name.split("/")
            for parent in parents:
                node = node.setdefault(parent, {})
            node[leaf] = value
    return state, iterator_state, provenance


def _pipeline() -> Pipeline:
    """The fixture's pipeline as it is built now."""
    source = MemorySource(MemorySourceConfig(), data={"x": np.arange(8, dtype=np.float32)})
    return Pipeline(source=source, stages=[], batch_size=2, rngs=nnx.Rngs(3), shuffle=True)


def test_the_fixture_was_saved_by_the_previous_layout() -> None:
    state, iterator_state, provenance = _saved()
    assert provenance["datarax_revision"].startswith("611bf97")
    assert provenance["worktree_clean"] is True
    assert {"rngs", "_shuffle_seed", "_shuffle_seeded", "index", "epoch"} <= set(state["source"])
    assert iterator_state["rng_counts"] == [1, 0, 1]


def test_a_saved_pipeline_state_is_refused_naming_its_layout_and_the_version_read() -> None:
    saved, _, _ = _saved()
    assert "version" not in saved
    with pytest.raises(ValueError, match=r"without a version.*module_state.*version 3"):
        _pipeline().set_state(saved)


def test_a_saved_session_state_is_refused_by_the_pipeline_naming_both_versions() -> None:
    _, iterator_state, _ = _saved()
    with pytest.raises(ValueError, match=r"version 2 \(the session layout.*reads version 3"):
        _pipeline().set_state(iterator_state)


def test_a_saved_iterator_state_is_refused_by_the_session() -> None:
    _, iterator_state, _ = _saved()
    session = _pipeline().session()
    with pytest.raises(ValueError, match="rng counts"):
        session.set_state(iterator_state)


def test_the_current_layout_round_trips() -> None:
    pipeline = _pipeline()
    batches = iter(pipeline)
    next(batches)
    state = pipeline.get_state()
    expected = np.asarray(next(batches).indices)
    resumed = _pipeline()
    resumed.set_state(state)
    np.testing.assert_array_equal(np.asarray(next(iter(resumed)).indices), expected)
