"""A state saved while the source held RNG, seed or counter state is refused, never upgraded.

The fixture holds a ``Pipeline.get_state()`` and an iterator ``get_state()`` as datarax
``611bf97`` saved them, from a ``MemorySource`` that held an ``nnx.Rngs``, a drawn shuffle seed
and its own position and epoch; its provenance is stored in the file. The pipeline now owns the
order and in-memory sources hold none of that state, so restoring either saved state into the
same pipeline built today raises, naming what the saved state has and the pipeline lacks. No
code converts the old layout.
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


def _paths(state: dict[str, Any], prefix: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    paths: set[tuple[str, ...]] = set()
    for key, value in state.items():
        here = (*prefix, str(key))
        paths |= _paths(value, here) if isinstance(value, dict) else {here}
    return paths


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


def test_a_saved_pipeline_state_is_refused_naming_what_the_pipeline_lacks() -> None:
    saved, _, _ = _saved()
    pipeline = _pipeline()
    present = {path[:end] for path in _paths(pipeline.get_state()) for end in range(len(path) + 1)}
    # Each saved path the pipeline lacks is named at its shallowest key the pipeline lacks.
    absent = {
        next(path[end - 1] for end in range(1, len(path) + 1) if path[:end] not in present)
        for path in _paths(saved)
        if path not in present
    }
    assert absent, "the control: the saved layout has source state the pipeline lacks"
    with pytest.raises(ValueError, match="structurally incompatible") as refused:
        pipeline.set_state(saved)
    cause = str(refused.value.__cause__)
    for name in absent:
        assert name in cause, f"{name!r} not named in {cause!r}"


def test_a_saved_iterator_state_is_refused() -> None:
    _, iterator_state, _ = _saved()
    session = _pipeline().session()
    with pytest.raises(ValueError, match="rng counts"):
        session.set_state(iterator_state)


def test_the_current_layout_round_trips() -> None:
    pipeline = _pipeline()
    pipeline.step()
    state = pipeline.get_state()
    expected = np.asarray(pipeline.step().indices)
    resumed = _pipeline()
    resumed.set_state(state)
    np.testing.assert_array_equal(np.asarray(resumed.step().indices), expected)
