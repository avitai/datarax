"""A pipeline is a checkpoint target: its tuned parameters and its place in the data resume.

Stages carry learnable parameters that training optimizes through the pipeline, so restoring a
pipeline must bring back those parameters as well as where iteration stands (position, epoch,
the epoch key, RNG counts and the source's own state), and nothing of the data. ``Pipeline``
implements the ``Checkpointable`` protocol with the state logic ``DataraxModule`` uses.
"""

from __future__ import annotations

import copy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from substrax.typing import Checkpointable

from datarax.checkpoint import IteratorCheckpoint
from datarax.core.config import ElementOperatorConfig
from datarax.core.operator import DIRECT_CALL_STREAM
from datarax.operators import ElementOperator
from datarax.pipeline import Pipeline
from datarax.sources.memory_source import MemorySource, MemorySourceConfig


_RECORDS = 20


class _Scale(nnx.Module):
    """A stage with one learnable parameter."""

    def __init__(self) -> None:
        self.factor = nnx.Param(jnp.float32(1.0))

    def __call__(self, batch: dict) -> dict:
        return {**batch, "x": batch["x"] * self.factor[...]}


def _jitter(element, key):
    return element.update_data(
        {"x": element.data["x"] + 0.1 * jax.random.normal(key, element.data["x"].shape)}
    )


def _build(*, stages: list[nnx.Module] | None = None) -> Pipeline:
    """A shuffled pipeline with a learnable stage and a stochastic operator, built the same way."""
    data = {"x": np.arange(2 * _RECORDS, dtype=np.float32).reshape(_RECORDS, 2)}
    source = MemorySource(MemorySourceConfig(shuffle=True), data=data, rngs=nnx.Rngs(0, shuffle=1))
    if stages is None:
        operator = ElementOperator(
            ElementOperatorConfig(stochastic=True, stream_name="aug"),
            fn=_jitter,
            rngs=nnx.Rngs(aug=2),
        )
        stages = [_Scale(), operator]
    return Pipeline(source=source, stages=stages, batch_size=4, rngs=nnx.Rngs(3))


def _scale(pipeline: Pipeline) -> _Scale:
    stage = pipeline.stages[0]
    assert isinstance(stage, _Scale)
    return stage


def _operator(pipeline: Pipeline) -> ElementOperator:
    stage = pipeline.stages[1]
    assert isinstance(stage, ElementOperator)
    return stage


def _tuned() -> Pipeline:
    """A pipeline three batches in, whose parameter an optimizer has moved."""
    pipeline = _build()
    for _ in range(3):
        pipeline.step()
    _scale(pipeline).factor[...] = jnp.float32(2.5)
    return pipeline


def _loss(pipeline: Pipeline) -> jax.Array:
    return jnp.sum(pipeline.step()["x"] ** 2)


@nnx.jit
def _train_step(pipeline: Pipeline) -> None:
    grads = nnx.grad(_loss, argnums=nnx.DiffState(0, nnx.Param))(pipeline)
    _scale(pipeline).factor[...] -= 1e-4 * grads["_stage_modules"]["stage_0"]["factor"][...]


def test_a_pipeline_is_checkpointable() -> None:
    assert isinstance(_build(), Checkpointable)


def test_the_state_holds_parameters_and_iteration_but_no_data() -> None:
    state = _tuned().get_state()
    leaves = jax.tree_util.tree_flatten_with_path(state)[0]
    paths = {jax.tree_util.keystr(path) for path, _ in leaves}

    assert "['_stage_modules']['stage_0']['factor']" in paths
    assert {"['_position']", "['_epoch']", "['_epoch_key_base']"} <= paths
    assert not any(getattr(leaf, "shape", ())[:1] == (_RECORDS,) for _, leaf in leaves)


def test_a_tuned_pipeline_restores_its_parameters_and_its_place(tmp_path: Path) -> None:
    tuned = _tuned()
    with IteratorCheckpoint(tmp_path) as checkpoint:
        checkpoint.save(tuned, step=3)
    restored = _build()
    with IteratorCheckpoint(tmp_path) as checkpoint:
        checkpoint.restore(restored)

    assert float(_scale(restored).factor[...]) == 2.5
    assert (int(restored._position[...]), int(restored._epoch[...])) == (12, 0)
    np.testing.assert_array_equal(np.asarray(restored.step()["x"]), np.asarray(tuned.step()["x"]))


def test_a_restored_pipeline_keeps_training_as_the_original_would(tmp_path: Path) -> None:
    """Training continued from a checkpoint follows the run that never stopped, step for step."""
    tuned = _tuned()
    with IteratorCheckpoint(tmp_path) as checkpoint:
        checkpoint.save(tuned, step=3)
    restored = _build()
    with IteratorCheckpoint(tmp_path) as checkpoint:
        checkpoint.restore(restored)

    for _ in range(3):
        _train_step(tuned)
        _train_step(restored)
    assert float(_scale(restored).factor[...]) != 2.5
    assert float(_scale(restored).factor[...]) == float(_scale(tuned).factor[...])


def test_a_restored_pipeline_scans_as_the_original_would(tmp_path: Path) -> None:
    tuned = _tuned()
    with IteratorCheckpoint(tmp_path) as checkpoint:
        checkpoint.save(tuned, step=3)
    restored = _build()
    with IteratorCheckpoint(tmp_path) as checkpoint:
        checkpoint.restore(restored)

    def total(batch: dict) -> jax.Array:
        return jnp.sum(batch["x"])

    np.testing.assert_array_equal(
        np.asarray(restored.scan(total, length=7)), np.asarray(tuned.scan(total, length=7))
    )


def test_a_save_during_iteration_holds_the_batches_served(tmp_path: Path) -> None:
    """A session writes its progress into the pipeline at every batch it yields."""
    pipeline = _build()
    session = pipeline.session()
    served = [next(session) for _ in range(3)]
    with IteratorCheckpoint(tmp_path) as checkpoint:
        checkpoint.save(pipeline, step=3)
    restored = _build()
    with IteratorCheckpoint(tmp_path) as checkpoint:
        checkpoint.restore(restored)

    assert len(served) == 3
    assert int(restored._position[...]) == 12
    np.testing.assert_array_equal(
        np.asarray(next(restored.session())["x"]), np.asarray(next(session)["x"])
    )


def test_a_pipeline_of_another_structure_is_refused() -> None:
    state = _tuned().get_state()
    other = _build(stages=[_Scale()])

    with pytest.raises(ValueError, match="structurally incompatible"):
        other.set_state(state)


def test_an_operator_s_earlier_layout_is_upgraded_inside_a_pipeline_checkpoint() -> None:
    """The pipeline offers each operator its own subtree, as a module checkpoint does."""
    state = copy.deepcopy(_tuned().get_state())
    operator_state = state["_stage_modules"]["stage_1"]
    base_key = operator_state["_base_key"]
    del operator_state["_rng_stream"]
    operator_state["rngs"] = {"aug": {"count": jnp.zeros((), jnp.uint32), "key": base_key}}

    restored = _build()
    restored.set_state(state)

    stream = _operator(restored)._rng_stream
    expected = jax.random.fold_in(base_key, DIRECT_CALL_STREAM)
    assert jnp.array_equal(jax.random.key_data(stream.key[...]), jax.random.key_data(expected))
