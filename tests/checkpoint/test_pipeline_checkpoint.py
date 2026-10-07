"""A pipeline checkpoint is its place in the data plus its stages' parameters, saved apart.

``Pipeline`` implements the ``Checkpointable`` protocol: ``get_state()`` is where iteration
stands (the host stage's versioned cursor) and holds no parameters and no data. Stages carry
learnable parameters and statistics that training moves, and those are the DAG's ``nnx`` state,
checkpointed as ``module_state(pipeline.dag)`` beside the cursor. ``step()`` and ``scan`` keep
their place in the pipeline's own Variables, so a run driven by ``step()`` checkpoints
``module_state(pipeline)``, which holds that place and the parameters together.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from substrax.checkpoint import OrbaxCheckpointStore
from substrax.typing import Checkpointable

from datarax.checkpoint import IteratorCheckpoint
from datarax.checkpoint.iterators import ITEM
from datarax.core.config import ElementOperatorConfig
from datarax.core.element_batch import Batch
from datarax.core.module import module_state, restore_module_state
from datarax.operators import ElementOperator
from datarax.pipeline import Pipeline
from datarax.sources.memory_source import MemorySource, MemorySourceConfig


_RECORDS = 20
_MODEL_ITEM = "model"  # the stages' parameters are trained, and saved, with the model


class _Scale(nnx.Module):
    """A stage with one learnable parameter."""

    def __init__(self) -> None:
        self.factor = nnx.Param(jnp.float32(1.0))

    def __call__(self, batch: Batch) -> Batch:
        return batch.replace(data={**batch.data, "x": batch["x"] * self.factor[...]})


def _jitter(element, key):
    return element.update_data(
        {"x": element.data["x"] + 0.1 * jax.random.normal(key, element.data["x"].shape)}
    )


def _build(*, stages: list[nnx.Module] | None = None) -> Pipeline:
    """A shuffled pipeline with a learnable stage and a stochastic operator, built the same way."""
    data = {"x": np.arange(2 * _RECORDS, dtype=np.float32).reshape(_RECORDS, 2)}
    source = MemorySource(MemorySourceConfig(), data=data)
    if stages is None:
        operator = ElementOperator(
            ElementOperatorConfig(stochastic=True, stream_name="aug"),
            fn=_jitter,
            rngs=nnx.Rngs(aug=2),
        )
        stages = [_Scale(), operator]
    return Pipeline(source=source, stages=stages, batch_size=4, rngs=nnx.Rngs(3), shuffle=True)


def _scale(pipeline: Pipeline) -> _Scale:
    stage = pipeline.stages[0]
    assert isinstance(stage, _Scale)
    return stage


def _tuned_iteration() -> tuple[Pipeline, Iterator[Batch]]:
    """A pipeline three batches into iteration, whose parameter an optimizer has moved."""
    pipeline = _build()
    batches = iter(pipeline)
    for _ in range(3):
        next(batches)
    _scale(pipeline).factor[...] = jnp.float32(2.5)
    return pipeline, batches


def _tuned_steps() -> Pipeline:
    """A pipeline three ``step()`` calls in, whose parameter an optimizer has moved."""
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
    _scale(pipeline).factor[...] -= 1e-4 * grads["dag"]["stages"]["stage_0"]["factor"][...]


def _save_and_restore_steps(tuned: Pipeline, tmp_path: Path) -> Pipeline:
    """Checkpoint a ``step()``-driven pipeline as its module state; restore it into a new one."""
    with OrbaxCheckpointStore(tmp_path) as store:
        store.save(3, {_MODEL_ITEM: module_state(tuned)})
        saved = store.restore(3).items[_MODEL_ITEM]
    restored = _build()
    restore_module_state(restored, saved)
    return restored


def test_a_pipeline_is_checkpointable() -> None:
    assert isinstance(_build(), Checkpointable)


def test_the_state_is_the_cursor_without_parameters_or_data() -> None:
    pipeline, _ = _tuned_iteration()
    state = pipeline.get_state()
    leaves = jax.tree_util.tree_flatten_with_path(state)[0]
    paths = {jax.tree_util.keystr(path) for path, _ in leaves}

    assert state["version"] == 3
    assert state["position"] == 12
    assert not any("dag" in path or "factor" in path or "_position" in path for path in paths)
    assert not any(getattr(leaf, "shape", ())[:1] == (_RECORDS,) for _, leaf in leaves)


def test_a_tuned_pipeline_restores_its_parameters_and_its_place(tmp_path: Path) -> None:
    tuned, batches = _tuned_iteration()
    with OrbaxCheckpointStore(tmp_path) as store:
        store.save(3, {ITEM: tuned.get_state(), _MODEL_ITEM: module_state(tuned.dag)})
        items = store.restore(3).items
    restored = _build()
    restored.set_state(items[ITEM])
    restore_module_state(restored.dag, items[_MODEL_ITEM])

    assert float(_scale(restored).factor[...]) == 2.5
    assert restored.get_state()["position"] == 12
    np.testing.assert_array_equal(
        np.asarray(next(iter(restored))["x"]), np.asarray(next(batches)["x"])
    )


def test_the_iteration_checkpoint_resumes_the_batches_not_yet_served(tmp_path: Path) -> None:
    """``IteratorCheckpoint`` saves ``get_state()``: the next batch is the one not yet taken."""
    pipeline = _build()
    batches = iter(pipeline)
    served = [next(batches) for _ in range(3)]
    with IteratorCheckpoint(tmp_path) as checkpoint:
        checkpoint.save(pipeline, step=3)
    restored = _build()
    with IteratorCheckpoint(tmp_path) as checkpoint:
        checkpoint.restore(restored)

    assert len(served) == 3
    np.testing.assert_array_equal(
        np.asarray(next(iter(restored))["x"]), np.asarray(next(batches)["x"])
    )


def test_a_step_driven_pipeline_restores_its_parameters_and_its_place(tmp_path: Path) -> None:
    tuned = _tuned_steps()
    restored = _save_and_restore_steps(tuned, tmp_path)

    assert float(_scale(restored).factor[...]) == 2.5
    assert (int(restored._position[...]), int(restored._epoch[...])) == (12, 0)
    np.testing.assert_array_equal(np.asarray(restored.step()["x"]), np.asarray(tuned.step()["x"]))


def test_a_restored_pipeline_keeps_training_as_the_original_would(tmp_path: Path) -> None:
    """Training continued from a checkpoint follows the run that never stopped, step for step."""
    tuned = _tuned_steps()
    restored = _save_and_restore_steps(tuned, tmp_path)

    for _ in range(3):
        _train_step(tuned)
        _train_step(restored)
    assert float(_scale(restored).factor[...]) != 2.5
    assert float(_scale(restored).factor[...]) == float(_scale(tuned).factor[...])


def test_a_restored_pipeline_scans_as_the_original_would(tmp_path: Path) -> None:
    tuned = _tuned_steps()
    restored = _save_and_restore_steps(tuned, tmp_path)

    def total(batch: dict) -> jax.Array:
        return jnp.sum(batch["x"])

    np.testing.assert_array_equal(
        np.asarray(restored.scan(total, length=7)), np.asarray(tuned.scan(total, length=7))
    )


def test_stages_of_another_structure_are_refused() -> None:
    saved = module_state(_build().dag)
    other = _build(stages=[_Scale()])

    with pytest.raises(ValueError, match="structurally incompatible"):
        restore_module_state(other.dag, saved)


def test_an_operator_state_carrying_a_stream_is_refused_inside_a_pipeline_checkpoint() -> None:
    """An operator's state is its base key and statistics; a subtree holding more is refused."""
    saved = module_state(_build().dag)
    operator_state = saved["stages"]["stage_1"]
    operator_state["_rng_stream"] = {
        "count": jnp.zeros((), jnp.uint32),
        "key": operator_state["_base_key"],
    }

    with pytest.raises(ValueError, match="structurally incompatible"):
        restore_module_state(_build().dag, saved)


def _host_values(state: dict) -> dict:
    """The state's leaves as NumPy arrays, keys as their key data, for exact comparison."""

    def to_host(leaf: jax.Array) -> np.ndarray:
        if isinstance(leaf, jax.Array) and jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key):
            return np.asarray(jax.random.key_data(leaf))
        return np.asarray(leaf)

    return jax.tree.map(to_host, state)


def test_a_refused_restore_of_the_stages_changes_nothing() -> None:
    """Validation covers every stage before any value is written, so a refusal is atomic."""
    saved = module_state(_tuned_steps().dag)
    saved["stages"]["stage_1"]["unexpected"] = jnp.zeros(())
    target = _build()
    before = _host_values(module_state(target.dag))

    with pytest.raises(ValueError, match="structurally incompatible"):
        restore_module_state(target.dag, saved)

    jax.tree.map(np.testing.assert_array_equal, _host_values(module_state(target.dag)), before)


def test_a_refused_restore_of_the_cursor_changes_nothing() -> None:
    pipeline, _ = _tuned_iteration()
    state = pipeline.get_state()
    state["fingerprint"]["batch_size"] = 8
    target, _ = _tuned_iteration()
    before = target.get_state()

    with pytest.raises(ValueError, match="batch_size"):
        target.set_state(state)

    assert target.get_state() == before
