"""Streaming iteration: batch validation and the compiled DAG session.

A forward-only source's batches are pulled on the host and handed to the stage
DAG. Before that hand-off the pipeline checks each batch against the source's
declared element spec (structure, per-element shapes, dtypes and one shared
batch length), and it refuses at the start of a pass a declaration the device
cannot hold as declared. Neither a mismatch nor a precision change reaches the
DAG silently.

The DAG runs as one compiled step per batch shape. The step covers the stages
and the position counter, never the source, so a source that swaps its backend
iterator between passes does not force a recompile, and the state the source
advances while pulling is never overwritten. Pipeline state is read before and
written back after every batch, so it is current at every yield. The step
carries RNG counts and plain ``nnx.Variable`` counters; a stage that writes any
other state, or changes the module structure, is refused instead of losing the
write. A source's declared spec is read once per source and precision mode,
because reading it can open a backend iterator.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule
from datarax.core.spec import SpecMismatchError
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.pipeline.pipeline import Pipeline


@dataclass(frozen=True)
class _StreamConfig(StructuralConfig):
    pass


class _ListStream(DataSourceModule):
    """Forward-only source over prepared batches, then an empty batch.

    Like the backend-iterator sources, it keeps its iterator as a plain
    attribute that ``reset`` replaces.
    """

    def __init__(
        self,
        batches: Sequence[dict[str, Any]],
        spec: dict[str, jax.ShapeDtypeStruct],
    ) -> None:
        super().__init__(_StreamConfig(stochastic=False))
        self._batches = nnx.data(list(batches))
        self._spec = spec
        self._backend: Iterator[dict[str, Any]] | None = None
        self.cursor = nnx.Variable(0)
        self.spec_calls = nnx.Variable(0)
        self.pulls = nnx.Variable(0)

    def supports_indexed_access(self) -> bool:
        return False

    def element_spec(self) -> dict[str, jax.ShapeDtypeStruct]:
        self.spec_calls.set_value(int(self.spec_calls.get_value()) + 1)
        return self._spec

    def reset(self) -> None:
        self._backend = None
        self.cursor.set_value(0)

    def get_batch(self, batch_size: int) -> dict[str, Any]:
        del batch_size
        self.pulls.set_value(int(self.pulls.get_value()) + 1)
        if self._backend is None:
            self._backend = iter(list(self._batches))
        batch = next(self._backend, None)
        if batch is None:
            return {}
        self.cursor.set_value(int(self.cursor.get_value()) + 1)
        return batch


class _CountingStage(nnx.Module):
    """Identity stage that records how many batches reached the DAG."""

    def __init__(self) -> None:
        self.calls = nnx.Variable(jnp.zeros((), jnp.int32))

    def __call__(self, batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
        self.calls[...] = self.calls[...] + 1
        return batch


class _NoiseStage(nnx.Module):
    """Stage drawing from its own RNG stream on every batch."""

    def __init__(self) -> None:
        self.rngs = nnx.Rngs(noise=0)

    def __call__(self, batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
        return {**batch, "x": batch["x"] + jax.random.normal(self.rngs.noise(), batch["x"].shape)}


class _StatisticsStage(nnx.Module):
    """Stage accumulating a total in batch statistics, which the step does not carry."""

    def __init__(self) -> None:
        self.total = nnx.BatchStat(jnp.zeros((), jnp.float32))

    def __call__(self, batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
        self.total[...] = self.total[...] + batch["x"].sum()
        return batch


class _GrowingStage(nnx.Module):
    """Stage adding state while it runs, which changes the module structure."""

    def __call__(self, batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
        self.seen = nnx.Variable(jnp.zeros((), jnp.int32))
        return batch


# Batch shapes whose Python stage body ran, recorded by _TracingStage.
_TRACED_SHAPES: list[tuple[int, ...]] = []


class _TracingStage(nnx.Module):
    """Identity stage recording the batch shape each time its Python body runs."""

    def __call__(self, batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
        _TRACED_SHAPES.append(tuple(batch["x"].shape))
        return batch


def _jitter(element: Any, key: jax.Array) -> Any:
    noise = jax.random.normal(key, element.data["x"].shape) * 0.1
    return element.update_data({"x": element.data["x"] + noise})


# Reference application: the DAG through nnx.jit on every batch. It is compiled
# like the streaming step, because eager and XLA-compiled float results may differ
# in the last bits.
_apply_per_batch = nnx.jit(Pipeline.__call__)

_SPEC = {
    "x": jax.ShapeDtypeStruct((3,), jnp.float32),
    "y": jax.ShapeDtypeStruct((), jnp.int32),
}


def _records(size: int) -> dict[str, jax.Array]:
    return {"x": jnp.ones((size, 3), jnp.float32), "y": jnp.zeros((size,), jnp.int32)}


def _pipeline(
    batches: Sequence[dict[str, Any]],
    spec: dict[str, jax.ShapeDtypeStruct] = _SPEC,
    batch_size: int = 2,
) -> tuple[Pipeline, _ListStream, _CountingStage]:
    source = _ListStream(batches, spec)
    stage = _CountingStage()
    pipeline = Pipeline(source=source, stages=[stage], batch_size=batch_size, rngs=nnx.Rngs(0))
    return pipeline, source, stage


def _stochastic_pipeline(batches: Sequence[dict[str, Any]]) -> Pipeline:
    jitter = ElementOperator(
        ElementOperatorConfig(stochastic=True, stream_name="jitter"),
        fn=_jitter,
        rngs=nnx.Rngs(0, jitter=1),
    )
    return Pipeline(
        source=_ListStream(batches, _SPEC),
        stages=[_NoiseStage(), jitter],
        batch_size=2,
        rngs=nnx.Rngs(0),
    )


# ---------------------------------------------------------------------------
# Batch validation
# ---------------------------------------------------------------------------


def test_matching_batches_flow_through_including_a_short_final_batch() -> None:
    pipeline, _, stage = _pipeline([_records(2), _records(2), _records(1)])

    outputs = list(pipeline)

    assert [out["x"].shape[0] for out in outputs] == [2, 2, 1]
    assert int(pipeline._position[...]) == 5
    assert int(stage.calls[...]) == 3


def test_the_declared_spec_is_read_once_per_source_and_precision_mode() -> None:
    pipeline, source, _ = _pipeline([_records(2), _records(2)])

    for _ in range(2):
        source.reset()
        assert len(list(pipeline)) == 2
    assert int(source.spec_calls.get_value()) == 1

    with jax.enable_x64(True):
        source.reset()
        assert len(list(pipeline)) == 2
    assert int(source.spec_calls.get_value()) == 2


def test_an_empty_batch_ends_iteration() -> None:
    pipeline, _, stage = _pipeline([])

    assert list(pipeline) == []
    assert int(stage.calls[...]) == 0


@pytest.mark.parametrize(
    ("batch", "fragment"),
    [
        ({"x": jnp.ones((2, 4), jnp.float32), "y": jnp.zeros((2,), jnp.int32)}, "['x']"),
        ({"x": np.ones((2, 3), np.float64), "y": np.zeros((2,), np.int32)}, "float64"),
        ({"x": jnp.ones((2, 3), jnp.float32), "y": jnp.zeros((5,), jnp.int32)}, "['y']"),
        (
            {
                "x": jnp.ones((2, 3), jnp.float32),
                "y": jnp.zeros((2,), jnp.int32),
                "z": jnp.ones((2,), jnp.float32),
            },
            "['z']",
        ),
        ({"x": jnp.ones((2, 3), jnp.float32)}, "['y']"),
        ({"x": jnp.ones((3, 3), jnp.float32), "y": jnp.zeros((3,), jnp.int32)}, "3 records"),
    ],
    ids=[
        "trailing-shape",
        "host-float64-declared-float32",
        "leaf-lengths-differ",
        "undeclared-field",
        "declared-field-missing",
        "larger-than-batch-size",
    ],
)
def test_a_batch_that_disagrees_with_the_declaration_stops_before_the_dag(
    batch: dict[str, Any], fragment: str
) -> None:
    pipeline, _, stage = _pipeline([batch])

    with pytest.raises(SpecMismatchError, match=re.escape(fragment)):
        list(pipeline)

    assert int(stage.calls[...]) == 0
    assert int(pipeline._position[...]) == 0


def test_a_declared_dtype_the_device_would_narrow_is_refused_before_any_pull() -> None:
    spec = {"x": jax.ShapeDtypeStruct((3,), np.float64)}
    pipeline, source, _ = _pipeline([{"x": np.ones((2, 3), np.float64)}], spec=spec)

    with pytest.raises(SpecMismatchError, match="jax_enable_x64"):
        list(pipeline)

    assert int(source.pulls.get_value()) == 0


def test_float64_streams_through_unchanged_when_x64_is_on() -> None:
    with jax.enable_x64(True):
        spec = {"x": jax.ShapeDtypeStruct((3,), np.float64)}
        pipeline, _, _ = _pipeline([{"x": np.ones((2, 3), np.float64)}], spec=spec)
        outputs = list(pipeline)

    assert outputs[0]["x"].dtype == np.float64


def test_a_declared_text_field_is_refused_with_its_name() -> None:
    spec = {"text": jax.ShapeDtypeStruct((), np.dtype("<U8")), **_SPEC}
    pipeline, _, _ = _pipeline([{**_records(2), "text": ["a", "b"]}], spec=spec)

    with pytest.raises(SpecMismatchError, match=re.escape("['text']")):
        list(pipeline)


# ---------------------------------------------------------------------------
# Compiled DAG session
# ---------------------------------------------------------------------------


def test_the_dag_compiles_once_per_batch_shape_across_passes_and_pipelines() -> None:
    _TRACED_SHAPES.clear()
    spec = {"x": jax.ShapeDtypeStruct((7,), jnp.float32)}
    batches = [{"x": jnp.ones((4, 7), jnp.float32)}] * 2 + [{"x": jnp.ones((1, 7), jnp.float32)}]

    for _ in range(2):
        source = _ListStream(batches, spec)
        pipeline = Pipeline(source=source, stages=[_TracingStage()], batch_size=4, rngs=nnx.Rngs(0))
        for _ in range(2):
            source.reset()
            assert len(list(pipeline)) == 3

    assert sorted(_TRACED_SHAPES) == [(1, 7), (4, 7)]


def test_stochastic_outputs_and_rng_streams_match_applying_the_dag_through_nnx_jit() -> None:
    batches = [_records(2), _records(2), _records(1)]
    streamed = _stochastic_pipeline(batches)
    reference = _stochastic_pipeline(batches)

    outputs = list(streamed)
    expected = []
    for batch in batches:
        expected.append(_apply_per_batch(reference, batch))
        reference._position[...] = reference._position[...] + jnp.int32(batch["x"].shape[0])

    for got, want in zip(outputs, expected, strict=True):
        np.testing.assert_array_equal(np.asarray(got["x"]), np.asarray(want["x"]))
    got_counts = [int(c) for c in jax.tree.leaves(nnx.state(streamed, nnx.RngCount))]
    want_counts = [int(c) for c in jax.tree.leaves(nnx.state(reference, nnx.RngCount))]
    assert got_counts == want_counts
    assert int(streamed._position[...]) == int(reference._position[...]) == 5


def test_pipeline_state_is_current_at_every_yield() -> None:
    pipeline, _, stage = _pipeline([_records(2), _records(1)])
    iterator = iter(pipeline)

    next(iterator)
    assert int(pipeline._position[...]) == 2
    assert int(stage.calls[...]) == 1

    next(iterator)
    assert int(pipeline._position[...]) == 3
    assert int(stage.calls[...]) == 2


def test_state_changed_between_batches_is_read_by_the_next_batch() -> None:
    pipeline, _, stage = _pipeline([_records(2), _records(2)])
    iterator = iter(pipeline)

    next(iterator)
    pipeline._position[...] = jnp.int32(10)
    stage.calls[...] = jnp.int32(7)
    next(iterator)

    assert int(pipeline._position[...]) == 12
    assert int(stage.calls[...]) == 8


def test_a_stage_writing_state_the_step_does_not_carry_is_refused() -> None:
    pipeline = Pipeline(
        source=_ListStream([_records(2)], _SPEC),
        stages=[_StatisticsStage()],
        batch_size=2,
        rngs=nnx.Rngs(0),
    )

    with pytest.raises(ValueError, match=re.escape("stage_0.total (BatchStat)")):
        list(pipeline)


def test_a_stage_changing_the_module_structure_is_refused() -> None:
    pipeline = Pipeline(
        source=_ListStream([_records(2)], _SPEC),
        stages=[_GrowingStage()],
        batch_size=2,
        rngs=nnx.Rngs(0),
    )

    with pytest.raises(ValueError, match="changed the module structure"):
        list(pipeline)


def test_the_source_keeps_the_state_its_pulls_advance() -> None:
    pipeline, source, _ = _pipeline([_records(2), _records(2)])

    list(pipeline)

    assert int(source.cursor.get_value()) == 2
    assert int(source.pulls.get_value()) == 3
