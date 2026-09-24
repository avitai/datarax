"""The end of an epoch: every row a pipeline serves is a record, never padding.

A random-access source keeps every batch of the compiled step at ``batch_size``, so what a
batch reaching the end of an epoch holds is the pipeline's contract, the two orderings
tf.data and Grain offer:

- ``drop_last=False`` (the default) is ``repeat().batch()``: the batch is completed from the
  head of the next epoch's order, so each epoch serves every record once and a batch may
  hold records of two epochs. A session serving ``num_epochs`` epochs stops after exactly
  ``num_epochs * N`` records, so its final batch may be short;
- ``drop_last=True`` is ``batch(drop_remainder=True).repeat()``: the records short of a full
  batch are skipped and the next batch starts the next epoch.

``step()`` and ``scan`` follow the same rule inside the compiled step, so they never run out.
"""

from __future__ import annotations

from itertools import islice

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from datarax.core.config import ElementOperatorConfig
from datarax.operators import ElementOperator
from datarax.pipeline import iteration, Pipeline, PipelineIterator
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from tests.pipeline.test_pipeline_streaming import _ListStream


_N = 100
_BATCH = 32


def _source(n: int = _N, *, shuffle: bool = False, seed: int = 0) -> MemorySource:
    return MemorySource(
        MemorySourceConfig(shuffle=shuffle),
        data={"x": np.arange(n, dtype=np.float32)},
        rngs=nnx.Rngs(seed, shuffle=seed),
    )


def _pipeline(
    n: int = _N, *, shuffle: bool = False, batch_size: int = _BATCH, **kwargs
) -> Pipeline:
    return Pipeline(
        source=_source(n, shuffle=shuffle),
        stages=[],
        batch_size=batch_size,
        rngs=nnx.Rngs(0),
        **kwargs,
    )


def _rows(batches: list[dict]) -> np.ndarray:
    return np.concatenate([np.asarray(batch["x"]) for batch in batches])


class TestNoPadding:
    def test_iteration_serves_each_record_once_and_a_short_final_batch(self) -> None:
        batches = list(_pipeline())

        assert [batch["x"].shape[0] for batch in batches] == [32, 32, 32, 4]
        np.testing.assert_array_equal(_rows(batches), np.arange(_N, dtype=np.float32))

    def test_no_batch_carries_a_mask(self) -> None:
        assert set(_pipeline().step()) == {"x"}
        assert all(set(batch) == {"x"} for batch in _pipeline())
        keys = _pipeline().scan(lambda batch: jnp.asarray(len(batch)), length=5)
        np.testing.assert_array_equal(np.asarray(keys), [1] * 5)

    def test_a_streaming_source_carries_no_mask(self) -> None:
        batches = [
            {"x": jnp.array([0.0, 1.0])},
            {"x": jnp.array([2.0, 3.0])},
            {"x": jnp.array([4.0])},
        ]
        source = _ListStream(batches, {"x": jax.ShapeDtypeStruct((), jnp.float32)})
        pipeline = Pipeline(source=source, stages=[], batch_size=2, rngs=nnx.Rngs(0))

        assert [set(batch) for batch in pipeline] == [{"x"}] * 3

    def test_drop_last_refuses_a_batch_larger_than_the_source(self) -> None:
        with pytest.raises(ValueError, match="batch_size <= len"):
            _pipeline(4, batch_size=8, drop_last=True)

    def test_a_batch_larger_than_the_source_spans_several_epochs(self) -> None:
        """Iteration serves each epoch's records once; the compiled step's batch spans epochs."""
        assert [np.asarray(b["x"]).tolist() for b in _pipeline(3, batch_size=4)] == [[0, 1, 2]]
        assert [np.asarray(b["x"]).tolist() for b in _pipeline(3, batch_size=4, num_epochs=2)] == [
            [0, 1, 2, 0],
            [1, 2],
        ]
        stepper = _pipeline(3, batch_size=8)
        np.testing.assert_array_equal(np.asarray(stepper.step()["x"]), [0, 1, 2, 0, 1, 2, 0, 1])
        assert (int(stepper._position[...]), int(stepper._epoch[...])) == (2, 2)
        np.testing.assert_array_equal(np.asarray(stepper.step()["x"]), [2, 0, 1, 2, 0, 1, 2, 0])


class TestCrossing:
    """``drop_last=False``: a batch reaching the end continues in the next epoch."""

    def test_the_boundary_batch_holds_the_tail_and_the_next_head(self) -> None:
        pipeline = _pipeline(num_epochs=2)
        batches = list(pipeline)

        assert len(pipeline) == len(batches) == 7  # ceil(200 / 32)
        assert [batch["x"].shape[0] for batch in batches] == [32] * 6 + [8]
        boundary = np.asarray(batches[3]["x"])
        np.testing.assert_array_equal(boundary[:4], np.arange(96, 100, dtype=np.float32))
        np.testing.assert_array_equal(boundary[4:], np.arange(28, dtype=np.float32))
        np.testing.assert_array_equal(_rows(batches), np.tile(np.arange(_N, dtype=np.float32), 2))

    def test_each_shuffled_epoch_is_one_permutation_and_the_head_is_the_next_one(self) -> None:
        epochs = _rows(list(_pipeline(shuffle=True, num_epochs=3))).reshape(3, _N)

        for epoch in epochs:
            assert sorted(epoch.tolist()) == list(range(_N))
        assert not np.array_equal(epochs[0], epochs[1])
        reference = _pipeline(shuffle=True, drop_last=True)
        reference.reset()
        np.testing.assert_array_equal(epochs[1][:_BATCH], np.asarray(reference.step()["x"]))

    def test_step_crosses_as_iteration_does_and_never_runs_out(self) -> None:
        pipeline = _pipeline(shuffle=True)
        stepped = _rows([pipeline.step() for _ in range(9)])
        streamed = _rows(list(islice(_pipeline(shuffle=True, num_epochs=None), 9)))

        np.testing.assert_array_equal(stepped, streamed)
        assert (int(pipeline._position[...]), int(pipeline._epoch[...])) == (88, 2)

    def test_iterator_state_reports_the_crossed_epoch(self) -> None:
        session = _pipeline(num_epochs=None).session()
        for _ in range(4):
            next(session)
        state = session.get_state()
        assert (state["position"], state["epoch"]) == (28, 1)

    def test_a_stream_has_no_length(self) -> None:
        pipeline = _pipeline(num_epochs=None)
        with pytest.raises(TypeError, match="no length"):
            len(pipeline)
        assert pipeline.batches_left() is None
        assert len(list(islice(pipeline, 13))) == 13


class TestDropLast:
    """``drop_last=True``: the records short of a batch are skipped each epoch."""

    def test_each_epoch_serves_its_full_batches(self) -> None:
        pipeline = _pipeline(drop_last=True, num_epochs=2)
        batches = list(pipeline)

        assert len(pipeline) == len(batches) == 6
        assert all(batch["x"].shape[0] == _BATCH for batch in batches)
        np.testing.assert_array_equal(_rows(batches), np.tile(np.arange(96, dtype=np.float32), 2))

    def test_step_past_the_last_full_batch_starts_the_next_epoch(self) -> None:
        pipeline = _pipeline(drop_last=True)
        for _ in range(3):
            pipeline.step()
        np.testing.assert_array_equal(
            np.asarray(pipeline.step()["x"]), np.arange(_BATCH, dtype=np.float32)
        )
        assert (int(pipeline._position[...]), int(pipeline._epoch[...])) == (32, 1)


class _Counter(nnx.Module):
    """A stage counting the records it sees."""

    def __init__(self) -> None:
        self.seen = nnx.Variable(jnp.zeros((), jnp.int32))

    def __call__(self, batch: dict) -> dict:
        self.seen[...] += batch["x"].shape[0]
        return batch


class TestRunEnd:
    def test_the_final_batch_is_computed_at_its_size(self) -> None:
        """A stage that counts what it sees counts ``num_epochs * N`` records, no more."""
        counter = _Counter()
        pipeline = Pipeline(
            source=_source(), stages=[counter], batch_size=_BATCH, rngs=nnx.Rngs(0), num_epochs=2
        )
        list(pipeline)
        assert int(counter.seen[...]) == 2 * _N

    def test_the_run_ends_at_the_end_of_its_last_epoch(self) -> None:
        pipeline = _pipeline()
        session = pipeline.session()
        list(session)

        assert (session.get_state()["position"], session.get_state()["epoch"]) == (_N, 0)
        assert list(pipeline) == []
        pipeline.reset()
        assert len(list(pipeline)) == 4

    def test_a_repeated_run_reuses_the_compiled_steps(self) -> None:
        iteration._SESSION_STEPS.clear()
        pipeline = _pipeline()
        list(pipeline)
        assert len(iteration._SESSION_STEPS) == 2  # the full batch and the short final one
        pipeline.reset()
        list(pipeline)
        assert len(iteration._SESSION_STEPS) == 2


class _Scale(nnx.Module):
    """A stage with a parameter, so gradients have somewhere to go."""

    def __init__(self) -> None:
        self.factor = nnx.Param(jnp.float32(2.0))

    def __call__(self, batch: dict) -> dict:
        return {**batch, "x": batch["x"] * self.factor[...]}


class _Model(nnx.Module):
    def __init__(self) -> None:
        self.weight = nnx.Param(jnp.float32(3.0))


def _at_the_boundary() -> tuple[Pipeline, np.ndarray]:
    """A shuffled pipeline two batches in, and the records its next (boundary) batch holds."""
    pipeline = Pipeline(
        source=_source(10, shuffle=True), stages=[_Scale()], batch_size=4, rngs=nnx.Rngs(0)
    )
    reference = Pipeline(
        source=_source(10, shuffle=True), stages=[], batch_size=4, rngs=nnx.Rngs(0)
    )
    for _ in range(2):
        pipeline.step()
        reference.step()
    return pipeline, np.asarray(reference.step()["x"])


class TestTransformsAtTheBoundary:
    """The crossing batch under the caller's transforms: the branch that runs, not only traces."""

    def test_a_jitted_gradient_reaches_the_model(self) -> None:
        pipeline, records = _at_the_boundary()
        model = _Model()

        @nnx.jit
        def train_step(model: _Model, pipeline: Pipeline) -> jax.Array:
            def loss(model: _Model, pipeline: Pipeline) -> jax.Array:
                return model.weight[...] * pipeline.step()["x"].sum()

            return nnx.grad(loss)(model, pipeline).weight[...]

        assert float(train_step(model, pipeline)) == pytest.approx(2.0 * records.sum())
        assert (int(pipeline._position[...]), int(pipeline._epoch[...])) == (2, 1)

    def test_the_gradient_reaches_the_stage(self) -> None:
        pipeline, records = _at_the_boundary()
        grads = nnx.grad(lambda pipeline: pipeline.step()["x"].sum())(pipeline)
        (factor,) = jax.tree.leaves(nnx.state(grads, nnx.Param))
        assert float(factor) == pytest.approx(records.sum())

    def test_crossing_epochs_adds_no_compiled_step_or_dispatch_entry(self) -> None:
        pipeline, _ = _at_the_boundary()
        session = Pipeline(
            source=_source(10, shuffle=True),
            stages=[_Scale()],
            batch_size=4,
            rngs=nnx.Rngs(0),
            num_epochs=None,
        ).session()
        next(session)
        steps = len(iteration._SESSION_STEPS)
        entries = session._pure_step._cache_size()
        for _ in range(12):  # 48 records: four epoch boundaries
            next(session)
            pipeline.step()
        assert len(iteration._SESSION_STEPS) == steps
        assert session._pure_step._cache_size() == entries

    def test_under_nnx_vmap_with_the_pipeline_broadcast(self) -> None:
        pipeline, records = _at_the_boundary()

        @nnx.vmap(in_axes=(None, 0))
        def scaled(pipeline: Pipeline, factor: jax.Array) -> jax.Array:
            return factor * pipeline.step()["x"].sum()

        out = np.asarray(scaled(pipeline, jnp.arange(3.0)))
        np.testing.assert_allclose(out, np.arange(3.0) * 2.0 * records.sum(), rtol=1e-6)


class TestScan:
    @pytest.mark.parametrize("drop_last", [False, True])
    def test_scan_rolls_over_as_step_does(self, drop_last: bool) -> None:
        scanned = _pipeline(shuffle=True, drop_last=drop_last).scan(
            lambda batch: batch["x"], length=9
        )
        stepper = _pipeline(shuffle=True, drop_last=drop_last)
        stepped = [np.asarray(stepper.step()["x"]) for _ in range(9)]

        np.testing.assert_array_equal(np.asarray(scanned), np.stack(stepped))


def _augment(element, key):
    noise = jax.random.normal(key, element.data["x"].shape)
    return element.update_data({"x": element.data["x"] + noise})


def _augmented(*, drop_last: bool) -> Pipeline:
    """A pipeline over zeros, so each served value is its record's augmentation."""
    operator = ElementOperator(
        ElementOperatorConfig(stochastic=True, stream_name="aug"),
        fn=_augment,
        rngs=nnx.Rngs(aug=7),
    )
    source = MemorySource(MemorySourceConfig(shuffle=False), data={"x": np.zeros(10, np.float32)})
    return Pipeline(
        source=source, stages=[operator], batch_size=4, rngs=nnx.Rngs(0), drop_last=drop_last
    )


class TestPerRecordEpoch:
    def test_a_head_record_is_augmented_as_its_own_epoch(self) -> None:
        """A record the boundary batch takes from epoch 1 is keyed on epoch 1, not 0."""
        crossing = _augmented(drop_last=False)
        crossing.step()
        crossing.step()
        boundary = np.asarray(crossing.step()["x"])  # records 8, 9 of epoch 0; 0, 1 of epoch 1
        reference = _augmented(drop_last=True)
        reference.reset()

        np.testing.assert_array_equal(boundary[2:], np.asarray(reference.step()["x"])[:2])
        epoch_zero_head = np.asarray(_augmented(drop_last=True).step()["x"])[:2]
        assert not np.allclose(boundary[2:], epoch_zero_head)


class TestCheckpointEveryNBatches:
    """A checkpoint taken after any number of batches resumes exactly."""

    @pytest.mark.parametrize("drop_last", [False, True])
    @pytest.mark.parametrize("every", [1, 2, 3, 7])
    def test_a_session_resumes_exactly(self, drop_last: bool, every: int) -> None:
        def session() -> PipelineIterator:
            return _pipeline(
                10, shuffle=True, batch_size=4, drop_last=drop_last, num_epochs=None
            ).session()

        running = session()
        for _ in range(every):
            next(running)
        state = running.get_state()
        expected = _rows(list(islice(running, 6)))
        resumed = session()
        resumed.set_state(state)
        np.testing.assert_array_equal(_rows(list(islice(resumed, 6))), expected)

    @pytest.mark.parametrize("drop_last", [False, True])
    @pytest.mark.parametrize("every", [1, 2, 3, 7])
    def test_a_scan_resumes_exactly(self, drop_last: bool, every: int) -> None:
        def pipeline() -> Pipeline:
            return _pipeline(10, shuffle=True, batch_size=4, drop_last=drop_last)

        running = pipeline()
        running.scan(lambda batch: batch["x"], length=every)
        saved = nnx.state(nnx.clone(running), nnx.Variable, graph=True)
        expected = running.scan(lambda batch: batch["x"], length=6)
        resumed = pipeline()
        nnx.update(resumed, saved)
        np.testing.assert_array_equal(
            np.asarray(resumed.scan(lambda batch: batch["x"], length=6)), np.asarray(expected)
        )
