"""Contracts for the compiled pipeline iteration session.

``iter(pipeline)`` over a random-access source returns a
``PipelineIterator``: a compiled fast loop that hoists the NNX
module-graph traversal out of the per-batch path. The contracts below pin
its equivalence with the plain ``step()`` path and its state semantics:

- Outputs are identical to stepping the pipeline directly, including
  stochastic pipelines (RNG counts carried through the session).
- Module state (position, RNG counts) is written back on exhaustion,
  ``close()``, garbage collection after an early break, and exceptions —
  so ``nnx.split``/checkpointing after leaving the loop sees the truth.
- ``get_state()``/``set_state()`` expose grain-style iterator state that
  is valid at every yield boundary and supports exact mid-epoch resume.
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.pipeline import Pipeline, PipelineIterator
from datarax.sources.memory_source import MemorySource, MemorySourceConfig


_N = 256
_BATCH = 32


def _data(n: int = _N) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    return {"x": rng.standard_normal((n, 8)).astype(np.float32)}


def _normalize(element, key=None):
    return element.update_data({"x": element.data["x"] * 2.0})


def _jitter(element, key):
    noise = jax.random.normal(key, element.data["x"].shape) * 0.1
    return element.update_data({"x": element.data["x"] + noise})


def _pipeline(*, stochastic: bool = False, n: int = _N, seed: int = 0) -> Pipeline:
    source = MemorySource(MemorySourceConfig(shuffle=False), data=_data(n), rngs=nnx.Rngs(seed))
    if stochastic:
        stage = ElementOperator(
            ElementOperatorConfig(stochastic=True, stream_name="jitter"),
            fn=_jitter,
            rngs=nnx.Rngs(seed, jitter=seed + 1),
        )
    else:
        stage = ElementOperator(
            ElementOperatorConfig(stochastic=False), fn=_normalize, rngs=nnx.Rngs(seed)
        )
    return Pipeline(source=source, stages=[stage], batch_size=_BATCH, rngs=nnx.Rngs(seed))


class _RunningTotal(nnx.Module):
    """Stage accumulating each batch's sum in batch statistics."""

    def __init__(self) -> None:
        self.total = nnx.BatchStat(jnp.zeros((), jnp.float32))

    def __call__(self, batch: dict) -> dict:
        self.total[...] = self.total[...] + batch["x"].sum()
        return batch


class _Scale(nnx.Module):
    """Learnable stage multiplying ``x`` by a parameter."""

    def __init__(self) -> None:
        self.factor = nnx.Param(jnp.float32(1.0))

    def __call__(self, batch: dict) -> dict:
        return {**batch, "x": batch["x"] * self.factor[...]}


class _GrowingStage(nnx.Module):
    """Stage adding state while it runs, which changes the module structure."""

    def __call__(self, batch: dict) -> dict:
        self.seen = nnx.Variable(jnp.zeros((), jnp.int32))
        return batch


def _pipeline_with(stage: nnx.Module) -> Pipeline:
    source = MemorySource(MemorySourceConfig(shuffle=False), data=_data(), rngs=nnx.Rngs(0))
    return Pipeline(source=source, stages=[stage], batch_size=_BATCH, rngs=nnx.Rngs(0))


def _session(pipeline: Pipeline) -> PipelineIterator:
    """iter() narrowed to the random-access session type."""
    iterator = iter(pipeline)
    assert isinstance(iterator, PipelineIterator)
    return iterator


def _epoch_via_step(pipeline: Pipeline) -> list[np.ndarray]:
    batches = []
    length = len(pipeline.source)
    while int(pipeline._position[...]) < length:
        batches.append(np.asarray(pipeline.step()["x"]))  # type: ignore[call-arg]
    return batches


# ---------------------------------------------------------------------------
# Equivalence with the plain step() path
# ---------------------------------------------------------------------------


class TestOutputEquivalence:
    """The compiled session reproduces step() outputs exactly."""

    def test_iterator_returns_pipeline_iterator(self):
        iterator = iter(_pipeline())
        assert isinstance(iterator, PipelineIterator)

    def test_deterministic_epoch_matches_step(self):
        expected = _epoch_via_step(_pipeline())
        got = [np.asarray(b["x"]) for b in _pipeline()]
        assert len(got) == len(expected) == _N // _BATCH
        for g, e in zip(got, expected):
            np.testing.assert_array_equal(g, e)

    def test_stochastic_epoch_matches_step(self):
        """RNG counts advance identically inside the compiled session."""
        expected = _epoch_via_step(_pipeline(stochastic=True))
        got = [np.asarray(b["x"]) for b in _pipeline(stochastic=True)]
        for g, e in zip(got, expected):
            np.testing.assert_array_equal(g, e)

    def test_wraparound_final_batch_matches_step(self):
        """Non-divisible dataset sizes wrap exactly like step()."""
        expected = _epoch_via_step(_pipeline(n=100))
        got = [np.asarray(b["x"]) for b in _pipeline(n=100)]
        assert len(got) == len(expected) == 4  # ceil(100 / 32)
        for g, e in zip(got, expected):
            np.testing.assert_array_equal(g, e)


# ---------------------------------------------------------------------------
# Module state write-back
# ---------------------------------------------------------------------------


class TestModuleStateWriteBack:
    """The live module reflects the session after any form of exit."""

    def test_position_after_exhaustion(self):
        pipeline = _pipeline()
        for _ in pipeline:
            pass
        assert int(pipeline._position[...]) == _N

    def test_position_after_explicit_close(self):
        pipeline = _pipeline()
        iterator = _session(pipeline)
        next(iterator)
        next(iterator)
        iterator.close()
        assert int(pipeline._position[...]) == 2 * _BATCH

    def test_close_is_idempotent_and_stops_iteration(self):
        pipeline = _pipeline()
        iterator = _session(pipeline)
        next(iterator)
        iterator.close()
        iterator.close()
        try:
            next(iterator)
            raise AssertionError("expected StopIteration after close")
        except StopIteration:
            pass

    def test_position_after_early_break(self):
        """A bare for/break writes back via prompt garbage collection."""
        pipeline = _pipeline()
        for index, _ in enumerate(pipeline):
            if index == 2:
                break
        assert int(pipeline._position[...]) == 3 * _BATCH

    def test_rng_counts_written_back(self):
        """After a session, manual step() continues the exact RNG stream."""
        via_iterator = _pipeline(stochastic=True)
        for index, _ in enumerate(via_iterator):
            if index == 3:
                break
        continued = np.asarray(via_iterator.step()["x"])  # type: ignore[call-arg]

        via_step = _pipeline(stochastic=True)
        for _ in range(4):
            via_step.step()  # type: ignore[call-arg]  # type: ignore[call-arg]
        expected = np.asarray(via_step.step()["x"])  # type: ignore[call-arg]
        np.testing.assert_array_equal(continued, expected)

    def test_sequential_reiteration_resumes_from_position(self):
        pipeline = _pipeline()
        first = [np.asarray(b["x"]) for b in pipeline]  # full epoch
        assert len(first) == _N // _BATCH
        again = [np.asarray(b["x"]) for b in pipeline]  # exhausted: no batches
        assert again == []
        pipeline._position[...] = jnp.int32(0)
        rewound = [np.asarray(b["x"]) for b in pipeline]
        assert len(rewound) == len(first)

    def test_checkpoint_round_trip_after_partial_iteration(self):
        """split/merge after leaving the loop resumes identically."""
        pipeline = _pipeline(stochastic=True)
        for index, _ in enumerate(pipeline):
            if index == 3:
                break
        graphdef, state = nnx.split(pipeline)
        restored = nnx.merge(graphdef, state)
        rest_restored = [np.asarray(b["x"]) for b in restored]

        reference = _pipeline(stochastic=True)
        for index, _ in enumerate(reference):
            if index == 3:
                break
        rest_reference = [np.asarray(b["x"]) for b in reference]
        assert len(rest_restored) == len(rest_reference)
        for g, e in zip(rest_restored, rest_reference):
            np.testing.assert_array_equal(g, e)


# ---------------------------------------------------------------------------
# Iterator state (grain-style get_state/set_state)
# ---------------------------------------------------------------------------


class TestIteratorState:
    """Iterator-owned state is valid at yield boundaries and resumable."""

    def test_get_state_shape(self):
        iterator = _session(_pipeline(stochastic=True))
        next(iterator)
        state = iterator.get_state()
        assert isinstance(state, dict)
        assert state["position"] == _BATCH
        assert type(state["position"]) is int
        assert isinstance(state["rng_counts"], list)
        assert all(isinstance(c, int) for c in state["rng_counts"])

    def test_state_survives_a_json_round_trip(self):
        """The state is JSON: what ``json.loads`` gives back resumes the same batches."""
        reference = _session(_pipeline(stochastic=True))
        for _ in range(3):
            next(reference)
        checkpoint = json.loads(json.dumps(reference.get_state()))
        expected = [np.asarray(next(reference)["x"]) for _ in range(3)]

        resumed = _session(_pipeline(stochastic=True))
        resumed.set_state(checkpoint)
        got = [np.asarray(next(resumed)["x"]) for _ in range(3)]
        for g, e in zip(got, expected, strict=True):
            np.testing.assert_array_equal(g, e)

    def test_state_shapes_stable_across_life(self):
        iterator = _session(_pipeline(stochastic=True))
        next(iterator)
        first = iterator.get_state()
        next(iterator)
        second = iterator.get_state()
        assert set(first) == set(second)
        assert len(first["rng_counts"]) == len(second["rng_counts"])

    def test_set_state_resumes_exactly(self):
        reference = _session(_pipeline(stochastic=True))
        for _ in range(3):
            next(reference)
        checkpoint = reference.get_state()
        expected = [np.asarray(next(reference)["x"]) for _ in range(3)]

        resumed = _session(_pipeline(stochastic=True))
        resumed.set_state(checkpoint)
        got = [np.asarray(next(resumed)["x"]) for _ in range(3)]
        for g, e in zip(got, expected):
            np.testing.assert_array_equal(g, e)

    def test_set_state_refuses_a_negative_position(self):
        iterator = _session(_pipeline(stochastic=True))
        next(iterator)
        state = iterator.get_state()
        state["position"] = -1

        with pytest.raises(ValueError, match="position"):
            iterator.set_state(state)

    def test_set_state_refuses_a_negative_epoch(self):
        iterator = _session(_pipeline(stochastic=True))
        next(iterator)
        state = iterator.get_state()
        state["epoch"] = -1

        with pytest.raises(ValueError, match="epoch"):
            iterator.set_state(state)


# ---------------------------------------------------------------------------
# Stage state
# ---------------------------------------------------------------------------


class TestStageState:
    """Every Variable a stage writes reaches the live module, as with step()."""

    def test_batch_statistics_written_by_a_stage_match_step(self):
        iterated_stage, stepped_stage = _RunningTotal(), _RunningTotal()
        for _ in _pipeline_with(iterated_stage):
            pass
        stepped = _pipeline_with(stepped_stage)
        for _ in range(_N // _BATCH):
            stepped.step()  # type: ignore[call-arg]

        assert float(stepped_stage.total[...]) != 0.0
        assert float(iterated_stage.total[...]) == float(stepped_stage.total[...])

    def test_a_stage_changing_the_module_structure_is_refused(self):
        with pytest.raises(ValueError, match="changed the module structure"):
            for _ in _pipeline_with(_GrowingStage()):
                pass

    def test_the_compiled_step_returns_only_the_state_it_writes(self):
        """Source payloads and RNG keys are never copied out of the step."""
        iterator = _session(_pipeline())
        _, (per_batch, staged) = iterator._pure_step(iterator._state, iterator._immutable_state)
        iterator.close()

        assert staged == {}
        assert iterator._position_index in per_batch


# ---------------------------------------------------------------------------
# Session/compile behavior
# ---------------------------------------------------------------------------


class TestSessionBehavior:
    """Sessions share compiled steps and honor structural changes."""

    def test_sessions_reuse_compiled_step(self):
        pipeline = _pipeline()
        first = _session(pipeline)
        next(first)
        first.close()
        second = _session(pipeline)
        next(second)
        second.close()
        assert second._pure_step is first._pure_step
        assert second._pure_step._cache_size() == 1

    def test_a_pipeline_of_the_same_structure_reuses_the_compiled_step(self):
        """A new pipeline instance with an identical module graph compiles nothing.

        Building a pipeline per epoch, fold or evaluation must not recompile each time.
        """
        first = _session(_pipeline())
        next(first)
        first.close()
        second = _session(_pipeline())
        next(second)
        second.close()
        assert second._pure_step is first._pure_step
        assert second._pure_step._cache_size() == 1

    def test_structural_change_compiles_fresh_session(self):
        """Static attributes (e.g. batch_size) key the compiled session."""
        pipeline = _pipeline()
        iterator = _session(pipeline)
        first = np.asarray(next(iterator)["x"])
        iterator.close()
        assert first.shape[0] == _BATCH

        pipeline.batch_size = _BATCH // 2
        smaller_session = _session(pipeline)
        smaller = np.asarray(next(smaller_session)["x"])
        smaller_session.close()
        assert smaller.shape[0] == _BATCH // 2
        assert smaller_session._pure_step is not iterator._pure_step


class TestMidLoopCheckpointing:
    """nnx.state(pipeline) inside the loop reflects consumed batches.

    This is the resumable-training pattern: checkpoint the pipeline
    module every N steps without leaving the loop, then resume from the
    snapshot and reproduce the exact remaining batch stream.
    """

    def test_module_state_live_at_yield_boundaries(self):
        pipeline = _pipeline(stochastic=True)
        snapshot = None
        for step, _ in enumerate(pipeline):
            if step == 2:
                snapshot = nnx.to_pure_dict(nnx.state(pipeline))
            if step == 4:
                break
        assert snapshot is not None
        assert int(snapshot["_position"]) == 3 * _BATCH

        restored = _pipeline(stochastic=True)
        state = nnx.state(restored)
        nnx.replace_by_pure_dict(state, snapshot)
        nnx.update(restored, state)
        resumed = [np.asarray(b["x"]) for b in restored]

        reference = _pipeline(stochastic=True)
        for step, _ in enumerate(reference):
            if step == 2:
                break
        rest = [np.asarray(b["x"]) for b in reference]
        assert len(resumed) == len(rest)
        for got, expected in zip(resumed, rest, strict=True):
            np.testing.assert_array_equal(got, expected)


class TestSessionRetracing:
    """New sessions reuse the compiled trace, not just the cached function.

    A fresh module presents host-typed leaves while later sessions carry
    device arrays; without canonicalization each session re-traced the
    jitted step (tens of milliseconds per iter() call).
    """

    def test_second_session_does_not_retrace(self):
        pipeline = _pipeline(stochastic=True)
        first = _session(pipeline)
        next(first)
        first.close()
        traces_after_first = first._pure_step._cache_size()

        second = _session(pipeline)
        next(second)
        second.close()
        assert second._pure_step is first._pure_step
        assert second._pure_step._cache_size() == traces_after_first == 1


class TestImmutableStaging:
    """Immutable state is device-staged once per pipeline, not per session.

    Re-staging per session re-uploads the entire source dataset to the
    device on every iter() call (tens of milliseconds per session for
    real datasets).
    """

    def test_second_session_reuses_staged_immutable(self):
        pipeline = _pipeline()
        first = _session(pipeline)
        next(first)
        first.close()
        second = _session(pipeline)
        next(second)
        second.close()
        first_leaves = jax.tree.leaves(
            first._immutable_state, is_leaf=lambda x: isinstance(x, nnx.Variable)
        )
        second_leaves = jax.tree.leaves(
            second._immutable_state, is_leaf=lambda x: isinstance(x, nnx.Variable)
        )
        assert all(a is b for a, b in zip(first_leaves, second_leaves, strict=True))

    def test_a_value_written_into_a_stage_parameter_reaches_the_next_session(self):
        """A write keeps the Variable object and replaces its value; staging must see it.

        Training updates a learnable stage's parameters between epochs; a session serving the
        staged copy would transform every later epoch with the first epoch's parameters.
        """
        stage = _Scale()
        pipeline = _pipeline_with(stage)
        first = np.asarray(next(_session(pipeline))["x"])

        pipeline.reset()
        stage.factor[...] = jnp.float32(5.0)
        second = np.asarray(next(_session(pipeline))["x"])

        pipeline.reset()
        stage.factor.set_value(jnp.float32(7.0))
        third = np.asarray(next(_session(pipeline))["x"])

        np.testing.assert_allclose(second, 5.0 * first, rtol=1e-6)
        np.testing.assert_allclose(third, 7.0 * first, rtol=1e-6)

    def test_swapped_source_data_restages(self):
        pipeline = _pipeline()
        first = _session(pipeline)
        batch_before = np.asarray(next(first)["x"])
        first.close()

        pipeline._position[...] = jnp.int32(0)
        source = pipeline.source
        assert isinstance(source, MemorySource)
        source.data = {"x": np.zeros((_N, 8), dtype=np.float32)}
        second = _session(pipeline)
        batch_after = np.asarray(next(second)["x"])
        second.close()
        assert not np.array_equal(batch_before, batch_after)
        np.testing.assert_array_equal(batch_after, np.zeros_like(batch_after))


class TestIteratorRngCounts:
    """``rng_counts`` holds one entry per stochastic operator and none for a deterministic one.

    Pipeline iteration keys each record on the operator's ``_base_key``, so an operator's own
    stream is never drawn from and its count stays 0; the entries that move belong to the
    pipeline and the source. The list's length therefore depends on how many operators are
    stochastic, not on how many streams the caller's ``Rngs`` happened to carry.
    """

    @staticmethod
    def _counts(pipeline: Pipeline) -> list[int]:
        iterator = _session(pipeline)
        counts = iterator.get_state()["rng_counts"]
        iterator.close()
        return counts

    def test_a_deterministic_operator_contributes_no_count(self):
        # The pipeline's own stream, then the source's.
        assert self._counts(_pipeline()) == [1, 0]

    def test_a_stochastic_operator_contributes_exactly_one(self):
        # The operator's private stream, the pipeline's, then the source's.
        assert self._counts(_pipeline(stochastic=True)) == [0, 1, 0]

    def test_operators_sharing_one_rngs_no_longer_share_a_count(self):
        shared = nnx.Rngs(jitter=0)
        stages = [
            ElementOperator(
                ElementOperatorConfig(stochastic=True, stream_name="jitter"),
                fn=_jitter,
                rngs=shared,
            )
            for _ in range(2)
        ]
        source = MemorySource(MemorySourceConfig(shuffle=False), data=_data(), rngs=nnx.Rngs(0))
        pipeline = Pipeline(source=source, stages=stages, batch_size=_BATCH, rngs=nnx.Rngs(0))

        assert self._counts(pipeline) == [0, 0, 1, 0]


class TestIteratorStateVersioning:
    """The iterator state names its format, and a state saved without one is upgraded."""

    def test_get_state_names_its_format_version(self):
        iterator = _session(_pipeline(stochastic=True))
        next(iterator)
        state = iterator.get_state()
        iterator.close()

        assert state["version"] == 2

    def test_a_state_from_the_earlier_layout_resumes_like_its_current_equivalent(self):
        reference = _session(_pipeline(stochastic=True))
        for _ in range(3):
            next(reference)
        current = reference.get_state()
        expected = [np.asarray(next(reference)["x"]) for _ in range(3)]
        reference.close()

        # The operator used to hold the caller's Rngs, whose `default` and `jitter` streams each
        # carried a count ahead of the pipeline's and the source's; it now contributes one entry.
        legacy = {
            "position": current["position"],
            "epoch": current["epoch"],
            "rng_counts": [0, 1, *current["rng_counts"][1:]],
        }

        resumed = _session(_pipeline(stochastic=True))
        resumed.set_state(legacy)
        got = [np.asarray(next(resumed)["x"]) for _ in range(3)]
        resumed.close()

        for g, e in zip(got, expected, strict=True):
            np.testing.assert_array_equal(g, e)

    def test_a_current_state_with_the_wrong_count_length_is_still_refused(self):
        iterator = _session(_pipeline(stochastic=True))
        next(iterator)
        state = iterator.get_state()
        state["rng_counts"] = [*state["rng_counts"], 0]

        with pytest.raises(ValueError, match="rng streams"):
            iterator.set_state(state)
        iterator.close()


class TestStateFingerprint:
    """Iterator state names the configuration that produced it, and ``set_state`` checks it."""

    def test_state_carries_the_configuration(self):
        state = _session(_pipeline()).get_state()

        assert state["version"] == 2
        assert state["fingerprint"] == {
            "batch_size": _BATCH,
            "length": _N,
            "drop_last": False,
            "num_epochs": 1,
            "shuffled": False,
        }

    def test_a_shuffled_source_names_its_order(self):
        source = MemorySource(MemorySourceConfig(shuffle=True), data=_data(), rngs=nnx.Rngs(0))
        pipeline = Pipeline(source=source, stages=[], batch_size=_BATCH, rngs=nnx.Rngs(0))
        assert _session(pipeline).get_state()["fingerprint"]["shuffled"] is True

    @pytest.mark.parametrize(
        ("field", "other"),
        [
            ("batch_size", 16),
            ("length", _N + 1),
            ("drop_last", True),
            ("num_epochs", None),
            ("shuffled", True),
        ],
    )
    def test_set_state_refuses_a_different_configuration(self, field: str, other: object):
        state = _session(_pipeline()).get_state()
        state["fingerprint"][field] = other

        with pytest.raises(ValueError, match=field):
            _session(_pipeline()).set_state(state)

    def test_a_version_1_state_without_a_fingerprint_is_accepted(self):
        session = _session(_pipeline())
        for _ in range(2):
            next(session)
        state = session.get_state()
        session.close()
        legacy = {k: v for k, v in state.items() if k != "fingerprint"}
        legacy["version"] = 1

        resumed = _session(_pipeline())
        resumed.set_state(legacy)
        assert resumed.get_state()["position"] == 2 * _BATCH
