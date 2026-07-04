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

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.pipeline import Pipeline, PipelineIterator
from datarax.pipeline.iteration import _session_cache_size
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
        assert type(state["position"]) is np.int64
        assert isinstance(state["rng_counts"], list)
        assert all(isinstance(c, int) for c in state["rng_counts"])

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
        assert _session_cache_size(pipeline) == 1

    def test_structural_change_compiles_fresh_session(self):
        """Static attributes (e.g. batch_size) key the compiled session."""
        pipeline = _pipeline()
        iterator = _session(pipeline)
        first = np.asarray(next(iterator)["x"])
        iterator.close()
        assert first.shape[0] == _BATCH

        pipeline.batch_size = _BATCH // 2
        iterator = _session(pipeline)
        smaller = np.asarray(next(iterator)["x"])
        iterator.close()
        assert smaller.shape[0] == _BATCH // 2
        assert _session_cache_size(pipeline) == 2


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
