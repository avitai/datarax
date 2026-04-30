"""End-to-end integration contracts for ``Pipeline``.

These tests cover behaviours that an integrating consumer (sibling
repos, downstream training scripts) relies on but that the more-focused
unit tests in ``test_pipeline_spike.py``, ``test_pipeline_scan.py``,
``test_pipeline_dag.py``, and ``test_pipeline_subclass.py`` do not
exercise:

A. Multi-epoch iteration:

   1. ``test_pipeline_multi_epoch_resets_to_deterministic_outputs`` —
      resetting ``_position`` and re-running ``scan`` reproduces the
      same per-step outputs.

B. Checkpointing (state round-trip):

   2. ``test_pipeline_state_round_trips_through_split_merge_after_iteration``
      — ``nnx.split`` then ``nnx.merge`` after several steps preserves
      ``_position`` and the rngs counter, so the rebuilt pipeline picks
      up exactly where the original left off.

C. Edge cases:

   3. ``test_pipeline_handles_empty_stages`` — Tier-1 with an empty
      ``stages`` list passes batches through unchanged.
   4. ``test_pipeline_handles_batch_size_larger_than_source`` — wraps
      around the source.
   5. ``test_pipeline_handles_single_element_source``.

D. Stochastic-stage interaction:

   6. ``test_pipeline_stochastic_stage_advances_across_steps`` — a
      stage that consumes from its own rngs sees a different key on
      each step (no key reuse).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax.pipeline import Pipeline
from datarax.sources.memory_source import MemorySource, MemorySourceConfig


def _source(num: int) -> MemorySource:
    return MemorySource(
        MemorySourceConfig(shuffle=False),
        {"x": jnp.arange(num, dtype=jnp.float32)},
    )


# ---------- A. Multi-epoch ----------


def test_pipeline_multi_epoch_resets_to_deterministic_outputs() -> None:
    pipeline = Pipeline(
        source=_source(16),
        stages=[],
        batch_size=4,
        rngs=nnx.Rngs(0),
    )

    def step_fn(batch: dict) -> jax.Array:
        return jnp.sum(batch["x"])

    epoch_1 = pipeline.scan(step_fn, length=4)
    pipeline._position[...] = jnp.int32(0)
    epoch_2 = pipeline.scan(step_fn, length=4)

    np.testing.assert_array_equal(np.asarray(epoch_1), np.asarray(epoch_2))


# ---------- B. Checkpoint round-trip ----------


def test_pipeline_state_round_trips_through_split_merge_after_iteration() -> None:
    """After several steps, a checkpointed state restores to the same position.

    Uses the canonical Orbax-style round-trip: ``nnx.to_pure_dict`` produces
    a checkpoint-friendly PyTree of arrays (deep copy semantics), and
    ``nnx.from_pure_dict`` + ``nnx.merge`` rebuilds an independent pipeline.
    """
    pipeline_a = Pipeline(
        source=_source(16),
        stages=[],
        batch_size=4,
        rngs=nnx.Rngs(0),
    )

    # Advance by 3 steps (positions 0, 4, 8 → next batch starts at 12)
    for _ in range(3):
        pipeline_a.step()  # type: ignore[reportCallIssue]

    # Capture state as a pure-dict checkpoint (deep-copies the variable values).
    graphdef, state = nnx.split(pipeline_a)
    checkpoint = nnx.to_pure_dict(state)

    # Rebuild from the checkpoint — independent of pipeline_a's variables.
    # We construct a fresh pipeline first, then replace its state with the
    # restored checkpoint values via nnx.replace_by_pure_dict.
    pipeline_b = Pipeline(
        source=_source(16),
        stages=[],
        batch_size=4,
        rngs=nnx.Rngs(0),
    )
    _, state_b = nnx.split(pipeline_b)
    nnx.replace_by_pure_dict(state_b, checkpoint)
    pipeline_b = nnx.merge(graphdef, state_b)

    # Rebuilt pipeline carries the saved position.
    assert int(pipeline_b._position[...]) == 12

    # Next step on the rebuilt pipeline produces the expected batch from position 12.
    next_b = pipeline_b.step()  # type: ignore[reportCallIssue]
    np.testing.assert_array_equal(np.asarray(next_b["x"]), np.array([12.0, 13.0, 14.0, 15.0]))


# ---------- C. Edge cases ----------


def test_pipeline_handles_empty_stages() -> None:
    pipeline = Pipeline(
        source=_source(8),
        stages=[],
        batch_size=4,
        rngs=nnx.Rngs(0),
    )

    out = pipeline.step()  # type: ignore[reportCallIssue]
    np.testing.assert_array_equal(np.asarray(out["x"]), np.array([0.0, 1.0, 2.0, 3.0]))


def test_pipeline_handles_batch_size_larger_than_source() -> None:
    pipeline = Pipeline(
        source=_source(4),
        stages=[],
        batch_size=8,
        rngs=nnx.Rngs(0),
    )

    out = pipeline.step()  # type: ignore[reportCallIssue]
    # batch_size=8 over a source of length 4 wraps: [0, 1, 2, 3, 0, 1, 2, 3]
    np.testing.assert_array_equal(
        np.asarray(out["x"]),
        np.array([0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]),
    )


def test_pipeline_handles_single_element_source() -> None:
    pipeline = Pipeline(
        source=_source(1),
        stages=[],
        batch_size=4,
        rngs=nnx.Rngs(0),
    )

    out = pipeline.step()  # type: ignore[reportCallIssue]
    # Source length 1 → every position wraps to index 0
    np.testing.assert_array_equal(np.asarray(out["x"]), np.array([0.0, 0.0, 0.0, 0.0]))


# ---------- D. Stochastic-stage interaction ----------


def test_pipeline_stochastic_stage_advances_across_steps() -> None:
    """A stage that consumes its own rngs sees a different key per step."""

    class _NoiseStage(nnx.Module):
        def __init__(self, *, rngs: nnx.Rngs) -> None:
            super().__init__()
            self.rngs = rngs
            self.last_key = nnx.Variable(jnp.zeros((2,), dtype=jnp.uint32))

        def __call__(self, batch: dict) -> dict:
            k = self.rngs()
            # Capture the raw key bits as a u32[2] so we can compare across steps.
            self.last_key[...] = jax.random.key_data(k)
            return batch

    stage = _NoiseStage(rngs=nnx.Rngs(42))
    pipeline = Pipeline(
        source=_source(16),
        stages=[stage],
        batch_size=4,
        rngs=nnx.Rngs(0),
    )

    pipeline.step()  # type: ignore[reportCallIssue]
    key_after_step_1 = np.asarray(stage.last_key[...]).copy()

    pipeline.step()  # type: ignore[reportCallIssue]
    key_after_step_2 = np.asarray(stage.last_key[...]).copy()

    assert not np.array_equal(key_after_step_1, key_after_step_2), (
        "stochastic stage must see a different rngs key on each step"
    )
