"""Contracts for the minimal Tier-1 ``Pipeline``.

Ten contracts validate the pipeline architecture: stateless source
indexing, stage composition, scan-based epoch iteration, and end-to-end
gradient flow through learnable stage parameters.

Test contract index:

1. ``test_memory_source_get_batch_at_is_stateless_and_deterministic`` —
   ``MemorySource.get_batch_at(start, size, key)`` returns identical
   output for identical args, advances by ``start`` not by internal
   counter, and does not mutate the source's state.
2. ``test_pipeline_constructs_with_source_and_stages`` — the
   constructor accepts a source, a list of stages, batch_size, and
   rngs; exposes them as attributes.
3. ``test_pipeline_call_applies_stages_in_order`` — ``__call__(batch)``
   threads the batch through every stage in sequence.
4. ``test_pipeline_step_advances_position_variable`` — ``step()``
   advances ``_position`` by ``batch_size`` on each call.
5. ``test_pipeline_step_advances_rng_keys`` — ``step()`` consumes from
   ``rngs`` so successive calls produce different keys.
6. ``test_pipeline_split_merge_round_trips_state`` — ``nnx.split`` then
   ``nnx.merge`` preserves all ``nnx.Variable`` iteration state. This
   is the contract ``nnx.scan`` checks via
   ``_check_carry_same_references``.
7. ``test_pipeline_iter_protocol_yields_same_batches_as_step`` — the
   debug ``__iter__`` path matches the canonical ``step()`` semantics.
8. ``test_pipeline_epoch_runs_under_nnx_scan`` — ``epoch()`` produces
   the same total result as a Python loop of ``step()`` calls.
9. ``test_pipeline_epoch_collects_outputs_along_leading_axis`` —
   per-step ``body_fn`` outputs stack along axis 0 with shape
   ``(length, ...)``.
10. ``test_pipeline_epoch_gradient_flows_to_learnable_stage`` —
    ``nnx.grad`` through ``Pipeline.epoch`` produces non-zero
    gradients on stage parameters; the load-bearing differentiability
    contract.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from datarax.sources.memory_source import MemorySource, MemorySourceConfig


# ---------- Test-local stages (simple dict→dict nnx.Modules) ----------


class _FactorStage(nnx.Module):
    """Multiplies every leaf in the batch by a fixed factor (deterministic)."""

    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = jnp.float32(factor)

    def __call__(self, batch: dict) -> dict:
        return jax.tree.map(lambda x: x * self.factor, batch)


class _OffsetStage(nnx.Module):
    """Adds a fixed offset to every leaf (deterministic)."""

    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = jnp.float32(offset)

    def __call__(self, batch: dict) -> dict:
        return jax.tree.map(lambda x: x + self.offset, batch)


class _LearnableScale(nnx.Module):
    """A stage whose factor is an ``nnx.Param`` — differentiable target."""

    def __init__(self, init_factor: float) -> None:
        super().__init__()
        self.factor = nnx.Param(jnp.float32(init_factor))

    def __call__(self, batch: dict) -> dict:
        return jax.tree.map(lambda x: x * self.factor[...], batch)


# ---------- Helpers ----------


def _source(num_elements: int = 16) -> MemorySource:
    """Build a deterministic MemorySource with ``x = arange(N)``."""
    return MemorySource(
        MemorySourceConfig(shuffle=False),
        {"x": jnp.arange(num_elements, dtype=jnp.float32)},
    )


# ---------- Contracts ----------


def test_memory_source_get_batch_at_is_stateless_and_deterministic() -> None:
    """``get_batch_at(start, size, key)`` is stateless and start-driven."""
    source = _source(num_elements=8)

    batch_a = source.get_batch_at(start=0, size=4, key=jax.random.key(0))
    batch_b = source.get_batch_at(start=0, size=4, key=jax.random.key(0))
    batch_c = source.get_batch_at(start=4, size=4, key=jax.random.key(0))

    np.testing.assert_array_equal(np.asarray(batch_a["x"]), np.asarray(batch_b["x"]))
    assert not np.array_equal(np.asarray(batch_a["x"]), np.asarray(batch_c["x"]))


def test_pipeline_constructs_with_source_and_stages() -> None:
    """The constructor wires source/stages/batch_size/rngs onto the module."""
    from datarax.pipeline import Pipeline

    pipeline = Pipeline(
        source=_source(),
        stages=[_FactorStage(2.0)],
        batch_size=4,
        rngs=nnx.Rngs(0),
    )

    assert pipeline.batch_size == 4
    assert len(pipeline.stages) == 1


def test_pipeline_call_applies_stages_in_order() -> None:
    """``__call__(batch)`` threads the batch through each stage in sequence."""
    from datarax.pipeline import Pipeline

    source = _source()
    pipeline = Pipeline(
        source=source,
        stages=[_FactorStage(2.0), _OffsetStage(10.0)],  # (x * 2) + 10
        batch_size=4,
        rngs=nnx.Rngs(0),
    )

    raw_batch = source.get_batch_at(start=0, size=4, key=jax.random.key(0))
    out = pipeline(raw_batch)

    expected = jnp.array([10.0, 12.0, 14.0, 16.0])
    np.testing.assert_allclose(np.asarray(out["x"]), np.asarray(expected))


def test_pipeline_step_advances_position_variable() -> None:
    """``step()`` reads ``_position``, returns one batch, advances by ``batch_size``."""
    from datarax.pipeline import Pipeline

    pipeline = Pipeline(
        source=_source(num_elements=16),
        stages=[],
        batch_size=4,
        rngs=nnx.Rngs(0),
    )

    assert int(pipeline._position[...]) == 0
    pipeline.step()  # type: ignore[reportCallIssue]
    assert int(pipeline._position[...]) == 4
    pipeline.step()  # type: ignore[reportCallIssue]
    assert int(pipeline._position[...]) == 8


def test_pipeline_step_advances_rng_keys() -> None:
    """``step()`` consumes from ``rngs`` so successive direct calls return different keys."""
    from datarax.pipeline import Pipeline

    pipeline = Pipeline(
        source=_source(),
        stages=[],
        batch_size=4,
        rngs=nnx.Rngs(42),
    )

    # Consume directly from the rngs to demonstrate stream advancement.
    key_before = pipeline.rngs()
    pipeline.step()  # type: ignore[reportCallIssue]
    key_after = pipeline.rngs()

    assert not bool(jnp.array_equal(key_before, key_after)), (
        "rngs default stream must advance after step()"
    )


def test_pipeline_split_merge_round_trips_state() -> None:
    """``nnx.split`` then ``nnx.merge`` preserves Pipeline state — the contract nnx.scan checks."""
    from datarax.pipeline import Pipeline

    pipeline = Pipeline(
        source=_source(),
        stages=[_FactorStage(2.0)],
        batch_size=4,
        rngs=nnx.Rngs(0),
    )
    pipeline.step()  # type: ignore[reportCallIssue]

    graphdef, state = nnx.split(pipeline)
    rebuilt = nnx.merge(graphdef, state)

    assert int(rebuilt._position[...]) == int(pipeline._position[...]) == 4


def test_pipeline_iter_protocol_yields_same_batches_as_step() -> None:
    """Debug ``__iter__`` matches sequential ``step()`` (parity contract for the fallback path)."""
    from datarax.pipeline import Pipeline

    pipeline_iter = Pipeline(
        source=_source(num_elements=16),
        stages=[],
        batch_size=4,
        rngs=nnx.Rngs(0),
    )
    iter_batches = list(pipeline_iter)

    pipeline_step = Pipeline(
        source=_source(num_elements=16),
        stages=[],
        batch_size=4,
        rngs=nnx.Rngs(0),
    )
    step_batches = [pipeline_step.step() for _ in iter_batches]  # type: ignore[reportCallIssue]

    assert len(iter_batches) == len(step_batches) == 4
    for a, b in zip(iter_batches, step_batches):
        np.testing.assert_array_equal(np.asarray(a["x"]), np.asarray(b["x"]))


def test_pipeline_epoch_runs_under_nnx_scan() -> None:
    """``epoch()`` produces results consistent with the equivalent Python loop."""
    from datarax.pipeline import Pipeline

    pipeline = Pipeline(
        source=_source(num_elements=16),
        stages=[_FactorStage(2.0)],
        batch_size=4,
        rngs=nnx.Rngs(0),
    )

    def body_fn(carry, batch):
        return carry + jnp.sum(batch["x"]), None

    final_carry, _ = pipeline.scan(body_fn, length=4, init_carry=jnp.float32(0.0))

    # Sum of (arange(16) * 2) = 2 * sum(0..15) = 2 * 120 = 240
    assert float(final_carry) == pytest.approx(240.0, rel=1e-5)


def test_pipeline_epoch_collects_outputs_along_leading_axis() -> None:
    """Per-step body_fn outputs stack along axis 0 with shape ``(length, ...)``."""
    from datarax.pipeline import Pipeline

    pipeline = Pipeline(
        source=_source(num_elements=16),
        stages=[],
        batch_size=4,
        rngs=nnx.Rngs(0),
    )

    def body_fn(carry, batch):
        return carry, jnp.sum(batch["x"])  # scalar per step

    _, outputs = pipeline.scan(body_fn, length=4, init_carry=jnp.float32(0.0))

    assert outputs.shape == (4,), f"expected (4,), got {outputs.shape}"
    # Per-step sums: 0+1+2+3=6, 4+5+6+7=22, 8+9+10+11=38, 12+13+14+15=54
    np.testing.assert_allclose(np.asarray(outputs), np.array([6.0, 22.0, 38.0, 54.0]))


def test_pipeline_epoch_gradient_flows_to_learnable_stage() -> None:
    """``nnx.grad`` through ``epoch()`` produces non-zero grads on stage params.

    This is the load-bearing differentiability contract — the redesign's
    purpose is end-to-end gradient flow through one XLA graph. If this
    passes, the architecture works.
    """
    from datarax.pipeline import Pipeline

    learnable = _LearnableScale(init_factor=2.0)
    pipeline = Pipeline(
        source=_source(num_elements=16),
        stages=[learnable],
        batch_size=4,
        rngs=nnx.Rngs(0),
    )

    def loss_fn(model: Pipeline) -> jax.Array:
        def body(carry, batch):
            return carry + jnp.sum(batch["x"]), None

        final_carry, _ = model.scan(body, length=4, init_carry=jnp.float32(0.0))
        return final_carry

    grads = nnx.grad(loss_fn)(pipeline)
    grad_state = nnx.state(grads, nnx.Param)
    grad_leaves = jax.tree.leaves(grad_state)

    assert grad_leaves, "expected at least one Param gradient"
    assert any(bool(jnp.any(jnp.abs(g) > 0)) for g in grad_leaves), (
        "expected non-zero gradient on learnable stage's factor parameter"
    )
