"""Contracts for Tier-3 — subclassing ``Pipeline`` for full topology control.

When the linear (``stages=...``) and DAG (``from_dag``) constructors are
not flexible enough — for runtime conditional branching, dynamic
dispatch, or composing multiple sub-pipelines — users subclass
``Pipeline`` and override ``__call__``. Pipeline is a regular
``nnx.Module``, so subclassing follows the standard Flax NNX pattern.

Test contract index:

1. ``test_subclass_can_override_call_for_custom_topology`` — a subclass
   that ignores ``stages`` and writes its own ``__call__`` produces the
   expected output.
2. ``test_subclass_inherits_step_and_scan`` — ``step()`` and ``scan()``
   work unchanged on subclasses (they call ``self(batch)`` which the
   subclass overrides).
3. ``test_subclass_with_nnx_cond_runtime_branching`` — a subclass using
   ``nnx.cond`` for runtime branching scans correctly under
   ``nnx.scan``; both branches participate in the trace.
4. ``test_subclass_can_hold_extra_state`` — additional ``nnx.Variable``
   fields on the subclass survive ``nnx.split`` / ``nnx.merge``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax.pipeline import Pipeline
from datarax.sources.memory_source import MemorySource, MemorySourceConfig


def _source(num_elements: int = 16) -> MemorySource:
    return MemorySource(
        MemorySourceConfig(shuffle=False),
        {"x": jnp.arange(num_elements, dtype=jnp.float32)},
    )


def test_subclass_can_override_call_for_custom_topology() -> None:
    class _CustomPipeline(Pipeline):
        """Pipeline that adds a per-batch constant before returning."""

        def __init__(self, *, source, batch_size, rngs, offset):
            super().__init__(source=source, batch_size=batch_size, rngs=rngs, stages=[])
            self.offset = jnp.float32(offset)

        def __call__(self, batch: dict) -> dict:
            return {**batch, "x": batch["x"] + self.offset}

    pipeline = _CustomPipeline(source=_source(), batch_size=4, rngs=nnx.Rngs(0), offset=100.0)
    out = pipeline.step()  # type: ignore[reportCallIssue]

    np.testing.assert_allclose(np.asarray(out["x"]), np.array([100.0, 101.0, 102.0, 103.0]))


def test_subclass_inherits_step_and_scan() -> None:
    """``step()`` and ``scan()`` use ``self(batch)`` so subclass overrides take effect."""

    class _Doubler(Pipeline):
        def __init__(self, *, source, batch_size, rngs):
            super().__init__(source=source, batch_size=batch_size, rngs=rngs, stages=[])

        def __call__(self, batch: dict) -> dict:
            return {**batch, "x": batch["x"] * 2.0}

    pipeline = _Doubler(source=_source(num_elements=16), batch_size=4, rngs=nnx.Rngs(0))

    def step_fn(batch: dict) -> jax.Array:
        return jnp.sum(batch["x"])

    outputs = pipeline.scan(step_fn, length=4)
    # arange(0,16) * 2 → per-step sums of 4 elements
    np.testing.assert_allclose(np.asarray(outputs), np.array([12.0, 44.0, 76.0, 108.0]))


def test_subclass_with_nnx_cond_runtime_branching() -> None:
    """A subclass using ``nnx.cond`` for runtime branching scans correctly."""

    class _ConditionalPipeline(Pipeline):
        def __init__(self, *, source, batch_size, rngs, threshold):
            super().__init__(source=source, batch_size=batch_size, rngs=rngs, stages=[])
            self.threshold = jnp.float32(threshold)

        def __call__(self, batch: dict) -> dict:
            mean = jnp.mean(batch["x"])

            def double_branch(b: dict) -> dict:
                return {**b, "x": b["x"] * 2.0}

            def half_branch(b: dict) -> dict:
                return {**b, "x": b["x"] * 0.5}

            new_batch = nnx.cond(mean > self.threshold, double_branch, half_branch, batch)
            return new_batch

    pipeline = _ConditionalPipeline(
        source=_source(num_elements=16),
        batch_size=4,
        rngs=nnx.Rngs(0),
        threshold=5.0,
    )

    def step_fn(batch: dict) -> jax.Array:
        return jnp.mean(batch["x"])

    outputs = pipeline.scan(step_fn, length=4)
    # arange(0,16) split into batches of 4. Means: 1.5, 5.5, 9.5, 13.5
    # Threshold 5.0: first batch halved (mean 0.75), rest doubled (means 11, 19, 27)
    np.testing.assert_allclose(np.asarray(outputs), np.array([0.75, 11.0, 19.0, 27.0]))


def test_subclass_can_hold_extra_state() -> None:
    """``nnx.Variable`` fields on a subclass survive ``nnx.split`` / ``nnx.merge``."""

    class _Counter(Pipeline):
        def __init__(self, *, source, batch_size, rngs):
            super().__init__(source=source, batch_size=batch_size, rngs=rngs, stages=[])
            self.batches_seen = nnx.Variable(jnp.int32(0))

        def __call__(self, batch: dict) -> dict:
            self.batches_seen[...] = self.batches_seen[...] + jnp.int32(1)
            return batch

    pipeline = _Counter(source=_source(), batch_size=4, rngs=nnx.Rngs(0))
    pipeline.step()  # type: ignore[reportCallIssue]
    pipeline.step()  # type: ignore[reportCallIssue]

    graphdef, state = nnx.split(pipeline)
    rebuilt = nnx.merge(graphdef, state)

    assert int(rebuilt.batches_seen[...]) == int(pipeline.batches_seen[...]) == 2
