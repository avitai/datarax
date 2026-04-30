"""Contracts for ``Pipeline.scan`` — the user-facing fast-path training API.

``scan`` replaces the prior ``epoch`` method and adds a ``modules`` parameter
so users can plug in their own ``nnx.Module`` instances (typically a model and
an ``nnx.Optimizer``) without writing ``nnx.StateAxes`` boilerplate. Pipeline
generates the underlying ``nnx.scan`` body internally.

Test contract index:

A. Without carry — ``step_fn(*modules, batch) -> output``:

   1. ``test_scan_no_carry_no_modules_returns_stacked_outputs`` —
      simplest form: ``step_fn(batch) -> y``; output stacked along axis 0.
   2. ``test_scan_no_carry_with_one_module_passes_module_to_step_fn`` —
      ``modules=(m,)`` causes ``step_fn(m, batch)`` to be called per step.
   3. ``test_scan_no_carry_with_multiple_modules_threads_all_through`` —
      ``modules=(m, opt)`` causes ``step_fn(m, opt, batch)`` per step.

B. With carry — ``step_fn(carry, *modules, batch) -> (new_carry, output)``:

   4. ``test_scan_with_carry_no_modules_returns_pair`` —
      ``init_carry=0.0`` causes ``step_fn(carry, batch)`` to be called.
   5. ``test_scan_with_carry_and_modules_threads_both`` —
      ``init_carry=0.0, modules=(m,)`` → ``step_fn(carry, m, batch)``.

C. Module state lifting (the load-bearing integration contract):

   6. ``test_scan_lifts_model_param_mutations_across_steps`` —
      a learnable module's ``nnx.Param`` updated inside ``step_fn``
      retains the updated value across iterations.
   7. ``test_scan_with_optimizer_module_persists_optimizer_state`` —
      ``nnx.Optimizer`` state survives the scan boundary.
   8. ``test_scan_gradient_flows_to_lifted_module_params`` —
      ``nnx.value_and_grad`` inside ``step_fn`` produces non-zero
      gradients on lifted module parameters.

D. Behavior parity with iterator (correctness):

   9. ``test_scan_total_matches_python_loop_total`` —
      ``scan`` and a Python loop over ``step()`` produce identical
      accumulated outputs for deterministic stages.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from datarax.pipeline import Pipeline
from datarax.sources.memory_source import MemorySource, MemorySourceConfig


# ---------- Helpers ----------


def _source(num_elements: int = 16) -> MemorySource:
    return MemorySource(
        MemorySourceConfig(shuffle=False),
        {"x": jnp.arange(num_elements, dtype=jnp.float32)},
    )


class _DoubleStage(nnx.Module):
    """Multiplies x by 2.0."""

    def __call__(self, batch: dict) -> dict:
        return {**batch, "x": batch["x"] * 2.0}


class _Adder(nnx.Module):
    """Holds an nnx.Param ``bias`` and adds it to ``x``."""

    def __init__(self, init_bias: float = 0.0) -> None:
        super().__init__()
        self.bias = nnx.Param(jnp.float32(init_bias))

    def __call__(self, batch: dict) -> jax.Array:
        return jnp.sum(batch["x"]) + self.bias[...]


# ---------- A. Without-carry contracts ----------


def test_scan_no_carry_no_modules_returns_stacked_outputs() -> None:
    pipeline = Pipeline(
        source=_source(num_elements=16),
        stages=[_DoubleStage()],
        batch_size=4,
        rngs=nnx.Rngs(0),
    )

    def step_fn(batch: dict) -> jax.Array:
        return jnp.sum(batch["x"])

    outputs = pipeline.scan(step_fn, length=4)

    # Per step: sum(arange(start, start+4) * 2) for start in 0,4,8,12 = 12, 44, 76, 108
    assert outputs.shape == (4,)
    np.testing.assert_allclose(np.asarray(outputs), np.array([12.0, 44.0, 76.0, 108.0]))


def test_scan_no_carry_with_one_module_passes_module_to_step_fn() -> None:
    pipeline = Pipeline(
        source=_source(num_elements=16),
        stages=[],
        batch_size=4,
        rngs=nnx.Rngs(0),
    )
    adder = _Adder(init_bias=10.0)

    def step_fn(adder: _Adder, batch: dict) -> jax.Array:
        return adder(batch)

    outputs = pipeline.scan(step_fn, modules=(adder,), length=4)

    # Per step: sum(arange(start, start+4)) + 10 → 6+10, 22+10, 38+10, 54+10
    np.testing.assert_allclose(np.asarray(outputs), np.array([16.0, 32.0, 48.0, 64.0]))


def test_scan_no_carry_with_multiple_modules_threads_all_through() -> None:
    pipeline = Pipeline(
        source=_source(num_elements=16),
        stages=[],
        batch_size=4,
        rngs=nnx.Rngs(0),
    )
    adder_a = _Adder(init_bias=1.0)
    adder_b = _Adder(init_bias=2.0)

    def step_fn(a: _Adder, b: _Adder, batch: dict) -> jax.Array:
        return a(batch) + b(batch)

    outputs = pipeline.scan(step_fn, modules=(adder_a, adder_b), length=4)

    # Per step: 2 * sum(batch) + 3 → 2*6+3, 2*22+3, 2*38+3, 2*54+3
    np.testing.assert_allclose(np.asarray(outputs), np.array([15.0, 47.0, 79.0, 111.0]))


# ---------- B. With-carry contracts ----------


def test_scan_with_carry_no_modules_returns_pair() -> None:
    pipeline = Pipeline(
        source=_source(num_elements=16),
        stages=[],
        batch_size=4,
        rngs=nnx.Rngs(0),
    )

    def step_fn(carry: jax.Array, batch: dict) -> tuple[jax.Array, jax.Array]:
        s = jnp.sum(batch["x"])
        return carry + s, s

    final_carry, outputs = pipeline.scan(step_fn, length=4, init_carry=jnp.float32(0.0))

    # Per step sums: 6, 22, 38, 54; total = 120
    assert float(final_carry) == pytest.approx(120.0)
    np.testing.assert_allclose(np.asarray(outputs), np.array([6.0, 22.0, 38.0, 54.0]))


def test_scan_with_carry_and_modules_threads_both() -> None:
    pipeline = Pipeline(
        source=_source(num_elements=16),
        stages=[],
        batch_size=4,
        rngs=nnx.Rngs(0),
    )
    adder = _Adder(init_bias=10.0)

    def step_fn(carry: jax.Array, adder: _Adder, batch: dict) -> tuple[jax.Array, jax.Array]:
        out = adder(batch)
        return carry + out, out

    final_carry, outputs = pipeline.scan(
        step_fn, modules=(adder,), length=4, init_carry=jnp.float32(0.0)
    )

    # Per step: 16, 32, 48, 64; total = 160
    assert float(final_carry) == pytest.approx(160.0)
    np.testing.assert_allclose(np.asarray(outputs), np.array([16.0, 32.0, 48.0, 64.0]))


# ---------- C. Module state lifting (the integration contract) ----------


def test_scan_lifts_model_param_mutations_across_steps() -> None:
    """A bias incremented inside step_fn retains the updated value next step."""
    pipeline = Pipeline(
        source=_source(num_elements=16),
        stages=[],
        batch_size=4,
        rngs=nnx.Rngs(0),
    )
    adder = _Adder(init_bias=0.0)

    def step_fn(adder: _Adder, batch: dict) -> jax.Array:
        # Mutate the parameter — additions accumulate across iterations.
        adder.bias[...] = adder.bias[...] + jnp.float32(1.0)
        return adder.bias[...]

    outputs = pipeline.scan(step_fn, modules=(adder,), length=5)

    # Bias should be 1, 2, 3, 4, 5 across the five steps.
    np.testing.assert_allclose(np.asarray(outputs), np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert float(adder.bias[...]) == pytest.approx(5.0)


def test_scan_with_optimizer_module_persists_optimizer_state() -> None:
    """nnx.Optimizer state (Adam moments) survives the scan boundary."""

    class _Linear(nnx.Module):
        def __init__(self, *, rngs: nnx.Rngs) -> None:
            super().__init__()
            self.lin = nnx.Linear(1, 1, rngs=rngs)

        def __call__(self, batch: dict) -> jax.Array:
            x = batch["x"].reshape(-1, 1)
            return jnp.mean(self.lin(x))

    model = _Linear(rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(1e-2), wrt=nnx.Param)

    pipeline = Pipeline(
        source=_source(num_elements=16),
        stages=[],
        batch_size=4,
        rngs=nnx.Rngs(0),
    )

    def step_fn(model: _Linear, optimizer: nnx.Optimizer, batch: dict) -> jax.Array:
        def loss_fn(m: _Linear) -> jax.Array:
            return model(batch) ** 2

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    outputs = pipeline.scan(step_fn, modules=(model, optimizer), length=4)

    # Each loss value is finite (model + optimizer state evolved through scan)
    assert outputs.shape == (4,)
    assert bool(jnp.all(jnp.isfinite(outputs)))


def test_scan_gradient_flows_to_lifted_module_params() -> None:
    """nnx.value_and_grad inside step_fn produces non-zero grads on lifted module params."""

    class _LearnableScale(nnx.Module):
        def __init__(self) -> None:
            super().__init__()
            self.factor = nnx.Param(jnp.float32(2.0))

        def __call__(self, batch: dict) -> jax.Array:
            return jnp.sum(batch["x"] * self.factor[...])

    model = _LearnableScale()
    pipeline = Pipeline(
        source=_source(num_elements=16),
        stages=[],
        batch_size=4,
        rngs=nnx.Rngs(0),
    )

    def step_fn(m: _LearnableScale, batch: dict) -> jax.Array:
        def loss_fn(model: _LearnableScale) -> jax.Array:
            return model(batch)

        _loss, grads = nnx.value_and_grad(loss_fn)(m)
        return grads.factor[...]

    grad_outputs = pipeline.scan(step_fn, modules=(model,), length=4)

    # Per step gradient w.r.t. factor = sum(batch["x"]) for each step's batch:
    # 6, 22, 38, 54 — all non-zero.
    assert grad_outputs.shape == (4,)
    np.testing.assert_allclose(np.asarray(grad_outputs), np.array([6.0, 22.0, 38.0, 54.0]))


# ---------- D. Parity with iterator ----------


def test_scan_with_shuffled_source_differentiates_through_data() -> None:
    """Shuffled source: gradients flow through data values, not through indices.

    Verifies the standard JAX semantic — ``jnp.take(values, integer_indices)``
    is differentiable w.r.t. ``values``. The shuffle decision is treated as a
    structural constant during each scan step; gradients on model parameters
    are correct even though sample order is randomized. Differentiability and
    sampling are orthogonal — the integer index choice is non-differentiable
    by design (and by JAX's int-array semantics), but the values at those
    indices flow through the gradient computation normally.
    """

    class _LearnableScale(nnx.Module):
        def __init__(self) -> None:
            super().__init__()
            self.factor = nnx.Param(jnp.float32(2.0))

        def __call__(self, batch: dict) -> jax.Array:
            return jnp.sum(batch["x"] * self.factor[...])

    shuffled_source = MemorySource(
        MemorySourceConfig(shuffle=True),
        {"x": jnp.arange(16, dtype=jnp.float32)},
        rngs=nnx.Rngs(shuffle=0),
    )
    pipeline = Pipeline(
        source=shuffled_source,
        stages=[],
        batch_size=4,
        rngs=nnx.Rngs(0),
    )
    model = _LearnableScale()

    def step_fn(m: _LearnableScale, batch: dict) -> jax.Array:
        def loss_fn(model: _LearnableScale) -> jax.Array:
            return model(batch)

        _loss, grads = nnx.value_and_grad(loss_fn)(m)
        return grads.factor[...]

    grad_outputs = pipeline.scan(step_fn, modules=(model,), length=4)

    # Every step has a finite, non-zero gradient — even with shuffled order
    # the per-step gradient w.r.t. factor equals sum(batch["x"]) for that
    # batch, and every batch contains some non-zero values.
    assert grad_outputs.shape == (4,)
    assert bool(jnp.all(jnp.isfinite(grad_outputs)))
    assert bool(jnp.all(jnp.abs(grad_outputs) > 0)), (
        "shuffled batches still contain values that contribute non-zero "
        "gradients to factor parameter"
    )


def test_scan_total_matches_python_loop_total() -> None:
    """`scan` and a Python loop produce identical accumulated outputs."""
    pipeline_scan = Pipeline(
        source=_source(num_elements=16),
        stages=[_DoubleStage()],
        batch_size=4,
        rngs=nnx.Rngs(0),
    )
    pipeline_loop = Pipeline(
        source=_source(num_elements=16),
        stages=[_DoubleStage()],
        batch_size=4,
        rngs=nnx.Rngs(0),
    )

    def step_fn(batch: dict) -> jax.Array:
        return jnp.sum(batch["x"])

    scan_outputs = pipeline_scan.scan(step_fn, length=4)

    loop_outputs = []
    for _ in range(4):
        loop_outputs.append(step_fn(pipeline_loop.step()))  # type: ignore[reportCallIssue]
    loop_outputs_arr = jnp.stack(loop_outputs)

    np.testing.assert_allclose(np.asarray(scan_outputs), np.asarray(loop_outputs_arr))


# ---------- E. Compiled-body cache contracts ----------


def test_scan_caches_compiled_body_across_calls_with_same_step_fn() -> None:
    """Same (step_fn, length, n_modules, has_init_carry) → cache hit.

    Without caching, every ``Pipeline.scan`` call rebuilds the
    ``@nnx.scan``-decorated body. The decorator is keyed on function
    identity in nnx.scan's internal JIT cache, so a fresh body forces
    re-tracing and re-compilation on every call. The cache eliminates
    that recompilation by reusing the same decorated body.
    """
    pipeline = Pipeline(
        source=_source(num_elements=16),
        stages=[_DoubleStage()],
        batch_size=4,
        rngs=nnx.Rngs(0),
    )

    def step_fn(batch: dict) -> jax.Array:
        return jnp.sum(batch["x"])

    pipeline.scan(step_fn, length=4)
    pipeline._position[...] = jnp.int32(0)
    cached_count = len(pipeline._scan_body_cache)
    assert cached_count == 1

    pipeline.scan(step_fn, length=4)
    assert len(pipeline._scan_body_cache) == cached_count, (
        "calling scan with the same step_fn and length must hit the cache"
    )


def test_scan_cache_key_distinguishes_different_step_fns() -> None:
    """Different step_fn identities produce separate cache entries."""
    pipeline = Pipeline(
        source=_source(num_elements=16),
        stages=[_DoubleStage()],
        batch_size=4,
        rngs=nnx.Rngs(0),
    )

    def step_a(batch: dict) -> jax.Array:
        return jnp.sum(batch["x"])

    def step_b(batch: dict) -> jax.Array:
        return jnp.mean(batch["x"])

    pipeline.scan(step_a, length=4)
    pipeline._position[...] = jnp.int32(0)
    pipeline.scan(step_b, length=4)
    assert len(pipeline._scan_body_cache) == 2


def test_scan_cache_key_distinguishes_different_lengths() -> None:
    """Different length values produce separate cache entries."""
    pipeline = Pipeline(
        source=_source(num_elements=16),
        stages=[_DoubleStage()],
        batch_size=4,
        rngs=nnx.Rngs(0),
    )

    def step_fn(batch: dict) -> jax.Array:
        return jnp.sum(batch["x"])

    pipeline.scan(step_fn, length=2)
    pipeline._position[...] = jnp.int32(0)
    pipeline.scan(step_fn, length=4)
    assert len(pipeline._scan_body_cache) == 2


def test_scan_cached_call_produces_identical_results() -> None:
    """Cached body must produce identical numerical output to the first call."""
    pipeline = Pipeline(
        source=_source(num_elements=16),
        stages=[_DoubleStage()],
        batch_size=4,
        rngs=nnx.Rngs(0),
    )

    def step_fn(batch: dict) -> jax.Array:
        return jnp.sum(batch["x"])

    first = pipeline.scan(step_fn, length=4)
    pipeline._position[...] = jnp.int32(0)
    second = pipeline.scan(step_fn, length=4)

    np.testing.assert_allclose(np.asarray(first), np.asarray(second))


def test_scan_cache_separates_carry_and_no_carry_variants() -> None:
    """``init_carry=None`` vs ``init_carry=value`` use different bodies."""
    pipeline = Pipeline(
        source=_source(num_elements=16),
        stages=[_DoubleStage()],
        batch_size=4,
        rngs=nnx.Rngs(0),
    )

    def step_no_carry(batch: dict) -> jax.Array:
        return jnp.sum(batch["x"])

    def step_with_carry(carry: jax.Array, batch: dict) -> tuple[jax.Array, jax.Array]:
        new_carry = carry + jnp.sum(batch["x"])
        return new_carry, new_carry

    pipeline.scan(step_no_carry, length=4)
    pipeline._position[...] = jnp.int32(0)
    pipeline.scan(step_with_carry, length=4, init_carry=jnp.float32(0.0))
    assert len(pipeline._scan_body_cache) == 2
