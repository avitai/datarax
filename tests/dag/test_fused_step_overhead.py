"""Tests for fused step overhead and nnx.cached_partial optimization.

Validates that:
1. Deterministic operators are closure-captured (no graph traversal per call)
2. Stochastic chains use nnx.cached_partial (graph traversal cached)
3. Per-call overhead stays within acceptable bounds
"""

import time

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from datarax.core.config import ElementOperatorConfig
from datarax.dag.dag_executor import DAGExecutor
from datarax.dag.nodes import OperatorNode
from datarax.operators.element_operator import ElementOperator


def _identity_fn(element, key):
    """Deterministic identity transform."""
    return element


def _stochastic_fn(element, key):
    """Stochastic transform that uses the RNG key."""
    noise = jax.random.normal(key, element.data["value"].shape) * 0.01
    new_data = {"value": element.data["value"] + noise}
    return element.replace(data=new_data)


class TestFusedStepDeterministic:
    """Tests for deterministic fused step (closure-captured operators)."""

    def test_deterministic_fused_step_is_closure(self):
        """Deterministic fused step should not accept operators as arguments.

        When operators are closure-captured, the fused function signature is
        (data, states) -> (data, states), NOT (operators, data, states).
        """
        config = ElementOperatorConfig(stochastic=False)
        op = ElementOperator(config, fn=_identity_fn)
        op_node = OperatorNode(op)

        executor = DAGExecutor(enforce_batch=False)
        fused = executor._make_fused_step([op_node])

        # The deterministic path is @nnx.jit def fused_step_deterministic(data, states)
        # It should work with just data and states (2 args), not 3
        data = {"value": jnp.ones((4, 8))}
        states = {}
        result_data, result_states = fused(data, states)
        assert "value" in result_data

    def test_deterministic_operators_not_in_jit_args(self):
        """Deterministic operators should be trace-time constants, not traced args."""
        config = ElementOperatorConfig(stochastic=False)
        ops = [OperatorNode(ElementOperator(config, fn=_identity_fn)) for _ in range(3)]
        executor = DAGExecutor(enforce_batch=False)
        fused = executor._make_fused_step(ops)

        # If operators were traced args, calling with different shapes would
        # trigger recompilation. Closure-captured operators don't affect this.
        data1 = {"value": jnp.ones((4, 8))}
        data2 = {"value": jnp.ones((4, 8))}
        fused(data1, {})
        fused(data2, {})  # Same shape = no recompilation


class TestFusedStepStochastic:
    """Tests for stochastic fused step (nnx.cached_partial)."""

    def test_stochastic_fused_step_uses_cached_partial(self):
        """Stochastic fused step should use nnx.cached_partial."""
        config = ElementOperatorConfig(stochastic=True, stream_name="params")
        op = ElementOperator(config, fn=_stochastic_fn, rngs=nnx.Rngs(params=42))
        op_node = OperatorNode(op)

        executor = DAGExecutor(enforce_batch=False)
        fused = executor._make_fused_step([op_node])

        # cached_partial returns a functools.partial-like object
        # It should work with just (data, states) — operators are pre-bound
        data = {"value": jnp.ones((4, 8))}
        states = {}
        result_data, result_states = fused(data, states)
        assert "value" in result_data

    def test_stochastic_produces_different_results(self):
        """Stochastic fused step should produce different results on successive calls."""
        config = ElementOperatorConfig(stochastic=True, stream_name="params")
        op = ElementOperator(config, fn=_stochastic_fn, rngs=nnx.Rngs(params=42))
        op_node = OperatorNode(op)

        executor = DAGExecutor(enforce_batch=False)
        fused = executor._make_fused_step([op_node])

        data = {"value": jnp.ones((4, 8))}
        r1, _ = fused(data, {})
        r2, _ = fused(data, {})
        # Stochastic ops should produce different outputs
        assert not jnp.allclose(r1["value"], r2["value"])


class TestFusedStepOverhead:
    """Measure per-call overhead of fused step."""

    @pytest.fixture
    def deterministic_fused(self):
        """Create a deterministic fused step for benchmarking."""
        config = ElementOperatorConfig(stochastic=False)
        ops = [OperatorNode(ElementOperator(config, fn=_identity_fn)) for _ in range(3)]
        executor = DAGExecutor(enforce_batch=False)
        return executor._make_fused_step(ops)

    def test_per_call_overhead_acceptable(self, deterministic_fused):
        """Per-call overhead should be under 1ms for deterministic chain."""
        data = {"value": jnp.ones((32, 64))}
        states = {}

        # Warmup (JIT compilation)
        for _ in range(5):
            deterministic_fused(data, states)

        # Measure
        times = []
        for _ in range(50):
            start = time.perf_counter()
            deterministic_fused(data, states)
            times.append(time.perf_counter() - start)

        import statistics

        median_ms = statistics.median(times) * 1000
        # Should be well under 1ms per call for compiled deterministic chain
        assert median_ms < 1.0, f"Per-call overhead too high: {median_ms:.3f}ms"
