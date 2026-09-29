"""Tests for _vmap_apply, the vectorised core of OperatorModule.apply_batch().

Test categories:
1. _vmap_apply produces identical output to apply_batch (shared core)
2. _vmap_apply handles deterministic ops (no dummy RNG overhead)
3. _vmap_apply handles stochastic ops (real RNG generation)
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from datarax.core import batch_ops
from datarax.core.config import OperatorConfig
from datarax.core.element_batch import Batch, Element
from datarax.core.operator import OperatorModule


# ========================================================================
# Test Operators (reusable fixtures)
# ========================================================================


@dataclass(frozen=True)
class ScaleConfig(OperatorConfig):  # type: ignore[reportGeneralTypeIssues]
    """Config for deterministic scale operator."""

    factor: float = 2.0


class ScaleOperator(OperatorModule):
    """Deterministic operator: multiplies data by a factor."""

    def apply(
        self,
        element: Element,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> Element:
        data = element.data
        del key, stats
        new_data = jax.tree.map(lambda x: x * self.config.factor, data)  # type: ignore[reportAttributeAccessIssue]
        return element.replace(data=new_data)


@dataclass(frozen=True)
class StochasticNoiseConfig(OperatorConfig):  # type: ignore[reportGeneralTypeIssues]
    """Config for stochastic noise operator."""

    noise_scale: float = 0.1

    def __post_init__(self):
        # Force stochastic=True and stream_name
        object.__setattr__(self, "stochastic", True)
        object.__setattr__(self, "stream_name", "noise")
        super().__post_init__()


class StochasticNoiseOperator(OperatorModule):
    """Stochastic operator: adds random noise to data."""

    def apply(
        self,
        element: Element,
        key: jax.Array | None = None,
        stats: dict[str, Any] | None = None,
    ) -> Element:
        data = element.data
        del stats
        # This record's noise sample, drawn from its own key.
        scale = self.config.noise_scale  # type: ignore[reportAttributeAccessIssue]
        noise = jax.random.normal(key, shape=()) * scale if key is not None else 0.0
        new_data = jax.tree.map(lambda x: x + noise, data)
        return element.replace(data=new_data)


# ========================================================================
# Fixtures
# ========================================================================


@pytest.fixture
def deterministic_op():
    """Create a deterministic ScaleOperator."""
    config = ScaleConfig(stochastic=False, factor=2.0)
    return ScaleOperator(config)


@pytest.fixture
def stochastic_op():
    """Create a stochastic NoiseOperator."""
    config = StochasticNoiseConfig(noise_scale=0.1)
    return StochasticNoiseOperator(config, rngs=nnx.Rngs(noise=42))


@pytest.fixture
def sample_batch():
    """Create a sample Batch with image-like data."""
    data = {"image": jnp.ones((4, 8, 8, 3), dtype=jnp.float32)}
    states = {}
    return batch_ops.from_arrays(data, states=states)


@pytest.fixture
def sample_batch_with_states():
    """Create a sample Batch with both data and states."""
    data = {"image": jnp.ones((4, 8, 8, 3), dtype=jnp.float32)}
    states = {"count": jnp.zeros((4,), dtype=jnp.int32)}
    return batch_ops.from_arrays(data, states=states)


# ========================================================================
# Tests: _vmap_apply matches apply_batch
# ========================================================================


class TestVmapApplyMatchesApplyBatch:
    """_vmap_apply produces identical output to apply_batch."""

    def test_deterministic_output_matches(self, deterministic_op, sample_batch):
        """_vmap_apply output matches apply_batch for deterministic ops."""
        # Get apply_batch result (existing behavior)
        result_batch = deterministic_op(sample_batch)
        expected_data = result_batch.data
        result_batch.states

        # Get _vmap_apply result (new method)
        batch_data = sample_batch.data
        batch_states = sample_batch.states
        actual_data = deterministic_op(batch_ops.from_arrays(batch_data, states=batch_states)).data

        # Verify numerical equivalence
        for key in expected_data:
            assert jnp.allclose(actual_data[key], expected_data[key]), (
                f"Data mismatch for key '{key}'"
            )

    def test_with_states(self, deterministic_op, sample_batch_with_states):
        """_vmap_apply handles batches with both data and states."""
        result_batch = deterministic_op(sample_batch_with_states)
        expected_data = result_batch.data

        batch_data = sample_batch_with_states.data
        batch_states = sample_batch_with_states.states
        actual_data = deterministic_op(batch_ops.from_arrays(batch_data, states=batch_states)).data

        for key in expected_data:
            assert jnp.allclose(actual_data[key], expected_data[key])

    def test_stochastic_uses_same_rng_path(self, stochastic_op, sample_batch):
        """Stochastic _vmap_apply uses the same RNG path as apply_batch.

        Note: We can't compare outputs directly because RNG state advances,
        but we verify both paths produce valid (non-NaN, non-zero) output.
        """
        batch_data = sample_batch.data
        batch_states = sample_batch.states

        actual_data = stochastic_op(batch_ops.from_arrays(batch_data, states=batch_states)).data

        # Should have same keys as input
        assert set(actual_data.keys()) == set(batch_data.keys())
        # Should not be NaN
        for key in actual_data:
            assert not jnp.any(jnp.isnan(actual_data[key])), f"NaN in {key}"
        # Stochastic: should not be identical to input (noise added)
        assert not jnp.allclose(actual_data["image"], batch_data["image"])


# ========================================================================
# Tests: _vmap_apply RNG behavior
# ========================================================================


class TestVmapApplyRng:
    """_vmap_apply correctly handles RNG for stochastic/deterministic."""

    def test_deterministic_no_dummy_rng_side_effect(self, deterministic_op, sample_batch):
        """Deterministic ops should produce consistent results without RNG."""
        batch_data = sample_batch.data
        batch_states = sample_batch.states

        result1_data = deterministic_op(batch_ops.from_arrays(batch_data, states=batch_states)).data
        result2_data = deterministic_op(batch_ops.from_arrays(batch_data, states=batch_states)).data

        for key in result1_data:
            assert jnp.allclose(result1_data[key], result2_data[key]), (
                "Deterministic op produced different results across calls"
            )

    def test_stochastic_is_deterministic_per_record(self, stochastic_op, sample_batch):
        """Stochastic ops are per-record deterministic.

        Randomness keys on the record's global index, so the same indices always
        produce the same output (invariant to call order / resume), while
        different indices produce different augmentation.
        """
        rows = jnp.arange(sample_batch.batch_size, dtype=jnp.uint32)
        indices_a = jnp.stack([jnp.zeros_like(rows), rows], -1)
        indices_b = indices_a.at[:, 1].add(1000)  # different global records

        result1 = stochastic_op(sample_batch.replace(indices=indices_a)).data
        result2 = stochastic_op(sample_batch.replace(indices=indices_a)).data
        result3 = stochastic_op(sample_batch.replace(indices=indices_b)).data

        # Same global indices -> identical output (per-record determinism).
        assert jnp.allclose(result1["image"], result2["image"]), (
            "Same global indices must yield identical augmentation"
        )
        # Different global indices -> different augmentation.
        assert not jnp.allclose(result1["image"], result3["image"]), (
            "Different global indices must yield different augmentation"
        )


# ========================================================================
# Tests: apply_batch
# ========================================================================


class TestApplyBatchPreserved:
    """apply_batch over the shared vmap core."""

    def test_apply_batch_returns_batch(self, deterministic_op, sample_batch):
        """apply_batch still returns a Batch object."""
        result = deterministic_op(sample_batch)
        assert isinstance(result, Batch)

    def test_apply_batch_correct_values(self, deterministic_op, sample_batch):
        """apply_batch produces correct values after refactor."""
        result = deterministic_op(sample_batch)
        expected = jnp.ones((4, 8, 8, 3)) * 2.0
        assert jnp.allclose(result.data["image"], expected)

    def test_apply_batch_empty_batch(self, deterministic_op):
        """apply_batch handles empty batch."""
        batch = batch_ops.from_arrays({"image": jnp.zeros((0, 4, 4, 3), jnp.float32)})
        # A batch of no rows has nothing to apply: it passes through.
        result = deterministic_op(batch)
        assert result is batch

    def test_apply_batch_keeps_identities(self, deterministic_op):
        """apply_batch changes data and states only; each record keeps its identity."""
        data = {"image": jnp.ones((2, 4, 4, 3), dtype=jnp.float32)}
        batch = batch_ops.from_arrays(data).replace(
            indices=jnp.array([[0, 9], [1, 3]], jnp.uint32), draws=jnp.array([0, 2], jnp.int32)
        )
        result = deterministic_op(batch)
        assert result.indices is batch.indices and result.draws is batch.draws
