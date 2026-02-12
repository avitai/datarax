"""Tests for _vmap_apply and _apply_on_raw extraction from apply_batch().

TDD RED phase: These tests define the contract for the new private methods
that extract the shared vmap core from OperatorModule.apply_batch().

Test categories:
1. _vmap_apply produces identical output to apply_batch (shared core)
2. _vmap_apply handles deterministic ops (no dummy RNG overhead)
3. _vmap_apply handles stochastic ops (real RNG generation)
4. _apply_on_raw returns raw dicts (not Batch objects)
5. _apply_on_raw matches apply_batch numerically
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from datarax.core.config import OperatorConfig
from datarax.core.element_batch import Batch
from datarax.core.operator import OperatorModule


# ========================================================================
# Test Operators (reusable fixtures)
# ========================================================================


@dataclass
class ScaleConfig(OperatorConfig):
    """Config for deterministic scale operator."""

    factor: float = 2.0


class ScaleOperator(OperatorModule):
    """Deterministic operator: multiplies data by a factor."""

    def apply(self, data, state, metadata, random_params=None, stats=None):
        new_data = jax.tree.map(lambda x: x * self.config.factor, data)
        return new_data, state, metadata


@dataclass
class StochasticNoiseConfig(OperatorConfig):
    """Config for stochastic noise operator."""

    noise_scale: float = 0.1

    def __post_init__(self):
        # Force stochastic=True and stream_name
        object.__setattr__(self, "stochastic", True)
        object.__setattr__(self, "stream_name", "noise")
        super().__post_init__()


class StochasticNoiseOperator(OperatorModule):
    """Stochastic operator: adds random noise to data."""

    def generate_random_params(self, rng, data_shapes):
        # data_shapes is a PyTree of shape tuples; use is_leaf to treat tuples as atoms
        shapes = jax.tree.leaves(data_shapes, is_leaf=lambda x: isinstance(x, tuple))
        batch_size = shapes[0][0]
        return jax.random.normal(rng, shape=(batch_size,)) * self.config.noise_scale

    def apply(self, data, state, metadata, random_params=None, stats=None):
        noise = random_params if random_params is not None else 0.0
        new_data = jax.tree.map(lambda x: x + noise, data)
        return new_data, state, metadata


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
    return Batch.from_parts(data=data, states=states, validate=False)


@pytest.fixture
def sample_batch_with_states():
    """Create a sample Batch with both data and states."""
    data = {"image": jnp.ones((4, 8, 8, 3), dtype=jnp.float32)}
    states = {"count": jnp.zeros((4,), dtype=jnp.int32)}
    return Batch.from_parts(data=data, states=states, validate=False)


# ========================================================================
# Tests: _vmap_apply matches apply_batch
# ========================================================================


class TestVmapApplyMatchesApplyBatch:
    """_vmap_apply produces identical output to apply_batch."""

    def test_deterministic_output_matches(self, deterministic_op, sample_batch):
        """_vmap_apply output matches apply_batch for deterministic ops."""
        # Get apply_batch result (existing behavior)
        result_batch = deterministic_op.apply_batch(sample_batch)
        expected_data = result_batch.data.get_value()
        result_batch.states.get_value()

        # Get _vmap_apply result (new method)
        batch_data = sample_batch.data.get_value()
        batch_states = sample_batch.states.get_value()
        actual_data, actual_states = deterministic_op._vmap_apply(batch_data, batch_states)

        # Verify numerical equivalence
        for key in expected_data:
            assert jnp.allclose(actual_data[key], expected_data[key]), (
                f"Data mismatch for key '{key}'"
            )

    def test_with_states(self, deterministic_op, sample_batch_with_states):
        """_vmap_apply handles batches with both data and states."""
        result_batch = deterministic_op.apply_batch(sample_batch_with_states)
        expected_data = result_batch.data.get_value()

        batch_data = sample_batch_with_states.data.get_value()
        batch_states = sample_batch_with_states.states.get_value()
        actual_data, actual_states = deterministic_op._vmap_apply(batch_data, batch_states)

        for key in expected_data:
            assert jnp.allclose(actual_data[key], expected_data[key])

    def test_stochastic_uses_same_rng_path(self, stochastic_op, sample_batch):
        """Stochastic _vmap_apply uses the same RNG path as apply_batch.

        Note: We can't compare outputs directly because RNG state advances,
        but we verify both paths produce valid (non-NaN, non-zero) output.
        """
        batch_data = sample_batch.data.get_value()
        batch_states = sample_batch.states.get_value()

        actual_data, actual_states = stochastic_op._vmap_apply(batch_data, batch_states)

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
        batch_data = sample_batch.data.get_value()
        batch_states = sample_batch.states.get_value()

        result1_data, _ = deterministic_op._vmap_apply(batch_data, batch_states)
        result2_data, _ = deterministic_op._vmap_apply(batch_data, batch_states)

        for key in result1_data:
            assert jnp.allclose(result1_data[key], result2_data[key]), (
                "Deterministic op produced different results across calls"
            )

    def test_stochastic_produces_different_results(self, stochastic_op, sample_batch):
        """Stochastic ops should produce different results on each call."""
        batch_data = sample_batch.data.get_value()
        batch_states = sample_batch.states.get_value()

        result1_data, _ = stochastic_op._vmap_apply(batch_data, batch_states)
        result2_data, _ = stochastic_op._vmap_apply(batch_data, batch_states)

        # With advancing RNG, results should differ
        assert not jnp.allclose(result1_data["image"], result2_data["image"]), (
            "Stochastic op produced identical results on successive calls"
        )


# ========================================================================
# Tests: _apply_on_raw returns raw dicts
# ========================================================================


class TestApplyOnRaw:
    """_apply_on_raw returns (dict, dict) not Batch objects."""

    def test_returns_tuple_of_dicts(self, deterministic_op, sample_batch):
        """_apply_on_raw returns (data_dict, states_dict) not Batch."""
        batch_data = sample_batch.data.get_value()
        batch_states = sample_batch.states.get_value()

        result = deterministic_op._apply_on_raw(batch_data, batch_states)

        assert isinstance(result, tuple), "Should return a tuple"
        assert len(result) == 2, "Should return (data, states)"
        result_data, result_states = result
        assert isinstance(result_data, dict), "data should be a dict"
        assert isinstance(result_states, dict), "states should be a dict"
        # Verify it's NOT a Batch
        assert not isinstance(result_data, Batch)

    def test_matches_apply_batch_numerically(self, deterministic_op, sample_batch):
        """_apply_on_raw produces same values as apply_batch."""
        # apply_batch result
        result_batch = deterministic_op.apply_batch(sample_batch)
        expected_data = result_batch.data.get_value()

        # _apply_on_raw result
        batch_data = sample_batch.data.get_value()
        batch_states = sample_batch.states.get_value()
        actual_data, actual_states = deterministic_op._apply_on_raw(batch_data, batch_states)

        for key in expected_data:
            assert jnp.allclose(actual_data[key], expected_data[key])

    def test_chainable_raw_dicts(self, deterministic_op, sample_batch):
        """_apply_on_raw output can be fed into another _apply_on_raw call."""
        batch_data = sample_batch.data.get_value()
        batch_states = sample_batch.states.get_value()

        # Chain two calls
        data1, states1 = deterministic_op._apply_on_raw(batch_data, batch_states)
        data2, states2 = deterministic_op._apply_on_raw(data1, states1)

        # ScaleOperator with factor=2.0, applied twice = 4.0x
        expected = jnp.ones((4, 8, 8, 3)) * 4.0
        assert jnp.allclose(data2["image"], expected)

    def test_with_empty_states(self, deterministic_op):
        """_apply_on_raw works with empty states dict."""
        data = {"image": jnp.ones((4, 8, 8, 3), dtype=jnp.float32)}
        states = {}

        result_data, result_states = deterministic_op._apply_on_raw(data, states)

        assert "image" in result_data
        assert jnp.allclose(result_data["image"], jnp.ones((4, 8, 8, 3)) * 2.0)

    def test_stochastic_on_raw(self, stochastic_op, sample_batch):
        """_apply_on_raw works with stochastic operators."""
        batch_data = sample_batch.data.get_value()
        batch_states = sample_batch.states.get_value()

        result_data, result_states = stochastic_op._apply_on_raw(batch_data, batch_states)

        assert isinstance(result_data, dict)
        assert "image" in result_data
        # Stochastic: output should differ from input
        assert not jnp.allclose(result_data["image"], batch_data["image"])


# ========================================================================
# Tests: apply_batch still works after refactor
# ========================================================================


class TestApplyBatchPreserved:
    """Verify apply_batch still works identically after refactor."""

    def test_apply_batch_returns_batch(self, deterministic_op, sample_batch):
        """apply_batch still returns a Batch object."""
        result = deterministic_op.apply_batch(sample_batch)
        assert isinstance(result, Batch)

    def test_apply_batch_correct_values(self, deterministic_op, sample_batch):
        """apply_batch produces correct values after refactor."""
        result = deterministic_op.apply_batch(sample_batch)
        expected = jnp.ones((4, 8, 8, 3)) * 2.0
        assert jnp.allclose(result.data.get_value()["image"], expected)

    def test_apply_batch_empty_batch(self, deterministic_op):
        """apply_batch handles empty batch."""
        batch = Batch([], validate=False)
        # Empty batch passthrough (existing behavior)
        result = deterministic_op.apply_batch(batch)
        assert result is batch  # Should return same object

    def test_apply_batch_preserves_metadata(self, deterministic_op):
        """apply_batch preserves metadata after refactor."""
        data = {"image": jnp.ones((2, 4, 4, 3), dtype=jnp.float32)}
        metadata_list = [{"source": "a"}, {"source": "b"}]
        batch = Batch.from_parts(data=data, states={}, metadata_list=metadata_list, validate=False)
        result = deterministic_op.apply_batch(batch)
        assert result._metadata_list.get_value() == metadata_list
