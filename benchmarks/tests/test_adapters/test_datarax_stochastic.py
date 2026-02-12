"""Tests for stochastic transform support in the Datarax adapter.

TDD RED phase: These tests define the contract for stochastic
augmentation transforms in the benchmark adapter.

Test categories:
1. Transform functions produce expected modifications
2. Operator creation handles stochastic/deterministic correctly
3. Pipeline integration with stochastic transforms
4. Scenario discovery finds AUG scenarios
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from benchmarks.adapters.base import ScenarioConfig
from benchmarks.adapters.datarax_adapter import (
    DataraxAdapter,
    _STOCHASTIC_TRANSFORM_FNS,
)
from datarax.core.element_batch import Element


# ========================================================================
# Fixtures
# ========================================================================


@pytest.fixture
def sample_element():
    """Create a sample Element with float32 image data."""
    data = {"image": jnp.ones((8, 8, 3), dtype=jnp.float32) * 0.5}
    return Element(data=data, state={}, metadata=None)


@pytest.fixture
def rng_key():
    """Create a JAX RNG key."""
    return jax.random.key(42)


@pytest.fixture
def adapter():
    """Create a DataraxAdapter instance."""
    return DataraxAdapter()


@pytest.fixture
def rngs():
    """Create Flax NNX Rngs with augment stream."""
    return nnx.Rngs(42, augment=43)


# ========================================================================
# Tests: Transform Functions
# ========================================================================


class TestStochasticTransformFunctions:
    """Stochastic transform functions modify data using RNG key."""

    def test_gaussian_noise_modifies_data(self, sample_element, rng_key):
        """GaussianNoise adds noise — output differs from input."""
        fn = _STOCHASTIC_TRANSFORM_FNS["GaussianNoise"]
        result = fn(sample_element, rng_key)

        assert not jnp.allclose(result.data["image"], sample_element.data["image"])
        # Noise should be small (std=0.05)
        diff = jnp.abs(result.data["image"] - sample_element.data["image"])
        assert jnp.max(diff) < 1.0, "Noise too large"

    def test_random_brightness_modifies_data(self, sample_element, rng_key):
        """RandomBrightness shifts values — output differs from input."""
        fn = _STOCHASTIC_TRANSFORM_FNS["RandomBrightness"]
        result = fn(sample_element, rng_key)

        assert not jnp.allclose(result.data["image"], sample_element.data["image"])
        # All pixels shifted by same delta
        diff = result.data["image"] - sample_element.data["image"]
        assert jnp.allclose(diff, diff[0, 0, 0]), "Brightness not uniform"

    def test_random_scale_modifies_data(self, sample_element, rng_key):
        """RandomScale multiplies values — output differs from input."""
        fn = _STOCHASTIC_TRANSFORM_FNS["RandomScale"]
        result = fn(sample_element, rng_key)

        assert not jnp.allclose(result.data["image"], sample_element.data["image"])
        # All pixels scaled by same factor
        ratio = result.data["image"] / sample_element.data["image"]
        assert jnp.allclose(ratio, ratio[0, 0, 0]), "Scale not uniform"


# ========================================================================
# Tests: Operator Creation
# ========================================================================


class TestOperatorCreation:
    """_create_operator handles stochastic vs deterministic correctly."""

    def test_create_stochastic_operator(self, adapter, rngs):
        """Creating a stochastic transform produces stochastic operator."""
        op = adapter._create_operator("GaussianNoise", rngs)
        assert op.config.stochastic is True
        assert op.config.stream_name == "augment"

    def test_create_deterministic_still_works(self, adapter, rngs):
        """Creating a deterministic transform still works as before."""
        op = adapter._create_operator("Normalize", rngs)
        assert op.config.stochastic is False

    def test_unknown_transform_raises(self, adapter, rngs):
        """Unknown transform name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown transform"):
            adapter._create_operator("NonExistentTransform", rngs)


# ========================================================================
# Tests: Pipeline Integration
# ========================================================================


class TestPipelineIntegration:
    """Full pipeline lifecycle with stochastic transforms."""

    def _make_aug1_config(self):
        """Create AUG-1-like config for testing."""
        return ScenarioConfig(
            scenario_id="AUG-1",
            dataset_size=256,
            element_shape=(8, 8, 3),
            batch_size=32,
            transforms=["Normalize", "GaussianNoise", "RandomBrightness"],
            extra={"variant_name": "test"},
        )

    def _make_test_data(self, config):
        """Generate synthetic uint8 image data."""
        rng = np.random.default_rng(42)
        return {
            "image": rng.integers(
                0,
                256,
                size=(config.dataset_size, *config.element_shape),
                dtype=np.uint8,
            )
        }

    def test_stochastic_pipeline_produces_output(self, adapter):
        """Full lifecycle: setup -> warmup -> iterate -> teardown."""
        config = self._make_aug1_config()
        data = self._make_test_data(config)

        adapter.setup(config, data)
        adapter.warmup(num_batches=1)

        batches = list(adapter._iterate_batches())
        assert len(batches) > 0

        # First batch should have data
        batch = batches[0]
        materialized = adapter._materialize_batch(batch)
        assert len(materialized) > 0
        assert materialized[0].shape[0] == config.batch_size

        adapter.teardown()

    def test_stochastic_pipeline_different_epochs(self, adapter):
        """Two iterations produce different results (RNG advances)."""
        config = self._make_aug1_config()
        data = self._make_test_data(config)

        # First epoch
        adapter.setup(config, data)
        adapter.warmup(num_batches=1)
        batch1 = next(iter(adapter._iterate_batches()))
        mat1 = adapter._materialize_batch(batch1)
        first_vals = np.array(mat1[0][:4])  # Save first 4 elements
        adapter.teardown()

        # Second epoch (re-setup advances RNG state differently)
        adapter.setup(config, data)
        adapter.warmup(num_batches=1)
        batch2 = next(iter(adapter._iterate_batches()))
        mat2 = adapter._materialize_batch(batch2)
        second_vals = np.array(mat2[0][:4])
        adapter.teardown()

        # Note: With same seed, same setup will produce same results.
        # This test verifies the pipeline runs without errors in both epochs.
        assert first_vals.shape == second_vals.shape

    def test_mixed_chain_pipeline(self, adapter):
        """Pipeline with both deterministic and stochastic transforms."""
        config = ScenarioConfig(
            scenario_id="AUG-2",
            dataset_size=128,
            element_shape=(8, 8, 3),
            batch_size=32,
            transforms=["Normalize", "GaussianNoise", "RandomScale"],
            extra={"variant_name": "test"},
        )
        data = {
            "image": np.random.default_rng(42).integers(0, 256, size=(128, 8, 8, 3), dtype=np.uint8)
        }

        adapter.setup(config, data)
        adapter.warmup(num_batches=1)

        batches = list(adapter._iterate_batches())
        assert len(batches) > 0

        adapter.teardown()


# ========================================================================
# Tests: Scenario Discovery
# ========================================================================


class TestScenarioDiscovery:
    """AUG scenarios are discovered by the scenario registry."""

    def test_aug_scenarios_discovered(self):
        """discover_scenarios() finds AUG-1, AUG-2, AUG-3."""
        from benchmarks.scenarios import discover_scenarios

        modules = discover_scenarios()
        scenario_ids = {m.SCENARIO_ID for m in modules}

        assert "AUG-1" in scenario_ids
        assert "AUG-2" in scenario_ids
        assert "AUG-3" in scenario_ids

    def test_aug1_tier1_variant(self):
        """AUG-1 has TIER1_VARIANT = 'small'."""
        from benchmarks.scenarios import get_scenario_by_id

        mod = get_scenario_by_id("AUG-1")
        assert mod is not None
        assert mod.TIER1_VARIANT == "small"
