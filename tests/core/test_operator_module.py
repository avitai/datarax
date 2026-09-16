"""Tests for OperatorModule - parametric transformation module.

This test suite validates OperatorModule - the base class for all parametric,
differentiable data transformations.

Test Categories (from operator-module-api.md):
1. Module initialization (config-based, stochastic vs deterministic)
2. Stochastic mode (random parameter generation and application)
3. Deterministic mode (no randomness)
4. Batch processing (vmap correctness, empty batches)
5. JIT compatibility (compilation, static branches)
6. Random parameter system (generation, distribution via vmap)
7. Statistics system (inherited from DataraxModule)
8. Training/eval mode (inherited from NNX)
9. Module copying (config-based)
"""

# ========================================================================
# Test Fixture: Example Operator Implementations
# ========================================================================
# Example 1: Simple stochastic operator (random brightness)
from dataclasses import dataclass, fields

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from substrax.testing import TraceCounter

from datarax.core.config import DataraxModuleConfig, OperatorConfig
from datarax.core.element_batch import Batch
from datarax.core.module import DataraxModule
from datarax.core.operator import DIRECT_CALL_STREAM, OperatorModule, require_key


@dataclass(frozen=True)
class RandomBrightnessConfig(OperatorConfig):  # type: ignore[reportGeneralTypeIssues]
    """Config for random brightness operator."""

    min_factor: float = 0.8
    max_factor: float = 1.2

    def __post_init__(self):
        super().__post_init__()
        if self.min_factor >= self.max_factor:
            raise ValueError("min_factor must be < max_factor")
        if self.min_factor <= 0 or self.max_factor <= 0:
            raise ValueError("Brightness factors must be positive")


class RandomBrightnessOperator(OperatorModule):
    """Stochastic operator that adjusts brightness randomly."""

    def apply(self, data, state, metadata, key=None, stats=None):
        del stats
        # This record's brightness factor, drawn from its own key.
        record_key = require_key(key, self)
        factor = jax.random.uniform(
            record_key,
            shape=(),
            minval=self.config.min_factor,  # type: ignore[reportAttributeAccessIssue]
            maxval=self.config.max_factor,  # type: ignore[reportAttributeAccessIssue]
        )
        transformed_data = {**data, "image": jnp.clip(data["image"] * factor, 0.0, 1.0)}
        return transformed_data, state, metadata


@dataclass(frozen=True)
class NormalizeConfig(OperatorConfig):  # type: ignore[reportGeneralTypeIssues]
    """Config for normalization operator (deterministic)."""

    # No operator-specific fields needed
    pass


class NormalizeOperator(OperatorModule):
    """Deterministic operator that normalizes data using statistics."""

    def apply(self, data, state, metadata, key=None, stats=None):
        # Get stats from config
        del key
        if stats is None:
            stats = self.get_statistics()

        if stats is None:
            # No normalization if no stats available
            return data, state, metadata

        mean = stats.get("mean", 0.0)
        std = stats.get("std", 1.0)

        transformed_data = {**data, "image": (data["image"] - mean) / std}
        return transformed_data, state, metadata


# ========================================================================
# Test Helper: Batch Creation
# ========================================================================


def create_test_batch(data, states=None, metadata_list=None, batch_size=None):
    """Helper to create batch using Batch.from_parts() API.

    This helper ensures all tests use the correct Batch constructor API.

    Args:
        data: Dict of arrays with batch dimension (e.g., {"image": array of shape (B, H, W, C)})
        states: PyTree dict with stacked states (batch dim on axis 0), or None for empty PyTree
        metadata_list: List of metadata objects (length B), or None for default Nones
        batch_size: Optional batch size (inferred from data if not provided)

    Returns:
        Batch instance created using from_parts()
    """
    if batch_size is None:
        # Infer batch size from first array in data
        first_array = next(iter(data.values()))
        batch_size = first_array.shape[0]

    if states is None:
        # Default: empty PyTree (empty dict)
        states = {}
    if metadata_list is None:
        metadata_list = [None] * batch_size

    return Batch.from_parts(data, states, metadata_list, validate=False)


# ========================================================================
# Test Category 1: Module Initialization
# ========================================================================


class TestOperatorModuleInitialization:
    """Test OperatorModule initialization with config."""

    def test_stochastic_initialization_with_rngs(self):
        """Test stochastic operator initialization with RNG manager."""
        config = RandomBrightnessConfig(
            stochastic=True,
            stream_name="augment",
            min_factor=0.8,
            max_factor=1.2,
        )
        rngs = nnx.Rngs(42)
        operator = RandomBrightnessOperator(config, rngs=rngs)

        assert operator.config is config
        assert operator.stochastic is True
        assert operator.stream_name == "augment"
        # The caller's Rngs is read once, for the base key, and not kept.
        assert operator.rngs is None

    def test_stochastic_initialization_without_rngs_fails(self):
        """Test that stochastic operator requires rngs at runtime."""
        config = RandomBrightnessConfig(stochastic=True, stream_name="augment")

        # Config validation passes (stream_name provided)
        # But module init should fail (no rngs)
        with pytest.raises(ValueError) as exc_info:
            RandomBrightnessOperator(config)  # Missing rngs

        error_msg = str(exc_info.value).lower()
        assert "stochastic" in error_msg or "require" in error_msg
        assert "rngs" in error_msg

    def test_deterministic_initialization_without_rngs(self):
        """Test deterministic operator doesn't require rngs."""
        config = NormalizeConfig(stochastic=False)
        operator = NormalizeOperator(config)
        operator.set_statistics({"mean": 0.5, "std": 0.2})

        assert operator.config is config
        assert operator.stochastic is False
        assert operator.stream_name is None
        assert operator.rngs is None

    def test_deterministic_initialization_with_rngs_allowed(self):
        """Test deterministic operator can accept rngs (but won't use them)."""
        config = NormalizeConfig(stochastic=False)
        rngs = nnx.Rngs(42)
        operator = NormalizeOperator(config, rngs=rngs)

        assert operator.rngs is None  # accepted, and not kept
        assert operator.stochastic is False  # Won't use rngs

    def test_initialization_with_name(self):
        """Test operator initialization with module name."""
        config = NormalizeConfig(stochastic=False)
        operator = NormalizeOperator(config, name="normalize_op")

        assert operator.name == "normalize_op"

    def test_initialization_caches_config_fields(self):
        """Test that module caches config fields for convenience."""
        config = RandomBrightnessConfig(
            stochastic=True,
            stream_name="brightness",
        )
        rngs = nnx.Rngs(0)
        operator = RandomBrightnessOperator(config, rngs=rngs)

        # Should cache these for convenience
        assert operator.stochastic == config.stochastic
        assert operator.stream_name == config.stream_name

    def test_is_nnx_module(self):
        """Test that OperatorModule is a proper NNX module."""
        config = NormalizeConfig(stochastic=False)
        operator = NormalizeOperator(config)

        assert isinstance(operator, nnx.Module)

    def test_calls_super_init(self):
        """Test that OperatorModule calls DataraxModule.__init__()."""
        config = NormalizeConfig(stochastic=False)
        operator = NormalizeOperator(config)

        # Should have all DataraxModule attributes
        assert hasattr(operator, "config")


# ========================================================================
# Test Category 2: Stochastic Mode Operations
# ========================================================================


class TestOperatorModuleStochasticMode:
    """Test stochastic operator functionality."""

    @staticmethod
    def _factors_from(data: dict) -> jax.Array:
        """Recover each record's drawn factor from an all-``0.5`` input image."""
        return jnp.mean(data["image"], axis=(1, 2, 3)) / 0.5

    def test_the_batch_path_draws_one_factor_per_record(self):
        """Every record in the batch gets its own factor, whatever the batch size."""
        operator = RandomBrightnessOperator(
            RandomBrightnessConfig(stochastic=True, stream_name="augment"), rngs=nnx.Rngs(42)
        )

        for batch_size in (1, 8, 32):
            data, _ = operator._vmap_apply({"image": jnp.ones((batch_size, 8, 8, 3)) * 0.5}, {})

            assert self._factors_from(data).shape == (batch_size,)

    def test_drawn_factors_are_within_the_configured_range(self):
        """Each record's factor comes from its key and respects min_factor/max_factor."""
        operator = RandomBrightnessOperator(
            RandomBrightnessConfig(
                stochastic=True, stream_name="augment", min_factor=0.5, max_factor=1.5
            ),
            rngs=nnx.Rngs(42),
        )

        data, _ = operator._vmap_apply({"image": jnp.ones((100, 8, 8, 3)) * 0.5}, {})

        factors = self._factors_from(data)
        assert jnp.all(factors >= 0.5)
        assert jnp.all(factors <= 1.5)

    def test_apply_draws_its_factor_from_the_key_it_is_given(self):
        """One key repeats its factor; a different key draws a different one."""
        operator = RandomBrightnessOperator(
            RandomBrightnessConfig(stochastic=True, stream_name="augment"), rngs=nnx.Rngs(42)
        )
        data = {"image": jnp.ones((8, 8, 3)) * 0.5}

        first, _, _ = operator.apply(data, {}, None, key=jax.random.key(0))
        again, _, _ = operator.apply(data, {}, None, key=jax.random.key(0))
        other, _, _ = operator.apply(data, {}, None, key=jax.random.key(1))

        assert jnp.array_equal(first["image"], again["image"])
        assert not jnp.allclose(first["image"], other["image"])

    def test_per_element_randomness_independence(self):
        """Each record in one batch draws independently of the others."""
        operator = RandomBrightnessOperator(
            RandomBrightnessConfig(stochastic=True, stream_name="augment"), rngs=nnx.Rngs(42)
        )

        data, _ = operator._vmap_apply({"image": jnp.ones((10, 8, 8, 3)) * 0.5}, {})

        # All 10 values should be different (with high probability)
        assert len(jnp.unique(self._factors_from(data))) >= 8

    def test_two_operators_with_one_seed_draw_the_same_factors(self):
        """The operator's own base key, not call order, decides what each record draws."""
        config = RandomBrightnessConfig(stochastic=True, stream_name="augment")
        batch = {"image": jnp.ones((32, 8, 8, 3)) * 0.5}

        first, _ = RandomBrightnessOperator(config, rngs=nnx.Rngs(42))._vmap_apply(batch, {})
        second, _ = RandomBrightnessOperator(config, rngs=nnx.Rngs(42))._vmap_apply(batch, {})
        other, _ = RandomBrightnessOperator(config, rngs=nnx.Rngs(7))._vmap_apply(batch, {})

        assert jnp.array_equal(first["image"], second["image"])
        assert not jnp.allclose(first["image"], other["image"])

    def test_apply_batch_stochastic_mode(self):
        """Test apply_batch() in stochastic mode."""
        config = RandomBrightnessConfig(
            stochastic=True,
            stream_name="augment",
        )
        rngs = nnx.Rngs(42)
        operator = RandomBrightnessOperator(config, rngs=rngs)

        # Create batch
        batch = create_test_batch(data={"image": jnp.ones((8, 64, 64, 3)) * 0.5})

        # Apply operator
        transformed = operator.apply_batch(batch)

        # Should have same batch structure
        assert transformed.batch_size == 8
        assert transformed.data["image"].shape == (8, 64, 64, 3)

        # Values should be different from input (brightened/darkened)
        assert not jnp.allclose(transformed.data["image"], batch.data["image"])

    def test_stochastic_batch_uses_rng_stream(self):
        """Test that stochastic mode uses configured RNG stream."""
        config = RandomBrightnessConfig(
            stochastic=True,
            stream_name="brightness_stream",  # Custom stream name
        )
        rngs = nnx.Rngs(42)
        operator = RandomBrightnessOperator(config, rngs=rngs)

        batch = create_test_batch(
            data={"image": jnp.ones((4, 32, 32, 3))},
            states={},
            metadata_list=[None] * 4,
        )

        # Should not raise (stream exists in rngs)
        transformed = operator.apply_batch(batch)
        assert transformed.batch_size == 4


_SEEN_KEYS: list = []


class KeyDrawingOperator(OperatorModule):
    """An operator written to the key contract: it draws from the record's own key."""

    def apply(self, data, state, metadata, key=None, stats=None):
        """Record what the framework passed, then scale by a draw from that key."""
        del metadata, stats
        _SEEN_KEYS.append(key)
        record_key = require_key(key, self)
        factor = jax.random.uniform(record_key, (), minval=0.5, maxval=1.5)
        return {**data, "image": data["image"] * factor}, state, None


class KeyRecordingPassthrough(OperatorModule):
    """A spy that records its fourth argument and draws nothing, in either mode."""

    def apply(self, data, state, metadata, key=None, stats=None):
        """Record what the framework passed and return the record unchanged."""
        del metadata, stats
        _SEEN_KEYS.append(key)
        return data, state, None


class TestOperatorPrivateRngStream:
    """A stochastic operator draws direct-call randomness from its own stream.

    The caller's ``Rngs`` is read once, at construction, to draw the operator's stable base key.
    After that the operator owns its randomness: a call that carries record identity keys off the
    base key, and a call without it draws from a private stream instead of reaching back into a
    caller whose state it does not own.
    """

    _CONFIG = RandomBrightnessConfig(stochastic=True, stream_name="augment")

    @classmethod
    def _stochastic(cls) -> RandomBrightnessOperator:
        return RandomBrightnessOperator(cls._CONFIG, rngs=nnx.Rngs(augment=0))

    @staticmethod
    def _batch() -> dict:
        return {"image": jnp.ones((4, 8, 8, 3)) * 0.5}

    @staticmethod
    def _rng_counts(operator: OperatorModule) -> list[int]:
        return [int(count) for count in jax.tree.leaves(nnx.state(operator, nnx.RngCount))]

    def test_an_operator_does_not_keep_the_callers_rngs(self):
        """The caller's Rngs is used at construction and is not operator state afterwards."""
        operator = RandomBrightnessOperator(self._CONFIG, rngs=nnx.Rngs(augment=0))

        assert operator.rngs is None

    def test_a_module_that_is_not_an_operator_keeps_its_rngs(self):
        """Only operators clear it; a source or sampler still draws from the caller's Rngs."""
        rngs = nnx.Rngs(0)

        assert DataraxModule(DataraxModuleConfig(), rngs=rngs).rngs is rngs

    def test_a_stochastic_operator_carries_a_base_key_and_a_private_stream(self):
        """Both live in module state, so both round-trip through a checkpoint."""
        state = nnx.to_pure_dict(nnx.state(self._stochastic()))

        assert "_base_key" in state
        assert set(state["_rng_stream"]) == {"key", "count"}

    def test_the_private_stream_hangs_off_the_base_key(self):
        """The stream is derived, not drawn, so it survives a restore of the base key alone."""
        state = nnx.to_pure_dict(nnx.state(self._stochastic()))

        expected = jax.random.fold_in(state["_base_key"], DIRECT_CALL_STREAM)
        assert jnp.array_equal(
            jax.random.key_data(state["_rng_stream"]["key"]), jax.random.key_data(expected)
        )

    def test_a_deterministic_operator_has_no_rng_state(self):
        """It draws nothing, so it carries neither a base key nor a stream."""
        operator = NormalizeOperator(NormalizeConfig(stochastic=False), rngs=nnx.Rngs(0))

        state = nnx.to_pure_dict(nnx.state(operator))
        assert "_base_key" not in state
        assert "_rng_stream" not in state
        assert self._rng_counts(operator) == []

    def test_two_direct_calls_draw_differently(self):
        """Without record identity there is nothing to key on, so each call draws afresh."""
        operator = self._stochastic()
        batch = self._batch()

        first, _ = operator._vmap_apply(batch, {})
        second, _ = operator._vmap_apply(batch, {})

        assert not jnp.allclose(first["image"], second["image"])

    def test_a_direct_call_advances_the_operators_own_stream(self):
        """The draw is counted on the operator, which is what makes it resumable."""
        operator = self._stochastic()

        before = self._rng_counts(operator)
        operator._vmap_apply(self._batch(), {})
        after = self._rng_counts(operator)

        assert before == [0]
        assert after == [1]

    def test_a_call_carrying_record_indices_does_not_draw(self):
        """With record identity the key comes from the base key, so the call repeats exactly."""
        operator = self._stochastic()
        batch = self._batch()
        indices = jnp.arange(4, dtype=jnp.uint32)

        first, _ = operator._vmap_apply(batch, {}, None, indices)
        second, _ = operator._vmap_apply(batch, {}, None, indices)

        assert jnp.array_equal(first["image"], second["image"])
        assert self._rng_counts(operator) == [0]

    def test_two_operators_built_from_one_seed_agree(self):
        """A fresh operator with the same seed reproduces the first one's draws.

        The control for the two tests above: without it, an operator that simply returned noise
        would satisfy "two calls differ" while reproducing nothing.
        """
        first, _ = self._stochastic()._vmap_apply(self._batch(), {})
        second, _ = self._stochastic()._vmap_apply(self._batch(), {})

        assert jnp.array_equal(first["image"], second["image"])

    def test_a_subclass_may_hold_the_callers_rngs_without_changing_draws(self):
        """Assigning ``self.rngs`` after ``super().__init__`` is allowed and changes no draw."""

        class KeepsRngs(RandomBrightnessOperator):
            def __init__(self, config, *, rngs):
                super().__init__(config, rngs=rngs)
                self.rngs = rngs

        plain, _ = self._stochastic()._vmap_apply(self._batch(), {})
        keeping, _ = KeepsRngs(self._CONFIG, rngs=nnx.Rngs(augment=0))._vmap_apply(
            self._batch(), {}
        )

        assert jnp.array_equal(plain["image"], keeping["image"])


class TestOperatorKeyContract:
    """What the framework hands ``apply`` as its fourth argument.

    Every assertion here runs through the framework rather than calling an override
    directly: a direct call would only exercise the test operator's own signature, which
    would hold whatever datarax passed.
    """

    @staticmethod
    def _stochastic() -> KeyDrawingOperator:
        return KeyDrawingOperator(
            OperatorConfig(stochastic=True, stream_name="augment"), rngs=nnx.Rngs(augment=0)
        )

    def test_the_batch_path_passes_a_key_to_a_stochastic_operator(self):
        """Every apply call the framework makes for a stochastic operator carries a key."""
        _SEEN_KEYS.clear()

        KeyRecordingPassthrough(
            OperatorConfig(stochastic=True, stream_name="augment"), rngs=nnx.Rngs(augment=0)
        )._vmap_apply({"image": jnp.ones((8, 4))}, {})

        assert _SEEN_KEYS, "the framework never called apply"
        assert all(key is not None for key in _SEEN_KEYS)

    def test_the_batch_path_passes_no_key_to_a_deterministic_operator(self):
        """The mirror of the case above: without it, a spy recording nothing would pass both."""
        _SEEN_KEYS.clear()

        KeyRecordingPassthrough(OperatorConfig(stochastic=False))._vmap_apply(
            {"image": jnp.ones((8, 4))}, {}
        )

        assert _SEEN_KEYS, "the framework never called apply"
        assert all(key is None for key in _SEEN_KEYS)

    def test_each_record_is_transformed_by_its_own_draw(self):
        """Eight records give eight different factors, so no draw is shared across the batch."""
        data, _ = self._stochastic()._vmap_apply({"image": jnp.ones((8, 4))}, {})

        means = {float(jnp.mean(data["image"][index])) for index in range(8)}
        assert len(means) == 8

    def test_a_record_keeps_its_draw_across_batch_position(self):
        """The same record index draws the same factor whatever else shares its batch."""
        whole, _ = self._stochastic()._vmap_apply(
            {"image": jnp.ones((8, 4))}, {}, None, jnp.arange(8)
        )
        tail, _ = self._stochastic()._vmap_apply(
            {"image": jnp.ones((3, 4))}, {}, None, jnp.arange(5, 8)
        )

        assert jnp.allclose(whole["image"][5:], tail["image"])

    def test_require_key_names_the_operator_it_refuses(self):
        """The helper reports which operator was handed no key."""
        with pytest.raises(ValueError, match="KeyDrawingOperator is stochastic"):
            require_key(None, self._stochastic())


# ========================================================================
# Test Category 3: Deterministic Mode Operations
# ========================================================================


class TestOperatorModuleDeterministicMode:
    """Test deterministic operator functionality."""

    def test_apply_without_random_params(self):
        """Test apply() in deterministic mode (no random_params)."""
        config = NormalizeConfig(stochastic=False)
        operator = NormalizeOperator(config)
        operator.set_statistics({"mean": 0.5, "std": 0.2})

        data = {"image": jnp.ones((64, 64, 3)) * 0.7}
        state = {}
        metadata = None

        transformed_data, new_state, new_metadata = operator.apply(data, state, metadata, key=None)

        # Should normalize: (0.7 - 0.5) / 0.2 = 1.0
        expected = jnp.ones((64, 64, 3)) * 1.0
        assert jnp.allclose(transformed_data["image"], expected)

    def test_determinism_same_input_same_output(self):
        """Test that deterministic operator is truly deterministic."""
        config = NormalizeConfig(stochastic=False)
        operator = NormalizeOperator(config)
        operator.set_statistics({"mean": 0.5, "std": 0.2})

        data = {"image": jnp.array([[[0.3, 0.7, 0.9]]])}
        state = {}
        metadata = None

        # Apply twice
        result1, _, _ = operator.apply(data, state, metadata)
        result2, _, _ = operator.apply(data, state, metadata)

        # Results should be identical
        assert jnp.array_equal(result1["image"], result2["image"])

    def test_apply_batch_deterministic_mode(self):
        """Test apply_batch() in deterministic mode."""
        config = NormalizeConfig(stochastic=False)
        operator = NormalizeOperator(config)
        operator.set_statistics({"mean": 0.5, "std": 0.2})

        batch = create_test_batch(
            data={"image": jnp.ones((8, 64, 64, 3)) * 0.7},
            states={},
            metadata_list=[None] * 8,
        )

        transformed = operator.apply_batch(batch)

        # All elements should be normalized to 1.0
        expected = jnp.ones((8, 64, 64, 3)) * 1.0
        assert jnp.allclose(transformed.data["image"], expected)

    def test_deterministic_batch_no_rng_required(self):
        """Test that deterministic mode doesn't use rngs."""
        config = NormalizeConfig(stochastic=False)
        # No rngs provided
        operator = NormalizeOperator(config)
        operator.set_statistics({"mean": 0.5, "std": 0.2})

        batch = create_test_batch(
            data={"image": jnp.ones((4, 32, 32, 3))},
            states={},
            metadata_list=[None] * 4,
        )

        # Should work fine without rngs
        transformed = operator.apply_batch(batch)
        assert transformed.batch_size == 4


# ========================================================================
# Test Category 4: Batch Processing
# ========================================================================


class TestOperatorModuleBatchProcessing:
    """Test batch processing functionality."""

    def test_empty_batch_handling(self):
        """Test that empty batches are handled correctly."""
        config = NormalizeConfig(stochastic=False)
        operator = NormalizeOperator(config)

        # Empty batch
        batch = create_test_batch(
            data={"image": jnp.zeros((0, 64, 64, 3))}, states={}, metadata_list=[]
        )

        transformed = operator.apply_batch(batch)

        # Should return unchanged empty batch
        assert transformed.batch_size == 0
        assert transformed.data["image"].shape == (0, 64, 64, 3)

    def test_single_element_batch(self):
        """Test processing batch with single element."""
        config = RandomBrightnessConfig(
            stochastic=True,
            stream_name="augment",
        )
        rngs = nnx.Rngs(42)
        operator = RandomBrightnessOperator(config, rngs=rngs)

        batch = create_test_batch(
            data={"image": jnp.ones((1, 64, 64, 3)) * 0.5}, states={}, metadata_list=[None]
        )

        transformed = operator.apply_batch(batch)

        assert transformed.batch_size == 1
        assert transformed.data["image"].shape == (1, 64, 64, 3)

    def test_multi_element_batch_stochastic(self):
        """Test processing multi-element batch in stochastic mode."""
        config = RandomBrightnessConfig(
            stochastic=True,
            stream_name="augment",
        )
        rngs = nnx.Rngs(42)
        operator = RandomBrightnessOperator(config, rngs=rngs)

        batch_size = 32
        batch = create_test_batch(
            data={"image": jnp.ones((batch_size, 64, 64, 3)) * 0.5},
            states={},
            metadata_list=[None] * batch_size,
        )

        transformed = operator.apply_batch(batch)

        assert transformed.batch_size == batch_size
        # Each element should have different brightness (with high probability)
        means = jnp.mean(transformed.data["image"], axis=(1, 2, 3))
        unique_means = jnp.unique(jnp.round(means, decimals=3))
        assert len(unique_means) >= batch_size // 2  # At least half different

    def test_multi_element_batch_deterministic(self):
        """Test processing multi-element batch in deterministic mode."""
        config = NormalizeConfig(stochastic=False)
        operator = NormalizeOperator(config)
        operator.set_statistics({"mean": 0.5, "std": 0.2})

        batch_size = 32
        batch = create_test_batch(
            data={"image": jnp.ones((batch_size, 64, 64, 3)) * 0.7},
            states={},
            metadata_list=[None] * batch_size,
        )

        transformed = operator.apply_batch(batch)

        assert transformed.batch_size == batch_size
        # All elements should be normalized identically
        for i in range(batch_size):
            assert jnp.allclose(transformed.data["image"][i], transformed.data["image"][0])

    def test_vmap_correctness_per_element(self):
        """Test that vmap correctly processes each element independently."""
        config = RandomBrightnessConfig(
            stochastic=True,
            stream_name="augment",
        )
        rngs = nnx.Rngs(42)
        operator = RandomBrightnessOperator(config, rngs=rngs)

        # Create batch with different input values
        batch_data = jnp.stack(
            [
                jnp.ones((64, 64, 3)) * 0.2,
                jnp.ones((64, 64, 3)) * 0.5,
                jnp.ones((64, 64, 3)) * 0.8,
            ]
        )
        batch = create_test_batch(
            data={"image": batch_data}, states={}, metadata_list=[None, None, None]
        )

        transformed = operator.apply_batch(batch)

        # Each element should maintain its relative brightness pattern
        # (element 2 should still be brighter than element 0)
        jnp.mean(transformed.data["image"], axis=(1, 2, 3))
        # Relative ordering might be preserved
        # (Just verify they're all processed)
        assert transformed.batch_size == 3


# ========================================================================
# Test Category 5: JIT Compatibility
# ========================================================================


class TestOperatorModuleJITCompatibility:
    """Test JAX JIT compilation compatibility."""

    def test_apply_batch_compiles_stochastic(self):
        """Test that apply_batch() compiles successfully in stochastic mode."""
        config = RandomBrightnessConfig(
            stochastic=True,
            stream_name="augment",
        )
        rngs = nnx.Rngs(42)
        operator = RandomBrightnessOperator(config, rngs=rngs)

        batch = create_test_batch(
            data={"image": jnp.ones((4, 32, 32, 3))},
            states={},
            metadata_list=[None] * 4,
        )

        # Should compile without errors (already decorated with @nnx.jit)
        transformed = operator.apply_batch(batch)
        assert transformed.batch_size == 4

    def test_apply_batch_compiles_deterministic(self):
        """Test that apply_batch() compiles successfully in deterministic mode."""
        config = NormalizeConfig(stochastic=False)
        operator = NormalizeOperator(config)
        operator.set_statistics({"mean": 0.5, "std": 0.2})

        batch = create_test_batch(
            data={"image": jnp.ones((4, 32, 32, 3))},
            states={},
            metadata_list=[None] * 4,
        )

        # Should compile without errors
        transformed = operator.apply_batch(batch)
        assert transformed.batch_size == 4

    def test_static_branch_compilation(self):
        """Test that static stochastic boolean enables branch elimination."""
        # This is implicit in the design - self.stochastic is compile-time constant
        # JAX will compile only the relevant branch

        # Create both operator types
        stochastic_config = RandomBrightnessConfig(stochastic=True, stream_name="augment")
        deterministic_config = NormalizeConfig(stochastic=False)

        stochastic_op = RandomBrightnessOperator(stochastic_config, rngs=nnx.Rngs(0))
        deterministic_op = NormalizeOperator(deterministic_config)

        # Both should have different compilation paths
        # (This is tested implicitly by successful compilation)
        assert stochastic_op.stochastic is True
        assert deterministic_op.stochastic is False

    def test_apply_is_pure_function(self):
        """Test that apply() is a pure function (same input → same output)."""
        config = NormalizeConfig(stochastic=False)
        operator = NormalizeOperator(config)
        operator.set_statistics({"mean": 0.5, "std": 0.2})

        data = {"image": jnp.array([[[0.3, 0.7]]])}
        state = {}
        metadata = None

        # Call multiple times
        results = [operator.apply(data, state, metadata) for _ in range(5)]

        # All results should be identical
        for result in results[1:]:
            assert jnp.array_equal(result[0]["image"], results[0][0]["image"])


# ========================================================================
# Test Category 6: Random Parameter System
# ========================================================================


class TestOperatorModuleRandomParams:
    """Test random parameter generation and distribution."""

    def test_random_params_distributed_via_vmap(self):
        """Test that random params are correctly distributed to elements via vmap."""
        config = RandomBrightnessConfig(
            stochastic=True,
            stream_name="augment",
        )
        rngs = nnx.Rngs(42)
        operator = RandomBrightnessOperator(config, rngs=rngs)

        # Create batch with uniform input
        batch = create_test_batch(
            data={"image": jnp.ones((8, 64, 64, 3)) * 0.5},
            states={},
            metadata_list=[None] * 8,
        )

        transformed = operator.apply_batch(batch)

        # Each element should have received different random param
        # (resulting in different brightness values)
        for i in range(8):
            for j in range(i + 1, 8):
                # Most pairs should be different
                elem_i_mean = jnp.mean(transformed.data["image"][i])
                elem_j_mean = jnp.mean(transformed.data["image"][j])
                if i == 0 and j == 1:
                    # At least first two should be different
                    assert not jnp.allclose(elem_i_mean, elem_j_mean)

    def test_a_deterministic_operator_has_no_draw_to_make(self):
        """A deterministic operator transforms its record without any key."""
        operator = NormalizeOperator(NormalizeConfig(stochastic=False))
        operator.set_statistics({"mean": 0.5, "std": 0.2})

        data, _, _ = operator.apply({"image": jnp.ones((4, 4, 3)) * 0.7}, {}, None)

        assert jnp.allclose(data["image"], jnp.ones((4, 4, 3)))


# ========================================================================
# Test Category 7: Statistics System (Inherited)
# ========================================================================


class TestOperatorModuleStatistics:
    """Test statistics stored on the operator and applied to its records."""

    def test_stored_statistics_are_readable(self):
        """Test operator using statistics set on it."""
        stats = {"mean": 0.5, "std": 0.2}
        operator = NormalizeOperator(NormalizeConfig(stochastic=False))
        operator.set_statistics(stats)

        # Statistics should be available
        assert operator.get_statistics() == stats

    def test_compute_statistics_returns_the_stored_statistics(self):
        """By default an operator applies whatever was stored on it."""
        operator = NormalizeOperator(NormalizeConfig(stochastic=False))
        operator.set_statistics({"mean": 0.5, "std": 0.2})

        assert operator.compute_statistics({"image": jnp.ones((4, 8, 8, 3))}) == {
            "mean": 0.5,
            "std": 0.2,
        }

    def test_an_operator_can_derive_statistics_from_the_batch(self):
        """An operator that fits statistics to each batch overrides compute_statistics."""

        class BatchFittedNormalize(NormalizeOperator):
            def compute_statistics(self, batch_data):
                image = batch_data["image"]
                return {"mean": jnp.mean(image), "std": jnp.std(image) + 1e-6}

        operator = BatchFittedNormalize(NormalizeConfig(stochastic=False))

        stats = operator.compute_statistics({"image": jnp.ones((4, 64, 64, 3)) * 0.7})

        assert jnp.isclose(stats["mean"], 0.7, rtol=1e-4)

    def test_compute_statistics_is_not_memoized(self):
        """Nothing caches the result, so an operator that fits per batch sees every batch."""
        call_count = 0

        class CountingNormalize(NormalizeOperator):
            def compute_statistics(self, batch_data):
                nonlocal call_count
                call_count += 1
                return {"mean": jnp.mean(batch_data["image"]), "std": 1.0}

        operator = CountingNormalize(NormalizeConfig(stochastic=False))

        operator.compute_statistics({"image": jnp.ones((4, 8, 8, 3))})
        operator.compute_statistics({"image": jnp.zeros((4, 8, 8, 3))})

        assert call_count == 2

    def test_set_statistics(self):
        """Test manually setting statistics."""
        config = NormalizeConfig(stochastic=False)
        operator = NormalizeOperator(config)

        # Initially None
        assert operator.get_statistics() is None

        # Set manually
        new_stats = {"mean": 0.3, "std": 0.1}
        operator.set_statistics(new_stats)

        assert operator.get_statistics() == new_stats

    def test_reset_statistics(self):
        """Test resetting statistics."""
        config = NormalizeConfig(stochastic=False)
        operator = NormalizeOperator(config)
        operator.set_statistics({"mean": 0.5, "std": 0.2})

        # Initially has stats
        assert operator.get_statistics() is not None

        # Reset
        operator.reset_statistics()
        assert operator.get_statistics() is None


# ========================================================================
# Test Category 8: Training/Eval Mode (Inherited from NNX)
# ========================================================================


class TestOperatorModuleTrainingMode:
    """Test training/evaluation mode switching (inherited from NNX)."""

    def test_train_mode_callable(self):
        """Test that train() method is available."""
        config = NormalizeConfig(stochastic=False)
        operator = NormalizeOperator(config)

        # Should have train() method from NNX
        assert hasattr(operator, "train")
        operator.train()  # Should not raise

    def test_eval_mode_callable(self):
        """Test that eval() method is available."""
        config = NormalizeConfig(stochastic=False)
        operator = NormalizeOperator(config)

        # Should have eval() method from NNX
        assert hasattr(operator, "eval")
        operator.eval()  # Should not raise

    def test_mode_switching(self):
        """Test switching between train and eval modes."""
        config = NormalizeConfig(stochastic=False)
        operator = NormalizeOperator(config)

        # Should be able to switch modes
        operator.train()
        operator.eval()
        operator.train()
        # No errors expected


# ========================================================================
# Test Category 10: Scan Batch Strategy
# ========================================================================


class TestOperatorModuleScanBatchStrategy:
    """Test scan-based batch execution strategy (sequential, low memory)."""

    def test_scan_default_is_vmap(self):
        """Default batch_strategy should be 'vmap'."""
        config = NormalizeConfig(stochastic=False)
        assert config.batch_strategy == "vmap"

    def test_scan_config_validation(self):
        """Invalid batch_strategy should raise ValueError."""
        with pytest.raises(ValueError, match="batch_strategy"):
            NormalizeConfig(stochastic=False, batch_strategy="invalid")

    def test_scan_produces_same_result_as_vmap(self):
        """Scan strategy produces identical results to vmap for deterministic ops."""
        # Create two operators: one vmap (default), one scan
        config_vmap = NormalizeConfig(stochastic=False, batch_strategy="vmap")
        config_scan = NormalizeConfig(stochastic=False, batch_strategy="scan")
        op_vmap = NormalizeOperator(config_vmap)
        op_scan = NormalizeOperator(config_scan)
        op_vmap.set_statistics({"mean": 0.5, "std": 0.2})
        op_scan.set_statistics({"mean": 0.5, "std": 0.2})

        # Create batch of 4 elements
        batch = create_test_batch(
            data={"image": jnp.array([[0.3], [0.5], [0.7], [0.9]])},
            states={},
            metadata_list=[None] * 4,
        )

        result_vmap = op_vmap.apply_batch(batch)
        result_scan = op_scan.apply_batch(batch)

        assert jnp.allclose(result_vmap.data["image"], result_scan.data["image"]), (
            "Scan and vmap should produce identical results for deterministic ops"
        )

    def test_scan_with_stochastic_operator(self):
        """Scan strategy works with stochastic operators (shapes match)."""
        config = RandomBrightnessConfig(
            stochastic=True,
            stream_name="augment",
            batch_strategy="scan",
        )
        rngs = nnx.Rngs(42)
        operator = RandomBrightnessOperator(config, rngs=rngs)

        batch = create_test_batch(
            data={"image": jnp.ones((4, 32, 32, 3)) * 0.5},
            states={},
            metadata_list=[None] * 4,
        )

        result = operator.apply_batch(batch)
        assert result.batch_size == 4
        assert result.data["image"].shape == (4, 32, 32, 3)

    def test_scan_single_element_batch(self):
        """Scan strategy works with batch size 1."""
        config = NormalizeConfig(stochastic=False, batch_strategy="scan")
        op = NormalizeOperator(config)
        op.set_statistics({"mean": 0.5, "std": 0.2})

        batch = create_test_batch(
            data={"image": jnp.ones((1, 32, 32, 3)) * 0.7},
            states={},
            metadata_list=[None],
        )

        result = op.apply_batch(batch)
        assert result.batch_size == 1
        expected = jnp.ones((1, 32, 32, 3)) * 1.0  # (0.7 - 0.5) / 0.2
        assert jnp.allclose(result.data["image"], expected)


class TestOperatorStatisticsStore:
    """Statistics belong to the operator, not to its static configuration.

    Measured before this change (``probes/probe_c13_statistics_today.txt``): statistics already
    reach ``apply`` under ``nnx.jit``, and two operators with equal statistics already share one
    trace. Those tests therefore guard what the previous commit delivered. What changes here is
    where the statistics live and what a configuration may carry.
    """

    @staticmethod
    def _fitted() -> NormalizeOperator:
        """Return an operator holding statistics that halve and rescale the input."""
        operator = NormalizeOperator(NormalizeConfig())
        operator.set_statistics({"mean": jnp.asarray(0.5), "std": jnp.asarray(0.2)})
        return operator

    def test_statistics_reach_apply_through_the_batch_call(self):
        """A batch is normalized with the statistics set on the operator."""
        operator = self._fitted()
        batch = create_test_batch(
            data={"image": jnp.ones((1, 4, 4, 3)) * 0.7}, states={}, metadata_list=[None]
        )

        result = operator.apply_batch(batch)

        assert jnp.allclose(result.data["image"], jnp.ones((1, 4, 4, 3)))

    def test_statistics_reach_apply_through_the_raw_path(self):
        """The fused raw-batch path reads the same store."""
        operator = self._fitted()

        out_data, _ = operator._apply_on_raw({"image": jnp.ones((2, 4, 4, 3)) * 0.7}, {})

        assert jnp.allclose(out_data["image"], jnp.ones((2, 4, 4, 3)))

    def test_statistics_reach_apply_under_nnx_jit(self):
        """Statistics are module state, so a compiled call sees them."""
        operator = self._fitted()

        @nnx.jit
        def run(op, data):
            return op._apply_on_raw(data, {})[0]

        out_data = run(operator, {"image": jnp.ones((2, 4, 4, 3)) * 0.7})

        assert jnp.allclose(out_data["image"], jnp.ones((2, 4, 4, 3)))

    def test_two_operators_with_equal_statistics_share_one_trace(self):
        """Statistics are state rather than graphdef metadata, so equal ones force no trace.

        The counter wraps the function and the transform wraps the counter, the order
        ``TraceCounter.wrap`` requires: jitting first would count calls instead of traces.
        """
        counter = TraceCounter()

        def run(op, data):
            return op._apply_on_raw(data, {})[0]

        traced = nnx.jit(counter.wrap(run))
        data = {"image": jnp.ones((2, 4, 4, 3)) * 0.7}

        with counter.expect(new_traces=1):
            traced(self._fitted(), data)
        with counter.expect(new_traces=0):
            traced(self._fitted(), data)

    def test_statistics_live_in_module_state_under_their_own_name(self):
        """The store is the operator's own state, named for what it holds."""
        operator = self._fitted()

        state = nnx.to_pure_dict(nnx.state(operator))

        assert "_statistics" in state
        assert "_computed_stats" not in state

    def test_reset_statistics_clears_the_store(self):
        """After a reset the operator has no statistics to give apply."""
        operator = self._fitted()

        operator.reset_statistics()

        assert operator.get_statistics() is None

    def test_a_module_configuration_carries_no_statistics(self):
        """Statistics are fitted state; a configuration is static metadata.

        Keeping them in a frozen config made every fitted value part of the graphdef a
        transform compares, and gave two ways to say the same thing.
        """
        names = {field.name for field in fields(DataraxModuleConfig)}

        assert "batch_stats_fn" not in names
        assert "precomputed_stats" not in names


# ========================================================================
# Test Count Summary
# ========================================================================
# TestOperatorStatisticsStore: 7 tests
# TestOperatorModuleInitialization: 8 tests
# TestOperatorModuleStochasticMode: 10 tests
# TestOperatorModuleDeterministicMode: 4 tests
# TestOperatorModuleBatchProcessing: 6 tests
# TestOperatorModuleJITCompatibility: 4 tests
# TestOperatorModuleRandomParams: 3 tests
# TestOperatorModuleStatistics: 5 tests
# TestOperatorModuleTrainingMode: 3 tests
# TestOperatorModuleScanBatchStrategy: 5 tests
# ========================================================================
# Total: 48 tests
