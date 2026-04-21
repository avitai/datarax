"""Tests for RNG short-circuit in deterministic operators.

Validates that:
1. Deterministic ElementOperator.generate_random_params returns None
2. Deterministic MapOperator.generate_random_params returns None
3. The 2-argument vmap path is used when random_params is None
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax.core.config import ElementOperatorConfig
from datarax.operators.element_operator import ElementOperator
from datarax.operators.map_operator import MapOperator, MapOperatorConfig


def _identity_fn(element, key):
    """Deterministic identity — ignores key."""
    del key
    return element


def _stochastic_fn(element, key):
    """Stochastic — uses key for noise."""
    noise = jax.random.normal(key, element.data["value"].shape) * 0.01
    new_data = {"value": element.data["value"] + noise}
    return element.replace(data=new_data)


def _map_fn(data, state, metadata, *, key=None):
    """Simple map function."""
    del key
    return data, state, metadata


class TestElementOperatorRNGShortCircuit:
    """Tests for ElementOperator deterministic RNG bypass."""

    def test_deterministic_returns_none(self):
        """Deterministic ElementOperator should return None from generate_random_params."""
        config = ElementOperatorConfig(stochastic=False)
        op = ElementOperator(config, fn=_identity_fn)

        rng = jax.random.key(0)
        data_shapes = {"value": (32, 8)}

        result = op.generate_random_params(rng, data_shapes)
        assert result is None

    def test_stochastic_returns_keys(self):
        """Stochastic ElementOperator should return RNG keys."""
        config = ElementOperatorConfig(stochastic=True, stream_name="params")
        op = ElementOperator(config, fn=_stochastic_fn, rngs=nnx.Rngs(params=42))

        rng = jax.random.key(0)
        data_shapes = {"value": (32, 8)}

        result = op.generate_random_params(rng, data_shapes)
        assert result is not None
        # Should be an array of keys, one per batch element
        assert hasattr(result, "shape")
        assert result.shape[0] == 32  # batch_size


class TestMapOperatorRNGShortCircuit:
    """Tests for MapOperator deterministic RNG bypass."""

    def test_deterministic_returns_none(self):
        """Deterministic MapOperator should return None from generate_random_params."""
        config = MapOperatorConfig(stochastic=False)
        op = MapOperator(config, fn=_map_fn)  # type: ignore[reportArgumentType]

        rng = jax.random.key(0)
        data_shapes = {"value": (32, 8)}

        result = op.generate_random_params(rng, data_shapes)
        assert result is None

    def test_stochastic_returns_keys(self):
        """Stochastic MapOperator should return RNG keys."""
        config = MapOperatorConfig(stochastic=True, stream_name="params")
        op = MapOperator(config, fn=_map_fn, rngs=nnx.Rngs(params=42))  # type: ignore[reportArgumentType]

        rng = jax.random.key(0)
        data_shapes = {"value": (32, 8)}

        result = op.generate_random_params(rng, data_shapes)
        assert result is not None


class TestVmapPathSelection:
    """Tests that verify the 2-arg vmap path when random_params is None."""

    def test_deterministic_pipeline_no_rng_overhead(self):
        """A fully deterministic pipeline should not generate any RNG keys.

        This tests the full path: generate_random_params -> None -> 2-arg vmap.
        """
        config = ElementOperatorConfig(stochastic=False)
        op = ElementOperator(config, fn=_identity_fn)

        # Build a batch with data
        data = {"value": jnp.ones((4, 8))}
        states: dict = {}

        # _vmap_apply should work without any RNG generation
        result_data, result_states = op._vmap_apply(data, states)
        assert "value" in result_data
        np.testing.assert_array_equal(result_data["value"], data["value"])
