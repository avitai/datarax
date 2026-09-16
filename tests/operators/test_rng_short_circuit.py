"""Tests for RNG short-circuit in deterministic operators.

Validates that:
1. Deterministic ElementOperator.generate_random_params returns None
2. Deterministic MapOperator.generate_random_params returns None
3. The 2-argument vmap path is used when random_params is None
4. A deterministic operator's apply receives no random parameters at all
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax.core.config import ElementOperatorConfig, OperatorConfig
from datarax.core.operator import OperatorModule
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

        element_keys = jax.random.split(jax.random.key(0), 32)  # one key per record
        data_shapes = {"value": (32, 8)}

        result = op.generate_random_params(element_keys, data_shapes)
        assert result is None

    def test_stochastic_returns_keys(self):
        """Stochastic ElementOperator should return RNG keys."""
        config = ElementOperatorConfig(stochastic=True, stream_name="params")
        op = ElementOperator(config, fn=_stochastic_fn, rngs=nnx.Rngs(params=42))

        element_keys = jax.random.split(jax.random.key(0), 32)  # one key per record
        data_shapes = {"value": (32, 8)}

        result = op.generate_random_params(element_keys, data_shapes)
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

        element_keys = jax.random.split(jax.random.key(0), 32)  # one key per record
        data_shapes = {"value": (32, 8)}

        result = op.generate_random_params(element_keys, data_shapes)
        assert result is None

    def test_stochastic_returns_keys(self):
        """Stochastic MapOperator should return RNG keys."""
        config = MapOperatorConfig(stochastic=True, stream_name="params")
        op = MapOperator(config, fn=_map_fn, rngs=nnx.Rngs(params=42))  # type: ignore[reportArgumentType]

        element_keys = jax.random.split(jax.random.key(0), 32)  # one key per record
        data_shapes = {"value": (32, 8)}

        result = op.generate_random_params(element_keys, data_shapes)
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


_SEEN_RANDOM_PARAMS: list[Any] = []


class _RecordingOperator(OperatorModule):
    """An operator that records the random parameters each apply call is given."""

    def apply(self, data, state, metadata, random_params=None, stats=None):
        """Record the random parameters and return the record unchanged."""
        del metadata, stats
        _SEEN_RANDOM_PARAMS.append(random_params)
        return data, state, None

    def generate_random_params(self, element_keys, data_shapes):
        """Return one key per record, as any stochastic operator does."""
        del data_shapes
        return element_keys


class TestDeterministicApplyReceivesNoRandomParameters:
    """What a deterministic operator's apply is handed, not just what it returns."""

    def test_deterministic_apply_is_given_none(self):
        """Every apply call a deterministic operator makes receives None.

        The assertion is over every recorded call rather than a call count: the output
        structure is discovered by calling apply as well, so counting calls would pin the
        discovery mechanism instead of the property.
        """
        _SEEN_RANDOM_PARAMS.clear()
        op = _RecordingOperator(OperatorConfig(stochastic=False))

        op._vmap_apply({"value": jnp.ones((4, 8))}, {})

        assert _SEEN_RANDOM_PARAMS
        assert all(params is None for params in _SEEN_RANDOM_PARAMS)

    def test_stochastic_apply_is_given_parameters(self):
        """The same spy sees parameters when the operator is stochastic.

        Without this, the test above would pass just as well against a spy that records
        nothing the operator is given. The two assertions are deliberately asymmetric: a
        deterministic operator has no call that receives parameters, while a stochastic one
        has at least one. Discovering the output structure calls apply with no parameters
        whatever the mode, so a stochastic operator is handed None once and its per-record
        keys once.
        """
        _SEEN_RANDOM_PARAMS.clear()
        op = _RecordingOperator(
            OperatorConfig(stochastic=True, stream_name="augment"),
            rngs=nnx.Rngs(augment=0),
        )

        op._vmap_apply({"value": jnp.ones((4, 8))}, {})

        assert _SEEN_RANDOM_PARAMS
        assert any(params is not None for params in _SEEN_RANDOM_PARAMS)
