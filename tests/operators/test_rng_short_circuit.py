"""Tests for what the framework hands a deterministic operator.

Validates that:

1. A deterministic operator's apply is passed no key at all
2. A stochastic operator's apply is passed one
3. The 2-argument vmap path is used when there is no key to map over
"""

import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax.core.config import ElementOperatorConfig, OperatorConfig
from datarax.core.operator import OperatorModule
from datarax.operators.element_operator import ElementOperator


def _identity_fn(element, key):
    """Deterministic identity — ignores key."""
    del key
    return element


class TestVmapPathSelection:
    """Tests that verify the 2-arg vmap path when there is no key."""

    def test_deterministic_pipeline_no_rng_overhead(self):
        """A fully deterministic pipeline derives no per-record keys at all."""
        config = ElementOperatorConfig(stochastic=False)
        op = ElementOperator(config, fn=_identity_fn)

        # Build a batch with data
        data = {"value": jnp.ones((4, 8))}
        states: dict = {}

        # _vmap_apply should work without deriving any key
        result_data, result_states = op._vmap_apply(data, states)
        assert "value" in result_data
        np.testing.assert_array_equal(result_data["value"], data["value"])


_SEEN_KEYS: list = []


class _RecordingOperator(OperatorModule):
    """An operator that records the fourth argument each apply call is given."""

    def apply(self, data, state, metadata, key=None, stats=None):
        """Record the key and return the record unchanged."""
        del metadata, stats
        _SEEN_KEYS.append(key)
        return data, state, None


class TestDeterministicApplyReceivesNoKey:
    """What the framework hands apply, not what an operator chooses to do with it."""

    def test_deterministic_apply_is_given_none(self):
        """Every apply call a deterministic operator makes receives None."""
        _SEEN_KEYS.clear()
        op = _RecordingOperator(OperatorConfig(stochastic=False))

        op._vmap_apply({"value": jnp.ones((4, 8))}, {})

        assert _SEEN_KEYS, "the framework never called apply"
        assert all(key is None for key in _SEEN_KEYS)

    def test_stochastic_apply_is_given_a_key(self):
        """The same spy sees a key when the operator is stochastic.

        Without this, the test above would pass just as well against a spy that records
        nothing it is given. Both assertions are now over *every* call: removing the
        output-structure probe left one apply call per mode, so a stochastic operator is
        never handed None.
        """
        _SEEN_KEYS.clear()
        op = _RecordingOperator(
            OperatorConfig(stochastic=True, stream_name="augment"),
            rngs=nnx.Rngs(augment=0),
        )

        op._vmap_apply({"value": jnp.ones((4, 8))}, {})

        assert _SEEN_KEYS, "the framework never called apply"
        assert all(key is not None for key in _SEEN_KEYS)
