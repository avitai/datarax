"""Tests for weight_key dynamic external weights in CompositeOperatorModule.

Tests the weight_key feature which allows WEIGHTED_PARALLEL composites to
receive weights from the data dict at call time (e.g., Gumbel-Softmax outputs
from a separate policy module). This enables the DADA use case: policy
logits -> softmax -> weights -> composite -> loss -> gradients propagate back.

Test Coverage:
- Config validation (mutual exclusivity with learnable_weights and explicit weights)
- Basic functionality (weighted sum correctness, key stripping, missing key error)
- Advanced scenarios (JIT compatibility, gradient flow, batch processing via __call__)
"""

import pytest
import jax
import jax.numpy as jnp
from flax import nnx

from datarax.operators.composite_operator import (
    CompositeOperatorModule,
    CompositeOperatorConfig,
    CompositionStrategy,
)
from datarax.operators.map_operator import MapOperator, MapOperatorConfig
from datarax.core.element_batch import Batch, Element


class TestWeightKeyConfig:
    """Config validation tests."""

    def test_weight_key_valid_config(self):
        """weight_key alone creates valid WEIGHTED_PARALLEL config."""
        rngs = nnx.Rngs(0)
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)
        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 3, rngs=rngs)

        # Should not raise — weight_key is a valid way to supply weights
        config = CompositeOperatorConfig(
            strategy=CompositionStrategy.WEIGHTED_PARALLEL,
            operators=[op1, op2],
            weight_key="op_weights",
        )
        assert config.weight_key == "op_weights"
        # weights should remain None (not auto-filled with equal weights)
        assert config.weights is None

    def test_weight_key_with_learnable_weights_raises(self):
        """weight_key + learnable_weights is mutually exclusive."""
        rngs = nnx.Rngs(0)
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)
        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 3, rngs=rngs)

        with pytest.raises(ValueError, match="weight_key.*learnable_weights"):
            CompositeOperatorConfig(
                strategy=CompositionStrategy.WEIGHTED_PARALLEL,
                operators=[op1, op2],
                weight_key="op_weights",
                learnable_weights=True,
            )

    def test_weight_key_with_explicit_weights_raises(self):
        """weight_key + explicit weights is mutually exclusive."""
        rngs = nnx.Rngs(0)
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)
        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 3, rngs=rngs)

        with pytest.raises(ValueError, match="weight_key.*explicit weights"):
            CompositeOperatorConfig(
                strategy=CompositionStrategy.WEIGHTED_PARALLEL,
                operators=[op1, op2],
                weight_key="op_weights",
                weights=[0.5, 0.5],
            )


class TestWeightKeyBasic:
    """Basic functionality."""

    def test_dynamic_weights_from_data(self):
        """Weights extracted from data[weight_key], weighted sum correct.

        2 MapOperators (x*2, x*3), weights [0.7, 0.3]:
        op1: 10*2 = 20, op2: 10*3 = 30
        result: 0.7*20 + 0.3*30 = 14 + 9 = 23.0
        """
        rngs = nnx.Rngs(0)
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)
        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 3, rngs=rngs)

        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.WEIGHTED_PARALLEL,
            operators=[op1, op2],
            weight_key="op_weights",
        )
        composite = CompositeOperatorModule(composite_config)

        data = {
            "value": jnp.array(10.0),
            "op_weights": jnp.array([0.7, 0.3]),
        }
        result_data, _, _ = composite.apply(data, {}, None)

        assert jnp.isclose(result_data["value"], 23.0)

    def test_weight_key_stripped_from_child_data(self):
        """Child operators do NOT receive the weight_key in their data.

        Uses a recording operator that captures which keys it received.
        """
        received_keys: list[set] = []

        class RecordingOperator(MapOperator):
            """MapOperator that records data keys it receives."""

            def apply(self, data, state, metadata, random_params=None, stats=None):
                received_keys.append(set(data.keys()))
                return super().apply(data, state, metadata, random_params, stats)

        rngs = nnx.Rngs(0)
        config1 = MapOperatorConfig(stochastic=False)
        op1 = RecordingOperator(config1, fn=lambda x, key: x, rngs=rngs)
        config2 = MapOperatorConfig(stochastic=False)
        op2 = RecordingOperator(config2, fn=lambda x, key: x, rngs=rngs)

        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.WEIGHTED_PARALLEL,
            operators=[op1, op2],
            weight_key="op_weights",
        )
        composite = CompositeOperatorModule(composite_config)

        data = {
            "value": jnp.array(1.0),
            "op_weights": jnp.array([0.5, 0.5]),
        }
        composite.apply(data, {}, None)

        # Both operators should have received data WITHOUT op_weights
        assert len(received_keys) == 2
        for keys in received_keys:
            assert "op_weights" not in keys
            assert "value" in keys

    def test_weight_key_missing_raises(self):
        """Missing weight_key in data -> clear ValueError."""
        rngs = nnx.Rngs(0)
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)
        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 3, rngs=rngs)

        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.WEIGHTED_PARALLEL,
            operators=[op1, op2],
            weight_key="op_weights",
        )
        composite = CompositeOperatorModule(composite_config)

        # Data dict missing the weight_key
        data = {"value": jnp.array(10.0)}

        with pytest.raises(ValueError, match="weight_key.*op_weights.*not found"):
            composite.apply(data, {}, None)


class TestWeightKeyAdvanced:
    """JIT, vmap, gradient flow."""

    def test_jit_compatible(self):
        """weight_key works inside @nnx.jit."""
        rngs = nnx.Rngs(0)
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)
        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 3, rngs=rngs)

        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.WEIGHTED_PARALLEL,
            operators=[op1, op2],
            weight_key="op_weights",
        )
        composite = CompositeOperatorModule(composite_config)

        @nnx.jit
        def jit_apply(model, data):
            result, _, _ = model.apply(data, {}, None)
            return result

        data = {
            "value": jnp.array(10.0),
            "op_weights": jnp.array([0.7, 0.3]),
        }
        result = jit_apply(composite, data)
        assert jnp.isclose(result["value"], 23.0)

    def test_gradient_flow_through_external_weights(self):
        """Gradients flow from loss through weights to an upstream parameter.

        This is the DADA use case: policy logits -> softmax -> weights -> loss.
        Verifies that d(loss)/d(logits) is non-zero.
        """
        rngs = nnx.Rngs(0)
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)
        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 3, rngs=rngs)

        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.WEIGHTED_PARALLEL,
            operators=[op1, op2],
            weight_key="op_weights",
        )
        composite = CompositeOperatorModule(composite_config)

        # Simulate policy logits as upstream learnable parameter
        logits = jnp.array([1.0, 0.0])  # Will softmax to ~[0.731, 0.269]

        def loss_fn(logits):
            weights = jax.nn.softmax(logits)
            data = {
                "value": jnp.array(10.0),
                "op_weights": weights,
            }
            graphdef, state = nnx.split(composite)
            model = nnx.merge(graphdef, state)
            result, _, _ = model.apply(data, {}, None)
            return result["value"]  # Loss = weighted sum output

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(logits)

        # Gradients should be non-zero — loss depends on logits through softmax -> weights
        assert jnp.sum(jnp.abs(grads)) > 0, "Gradients are zero — pipeline is not differentiable!"
        # op1 output (20) != op2 output (30), so shifting weights changes the result
        # d(loss)/d(logits) should have opposite signs for the two entries
        assert grads[0] * grads[1] < 0, f"Expected opposite-sign gradients, got {grads}"

    def test_batch_processing_via_call(self):
        """weight_key works through __call__ -> apply_batch -> vmap path."""
        rngs = nnx.Rngs(0)
        config1 = MapOperatorConfig(stochastic=False)
        op1 = MapOperator(config1, fn=lambda x, key: x * 2, rngs=rngs)
        config2 = MapOperatorConfig(stochastic=False)
        op2 = MapOperator(config2, fn=lambda x, key: x * 3, rngs=rngs)

        composite_config = CompositeOperatorConfig(
            strategy=CompositionStrategy.WEIGHTED_PARALLEL,
            operators=[op1, op2],
            weight_key="op_weights",
        )
        composite = CompositeOperatorModule(composite_config)

        # Create batch with weight_key in each element's data
        batch = Batch(
            [
                Element(
                    data={
                        "value": jnp.array([1.0]),
                        "op_weights": jnp.array([0.7, 0.3]),
                    }
                ),
                Element(
                    data={
                        "value": jnp.array([2.0]),
                        "op_weights": jnp.array([0.3, 0.7]),
                    }
                ),
            ]
        )

        result_batch = composite(batch)
        result_data = result_batch.get_data()

        # Element 0: 0.7*(1*2) + 0.3*(1*3) = 1.4 + 0.9 = 2.3
        # Element 1: 0.3*(2*2) + 0.7*(2*3) = 1.2 + 4.2 = 5.4
        assert jnp.isclose(result_data["value"][0, 0], 2.3, atol=1e-5)
        assert jnp.isclose(result_data["value"][1, 0], 5.4, atol=1e-5)
