"""Test suite for CrossModalOperator.

Tests the base class for cross-modal operators.
Follows TDD approach - tests written first (RED phase).
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from datarax.core.cross_modal import CrossModalOperator, CrossModalOperatorConfig


class ConcreteFusionOperator(CrossModalOperator):
    """Concrete fusion implementation for testing."""

    def apply(self, data, state, metadata, random_params=None, stats=None):
        """Simple fusion - concatenate inputs."""
        inputs = self._extract_inputs(data)
        # Simple concatenation fusion
        fused = jnp.concatenate(inputs, axis=-1)
        outputs = [fused]
        result = self._store_outputs(data, outputs)
        return result, state, metadata


class StochasticContrastiveOperator(CrossModalOperator):
    """Stochastic contrastive implementation for testing."""

    def apply(self, data, state, metadata, random_params=None, stats=None):
        """Compute similarity with optional noise."""
        inputs = self._extract_inputs(data)
        noise = random_params if random_params is not None else 0.0

        # Compute cosine similarity with noise
        emb1, emb2 = inputs[0], inputs[1]
        similarity = jnp.dot(emb1, emb2) / (jnp.linalg.norm(emb1) * jnp.linalg.norm(emb2))
        similarity = similarity + noise

        outputs = [similarity]
        result = self._store_outputs(data, outputs)
        return result, state, metadata

    def generate_random_params(self, rng, data_shapes):
        """Generate random noise for contrastive learning."""
        # Get batch size from first input field
        first_field = self.config.input_fields[0]
        batch_size = data_shapes[first_field][0]
        return jax.random.normal(rng, (batch_size,)) * 0.01


class TestCrossModalOperatorInitialization:
    """Test CrossModalOperator initialization."""

    def test_deterministic_initialization(self):
        """Deterministic operator initialization should succeed."""
        config = CrossModalOperatorConfig(
            input_fields=["image_emb", "text_emb"],
            output_fields=["fused_emb"],
        )
        rngs = nnx.Rngs(0)
        operator = ConcreteFusionOperator(config, rngs=rngs)

        assert operator.config.input_fields == ["image_emb", "text_emb"]
        assert operator.config.output_fields == ["fused_emb"]
        assert operator.stochastic is False
        assert operator.stream_name is None

    def test_stochastic_initialization(self):
        """Stochastic operator initialization should succeed."""
        config = CrossModalOperatorConfig(
            input_fields=["anchor_emb", "positive_emb"],
            output_fields=["similarity"],
            stochastic=True,
            stream_name="contrastive",
        )
        rngs = nnx.Rngs(0, contrastive=1)
        operator = StochasticContrastiveOperator(config, rngs=rngs)

        assert operator.config.input_fields == ["anchor_emb", "positive_emb"]
        assert operator.config.output_fields == ["similarity"]
        assert operator.stochastic is True
        assert operator.stream_name == "contrastive"

    def test_stochastic_without_rngs_raises_error(self):
        """Stochastic operator without rngs should raise ValueError."""
        config = CrossModalOperatorConfig(
            input_fields=["input1", "input2"],
            output_fields=["output"],
            stochastic=True,
            stream_name="fusion",
        )
        with pytest.raises(ValueError, match="Stochastic operators require rngs"):
            StochasticContrastiveOperator(config, rngs=None)

    def test_initialization_with_name(self):
        """Initialization with custom name should work."""
        config = CrossModalOperatorConfig(
            input_fields=["input1", "input2"],
            output_fields=["output"],
        )
        rngs = nnx.Rngs(0)
        operator = ConcreteFusionOperator(config, rngs=rngs, name="my_fusion_op")

        assert operator.name == "my_fusion_op"


class TestCrossModalOperatorAbstractMethods:
    """Test that abstract methods must be implemented."""

    def test_base_class_apply_raises_not_implemented(self):
        """Calling apply() on base class should raise NotImplementedError."""
        config = CrossModalOperatorConfig(
            input_fields=["input1", "input2"],
            output_fields=["output"],
        )
        rngs = nnx.Rngs(0)
        operator = CrossModalOperator(config, rngs=rngs)

        data = {"input1": jnp.ones((128,)), "input2": jnp.ones((128,))}
        state = {}
        metadata = None

        with pytest.raises(NotImplementedError, match="must implement apply"):
            operator.apply(data, state, metadata)

    def test_stochastic_without_generate_random_params_raises_error(self):
        """Stochastic operator without generate_random_params should raise error."""

        class IncompleteStochasticOperator(CrossModalOperator):
            """Missing generate_random_params implementation."""

            def apply(self, data, state, metadata, random_params=None, stats=None):
                return data, state, metadata

        config = CrossModalOperatorConfig(
            input_fields=["input1", "input2"],
            output_fields=["output"],
            stochastic=True,
            stream_name="fusion",
        )
        rngs = nnx.Rngs(0, fusion=1)
        operator = IncompleteStochasticOperator(config, rngs=rngs)

        data_shapes = {"input1": (4, 128), "input2": (4, 128)}
        rng = jax.random.PRNGKey(0)

        with pytest.raises(NotImplementedError, match="does not implement generate_random_params"):
            operator.generate_random_params(rng, data_shapes)


class TestCrossModalOperatorHelperMethods:
    """Test helper methods for input/output handling."""

    def test_extract_inputs_single_field(self):
        """_extract_inputs should extract single input field."""
        config = CrossModalOperatorConfig(
            input_fields=["embedding"],
            output_fields=["output"],
        )
        rngs = nnx.Rngs(0)
        operator = ConcreteFusionOperator(config, rngs=rngs)

        data = {"embedding": jnp.array([1, 2, 3]), "other": jnp.array([4, 5, 6])}
        inputs = operator._extract_inputs(data)

        assert len(inputs) == 1
        assert jnp.array_equal(inputs[0], jnp.array([1, 2, 3]))

    def test_extract_inputs_multiple_fields(self):
        """_extract_inputs should extract multiple input fields in order."""
        config = CrossModalOperatorConfig(
            input_fields=["image_emb", "text_emb", "audio_emb"],
            output_fields=["fused"],
        )
        rngs = nnx.Rngs(0)
        operator = ConcreteFusionOperator(config, rngs=rngs)

        data = {
            "image_emb": jnp.array([1, 2]),
            "text_emb": jnp.array([3, 4]),
            "audio_emb": jnp.array([5, 6]),
            "unused": jnp.array([7, 8]),
        }
        inputs = operator._extract_inputs(data)

        assert len(inputs) == 3
        assert jnp.array_equal(inputs[0], jnp.array([1, 2]))
        assert jnp.array_equal(inputs[1], jnp.array([3, 4]))
        assert jnp.array_equal(inputs[2], jnp.array([5, 6]))

    def test_store_outputs_single_field(self):
        """_store_outputs should store single output field."""
        config = CrossModalOperatorConfig(
            input_fields=["input"],
            output_fields=["fused_output"],
        )
        rngs = nnx.Rngs(0)
        operator = ConcreteFusionOperator(config, rngs=rngs)

        data = {"input": jnp.array([1, 2, 3])}
        outputs = [jnp.array([4, 5, 6])]
        result = operator._store_outputs(data, outputs)

        assert jnp.array_equal(result["input"], jnp.array([1, 2, 3]))
        assert jnp.array_equal(result["fused_output"], jnp.array([4, 5, 6]))

    def test_store_outputs_multiple_fields(self):
        """_store_outputs should store multiple output fields."""
        config = CrossModalOperatorConfig(
            input_fields=["input1", "input2"],
            output_fields=["fused", "similarity", "alignment"],
        )
        rngs = nnx.Rngs(0)
        operator = ConcreteFusionOperator(config, rngs=rngs)

        data = {"input1": jnp.array([1]), "input2": jnp.array([2])}
        outputs = [jnp.array([3]), jnp.array([0.9]), jnp.array([0.8])]
        result = operator._store_outputs(data, outputs)

        assert jnp.array_equal(result["input1"], jnp.array([1]))
        assert jnp.array_equal(result["input2"], jnp.array([2]))
        assert jnp.array_equal(result["fused"], jnp.array([3]))
        assert jnp.allclose(result["similarity"], jnp.array([0.9]))
        assert jnp.allclose(result["alignment"], jnp.array([0.8]))

    def test_store_outputs_preserves_original_data(self):
        """_store_outputs should preserve original data fields."""
        config = CrossModalOperatorConfig(
            input_fields=["input1"],
            output_fields=["output1"],
        )
        rngs = nnx.Rngs(0)
        operator = ConcreteFusionOperator(config, rngs=rngs)

        data = {"input1": jnp.array([1, 2]), "other_field": jnp.array([9, 10])}
        outputs = [jnp.array([3, 4])]
        result = operator._store_outputs(data, outputs)

        # Original fields should be preserved
        assert jnp.array_equal(result["input1"], jnp.array([1, 2]))
        assert jnp.array_equal(result["other_field"], jnp.array([9, 10]))
        # New field should be added
        assert jnp.array_equal(result["output1"], jnp.array([3, 4]))


class TestCrossModalOperatorApply:
    """Test apply() method implementation."""

    def test_apply_fusion_operator(self):
        """Apply should work for fusion operator."""
        config = CrossModalOperatorConfig(
            input_fields=["emb1", "emb2"],
            output_fields=["fused"],
        )
        rngs = nnx.Rngs(0)
        operator = ConcreteFusionOperator(config, rngs=rngs)

        data = {
            "emb1": jnp.array([1.0, 2.0]),
            "emb2": jnp.array([3.0, 4.0]),
        }
        state = {}
        metadata = None

        result_data, result_state, result_metadata = operator.apply(data, state, metadata)

        # Should concatenate embeddings
        expected = jnp.array([1.0, 2.0, 3.0, 4.0])
        assert jnp.allclose(result_data["fused"], expected)
        # Original data should be preserved
        assert jnp.allclose(result_data["emb1"], jnp.array([1.0, 2.0]))
        assert jnp.allclose(result_data["emb2"], jnp.array([3.0, 4.0]))

    def test_apply_contrastive_operator(self):
        """Apply should work for contrastive operator."""
        config = CrossModalOperatorConfig(
            input_fields=["anchor", "positive"],
            output_fields=["similarity"],
        )
        rngs = nnx.Rngs(0)
        operator = StochasticContrastiveOperator(config, rngs=rngs)

        # Orthogonal vectors (similarity should be 0)
        data = {
            "anchor": jnp.array([1.0, 0.0]),
            "positive": jnp.array([0.0, 1.0]),
        }
        state = {}
        metadata = None

        result_data, result_state, result_metadata = operator.apply(data, state, metadata)

        # Similarity should be close to 0
        assert jnp.allclose(result_data["similarity"], 0.0, atol=0.01)


class TestCrossModalOperatorStochastic:
    """Test stochastic operator behavior."""

    def test_stochastic_operator_generates_random_params(self):
        """Stochastic operator should generate random parameters."""
        config = CrossModalOperatorConfig(
            input_fields=["anchor", "positive"],
            output_fields=["similarity"],
            stochastic=True,
            stream_name="contrastive",
        )
        rngs = nnx.Rngs(0, contrastive=1)
        operator = StochasticContrastiveOperator(config, rngs=rngs)

        data_shapes = {"anchor": (8, 128), "positive": (8, 128)}
        rng = jax.random.PRNGKey(42)

        random_params = operator.generate_random_params(rng, data_shapes)

        # Should generate 8 random noise values (batch size = 8)
        assert random_params.shape == (8,)

    def test_stochastic_operator_apply_uses_random_params(self):
        """Stochastic operator apply should use provided random params."""
        config = CrossModalOperatorConfig(
            input_fields=["anchor", "positive"],
            output_fields=["similarity"],
            stochastic=True,
            stream_name="contrastive",
        )
        rngs = nnx.Rngs(0, contrastive=1)
        operator = StochasticContrastiveOperator(config, rngs=rngs)

        data = {
            "anchor": jnp.array([1.0, 0.0]),
            "positive": jnp.array([0.0, 1.0]),
        }
        state = {}
        metadata = None

        # Apply with specific noise
        result_data, _, _ = operator.apply(data, state, metadata, random_params=0.5)

        # Similarity should be 0.0 + 0.5 = 0.5
        assert jnp.allclose(result_data["similarity"], 0.5, atol=0.01)


class TestCrossModalOperatorJAXCompatibility:
    """Test JAX transformation compatibility."""

    def test_operator_is_jit_compatible(self):
        """Operator apply should be JIT compatible."""
        config = CrossModalOperatorConfig(
            input_fields=["emb1", "emb2"],
            output_fields=["fused"],
        )
        rngs = nnx.Rngs(0)
        operator = ConcreteFusionOperator(config, rngs=rngs)

        @jax.jit
        def jitted_apply(data, state, metadata):
            return operator.apply(data, state, metadata)

        data = {
            "emb1": jnp.array([1.0, 2.0]),
            "emb2": jnp.array([3.0, 4.0]),
        }
        state = {}
        metadata = None

        result_data, _, _ = jitted_apply(data, state, metadata)

        expected = jnp.array([1.0, 2.0, 3.0, 4.0])
        assert jnp.allclose(result_data["fused"], expected)

    def test_operator_is_vmap_compatible(self):
        """Operator apply should be vmap compatible."""
        config = CrossModalOperatorConfig(
            input_fields=["emb1", "emb2"],
            output_fields=["fused"],
        )
        rngs = nnx.Rngs(0)
        operator = ConcreteFusionOperator(config, rngs=rngs)

        # Batch of 3 embedding pairs
        batch_data = {
            "emb1": jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            "emb2": jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        }
        batch_state = [{}, {}, {}]

        def apply_single(emb1, emb2, state):
            data_dict = {"emb1": emb1, "emb2": emb2}
            result_data, result_state, _ = operator.apply(data_dict, state, None)
            return result_data["fused"], result_state

        vmapped_apply = jax.vmap(apply_single, in_axes=(0, 0, 0))
        result_fused, result_states = vmapped_apply(
            batch_data["emb1"], batch_data["emb2"], batch_state
        )

        # Check shape: (3, 4) - 3 batches, 4 features (2+2)
        assert result_fused.shape == (3, 4)

        # Check first batch result
        expected_first = jnp.array([1.0, 2.0, 0.1, 0.2])
        assert jnp.allclose(result_fused[0], expected_first)
