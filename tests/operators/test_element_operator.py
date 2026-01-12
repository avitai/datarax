"""Tests for ElementOperator - operator for element-level transformations.

This test suite validates ElementOperator which applies user-provided element
transformation functions to entire Element structures (data + state + metadata).

Key Difference from MapOperator:
- MapOperator: fn(array_leaf, key) -> array_leaf (per-array-leaf transformation)
- ElementOperator: fn(element, key) -> element (per-element transformation)

ElementOperator allows coordinated transformations across multiple fields,
access to state/metadata, and complex augmentation pipelines.

Test Categories:
1. Config validation (ElementOperatorConfig)
2. Initialization (basic, stochastic, deterministic)
3. Basic transformations (identity, data modification)
4. State/metadata access (read and modify)
5. Stochastic mode (RNG key handling)
6. Edge cases (empty data, missing fields)
7. JAX compatibility (jit, vmap)
8. Integration tests
"""

from typing import Any, cast

import pytest
import jax
import jax.numpy as jnp
from flax import nnx

from datarax.core.config import ElementOperatorConfig
from datarax.operators.element_operator import ElementOperator
from datarax.core.element_batch import Batch, Element


# ========================================================================
# Test Helper: Batch Creation
# ========================================================================


def create_test_batch(data, states=None, metadata_list=None, batch_size=None):
    """Helper to create batch using Batch.from_parts() API."""
    if batch_size is None:
        first_array = jax.tree.leaves(data)[0]
        batch_size = first_array.shape[0]

    if states is None:
        states = {}
    if metadata_list is None:
        metadata_list = [None] * batch_size

    return Batch.from_parts(data, states, metadata_list, validate=False)


# ========================================================================
# Test Category 1: Config Validation
# ========================================================================


class TestElementOperatorConfig:
    """Test ElementOperatorConfig validation."""

    def test_config_inheritance(self):
        """Config inherits from OperatorConfig."""
        from datarax.core.config import OperatorConfig

        config = ElementOperatorConfig(stochastic=False)
        assert isinstance(config, OperatorConfig)

    def test_deterministic_config(self):
        """Deterministic config is valid."""
        config = ElementOperatorConfig(stochastic=False)
        assert config.stochastic is False
        assert config.stream_name is None

    def test_stochastic_config(self):
        """Stochastic config requires stream_name."""
        config = ElementOperatorConfig(stochastic=True, stream_name="augment")
        assert config.stochastic is True
        assert config.stream_name == "augment"

    def test_stochastic_without_stream_name_fails(self):
        """Stochastic=True without stream_name raises ValueError."""
        with pytest.raises(ValueError, match="stream_name"):
            ElementOperatorConfig(stochastic=True)

    def test_deterministic_with_stream_name_fails(self):
        """Deterministic with stream_name raises ValueError."""
        with pytest.raises(ValueError, match="stream_name"):
            ElementOperatorConfig(stochastic=False, stream_name="augment")


# ========================================================================
# Test Category 2: Initialization
# ========================================================================


class TestElementOperatorInit:
    """Test ElementOperator initialization."""

    def test_init_deterministic(self):
        """Deterministic mode initialization."""

        def identity(element, key):
            return element

        config = ElementOperatorConfig(stochastic=False)
        rngs = nnx.Rngs(0)
        op = ElementOperator(config, fn=identity, rngs=rngs)

        assert op.fn == identity
        assert op.stochastic is False
        assert op.stream_name is None

    def test_init_stochastic(self):
        """Stochastic mode initialization."""

        def add_noise(element, key):
            return element

        config = ElementOperatorConfig(stochastic=True, stream_name="augment")
        rngs = nnx.Rngs(0, augment=1)
        op = ElementOperator(config, fn=add_noise, rngs=rngs)

        assert op.fn == add_noise
        assert op.stochastic is True
        assert op.stream_name == "augment"

    def test_init_stochastic_requires_rngs(self):
        """Stochastic mode requires rngs parameter."""

        def add_noise(element, key):
            return element

        config = ElementOperatorConfig(stochastic=True, stream_name="augment")

        with pytest.raises(ValueError, match="rngs"):
            ElementOperator(config, fn=add_noise, rngs=None)

    def test_init_with_custom_name(self):
        """Custom name is stored."""

        def identity(element, key):
            return element

        config = ElementOperatorConfig(stochastic=False)
        rngs = nnx.Rngs(0)
        op = ElementOperator(config, fn=identity, rngs=rngs, name="MyOperator")

        assert op.name == "MyOperator"


# ========================================================================
# Test Category 3: Basic Transformations
# ========================================================================


class TestElementOperatorBasicTransformations:
    """Test basic transformation functionality."""

    def test_identity_transformation(self):
        """Identity function passes data through unchanged."""

        def identity(element, key):
            return element

        config = ElementOperatorConfig(stochastic=False)
        rngs = nnx.Rngs(0)
        op = ElementOperator(config, fn=identity, rngs=rngs)

        batch_data = {"image": jnp.array([[1.0, 2.0, 3.0]])}
        batch = create_test_batch(batch_data)

        result = op(batch)

        assert jnp.allclose(result.get_data()["image"], batch_data["image"])

    def test_data_multiplication(self):
        """Function can multiply data fields."""

        def double_data(element, key):
            new_data = jax.tree.map(lambda x: x * 2.0, element.data)
            return element.replace(data=new_data)

        config = ElementOperatorConfig(stochastic=False)
        rngs = nnx.Rngs(0)
        op = ElementOperator(config, fn=double_data, rngs=rngs)

        batch_data = {"image": jnp.array([[1.0, 2.0, 3.0]])}
        batch = create_test_batch(batch_data)

        result = op(batch)

        expected = jnp.array([[2.0, 4.0, 6.0]])
        assert jnp.allclose(result.get_data()["image"], expected)

    def test_data_addition(self):
        """Function can add constants to data."""

        def add_one(element, key):
            new_data = {"value": element.data["value"] + 1.0}
            return element.replace(data=new_data)

        config = ElementOperatorConfig(stochastic=False)
        rngs = nnx.Rngs(0)
        op = ElementOperator(config, fn=add_one, rngs=rngs)

        batch_data = {"value": jnp.array([[10.0, 20.0]])}
        batch = create_test_batch(batch_data)

        result = op(batch)

        expected = jnp.array([[11.0, 21.0]])
        assert jnp.allclose(result.get_data()["value"], expected)

    def test_multiple_fields(self):
        """Function can transform multiple fields."""

        def normalize_all(element, key):
            new_data = {
                "image": element.data["image"] / 255.0,
                "mask": element.data["mask"] * 2.0,
            }
            return element.replace(data=new_data)

        config = ElementOperatorConfig(stochastic=False)
        rngs = nnx.Rngs(0)
        op = ElementOperator(config, fn=normalize_all, rngs=rngs)

        batch_data = {
            "image": jnp.array([[255.0, 127.5]]),
            "mask": jnp.array([[0.5, 1.0]]),
        }
        batch = create_test_batch(batch_data)

        result = op(batch)
        result_data = result.get_data()

        assert jnp.allclose(result_data["image"], jnp.array([[1.0, 0.5]]))
        assert jnp.allclose(result_data["mask"], jnp.array([[1.0, 2.0]]))

    def test_batch_processing(self):
        """Works correctly with batch of multiple elements."""

        def triple(element, key):
            new_data = jax.tree.map(lambda x: x * 3.0, element.data)
            return element.replace(data=new_data)

        config = ElementOperatorConfig(stochastic=False)
        rngs = nnx.Rngs(0)
        op = ElementOperator(config, fn=triple, rngs=rngs)

        # Batch of 4 elements
        batch_data = {"value": jnp.ones((4, 3))}
        batch = create_test_batch(batch_data)

        result = op(batch)

        expected = jnp.ones((4, 3)) * 3.0
        assert jnp.allclose(result.get_data()["value"], expected)


# ========================================================================
# Test Category 4: State and Metadata Access
# ========================================================================


class TestElementOperatorStateMetadata:
    """Test state and metadata access and modification."""

    def test_state_passthrough(self):
        """State passes through when not modified."""

        def data_only(element, key):
            new_data = {"value": element.data["value"] + 1.0}
            return element.replace(data=new_data)

        config = ElementOperatorConfig(stochastic=False)
        rngs = nnx.Rngs(0)
        op = ElementOperator(config, fn=data_only, rngs=rngs)

        batch_data = {"value": jnp.array([[1.0]])}
        batch_states = {"counter": jnp.array([[100.0]])}
        batch = Batch.from_parts(batch_data, batch_states, [None], validate=False)

        result = op(batch)

        # State should be unchanged
        states = cast(dict[str, Any], result.get_states())
        assert jnp.allclose(states["counter"], jnp.array([[100.0]]))

    def test_state_modification(self):
        """Function can modify state."""

        def increment_counter(element, key):
            new_state = {"counter": element.state["counter"] + 1.0}
            return element.replace(state=new_state)

        config = ElementOperatorConfig(stochastic=False)
        rngs = nnx.Rngs(0)
        op = ElementOperator(config, fn=increment_counter, rngs=rngs)

        batch_data = {"value": jnp.array([[1.0]])}
        batch_states = {"counter": jnp.array([[0.0]])}
        batch = Batch.from_parts(batch_data, batch_states, [None], validate=False)

        result = op(batch)

        # State should be incremented
        states = cast(dict[str, Any], result.get_states())
        assert jnp.allclose(states["counter"], jnp.array([[1.0]]))

    def test_metadata_passthrough(self):
        """Metadata passes through unchanged (not vmapped)."""

        def identity(element, key):
            return element

        config = ElementOperatorConfig(stochastic=False)
        rngs = nnx.Rngs(0)
        op = ElementOperator(config, fn=identity, rngs=rngs)

        batch_data = {"value": jnp.array([[1.0]])}
        metadata_list = [{"filename": "test.jpg", "index": 42}]
        batch = Batch.from_parts(batch_data, {}, metadata_list, validate=False)

        result = op(batch)

        # Metadata should be unchanged
        assert result._metadata_list[0] == {"filename": "test.jpg", "index": 42}

    def test_coordinated_transformation(self):
        """Function can coordinate transformation across data/state."""

        def flip_and_track(element, key):
            # Flip data
            new_data = {"image": -element.data["image"]}
            # Track in state that we flipped
            new_state = {"flipped": jnp.array(1.0)}
            return element.replace(data=new_data, state=new_state)

        config = ElementOperatorConfig(stochastic=False)
        rngs = nnx.Rngs(0)
        op = ElementOperator(config, fn=flip_and_track, rngs=rngs)

        batch_data = {"image": jnp.array([[1.0, 2.0]])}
        batch_states = {"flipped": jnp.array([[0.0]])}
        batch = Batch.from_parts(batch_data, batch_states, [None], validate=False)

        result = op(batch)

        # Data should be flipped
        assert jnp.allclose(result.get_data()["image"], jnp.array([[-1.0, -2.0]]))
        # State should track the flip
        states = cast(dict[str, Any], result.get_states())
        assert jnp.allclose(states["flipped"], jnp.array([[1.0]]))


# ========================================================================
# Test Category 5: Stochastic Mode
# ========================================================================


class TestElementOperatorStochastic:
    """Test stochastic mode with RNG key handling."""

    def test_stochastic_adds_noise(self):
        """Stochastic mode passes key for randomness."""

        def add_noise(element, key):
            noise = jax.random.normal(key, element.data["value"].shape) * 0.1
            new_data = {"value": element.data["value"] + noise}
            return element.replace(data=new_data)

        config = ElementOperatorConfig(stochastic=True, stream_name="augment")
        rngs = nnx.Rngs(0, augment=42)
        op = ElementOperator(config, fn=add_noise, rngs=rngs)

        original = jnp.ones((3, 4))
        batch_data = {"value": original}
        batch = create_test_batch(batch_data)

        result = op(batch)

        # Output should differ from input (noise added)
        assert not jnp.allclose(result.get_data()["value"], original)
        # But should be close (noise is small)
        assert jnp.allclose(result.get_data()["value"], original, atol=0.5)

    def test_stochastic_reproducibility(self):
        """Same seed produces same output."""

        def add_noise(element, key):
            noise = jax.random.normal(key, element.data["value"].shape) * 0.1
            new_data = {"value": element.data["value"] + noise}
            return element.replace(data=new_data)

        config = ElementOperatorConfig(stochastic=True, stream_name="augment")

        # Two operators with same seed
        rngs1 = nnx.Rngs(0, augment=42)
        op1 = ElementOperator(config, fn=add_noise, rngs=rngs1)

        rngs2 = nnx.Rngs(0, augment=42)
        op2 = ElementOperator(config, fn=add_noise, rngs=rngs2)

        batch_data = {"value": jnp.ones((2, 3))}
        batch = create_test_batch(batch_data)

        result1 = op1(batch)
        result2 = op2(batch)

        # Should produce identical outputs
        assert jnp.allclose(result1.get_data()["value"], result2.get_data()["value"])

    def test_stochastic_different_seeds_differ(self):
        """Different seeds produce different outputs."""

        def add_noise(element, key):
            noise = jax.random.normal(key, element.data["value"].shape) * 0.1
            new_data = {"value": element.data["value"] + noise}
            return element.replace(data=new_data)

        config = ElementOperatorConfig(stochastic=True, stream_name="augment")

        # Two operators with different seeds
        rngs1 = nnx.Rngs(0, augment=1)
        op1 = ElementOperator(config, fn=add_noise, rngs=rngs1)

        rngs2 = nnx.Rngs(0, augment=999)
        op2 = ElementOperator(config, fn=add_noise, rngs=rngs2)

        batch_data = {"value": jnp.ones((2, 3))}
        batch = create_test_batch(batch_data)

        result1 = op1(batch)
        result2 = op2(batch)

        # Should produce different outputs
        assert not jnp.allclose(result1.get_data()["value"], result2.get_data()["value"])

    def test_deterministic_ignores_key(self):
        """Deterministic mode ignores key (always same output)."""

        def multiply(element, key):
            # Ignores key - deterministic operation
            new_data = {"value": element.data["value"] * 2.0}
            return element.replace(data=new_data)

        config = ElementOperatorConfig(stochastic=False)

        # Different seeds shouldn't matter
        rngs1 = nnx.Rngs(1)
        op1 = ElementOperator(config, fn=multiply, rngs=rngs1)

        rngs2 = nnx.Rngs(999)
        op2 = ElementOperator(config, fn=multiply, rngs=rngs2)

        batch_data = {"value": jnp.array([[1.0, 2.0]])}
        batch = create_test_batch(batch_data)

        result1 = op1(batch)
        result2 = op2(batch)

        # Should produce identical deterministic outputs
        expected = jnp.array([[2.0, 4.0]])
        assert jnp.allclose(result1.get_data()["value"], expected)
        assert jnp.allclose(result2.get_data()["value"], expected)


# ========================================================================
# Test Category 6: Edge Cases
# ========================================================================


class TestElementOperatorEdgeCases:
    """Test edge cases."""

    def test_empty_data(self):
        """Handles empty data dict."""

        def identity(element, key):
            return element

        config = ElementOperatorConfig(stochastic=False)
        rngs = nnx.Rngs(0)
        op = ElementOperator(config, fn=identity, rngs=rngs)

        batch = Batch([Element(data={}, state={}, metadata=None)])

        result = op(batch)

        assert result.get_data() == {}
        assert result.get_states() == {}

    def test_nested_data_structure(self):
        """Handles nested data structures (preserving structure)."""

        def transform_nested(element, key):
            # Transform nested data while preserving structure
            image = element.data["features"]["image"] * 2.0
            depth = element.data["features"]["depth"] + 1.0
            new_data = {"features": {"image": image, "depth": depth}}
            return element.replace(data=new_data)

        config = ElementOperatorConfig(stochastic=False)
        rngs = nnx.Rngs(0)
        op = ElementOperator(config, fn=transform_nested, rngs=rngs)

        batch_data = {
            "features": {
                "image": jnp.array([[1.0, 2.0]]),
                "depth": jnp.array([[3.0, 4.0]]),
            }
        }
        batch = create_test_batch(batch_data)

        result = op(batch)
        result_data = result.get_data()

        assert jnp.allclose(result_data["features"]["image"], jnp.array([[2.0, 4.0]]))
        assert jnp.allclose(result_data["features"]["depth"], jnp.array([[4.0, 5.0]]))

    def test_single_element_batch(self):
        """Works with single-element batch."""

        def negate(element, key):
            new_data = jax.tree.map(lambda x: -x, element.data)
            return element.replace(data=new_data)

        config = ElementOperatorConfig(stochastic=False)
        rngs = nnx.Rngs(0)
        op = ElementOperator(config, fn=negate, rngs=rngs)

        batch_data = {"value": jnp.array([[5.0]])}  # Single element
        batch = create_test_batch(batch_data)

        result = op(batch)

        assert jnp.allclose(result.get_data()["value"], jnp.array([[-5.0]]))

    def test_large_batch(self):
        """Works with large batch."""

        def add_ten(element, key):
            new_data = {"value": element.data["value"] + 10.0}
            return element.replace(data=new_data)

        config = ElementOperatorConfig(stochastic=False)
        rngs = nnx.Rngs(0)
        op = ElementOperator(config, fn=add_ten, rngs=rngs)

        batch_data = {"value": jnp.zeros((100, 50))}  # Large batch
        batch = create_test_batch(batch_data)

        result = op(batch)

        assert jnp.allclose(result.get_data()["value"], jnp.ones((100, 50)) * 10.0)


# ========================================================================
# Test Category 7: JAX Compatibility
# ========================================================================


class TestElementOperatorJAXCompatibility:
    """Test JAX transformation compatibility."""

    def test_jit_deterministic(self):
        """JIT compilation works in deterministic mode."""

        def normalize(element, key):
            new_data = {"value": (element.data["value"] - 0.5) / 0.5}
            return element.replace(data=new_data)

        config = ElementOperatorConfig(stochastic=False)
        rngs = nnx.Rngs(0)
        op = ElementOperator(config, fn=normalize, rngs=rngs)

        batch_data = {"value": jnp.ones((2, 3))}
        batch = create_test_batch(batch_data)

        # JIT compile with nnx.jit (pass module as argument, not closure)
        @nnx.jit
        def jitted_apply(model, batch):
            return model(batch)

        result = jitted_apply(op, batch)

        expected = (jnp.ones((2, 3)) - 0.5) / 0.5
        assert jnp.allclose(result.data.get_value()["value"], expected)

    def test_jit_stochastic(self):
        """JIT compilation works in stochastic mode."""

        def add_noise(element, key):
            noise = jax.random.normal(key, element.data["value"].shape) * 0.1
            new_data = {"value": element.data["value"] + noise}
            return element.replace(data=new_data)

        config = ElementOperatorConfig(stochastic=True, stream_name="augment")
        rngs = nnx.Rngs(0, augment=42)
        op = ElementOperator(config, fn=add_noise, rngs=rngs)

        batch_data = {"value": jnp.ones((2, 3))}
        batch = create_test_batch(batch_data)

        # JIT compile using nnx.jit (passes operator as argument)
        @nnx.jit
        def apply_op(operator, batch):
            return operator(batch)

        result = apply_op(op, batch)

        # Should produce noise
        assert not jnp.allclose(result.data.get_value()["value"], jnp.ones((2, 3)))

    def test_function_purity(self):
        """User function can be pure (no side effects)."""

        def pure_transform(element, key):
            # Pure function - no external state access
            return element.replace(data={"value": element.data["value"] ** 2})

        config = ElementOperatorConfig(stochastic=False)
        rngs = nnx.Rngs(0)
        op = ElementOperator(config, fn=pure_transform, rngs=rngs)

        batch_data = {"value": jnp.array([[2.0, 3.0]])}
        batch = create_test_batch(batch_data)

        result = op(batch)

        assert jnp.allclose(result.get_data()["value"], jnp.array([[4.0, 9.0]]))


# ========================================================================
# Test Category 8: Integration Tests
# ========================================================================


class TestElementOperatorIntegration:
    """Integration tests for real-world usage patterns."""

    def test_chaining_operators(self):
        """Can chain multiple ElementOperators."""

        def normalize(element, key):
            new_data = {"value": element.data["value"] / 255.0}
            return element.replace(data=new_data)

        def center(element, key):
            new_data = {"value": element.data["value"] - 0.5}
            return element.replace(data=new_data)

        config = ElementOperatorConfig(stochastic=False)
        rngs = nnx.Rngs(0)

        op1 = ElementOperator(config, fn=normalize, rngs=rngs)
        op2 = ElementOperator(config, fn=center, rngs=rngs)

        batch_data = {"value": jnp.array([[255.0, 127.5]])}
        batch = create_test_batch(batch_data)

        # Chain: normalize then center
        result = op2(op1(batch))

        # 255/255 - 0.5 = 0.5, 127.5/255 - 0.5 = 0.0
        expected = jnp.array([[0.5, 0.0]])
        assert jnp.allclose(result.get_data()["value"], expected, atol=1e-5)

    def test_coordinated_augmentation(self):
        """Real-world coordinated augmentation pattern."""

        def augment_image_and_mask(element, key):
            # Flip both image and mask together
            image = element.data["image"]
            mask = element.data["mask"]

            # Flip horizontally (simulated)
            flipped_image = image[..., ::-1]
            flipped_mask = mask[..., ::-1]

            new_data = {"image": flipped_image, "mask": flipped_mask}
            return element.replace(data=new_data)

        config = ElementOperatorConfig(stochastic=False)
        rngs = nnx.Rngs(0)
        op = ElementOperator(config, fn=augment_image_and_mask, rngs=rngs)

        batch_data = {
            "image": jnp.array([[[1.0, 2.0, 3.0]]]),  # Shape (1, 1, 3)
            "mask": jnp.array([[[0.0, 1.0, 0.0]]]),  # Shape (1, 1, 3)
        }
        batch = create_test_batch(batch_data)

        result = op(batch)
        result_data = result.get_data()

        # Both should be flipped
        assert jnp.allclose(result_data["image"], jnp.array([[[3.0, 2.0, 1.0]]]))
        assert jnp.allclose(result_data["mask"], jnp.array([[[0.0, 1.0, 0.0]]]))

    def test_random_augmentation_pipeline(self):
        """Stochastic augmentation pipeline."""

        def random_augment(element, key):
            # Add random noise based on key
            key1, key2 = jax.random.split(key)

            image = element.data["image"]
            noise = jax.random.normal(key1, image.shape) * 0.01

            new_data = {"image": image + noise}
            return element.replace(data=new_data)

        config = ElementOperatorConfig(stochastic=True, stream_name="augment")
        rngs = nnx.Rngs(0, augment=123)
        op = ElementOperator(config, fn=random_augment, rngs=rngs)

        batch_data = {"image": jnp.ones((4, 8, 8, 3))}  # 4 images
        batch = create_test_batch(batch_data)

        result = op(batch)

        # Should add noise
        assert not jnp.allclose(result.get_data()["image"], jnp.ones((4, 8, 8, 3)))
        # But stay close
        assert jnp.allclose(result.get_data()["image"], jnp.ones((4, 8, 8, 3)), atol=0.1)
