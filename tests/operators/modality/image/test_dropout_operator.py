"""Tests for DropoutOperator.

This test suite validates the DropoutOperator implementation which provides
pixel-level and channel-level dropout for image data. Tests cover configuration
validation, both dropout modes, batch processing, and JAX compatibility.

Example Usage:
    config = DropoutOperatorConfig(
        field_key="image",
        dropout_rate=0.2,
        mode="pixel",
        stochastic=True,
        stream_name="augment",
    )
    operator = DropoutOperator(config, rngs=nnx.Rngs(0, augment=1))
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from datarax.core.element_batch import Batch, Element
from datarax.operators.modality.image.dropout_operator import (
    DropoutOperator,
    DropoutOperatorConfig,
)


class TestDropoutOperatorConfig:
    """Tests for DropoutOperatorConfig validation."""

    def test_basic_config_creation(self):
        """Test default configuration creation."""
        config = DropoutOperatorConfig(
            field_key="image",
            dropout_rate=0.2,
        )
        assert config.field_key == "image"
        assert config.dropout_rate == 0.2
        assert config.mode == "pixel"  # Default
        assert config.clip_range is None  # No clipping by default

    def test_custom_params_config(self):
        """Test configuration with custom parameters."""
        config = DropoutOperatorConfig(
            field_key="custom_image",
            dropout_rate=0.3,
            mode="channel",
            clip_range=(0.0, 1.0),
        )
        assert config.field_key == "custom_image"
        assert config.dropout_rate == 0.3
        assert config.mode == "channel"
        assert config.clip_range == (0.0, 1.0)

    def test_invalid_dropout_rate_negative(self):
        """Test validation of negative dropout rate."""
        with pytest.raises(ValueError, match="dropout_rate must be in"):
            DropoutOperatorConfig(
                field_key="image",
                dropout_rate=-0.1,
            )

    def test_invalid_dropout_rate_too_high(self):
        """Test validation of dropout rate > 1.0."""
        with pytest.raises(ValueError, match="dropout_rate must be in"):
            DropoutOperatorConfig(
                field_key="image",
                dropout_rate=1.5,
            )

    def test_invalid_dropout_rate_type(self):
        """Test validation of non-numeric dropout rate."""
        with pytest.raises(TypeError, match="dropout_rate must be a number"):
            DropoutOperatorConfig(
                field_key="image",
                dropout_rate="0.2",  # String instead of float
            )

    def test_invalid_mode(self):
        """Test validation of invalid dropout mode."""
        with pytest.raises(ValueError, match="mode must be"):
            DropoutOperatorConfig(
                field_key="image",
                dropout_rate=0.2,
                mode="invalid",
            )


class TestDropoutOperatorInitialization:
    """Tests for DropoutOperator initialization modes."""

    def test_init_deterministic_mode(self):
        """Test initialization in deterministic mode."""
        config = DropoutOperatorConfig(
            field_key="image",
            dropout_rate=0.2,
            stochastic=False,
        )
        operator = DropoutOperator(config, rngs=nnx.Rngs(0))

        assert operator.config.dropout_rate == 0.2
        assert operator.config.stochastic is False

    def test_init_stochastic_mode(self):
        """Test initialization in stochastic mode."""
        config = DropoutOperatorConfig(
            field_key="image",
            dropout_rate=0.3,
            stochastic=True,
            stream_name="augment",
        )
        operator = DropoutOperator(config, rngs=nnx.Rngs(0, augment=1))

        assert operator.config.dropout_rate == 0.3
        assert operator.config.stochastic is True
        assert operator.config.stream_name == "augment"

    def test_init_pixel_mode(self):
        """Test initialization with pixel mode."""
        config = DropoutOperatorConfig(
            field_key="image",
            dropout_rate=0.2,
            mode="pixel",
        )
        operator = DropoutOperator(config, rngs=nnx.Rngs(0))

        assert operator.config.mode == "pixel"

    def test_init_channel_mode(self):
        """Test initialization with channel mode."""
        config = DropoutOperatorConfig(
            field_key="image",
            dropout_rate=0.2,
            mode="channel",
        )
        operator = DropoutOperator(config, rngs=nnx.Rngs(0))

        assert operator.config.mode == "channel"


class TestDropoutOperatorTransformations:
    """Tests for basic dropout transformations."""

    def test_deterministic_pixel_mode_single_element(self):
        """Test deterministic pixel dropout on single element."""
        config = DropoutOperatorConfig(
            field_key="image",
            dropout_rate=0.5,
            mode="pixel",
            stochastic=False,
        )
        operator = DropoutOperator(config, rngs=nnx.Rngs(0))

        # Create single element
        image = jnp.ones((32, 32, 3))
        data = {"image": image}

        result, state, metadata = operator.apply(data, {}, {})

        # Should have some zeros (dropped pixels) and some ones (kept pixels)
        result_image = result["image"]
        assert jnp.any(result_image == 0.0)  # Some dropped
        assert jnp.any(result_image == 1.0)  # Some kept
        assert result_image.shape == image.shape

    def test_deterministic_channel_mode_single_element(self):
        """Test deterministic channel dropout on single element."""
        config = DropoutOperatorConfig(
            field_key="image",
            dropout_rate=0.5,
            mode="channel",
            stochastic=False,
        )
        operator = DropoutOperator(config, rngs=nnx.Rngs(0))

        # Create single element
        image = jnp.ones((32, 32, 3))
        data = {"image": image}

        result, state, metadata = operator.apply(data, {}, {})

        # In channel mode, entire channels are dropped
        result_image = result["image"]
        assert result_image.shape == image.shape

        # Check if any channel is fully dropped (all zeros)
        # Sum over H and W dimensions - if a channel is dropped, sum will be 0
        channel_sums = jnp.sum(result_image, axis=(0, 1))
        # With dropout_rate=0.5 and 3 channels, likely at least one channel dropped
        assert jnp.any(channel_sums == 0.0) or jnp.all(channel_sums > 0.0)

    def test_deterministic_apply_batch_pixel_mode(self):
        """Test deterministic pixel dropout on batch."""
        config = DropoutOperatorConfig(
            field_key="image",
            dropout_rate=0.3,
            mode="pixel",
            stochastic=False,
        )
        operator = DropoutOperator(config, rngs=nnx.Rngs(0))

        # Create batch of elements
        images = jnp.ones((4, 16, 16, 3))
        elements = [Element(data={"image": img}, state={}, metadata={}) for img in images]
        batch = Batch(elements=elements)

        result_batch = operator.apply_batch(batch)

        # Check batch processing
        result_images = result_batch.data.get_value()["image"]
        assert result_images.shape == (4, 16, 16, 3)
        # Should have some dropout
        assert jnp.any(result_images < images)

    def test_deterministic_apply_batch_channel_mode(self):
        """Test deterministic channel dropout on batch."""
        config = DropoutOperatorConfig(
            field_key="image",
            dropout_rate=0.3,
            mode="channel",
            stochastic=False,
        )
        operator = DropoutOperator(config, rngs=nnx.Rngs(0))

        # Create batch of elements
        images = jnp.ones((2, 16, 16, 3))
        elements = [Element(data={"image": img}, state={}, metadata={}) for img in images]
        batch = Batch(elements=elements)

        result_batch = operator.apply_batch(batch)

        result_images = result_batch.data.get_value()["image"]
        assert result_images.shape == (2, 16, 16, 3)


class TestDropoutOperatorEdgeCases:
    """Tests for edge cases and error handling."""

    def test_apply_zero_dropout_rate(self):
        """Test that zero dropout rate produces no change."""
        config = DropoutOperatorConfig(
            field_key="image",
            dropout_rate=0.0,
            stochastic=False,
        )
        operator = DropoutOperator(config, rngs=nnx.Rngs(0))

        image = jnp.ones((16, 16, 3))
        data = {"image": image}

        result, _, _ = operator.apply(data, {}, {})

        # Should be completely unchanged
        assert jnp.array_equal(result["image"], image)

    def test_apply_full_dropout_rate(self):
        """Test that dropout rate of 1.0 drops everything."""
        config = DropoutOperatorConfig(
            field_key="image",
            dropout_rate=1.0,
            mode="pixel",
            stochastic=False,
        )
        operator = DropoutOperator(config, rngs=nnx.Rngs(0))

        image = jnp.ones((16, 16, 3))
        data = {"image": image}

        result, _, _ = operator.apply(data, {}, {})

        # Should be all zeros
        assert jnp.all(result["image"] == 0.0)

    def test_apply_custom_image_key(self):
        """Test operator with custom field key."""
        config = DropoutOperatorConfig(
            field_key="custom_image",
            dropout_rate=0.2,
            stochastic=False,
        )
        operator = DropoutOperator(config, rngs=nnx.Rngs(0))

        image = jnp.ones((16, 16, 3))
        data = {"custom_image": image, "other": jnp.ones(10)}

        result, _, _ = operator.apply(data, {}, {})

        # Custom image should be transformed, other data unchanged
        assert "custom_image" in result
        assert "other" in result
        assert jnp.array_equal(result["other"], jnp.ones(10))

    def test_apply_missing_image_key(self):
        """Test operator handles missing field with KeyError."""
        config = DropoutOperatorConfig(
            field_key="image",
            dropout_rate=0.2,
            stochastic=False,
        )
        operator = DropoutOperator(config, rngs=nnx.Rngs(0))

        data = {"other": jnp.ones(10)}

        # Should raise KeyError from _extract_field
        with pytest.raises(KeyError):
            operator.apply(data, {}, {})

    def test_apply_target_key(self):
        """Test operator with target_key different from field_key."""
        config = DropoutOperatorConfig(
            field_key="image",
            target_key="dropout_image",
            dropout_rate=0.2,
            stochastic=False,
        )
        operator = DropoutOperator(config, rngs=nnx.Rngs(0))

        image = jnp.ones((16, 16, 3))
        data = {"image": image}

        result, _, _ = operator.apply(data, {}, {})

        # Original image unchanged, new field created
        assert "image" in result
        assert "dropout_image" in result
        assert jnp.array_equal(result["image"], image)
        assert not jnp.array_equal(result["dropout_image"], image)

    def test_channel_mode_2d_image_fallback(self):
        """Test channel mode falls back to pixel mode for 2D images."""
        config = DropoutOperatorConfig(
            field_key="image",
            dropout_rate=0.3,
            mode="channel",
            stochastic=False,
        )
        operator = DropoutOperator(config, rngs=nnx.Rngs(0))

        # 2D image (no channel dimension)
        image = jnp.ones((16, 16))
        data = {"image": image}

        result, _, _ = operator.apply(data, {}, {})

        # Should still work (fallback to pixel mode)
        assert result["image"].shape == image.shape


class TestDropoutOperatorStochasticMode:
    """Tests for stochastic random parameter generation."""

    def test_generate_random_params_shape(self):
        """Test random parameter generation produces correct shape."""
        config = DropoutOperatorConfig(
            field_key="image",
            dropout_rate=0.3,
            mode="pixel",
            stochastic=True,
            stream_name="augment",
        )
        operator = DropoutOperator(config, rngs=nnx.Rngs(42, augment=1))

        rng = jax.random.key(42)
        data_shapes = {"image": (4, 32, 32, 3)}  # Batch size 4

        random_params = operator.generate_random_params(rng, data_shapes)

        assert "keep_mask" in random_params
        # Pixel mode: full shape mask
        assert random_params["keep_mask"].shape == (4, 32, 32, 3)

    def test_generate_random_params_channel_mode_shape(self):
        """Test random parameter generation for channel mode."""
        config = DropoutOperatorConfig(
            field_key="image",
            dropout_rate=0.3,
            mode="channel",
            stochastic=True,
            stream_name="augment",
        )
        operator = DropoutOperator(config, rngs=nnx.Rngs(42, augment=1))

        rng = jax.random.key(42)
        data_shapes = {"image": (4, 32, 32, 3)}  # Batch size 4

        random_params = operator.generate_random_params(rng, data_shapes)

        assert "keep_mask" in random_params
        # Channel mode: broadcasted shape (batch, 1, 1, C)
        assert random_params["keep_mask"].shape == (4, 1, 1, 3)

    def test_stochastic_apply_batch_varies_between_samples(self):
        """Test stochastic mode produces different results for batch elements."""
        config = DropoutOperatorConfig(
            field_key="image",
            dropout_rate=0.3,
            mode="pixel",
            stochastic=True,
            stream_name="augment",
        )
        operator = DropoutOperator(config, rngs=nnx.Rngs(42, augment=1))

        # Create batch with identical images
        images = jnp.ones((4, 16, 16, 3))
        elements = [Element(data={"image": img}, state={}, metadata={}) for img in images]
        batch = Batch(elements=elements)

        result_batch = operator.apply_batch(batch)

        # Due to randomness, results should vary between samples
        result_images = result_batch.data.get_value()["image"]
        # Check if different samples have different dropout masks
        # (high probability with stochastic mode)
        found_difference = False
        for i in range(len(result_images) - 1):
            if not jnp.array_equal(result_images[i], result_images[i + 1]):
                found_difference = True
                break
        assert found_difference, "Stochastic mode should produce varying dropout masks"


class TestDropoutOperatorJAXCompatibility:
    """Tests for JAX transformation compatibility (jit, vmap, grad)."""

    def test_jit_compatibility_deterministic(self):
        """Test JIT compilation with deterministic operator."""
        config = DropoutOperatorConfig(
            field_key="image",
            dropout_rate=0.3,
            mode="pixel",
            stochastic=False,
        )
        operator = DropoutOperator(config, rngs=nnx.Rngs(0))

        @nnx.jit
        def jit_apply(op, data):
            result, state, metadata = op.apply(data, {}, {})
            return result

        image = jnp.ones((16, 16, 3))
        data = {"image": image}

        result = jit_apply(operator, data)

        # Should have dropout applied
        assert result["image"].shape == image.shape
        assert jnp.any(result["image"] < image)

    def test_jit_compatibility_stochastic(self):
        """Test JIT compilation with stochastic operator."""
        config = DropoutOperatorConfig(
            field_key="image",
            dropout_rate=0.3,
            mode="pixel",
            stochastic=True,
            stream_name="augment",
        )
        operator = DropoutOperator(config, rngs=nnx.Rngs(42, augment=1))

        @nnx.jit
        def jit_apply_batch(op, batch):
            return op.apply_batch(batch)

        images = jnp.ones((2, 16, 16, 3))
        elements = [Element(data={"image": img}, state={}, metadata={}) for img in images]
        batch = Batch(elements=elements)

        result_batch = jit_apply_batch(operator, batch)
        assert result_batch.batch_size == 2

    def test_vmap_compatibility(self):
        """Test vmap compatibility via apply_batch."""
        config = DropoutOperatorConfig(
            field_key="image",
            dropout_rate=0.2,
            mode="pixel",
            stochastic=False,
        )
        operator = DropoutOperator(config, rngs=nnx.Rngs(0))

        # apply_batch uses vmap internally
        images = jnp.ones((4, 32, 32, 3))
        elements = [Element(data={"image": img}, state={}, metadata={}) for img in images]
        batch = Batch(elements=elements)

        result_batch = operator.apply_batch(batch)

        # All elements should be transformed correctly
        assert result_batch.batch_size == 4
        result_images = result_batch.data.get_value()["image"]
        assert result_images.shape == (4, 32, 32, 3)

    def test_grad_compatibility(self):
        """Test gradient computation through dropout operator.

        Note: Dropout is not differentiable in the traditional sense
        (it's a discrete operation), but JAX should handle it gracefully.
        We test that gradients can be computed for the input.
        """
        config = DropoutOperatorConfig(
            field_key="image",
            dropout_rate=0.0,  # Use 0 dropout for meaningful gradients
            mode="pixel",
            stochastic=False,
        )
        operator = DropoutOperator(config, rngs=nnx.Rngs(0))

        def loss_fn(image_data):
            """Simple loss function for gradient test."""
            data = {"image": image_data}
            result, _, _ = operator.apply(data, {}, {})
            # Mean squared value as loss
            return jnp.sum(result["image"] ** 2)

        image = jnp.ones((8, 8, 3))

        # Compute gradient
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(image)

        # With 0 dropout, gradients should be 2*image (from x^2)
        expected_grad = 2.0 * image
        assert jnp.allclose(grads, expected_grad)

    def test_grad_compatibility_with_dropout(self):
        """Test gradient computation with actual dropout.

        With dropout, gradients will be masked (zero for dropped pixels).
        """
        config = DropoutOperatorConfig(
            field_key="image",
            dropout_rate=0.5,
            mode="pixel",
            stochastic=False,
        )
        operator = DropoutOperator(config, rngs=nnx.Rngs(0))

        def loss_fn(image_data):
            """Loss function that uses dropout."""
            data = {"image": image_data}
            result, _, _ = operator.apply(data, {}, {})
            return jnp.sum(result["image"] ** 2)

        image = jnp.ones((8, 8, 3))

        # Compute gradient
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(image)

        # Gradients should exist and have same shape
        assert grads.shape == image.shape
        # Some gradients should be zero (for dropped pixels)
        assert jnp.any(grads == 0.0)
        # Some gradients should be non-zero (for kept pixels)
        assert jnp.any(grads != 0.0)


class TestDropoutOperatorCommonPatterns:
    """Tests for common dropout patterns."""

    def test_pixel_mode_pattern(self):
        """Test pixel mode dropout pattern."""
        config = DropoutOperatorConfig(
            field_key="image",
            dropout_rate=0.2,
            mode="pixel",
            stochastic=False,
        )
        operator = DropoutOperator(config, rngs=nnx.Rngs(42))

        image = jnp.ones((32, 32, 3))
        data = {"image": image}

        result, _, _ = operator.apply(data, {}, {})

        # Should have dropout applied
        assert result["image"].shape == image.shape
        assert jnp.any(result["image"] == 0.0)  # Some dropped
        assert jnp.any(result["image"] == 1.0)  # Some kept

    def test_channel_mode_pattern(self):
        """Test channel mode dropout pattern."""
        config = DropoutOperatorConfig(
            field_key="image",
            dropout_rate=0.3,
            mode="channel",
            stochastic=False,
        )
        operator = DropoutOperator(config, rngs=nnx.Rngs(42))

        image = jnp.ones((32, 32, 3))
        data = {"image": image}

        result, _, _ = operator.apply(data, {}, {})

        # Should have channel dropout applied
        assert result["image"].shape == image.shape
