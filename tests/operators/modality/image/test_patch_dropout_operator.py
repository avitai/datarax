"""Tests for PatchDropoutOperator.

This test suite validates the PatchDropoutOperator implementation which provides
random rectangular patch dropout for image data. Tests cover configuration
validation, various patch sizes, batch processing, and JAX compatibility.

Example Usage:
    config = PatchDropoutOperatorConfig(
        field_key="image",
        num_patches=4,
        patch_size=(8, 8),
        drop_value=0.0,
        stochastic=True,
        stream_name="augment",
    )
    operator = PatchDropoutOperator(config, rngs=nnx.Rngs(0, augment=1))
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from datarax.core.element_batch import Batch, Element
from datarax.operators.modality.image.patch_dropout_operator import (
    PatchDropoutOperator,
    PatchDropoutOperatorConfig,
)


class TestPatchDropoutOperatorConfig:
    """Tests for PatchDropoutOperatorConfig validation."""

    def test_basic_config_creation(self):
        """Test default configuration creation."""
        config = PatchDropoutOperatorConfig(
            field_key="image",
            num_patches=4,
        )
        assert config.field_key == "image"
        assert config.num_patches == 4
        assert config.patch_size == (8, 8)  # Default
        assert config.drop_value == 0.0  # Default
        assert config.clip_range is None  # No clipping by default

    def test_custom_params_config(self):
        """Test configuration with custom parameters."""
        config = PatchDropoutOperatorConfig(
            field_key="custom_image",
            num_patches=8,
            patch_size=(16, 16),
            drop_value=0.5,
            clip_range=(0.0, 1.0),
        )
        assert config.field_key == "custom_image"
        assert config.num_patches == 8
        assert config.patch_size == (16, 16)
        assert config.drop_value == 0.5
        assert config.clip_range == (0.0, 1.0)

    def test_config_with_int_patch_size(self):
        """Test configuration with integer patch_size (backward compat)."""
        config = PatchDropoutOperatorConfig(
            field_key="image",
            num_patches=4,
            patch_size=12,  # Int instead of tuple
        )
        # Should be converted to tuple
        assert config.patch_size == (12, 12)

    def test_invalid_num_patches_negative(self):
        """Test validation of negative num_patches."""
        with pytest.raises(ValueError, match="num_patches must be non-negative"):
            PatchDropoutOperatorConfig(
                field_key="image",
                num_patches=-1,
            )

    def test_invalid_num_patches_type(self):
        """Test validation of non-integer num_patches."""
        with pytest.raises(TypeError, match="num_patches must be an integer"):
            PatchDropoutOperatorConfig(
                field_key="image",
                num_patches=4.5,
            )

    def test_invalid_patch_size_negative(self):
        """Test validation of negative patch_size."""
        with pytest.raises(ValueError, match="patch_size must be positive"):
            PatchDropoutOperatorConfig(
                field_key="image",
                num_patches=4,
                patch_size=-8,
            )

    def test_invalid_patch_size_tuple_negative(self):
        """Test validation of negative dimensions in patch_size tuple."""
        with pytest.raises(ValueError, match="patch_size dimensions must be positive"):
            PatchDropoutOperatorConfig(
                field_key="image",
                num_patches=4,
                patch_size=(8, -8),
            )

    def test_invalid_patch_size_wrong_length(self):
        """Test validation of patch_size tuple with wrong length."""
        with pytest.raises(ValueError, match="patch_size must be a tuple of length 2"):
            PatchDropoutOperatorConfig(
                field_key="image",
                num_patches=4,
                patch_size=(8, 8, 8),
            )

    def test_invalid_drop_value_type(self):
        """Test validation of non-numeric drop_value."""
        with pytest.raises(TypeError, match="drop_value must be a number"):
            PatchDropoutOperatorConfig(
                field_key="image",
                num_patches=4,
                drop_value="zero",
            )


class TestPatchDropoutOperatorInitialization:
    """Tests for PatchDropoutOperator initialization modes."""

    def test_init_deterministic_mode(self):
        """Test initialization in deterministic mode."""
        config = PatchDropoutOperatorConfig(
            field_key="image",
            num_patches=4,
            patch_size=(8, 8),
            stochastic=False,
        )
        operator = PatchDropoutOperator(config, rngs=nnx.Rngs(0))

        assert operator.config.num_patches == 4
        assert operator.config.patch_size == (8, 8)
        assert operator.config.stochastic is False

    def test_init_stochastic_mode(self):
        """Test initialization in stochastic mode."""
        config = PatchDropoutOperatorConfig(
            field_key="image",
            num_patches=8,
            patch_size=(16, 16),
            stochastic=True,
            stream_name="augment",
        )
        operator = PatchDropoutOperator(config, rngs=nnx.Rngs(0, augment=1))

        assert operator.config.num_patches == 8
        assert operator.config.patch_size == (16, 16)
        assert operator.config.stochastic is True
        assert operator.config.stream_name == "augment"

    def test_init_square_patches(self):
        """Test initialization with square patches."""
        config = PatchDropoutOperatorConfig(
            field_key="image",
            num_patches=4,
            patch_size=10,  # Int for square
        )
        operator = PatchDropoutOperator(config, rngs=nnx.Rngs(0))

        assert operator.config.patch_size == (10, 10)

    def test_init_rectangular_patches(self):
        """Test initialization with rectangular patches."""
        config = PatchDropoutOperatorConfig(
            field_key="image",
            num_patches=4,
            patch_size=(8, 16),  # Rectangular
        )
        operator = PatchDropoutOperator(config, rngs=nnx.Rngs(0))

        assert operator.config.patch_size == (8, 16)


class TestPatchDropoutOperatorTransformations:
    """Tests for basic patch dropout transformations."""

    def test_deterministic_single_element(self):
        """Test deterministic patch dropout on single element."""
        config = PatchDropoutOperatorConfig(
            field_key="image",
            num_patches=2,
            patch_size=(4, 4),
            drop_value=0.0,
            stochastic=False,
        )
        operator = PatchDropoutOperator(config, rngs=nnx.Rngs(0))

        # Create single element
        image = jnp.ones((32, 32, 3))
        data = {"image": image}

        result, state, metadata = operator.apply(data, {}, {})

        # Should have some zeros (dropped patches)
        result_image = result["image"]
        assert jnp.any(result_image == 0.0)  # Some patches dropped
        assert jnp.any(result_image == 1.0)  # Some pixels kept
        assert result_image.shape == image.shape

    def test_deterministic_apply_batch(self):
        """Test deterministic patch dropout on batch."""
        config = PatchDropoutOperatorConfig(
            field_key="image",
            num_patches=3,
            patch_size=(8, 8),
            drop_value=0.0,
            stochastic=False,
        )
        operator = PatchDropoutOperator(config, rngs=nnx.Rngs(0))

        # Create batch of elements
        images = jnp.ones((4, 64, 64, 3))
        elements = [Element(data={"image": img}, state={}, metadata={}) for img in images]
        batch = Batch(elements=elements)

        result_batch = operator.apply_batch(batch)

        # Check batch processing
        result_images = result_batch.data.get_value()["image"]
        assert result_images.shape == (4, 64, 64, 3)
        # Should have some patches dropped
        assert jnp.any(result_images < images)

    def test_custom_drop_value(self):
        """Test patch dropout with custom drop value."""
        config = PatchDropoutOperatorConfig(
            field_key="image",
            num_patches=2,
            patch_size=(8, 8),
            drop_value=0.5,  # Gray instead of black
            stochastic=False,
        )
        operator = PatchDropoutOperator(config, rngs=nnx.Rngs(0))

        image = jnp.ones((32, 32, 3))
        data = {"image": image}

        result, _, _ = operator.apply(data, {}, {})

        # Should have patches with value 0.5
        result_image = result["image"]
        assert jnp.any(result_image == 0.5)  # Dropped patches

    def test_2d_image_support(self):
        """Test patch dropout on 2D (grayscale) images."""
        config = PatchDropoutOperatorConfig(
            field_key="image",
            num_patches=2,
            patch_size=(4, 4),
            drop_value=0.0,
            stochastic=False,
        )
        operator = PatchDropoutOperator(config, rngs=nnx.Rngs(0))

        # 2D image (no channel dimension)
        image = jnp.ones((32, 32))
        data = {"image": image}

        result, _, _ = operator.apply(data, {}, {})

        # Should still be 2D
        result_image = result["image"]
        assert result_image.shape == image.shape
        assert result_image.ndim == 2


class TestPatchDropoutOperatorEdgeCases:
    """Tests for edge cases and error handling."""

    def test_apply_zero_num_patches(self):
        """Test that zero num_patches produces no change."""
        config = PatchDropoutOperatorConfig(
            field_key="image",
            num_patches=0,
            stochastic=False,
        )
        operator = PatchDropoutOperator(config, rngs=nnx.Rngs(0))

        image = jnp.ones((16, 16, 3))
        data = {"image": image}

        result, _, _ = operator.apply(data, {}, {})

        # Should be completely unchanged
        assert jnp.array_equal(result["image"], image)

    def test_apply_patch_too_large(self):
        """Test that oversized patches are handled gracefully."""
        config = PatchDropoutOperatorConfig(
            field_key="image",
            num_patches=2,
            patch_size=(64, 64),  # Larger than image
            stochastic=False,
        )
        operator = PatchDropoutOperator(config, rngs=nnx.Rngs(0))

        image = jnp.ones((32, 32, 3))  # Smaller than patch
        data = {"image": image}

        result, _, _ = operator.apply(data, {}, {})

        # Should be unchanged (patches don't fit)
        assert jnp.array_equal(result["image"], image)

    def test_apply_custom_image_key(self):
        """Test operator with custom field key."""
        config = PatchDropoutOperatorConfig(
            field_key="custom_image",
            num_patches=2,
            patch_size=(4, 4),
            stochastic=False,
        )
        operator = PatchDropoutOperator(config, rngs=nnx.Rngs(0))

        image = jnp.ones((16, 16, 3))
        data = {"custom_image": image, "other": jnp.ones(10)}

        result, _, _ = operator.apply(data, {}, {})

        # Custom image should be transformed, other data unchanged
        assert "custom_image" in result
        assert "other" in result
        assert jnp.array_equal(result["other"], jnp.ones(10))

    def test_apply_missing_image_key(self):
        """Test operator handles missing field with KeyError."""
        config = PatchDropoutOperatorConfig(
            field_key="image",
            num_patches=2,
            stochastic=False,
        )
        operator = PatchDropoutOperator(config, rngs=nnx.Rngs(0))

        data = {"other": jnp.ones(10)}

        # Should raise KeyError from _extract_field
        with pytest.raises(KeyError):
            operator.apply(data, {}, {})

    def test_apply_target_key(self):
        """Test operator with target_key different from field_key."""
        config = PatchDropoutOperatorConfig(
            field_key="image",
            target_key="patched_image",
            num_patches=2,
            patch_size=(4, 4),
            stochastic=False,
        )
        operator = PatchDropoutOperator(config, rngs=nnx.Rngs(0))

        image = jnp.ones((16, 16, 3))
        data = {"image": image}

        result, _, _ = operator.apply(data, {}, {})

        # Original image unchanged, new field created
        assert "image" in result
        assert "patched_image" in result
        assert jnp.array_equal(result["image"], image)
        assert not jnp.array_equal(result["patched_image"], image)

    def test_invalid_image_shape(self):
        """Test operator rejects invalid image shapes."""
        config = PatchDropoutOperatorConfig(
            field_key="image",
            num_patches=2,
            stochastic=False,
        )
        operator = PatchDropoutOperator(config, rngs=nnx.Rngs(0))

        # 1D array (invalid)
        image = jnp.ones(32)
        data = {"image": image}

        with pytest.raises(ValueError, match="Expected 2D or 3D image"):
            operator.apply(data, {}, {})


class TestPatchDropoutOperatorStochasticMode:
    """Tests for stochastic random parameter generation."""

    def test_generate_random_params_shape(self):
        """Test random parameter generation produces correct shape."""
        config = PatchDropoutOperatorConfig(
            field_key="image",
            num_patches=4,
            patch_size=(8, 8),
            stochastic=True,
            stream_name="augment",
        )
        operator = PatchDropoutOperator(config, rngs=nnx.Rngs(42, augment=1))

        rng = jax.random.key(42)
        data_shapes = {"image": (3, 32, 32, 3)}  # Batch size 3

        random_params = operator.generate_random_params(rng, data_shapes)

        assert "patch_positions" in random_params
        # Shape: (batch_size, num_patches, 2) where last dim is (y, x)
        assert random_params["patch_positions"].shape == (3, 4, 2)

    def test_generate_random_params_values(self):
        """Test random parameters have valid values."""
        config = PatchDropoutOperatorConfig(
            field_key="image",
            num_patches=4,
            patch_size=(8, 8),
            stochastic=True,
            stream_name="augment",
        )
        operator = PatchDropoutOperator(config, rngs=nnx.Rngs(42, augment=1))

        rng = jax.random.key(42)
        data_shapes = {"image": (10, 32, 32, 3)}

        random_params = operator.generate_random_params(rng, data_shapes)

        positions = random_params["patch_positions"]
        # All positions should be within valid range
        # Max valid position is (image_size - patch_size)
        assert jnp.all(positions[:, :, 0] >= 0)  # y >= 0
        assert jnp.all(positions[:, :, 0] <= 32 - 8)  # y <= max_y
        assert jnp.all(positions[:, :, 1] >= 0)  # x >= 0
        assert jnp.all(positions[:, :, 1] <= 32 - 8)  # x <= max_x

    def test_stochastic_apply_batch_varies_between_samples(self):
        """Test stochastic mode produces different results for batch elements."""
        config = PatchDropoutOperatorConfig(
            field_key="image",
            num_patches=4,
            patch_size=(8, 8),
            drop_value=0.0,
            stochastic=True,
            stream_name="augment",
        )
        operator = PatchDropoutOperator(config, rngs=nnx.Rngs(42, augment=1))

        # Create batch with identical images
        images = jnp.ones((4, 32, 32, 3))
        elements = [Element(data={"image": img}, state={}, metadata={}) for img in images]
        batch = Batch(elements=elements)

        result_batch = operator.apply_batch(batch)

        # Due to randomness, patch positions should vary between samples
        result_images = result_batch.data.get_value()["image"]
        # Check if different samples have different dropout patterns
        # (high probability with stochastic mode)
        found_difference = False
        for i in range(len(result_images) - 1):
            if not jnp.array_equal(result_images[i], result_images[i + 1]):
                found_difference = True
                break
        assert found_difference, "Stochastic mode should produce varying patch positions"


class TestPatchDropoutOperatorJAXCompatibility:
    """Tests for JAX transformation compatibility (jit, vmap, grad)."""

    def test_jit_compatibility_deterministic(self):
        """Test JIT compilation with deterministic operator."""
        config = PatchDropoutOperatorConfig(
            field_key="image",
            num_patches=2,
            patch_size=(4, 4),
            stochastic=False,
        )
        operator = PatchDropoutOperator(config, rngs=nnx.Rngs(0))

        @nnx.jit
        def jit_apply(op, data):
            result, state, metadata = op.apply(data, {}, {})
            return result

        image = jnp.ones((16, 16, 3))
        data = {"image": image}

        result = jit_apply(operator, data)

        # Should have patches dropped
        assert result["image"].shape == image.shape
        assert jnp.any(result["image"] < image)

    def test_jit_compatibility_stochastic(self):
        """Test JIT compilation with stochastic operator."""
        config = PatchDropoutOperatorConfig(
            field_key="image",
            num_patches=2,
            patch_size=(4, 4),
            stochastic=True,
            stream_name="augment",
        )
        operator = PatchDropoutOperator(config, rngs=nnx.Rngs(42, augment=1))

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
        config = PatchDropoutOperatorConfig(
            field_key="image",
            num_patches=2,
            patch_size=(4, 4),
            stochastic=False,
        )
        operator = PatchDropoutOperator(config, rngs=nnx.Rngs(0))

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
        """Test gradient computation through patch dropout operator.

        Note: Patch dropout uses dynamic_update_slice which has well-defined
        gradients. We test that gradients can be computed.
        """
        config = PatchDropoutOperatorConfig(
            field_key="image",
            num_patches=0,  # Use 0 patches for meaningful gradients
            stochastic=False,
        )
        operator = PatchDropoutOperator(config, rngs=nnx.Rngs(0))

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

        # With 0 patches, gradients should be 2*image (from x^2)
        expected_grad = 2.0 * image
        assert jnp.allclose(grads, expected_grad)


class TestPatchDropoutOperatorCommonPatterns:
    """Tests for common patch dropout patterns."""

    def test_square_patch_pattern(self):
        """Test square patch pattern with integer patch_size."""
        config = PatchDropoutOperatorConfig(
            field_key="image",
            num_patches=4,
            patch_size=8,  # Int instead of tuple
            drop_value=0.0,
            stochastic=False,
        )
        operator = PatchDropoutOperator(config, rngs=nnx.Rngs(42))

        # Should be converted to tuple
        assert operator.config.patch_size == (8, 8)

        image = jnp.ones((32, 32, 3))
        data = {"image": image}

        result, _, _ = operator.apply(data, {}, {})

        # Should have patches dropped
        assert result["image"].shape == image.shape
        assert jnp.any(result["image"] == 0.0)

    def test_rectangular_patch_pattern(self):
        """Test rectangular patch pattern with tuple patch_size."""
        config = PatchDropoutOperatorConfig(
            field_key="image",
            num_patches=2,
            patch_size=(4, 8),
            drop_value=0.0,
            stochastic=False,
        )
        operator = PatchDropoutOperator(config, rngs=nnx.Rngs(42))

        image = jnp.ones((32, 32, 3))
        data = {"image": image}

        result, _, _ = operator.apply(data, {}, {})

        # Should have rectangular patches dropped
        assert result["image"].shape == image.shape
