"""Tests for NoiseOperator - Unified noise augmentation operator.

This test suite covers all three noise modes:
- Gaussian: Additive Gaussian noise
- Salt & Pepper: Impulse noise (random pixels to min/max)
- Poisson: Shot noise (photon noise simulation)
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from datarax.operators.modality.image.noise_operator import (
    NoiseOperator,
    NoiseOperatorConfig,
)


class TestNoiseOperatorConfig:
    """Test suite for NoiseOperatorConfig validation."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = NoiseOperatorConfig(field_key="image")
        assert config.field_key == "image"
        assert config.mode == "gaussian"
        assert config.noise_std == 0.05
        assert config.noise_mean == 0.0
        assert config.salt_prob == 0.01
        assert config.pepper_prob == 0.01
        assert config.salt_value is None
        assert config.pepper_value is None
        assert config.lam_scale == 1.0
        assert config.clip_range == (0.0, 1.0)
        # stochastic defaults to False (inherited from ModalityOperatorConfig)

    def test_config_gaussian_mode(self):
        """Test Gaussian mode configuration."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="gaussian",
            noise_std=0.1,
            noise_mean=0.02,
            clip_range=(0.0, 1.0),
        )
        assert config.mode == "gaussian"
        assert config.noise_std == 0.1
        assert config.noise_mean == 0.02

    def test_config_salt_pepper_mode(self):
        """Test salt & pepper mode configuration."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="salt_pepper",
            salt_prob=0.02,
            pepper_prob=0.03,
            salt_value=255.0,
            pepper_value=0.0,
        )
        assert config.mode == "salt_pepper"
        assert config.salt_prob == 0.02
        assert config.pepper_prob == 0.03
        assert config.salt_value == 255.0
        assert config.pepper_value == 0.0

    def test_config_poisson_mode(self):
        """Test Poisson mode configuration."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="poisson",
            lam_scale=2.0,
            clip_range=(0.0, 1.0),
        )
        assert config.mode == "poisson"
        assert config.lam_scale == 2.0

    def test_config_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(
            ValueError, match="mode must be 'gaussian', 'salt_pepper', or 'poisson'"
        ):
            NoiseOperatorConfig(field_key="image", mode="invalid")

    def test_config_negative_noise_std(self):
        """Test that negative noise_std raises ValueError."""
        with pytest.raises(ValueError, match="noise_std must be non-negative"):
            NoiseOperatorConfig(
                field_key="image",
                mode="gaussian",
                noise_std=-0.1,
            )

    def test_config_invalid_salt_prob(self):
        """Test that invalid salt_prob raises ValueError."""
        with pytest.raises(ValueError, match="salt_prob must be in \\[0.0, 1.0\\]"):
            NoiseOperatorConfig(
                field_key="image",
                mode="salt_pepper",
                salt_prob=-0.1,
            )

        with pytest.raises(ValueError, match="salt_prob must be in \\[0.0, 1.0\\]"):
            NoiseOperatorConfig(
                field_key="image",
                mode="salt_pepper",
                salt_prob=1.5,
            )

    def test_config_invalid_pepper_prob(self):
        """Test that invalid pepper_prob raises ValueError."""
        with pytest.raises(ValueError, match="pepper_prob must be in \\[0.0, 1.0\\]"):
            NoiseOperatorConfig(
                field_key="image",
                mode="salt_pepper",
                pepper_prob=-0.1,
            )

        with pytest.raises(ValueError, match="pepper_prob must be in \\[0.0, 1.0\\]"):
            NoiseOperatorConfig(
                field_key="image",
                mode="salt_pepper",
                pepper_prob=1.5,
            )

    def test_config_prob_sum_exceeds_one(self):
        """Test that salt_prob + pepper_prob > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="Sum of salt_prob and pepper_prob cannot exceed 1.0"):
            NoiseOperatorConfig(
                field_key="image",
                mode="salt_pepper",
                salt_prob=0.6,
                pepper_prob=0.6,
            )

    def test_config_invalid_salt_value_type(self):
        """Test that invalid salt_value type raises TypeError."""
        with pytest.raises(TypeError, match="salt_value must be a number or None"):
            NoiseOperatorConfig(
                field_key="image",
                mode="salt_pepper",
                salt_value="invalid",
            )

    def test_config_invalid_pepper_value_type(self):
        """Test that invalid pepper_value type raises TypeError."""
        with pytest.raises(TypeError, match="pepper_value must be a number or None"):
            NoiseOperatorConfig(
                field_key="image",
                mode="salt_pepper",
                pepper_value="invalid",
            )

    def test_config_non_positive_lam_scale(self):
        """Test that non-positive lam_scale raises ValueError."""
        with pytest.raises(ValueError, match="lam_scale must be positive"):
            NoiseOperatorConfig(
                field_key="image",
                mode="poisson",
                lam_scale=0.0,
            )

        with pytest.raises(ValueError, match="lam_scale must be positive"):
            NoiseOperatorConfig(
                field_key="image",
                mode="poisson",
                lam_scale=-1.0,
            )


class TestNoiseOperatorInitialization:
    """Test suite for NoiseOperator initialization."""

    def test_init_gaussian_deterministic(self):
        """Test initialization with Gaussian mode (deterministic)."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="gaussian",
            noise_std=0.1,
            stochastic=False,
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))

        assert operator.config.mode == "gaussian"
        assert operator.config.noise_std == 0.1
        assert operator.config.stochastic is False

    def test_init_gaussian_stochastic(self):
        """Test initialization with Gaussian mode (stochastic)."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="gaussian",
            noise_std=0.1,
            stochastic=True,
            stream_name="augment",
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(augment=0))

        assert operator.config.stochastic is True
        assert hasattr(operator, "rngs")

    def test_init_salt_pepper_deterministic(self):
        """Test initialization with salt & pepper mode (deterministic)."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="salt_pepper",
            salt_prob=0.02,
            pepper_prob=0.02,
            stochastic=False,
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))

        assert operator.config.mode == "salt_pepper"
        assert operator.config.salt_prob == 0.02
        assert operator.config.stochastic is False

    def test_init_salt_pepper_stochastic(self):
        """Test initialization with salt & pepper mode (stochastic)."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="salt_pepper",
            salt_prob=0.02,
            pepper_prob=0.02,
            stochastic=True,
            stream_name="augment",
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(augment=0))

        assert operator.config.stochastic is True
        assert hasattr(operator, "rngs")

    def test_init_poisson_deterministic(self):
        """Test initialization with Poisson mode (deterministic)."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="poisson",
            lam_scale=1.0,
            stochastic=False,
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))

        assert operator.config.mode == "poisson"
        assert operator.config.lam_scale == 1.0
        assert operator.config.stochastic is False

    def test_init_poisson_stochastic(self):
        """Test initialization with Poisson mode (stochastic)."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="poisson",
            lam_scale=1.0,
            stochastic=True,
            stream_name="augment",
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(augment=0))

        assert operator.config.stochastic is True
        assert hasattr(operator, "rngs")


class TestNoiseOperatorGaussianTransformations:
    """Test suite for Gaussian noise transformations."""

    def test_gaussian_apply_single_element(self):
        """Test Gaussian noise application to single element."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="gaussian",
            noise_std=0.1,
            noise_mean=0.0,
            stochastic=False,
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))

        # Single image (H, W, C)
        data = {"image": jnp.ones((32, 32, 3)) * 0.5}
        state = {}
        metadata = {}

        result, new_state, new_metadata = operator.apply(data, state, metadata)

        assert result["image"].shape == (32, 32, 3)
        assert new_state == state
        assert new_metadata == metadata
        # Should be different due to noise (deterministic mode uses fixed seed)
        assert not jnp.allclose(result["image"], data["image"])
        # Should be within reasonable bounds
        assert jnp.all(result["image"] >= 0.0)
        assert jnp.all(result["image"] <= 1.0)

    def test_gaussian_apply_batch(self):
        """Test Gaussian noise application to batch."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="gaussian",
            noise_std=0.1,
            stochastic=True,
            stream_name="augment",
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(augment=42))

        # Create batch of elements
        from datarax.core.element_batch import Batch, Element

        images = jnp.ones((4, 32, 32, 3)) * 0.5
        elements = [Element(data={"image": img}, state={}, metadata={}) for img in images]
        batch = Batch(elements=elements)

        result_batch = operator.apply_batch(batch)

        result_images = result_batch.data.get_value()["image"]
        assert result_images.shape == (4, 32, 32, 3)
        assert not jnp.allclose(result_images, images)
        # Each image should have different noise
        assert not jnp.allclose(result_images[0], result_images[1])

    def test_gaussian_zero_std(self):
        """Test Gaussian noise with zero std (no noise)."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="gaussian",
            noise_std=0.0,
            stochastic=False,
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))

        data = {"image": jnp.ones((32, 32, 3)) * 0.5}
        state = {}
        metadata = {}

        result, _, _ = operator.apply(data, state, metadata)

        # Should be unchanged with zero noise
        assert jnp.allclose(result["image"], data["image"])

    def test_gaussian_with_mean_offset(self):
        """Test Gaussian noise with non-zero mean."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="gaussian",
            noise_std=0.01,
            noise_mean=0.1,
            stochastic=False,
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))

        data = {"image": jnp.ones((32, 32, 3)) * 0.5}
        state = {}
        metadata = {}

        result, _, _ = operator.apply(data, state, metadata)

        # Mean should be approximately increased by noise_mean
        assert jnp.mean(result["image"]) > jnp.mean(data["image"])

    def test_gaussian_clipping(self):
        """Test Gaussian noise with clipping enabled."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="gaussian",
            noise_std=2.0,  # Large std to test clipping
            clip_range=(0.0, 1.0),
            stochastic=False,
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))

        data = {"image": jnp.ones((32, 32, 3)) * 0.9}
        state = {}
        metadata = {}

        result, _, _ = operator.apply(data, state, metadata)

        # Values should be clipped to [0, 1]
        assert jnp.all(result["image"] >= 0.0)
        assert jnp.all(result["image"] <= 1.0)


class TestNoiseOperatorSaltPepperTransformations:
    """Test suite for Salt & Pepper noise transformations."""

    def test_salt_pepper_apply_single_element(self):
        """Test salt & pepper noise application to single element."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="salt_pepper",
            salt_prob=0.05,
            pepper_prob=0.05,
            stochastic=False,
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))

        data = {"image": jnp.ones((32, 32, 3)) * 0.5}
        state = {}
        metadata = {}

        result, new_state, new_metadata = operator.apply(data, state, metadata)

        assert result["image"].shape == (32, 32, 3)
        assert new_state == state
        assert new_metadata == metadata
        # Should have some pixels at extremes (0 or 1)
        assert not jnp.allclose(result["image"], data["image"])

    def test_salt_pepper_apply_batch(self):
        """Test salt & pepper noise application to batch."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="salt_pepper",
            salt_prob=0.05,
            pepper_prob=0.05,
            stochastic=True,
            stream_name="augment",
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(augment=42))

        # Create batch of elements
        from datarax.core.element_batch import Batch, Element

        images = jnp.ones((4, 32, 32, 3)) * 0.5
        elements = [Element(data={"image": img}, state={}, metadata={}) for img in images]
        batch = Batch(elements=elements)

        result_batch = operator.apply_batch(batch)

        result_images = result_batch.data.get_value()["image"]
        assert result_images.shape == (4, 32, 32, 3)
        assert not jnp.allclose(result_images, images)

    def test_salt_pepper_zero_noise(self):
        """Test salt & pepper with zero noise probabilities."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="salt_pepper",
            salt_prob=0.0,
            pepper_prob=0.0,
            stochastic=False,
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))

        data = {"image": jnp.ones((32, 32, 3)) * 0.5}
        state = {}
        metadata = {}

        result, _, _ = operator.apply(data, state, metadata)

        # Should be unchanged with zero noise
        assert jnp.allclose(result["image"], data["image"])

    def test_salt_pepper_only_salt(self):
        """Test salt & pepper with only salt noise."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="salt_pepper",
            salt_prob=0.2,
            pepper_prob=0.0,
            stochastic=False,
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))

        data = {"image": jnp.ones((32, 32, 3)) * 0.5}
        state = {}
        metadata = {}

        result, _, _ = operator.apply(data, state, metadata)

        # Should have some salt pixels (value 1.0)
        assert jnp.any(result["image"] == 1.0)

    def test_salt_pepper_only_pepper(self):
        """Test salt & pepper with only pepper noise."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="salt_pepper",
            salt_prob=0.0,
            pepper_prob=0.2,
            stochastic=False,
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))

        data = {"image": jnp.ones((32, 32, 3)) * 0.5}
        state = {}
        metadata = {}

        result, _, _ = operator.apply(data, state, metadata)

        # Should have some pepper pixels (value 0.0)
        assert jnp.any(result["image"] == 0.0)

    def test_salt_pepper_custom_values(self):
        """Test salt & pepper with custom salt/pepper values."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="salt_pepper",
            salt_prob=0.1,
            pepper_prob=0.1,
            salt_value=255.0,
            pepper_value=0.0,
            clip_range=None,  # Disable clipping to test custom values
            stochastic=False,
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))

        data = {"image": jnp.ones((32, 32, 3)) * 128.0}
        state = {}
        metadata = {}

        result, _, _ = operator.apply(data, state, metadata)

        # Should have pixels with custom salt/pepper values
        assert jnp.any(result["image"] == 255.0) or jnp.any(result["image"] == 0.0)

    def test_salt_pepper_auto_detect_range(self):
        """Test salt & pepper auto-detection of image range."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="salt_pepper",
            salt_prob=0.2,
            pepper_prob=0.0,
            salt_value=None,  # Auto-detect
            pepper_value=None,
            clip_range=None,
            stochastic=False,
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))

        # Test with [0, 255] range
        data_255 = {"image": jnp.ones((32, 32, 3)) * 128.0}
        result_255, _, _ = operator.apply(data_255, {}, {})
        # Auto-detect should use 255.0 for salt
        assert jnp.any(result_255["image"] == 255.0)

        # Test with [0, 1] range
        data_01 = {"image": jnp.ones((32, 32, 3)) * 0.5}
        result_01, _, _ = operator.apply(data_01, {}, {})
        # Auto-detect should use 1.0 for salt
        assert jnp.any(result_01["image"] == 1.0)


class TestNoiseOperatorPoissonTransformations:
    """Test suite for Poisson noise transformations."""

    def test_poisson_apply_single_element(self):
        """Test Poisson noise application to single element."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="poisson",
            lam_scale=1.0,
            stochastic=False,
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))

        data = {"image": jnp.ones((32, 32, 3)) * 0.5}
        state = {}
        metadata = {}

        result, new_state, new_metadata = operator.apply(data, state, metadata)

        assert result["image"].shape == (32, 32, 3)
        assert new_state == state
        assert new_metadata == metadata
        # Should be different due to Poisson noise
        assert not jnp.allclose(result["image"], data["image"])

    def test_poisson_apply_batch(self):
        """Test Poisson noise application to batch."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="poisson",
            lam_scale=1.0,
            stochastic=True,
            stream_name="augment",
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(augment=42))

        # Create batch of elements
        from datarax.core.element_batch import Batch, Element

        images = jnp.ones((4, 32, 32, 3)) * 0.5
        elements = [Element(data={"image": img}, state={}, metadata={}) for img in images]
        batch = Batch(elements=elements)

        result_batch = operator.apply_batch(batch)

        result_images = result_batch.data.get_value()["image"]
        assert result_images.shape == (4, 32, 32, 3)
        assert not jnp.allclose(result_images, images)

    def test_poisson_different_scales(self):
        """Test Poisson noise with different lambda scales."""
        # Low scale (less noise)
        config_low = NoiseOperatorConfig(
            field_key="image",
            mode="poisson",
            lam_scale=0.1,
            stochastic=False,
        )
        operator_low = NoiseOperator(config_low, rngs=nnx.Rngs(0))

        # High scale (more noise)
        config_high = NoiseOperatorConfig(
            field_key="image",
            mode="poisson",
            lam_scale=10.0,
            stochastic=False,
        )
        operator_high = NoiseOperator(config_high, rngs=nnx.Rngs(0))

        data = {"image": jnp.ones((32, 32, 3)) * 0.5}
        state = {}
        metadata = {}

        result_low, _, _ = operator_low.apply(data, state, metadata)
        result_high, _, _ = operator_high.apply(data, state, metadata)

        assert result_low["image"].shape == (32, 32, 3)
        assert result_high["image"].shape == (32, 32, 3)

    def test_poisson_01_range(self):
        """Test Poisson noise with [0, 1] range images."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="poisson",
            lam_scale=1.0,
            stochastic=False,
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))

        data = {"image": jnp.ones((32, 32, 3)) * 0.5}
        state = {}
        metadata = {}

        result, _, _ = operator.apply(data, state, metadata)

        assert result["image"].shape == (32, 32, 3)
        # Should be within reasonable bounds
        assert jnp.all(result["image"] >= 0.0)
        assert jnp.all(result["image"] <= 1.0)

    def test_poisson_255_range(self):
        """Test Poisson noise with [0, 255] range images."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="poisson",
            lam_scale=1.0,
            clip_range=(0.0, 255.0),
            stochastic=False,
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))

        data = {"image": jnp.ones((32, 32, 3)) * 128.0}
        state = {}
        metadata = {}

        result, _, _ = operator.apply(data, state, metadata)

        assert result["image"].shape == (32, 32, 3)
        # Should be within [0, 255] range after clipping
        assert jnp.all(result["image"] >= 0.0)
        assert jnp.all(result["image"] <= 255.0)

    def test_poisson_negative_values(self):
        """Test Poisson noise with negative input values."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="poisson",
            lam_scale=1.0,
            clip_range=None,
            stochastic=False,
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))

        # Poisson requires non-negative values, operator should clamp to 0
        data = {"image": jnp.ones((32, 32, 3)) * -0.5}
        state = {}
        metadata = {}

        result, _, _ = operator.apply(data, state, metadata)

        # Output should be non-negative (clamped internally)
        assert jnp.all(result["image"] >= 0.0)


class TestNoiseOperatorEdgeCases:
    """Test suite for edge cases."""

    def test_missing_field_key(self):
        """Test behavior when field_key is missing from data."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="gaussian",
            stochastic=False,
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))

        data = {"other_field": jnp.ones((32, 32, 3))}
        state = {}
        metadata = {}

        # Should raise KeyError
        with pytest.raises(KeyError):
            operator.apply(data, state, metadata)

    def test_different_image_shapes(self):
        """Test with different image shapes (grayscale, RGB, different sizes)."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="gaussian",
            noise_std=0.05,
            stochastic=False,
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))

        # Grayscale
        data_gray = {"image": jnp.ones((28, 28, 1)) * 0.5}
        result_gray, _, _ = operator.apply(data_gray, {}, {})
        assert result_gray["image"].shape == (28, 28, 1)

        # RGB
        data_rgb = {"image": jnp.ones((32, 32, 3)) * 0.5}
        result_rgb, _, _ = operator.apply(data_rgb, {}, {})
        assert result_rgb["image"].shape == (32, 32, 3)

        # Large image
        data_large = {"image": jnp.ones((256, 256, 3)) * 0.5}
        result_large, _, _ = operator.apply(data_large, {}, {})
        assert result_large["image"].shape == (256, 256, 3)

    def test_custom_field_key(self):
        """Test with custom field key."""
        config = NoiseOperatorConfig(
            field_key="custom_image",
            mode="gaussian",
            noise_std=0.1,
            stochastic=False,
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))

        data = {
            "custom_image": jnp.ones((32, 32, 3)) * 0.5,
            "other": jnp.ones((10,)),
        }
        state = {}
        metadata = {}

        result, _, _ = operator.apply(data, state, metadata)

        # Custom image should be transformed
        assert not jnp.allclose(result["custom_image"], data["custom_image"])
        # Other fields should be unchanged
        assert jnp.allclose(result["other"], data["other"])

    def test_nested_field_access(self):
        """Test with nested field access using dot notation."""
        config = NoiseOperatorConfig(
            field_key="data.image",
            mode="gaussian",
            noise_std=0.1,
            stochastic=False,
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))

        data = {"data": {"image": jnp.ones((32, 32, 3)) * 0.5}}
        state = {}
        metadata = {}

        result, _, _ = operator.apply(data, state, metadata)

        # Nested field should be transformed
        assert not jnp.allclose(result["data"]["image"], data["data"]["image"])

    def test_no_clipping(self):
        """Test with clip_range=None to disable clipping."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="gaussian",
            noise_std=2.0,
            clip_range=None,
            stochastic=False,
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))

        data = {"image": jnp.ones((32, 32, 3)) * 0.9}
        state = {}
        metadata = {}

        result, _, _ = operator.apply(data, state, metadata)

        # Values might exceed [0, 1] without clipping
        assert result["image"].shape == (32, 32, 3)


class TestNoiseOperatorStochasticMode:
    """Test suite for stochastic mode and random parameter generation."""

    def test_generate_random_params_gaussian(self):
        """Test random parameter generation for Gaussian mode."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="gaussian",
            noise_std=0.1,
            stochastic=True,
            stream_name="augment",
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))

        rng = jax.random.key(42)
        data_shapes = {"image": (4, 32, 32, 3)}

        random_params = operator.generate_random_params(rng, data_shapes)

        assert "noise" in random_params
        assert random_params["noise"].shape == (4, 32, 32, 3)

    def test_generate_random_params_salt_pepper(self):
        """Test random parameter generation for salt & pepper mode."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="salt_pepper",
            salt_prob=0.02,
            pepper_prob=0.02,
            stochastic=True,
            stream_name="augment",
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))

        rng = jax.random.key(42)
        data_shapes = {"image": (4, 32, 32, 3)}

        random_params = operator.generate_random_params(rng, data_shapes)

        assert "noise_mask" in random_params
        assert random_params["noise_mask"].shape == (4, 32, 32, 3)

    def test_generate_random_params_poisson(self):
        """Test random parameter generation for Poisson mode."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="poisson",
            lam_scale=1.0,
            stochastic=True,
            stream_name="augment",
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))

        rng = jax.random.key(42)
        data_shapes = {"image": (4, 32, 32, 3)}

        random_params = operator.generate_random_params(rng, data_shapes)

        assert "poisson_rngs" in random_params
        assert random_params["poisson_rngs"].shape == (4,)

    def test_generate_random_params_missing_field(self):
        """Test that missing field_key raises KeyError."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="gaussian",
            stochastic=True,
            stream_name="augment",
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))

        rng = jax.random.key(42)
        data_shapes = {"other": (4, 32, 32, 3)}

        with pytest.raises(KeyError, match="Field key 'image' not found in data_shapes"):
            operator.generate_random_params(rng, data_shapes)

    def test_stochastic_batch_different_noise(self):
        """Test that stochastic mode produces different noise for each batch element."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="gaussian",
            noise_std=0.1,
            stochastic=True,
            stream_name="augment",
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(augment=42))

        # Create batch of elements
        from datarax.core.element_batch import Batch, Element

        images = jnp.ones((4, 32, 32, 3)) * 0.5
        elements = [Element(data={"image": img}, state={}, metadata={}) for img in images]
        batch = Batch(elements=elements)

        result_batch = operator.apply_batch(batch)
        result_images = result_batch.data.value["image"]

        # Each batch element should have different noise
        assert not jnp.allclose(result_images[0], result_images[1])
        assert not jnp.allclose(result_images[1], result_images[2])


class TestNoiseOperatorJAXCompatibility:
    """Test suite for JAX transformation compatibility."""

    def test_jit_compatibility_gaussian(self):
        """Test JIT compilation with Gaussian noise."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="gaussian",
            noise_std=0.1,
            stochastic=False,
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))

        @nnx.jit
        def jitted_apply(op, data, state, metadata):
            return op.apply(data, state, metadata)

        data = {"image": jnp.ones((32, 32, 3)) * 0.5}
        result, _, _ = jitted_apply(operator, data, {}, {})

        assert result["image"].shape == (32, 32, 3)

    def test_jit_compatibility_salt_pepper(self):
        """Test JIT compilation with salt & pepper noise."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="salt_pepper",
            salt_prob=0.05,
            pepper_prob=0.05,
            stochastic=False,
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))

        @nnx.jit
        def jitted_apply(op, data, state, metadata):
            return op.apply(data, state, metadata)

        data = {"image": jnp.ones((32, 32, 3)) * 0.5}
        result, _, _ = jitted_apply(operator, data, {}, {})

        assert result["image"].shape == (32, 32, 3)

    def test_jit_compatibility_poisson(self):
        """Test JIT compilation with Poisson noise."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="poisson",
            lam_scale=1.0,
            stochastic=False,
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))

        @nnx.jit
        def jitted_apply(op, data, state, metadata):
            return op.apply(data, state, metadata)

        data = {"image": jnp.ones((32, 32, 3)) * 0.5}
        result, _, _ = jitted_apply(operator, data, {}, {})

        assert result["image"].shape == (32, 32, 3)

    def test_vmap_compatibility(self):
        """Test vmap compatibility through apply_batch."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="gaussian",
            noise_std=0.1,
            stochastic=True,
            stream_name="augment",
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(augment=42))

        # Create batch of elements
        from datarax.core.element_batch import Batch, Element

        images = jnp.ones((8, 32, 32, 3)) * 0.5
        elements = [Element(data={"image": img}, state={}, metadata={}) for img in images]
        batch = Batch(elements=elements)

        result_batch = operator.apply_batch(batch)
        result_images = result_batch.data.value["image"]

        assert result_images.shape == (8, 32, 32, 3)

    def test_grad_compatibility_gaussian(self):
        """Test gradient computation compatibility with Gaussian noise."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="gaussian",
            noise_std=0.1,
            stochastic=False,
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))

        def loss_fn(data):
            result, _, _ = operator.apply(data, {}, {})
            return jnp.sum(result["image"] ** 2)

        data = {"image": jnp.ones((32, 32, 3)) * 0.5}
        grads = jax.grad(loss_fn)(data)

        assert "image" in grads
        assert grads["image"].shape == (32, 32, 3)

    def test_grad_compatibility_salt_pepper(self):
        """Test gradient computation compatibility with salt & pepper noise."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="salt_pepper",
            salt_prob=0.05,
            pepper_prob=0.05,
            stochastic=False,
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))

        def loss_fn(data):
            result, _, _ = operator.apply(data, {}, {})
            return jnp.sum(result["image"] ** 2)

        data = {"image": jnp.ones((32, 32, 3)) * 0.5}
        grads = jax.grad(loss_fn)(data)

        assert "image" in grads
        assert grads["image"].shape == (32, 32, 3)

    def test_grad_compatibility_poisson(self):
        """Test gradient computation compatibility with Poisson noise."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="poisson",
            lam_scale=1.0,
            stochastic=False,
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))

        def loss_fn(data):
            result, _, _ = operator.apply(data, {}, {})
            return jnp.sum(result["image"] ** 2)

        data = {"image": jnp.ones((32, 32, 3)) * 0.5}
        grads = jax.grad(loss_fn)(data)

        assert "image" in grads
        assert grads["image"].shape == (32, 32, 3)


class TestNoiseOperatorCommonPatterns:
    """Tests for common noise operator patterns."""

    def test_gaussian_noise_pattern(self):
        """Test Gaussian noise pattern with clipping."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="gaussian",
            noise_std=0.05,
            noise_mean=0.0,
            clip_range=(0.0, 1.0),
            stochastic=False,
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))

        data = {"image": jnp.ones((32, 32, 3)) * 0.5}
        result, _, _ = operator.apply(data, {}, {})

        # Should produce reasonable output with noise applied
        assert result["image"].shape == (32, 32, 3)
        assert jnp.all(result["image"] >= 0.0)
        assert jnp.all(result["image"] <= 1.0)
        assert not jnp.allclose(result["image"], data["image"])

    def test_salt_pepper_noise_pattern(self):
        """Test salt-and-pepper noise pattern with auto-detected values."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="salt_pepper",
            salt_prob=0.01,
            pepper_prob=0.01,
            salt_value=None,  # Auto-detect
            pepper_value=None,
            stochastic=False,
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))

        data = {"image": jnp.ones((32, 32, 3)) * 0.5}
        result, _, _ = operator.apply(data, {}, {})

        # Should produce reasonable output with noise applied
        assert result["image"].shape == (32, 32, 3)
        assert not jnp.allclose(result["image"], data["image"])

    def test_poisson_noise_pattern(self):
        """Test Poisson noise pattern with clipping."""
        config = NoiseOperatorConfig(
            field_key="image",
            mode="poisson",
            lam_scale=1.0,
            clip_range=(0.0, 1.0),
            stochastic=False,
        )
        operator = NoiseOperator(config, rngs=nnx.Rngs(0))

        data = {"image": jnp.ones((32, 32, 3)) * 0.5}
        result, _, _ = operator.apply(data, {}, {})

        # Should produce reasonable output with noise applied
        assert result["image"].shape == (32, 32, 3)
        assert jnp.all(result["image"] >= 0.0)
        assert jnp.all(result["image"] <= 1.0)
        assert not jnp.allclose(result["image"], data["image"])
