"""Tests for ContrastOperator.

Tests cover:
- Configuration validation and initialization
- Basic contrast transformations
- Stochastic mode
- JAX transformation compatibility
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from datarax.operators.modality.image.contrast_operator import (
    ContrastOperator,
    ContrastOperatorConfig,
)


class TestContrastOperatorConfig:
    """Test ContrastOperatorConfig validation and initialization."""

    def test_basic_config_creation(self):
        """A stochastic config keeps the contrast range it is given."""
        config = ContrastOperatorConfig(
            field_key="image",
            contrast_range=(0.8, 1.2),
            stochastic=True,
            stream_name="augment",
        )

        assert config.field_key == "image"
        assert config.contrast_range == (0.8, 1.2)
        assert config.contrast_factor is None
        assert config.clip_range == (0.0, 1.0)  # Default value

    def test_default_parameters_follow_the_mode(self):
        """Deterministic mode defaults to factor 1.0, stochastic mode to the range (0.8, 1.2)."""
        deterministic = ContrastOperatorConfig(field_key="image")
        stochastic = ContrastOperatorConfig(
            field_key="image", stochastic=True, stream_name="augment"
        )

        assert (deterministic.contrast_factor, deterministic.contrast_range) == (1.0, None)
        assert (stochastic.contrast_factor, stochastic.contrast_range) == (None, (0.8, 1.2))

    def test_invalid_contrast_range(self):
        """Test validation of contrast_range parameter."""
        with pytest.raises(ValueError, match="contrast_range must be.*with min <= max"):
            ContrastOperatorConfig(
                field_key="image",
                contrast_range=(1.2, 0.8),  # Invalid: min > max
                stochastic=True,
                stream_name="augment",
            )


class TestContrastOperatorInitialization:
    """Test ContrastOperator initialization."""

    def test_deterministic_initialization(self):
        """Test initialization in deterministic mode."""
        config = ContrastOperatorConfig(
            field_key="image",
            stochastic=False,
        )
        rngs = nnx.Rngs(0)
        operator = ContrastOperator(config, rngs=rngs)
        assert operator.config.stochastic is False

    def test_stochastic_initialization(self):
        """Test initialization in stochastic mode."""
        config = ContrastOperatorConfig(
            field_key="image",
            stochastic=True,
            stream_name="augment",
        )
        rngs = nnx.Rngs(0, augment=1)
        operator = ContrastOperator(config, rngs=rngs)
        assert operator.config.stochastic is True


class TestContrastOperatorTransformations:
    """Test transformation functionality."""

    def test_contrast_adjustment(self):
        """Test contrast adjustment transformation."""
        config = ContrastOperatorConfig(
            field_key="image",
            contrast_factor=1.5,
            stochastic=False,
        )
        rngs = nnx.Rngs(0)
        operator = ContrastOperator(config, rngs=rngs)

        # Create sample data (2x2 to have spatial variance)
        val1, val2 = 0.4, 0.6
        image = jnp.array([[val1, val2], [val1, val2]])
        image = image[..., None]  # (2, 2, 1)
        data = {"image": image}
        state = {}
        metadata = {}

        # Apply transformation
        result, _, _ = operator.apply(data, state, metadata)

        # Verify contrast adjustment (around mean 0.5)
        # Expected: 0.5 + 1.5 * (val - 0.5)
        # 0.4 -> 0.5 + 1.5 * (-0.1) = 0.35
        # 0.6 -> 0.5 + 1.5 * (0.1) = 0.65
        expected = jnp.array([[0.35, 0.65], [0.35, 0.65]])[..., None]
        assert jnp.allclose(result["image"], expected, atol=1e-6)

    def test_uniform_image_is_returned_unchanged(self):
        """A uniform image has no contrast to adjust, so the operator returns it as it is."""
        config = ContrastOperatorConfig(field_key="image", contrast_factor=1.5, stochastic=False)
        operator = ContrastOperator(config, rngs=nnx.Rngs(0))
        image = jnp.full((4, 4, 1), 0.5)

        result, _, _ = operator.apply({"image": image}, {}, {})

        assert result["image"].shape == image.shape
        assert jnp.array_equal(result["image"], image)

    def test_clip_range_applied(self):
        """Test that clip_range is applied."""
        config = ContrastOperatorConfig(
            field_key="image",
            contrast_factor=10.0,  # Large factor to trigger clipping
            clip_range=(0.0, 1.0),
            stochastic=False,
        )
        rngs = nnx.Rngs(0)
        operator = ContrastOperator(config, rngs=rngs)

        # Image with variance
        image = jnp.array([[[0.0, 0.5, 1.0]]])
        data = {"image": image}

        result, _, _ = operator.apply(data, {}, {})

        assert jnp.all(result["image"] >= 0.0)
        assert jnp.all(result["image"] <= 1.0)


class TestContrastOperatorStochastic:
    """Test stochastic mode."""

    def test_each_record_draws_its_own_contrast_factor(self):
        """The batch path gives every record its own factor, and identical inputs diverge."""
        config = ContrastOperatorConfig(
            field_key="image",
            contrast_range=(0.5, 1.5),
            stochastic=True,
            stream_name="augment",
        )
        operator = ContrastOperator(config, rngs=nnx.Rngs(0, augment=1))

        batch_size = 10
        # A two-tone image, so a change of contrast is visible in the output
        image = jnp.zeros((32, 32, 3)).at[:16].set(1.0)
        data, _ = operator._vmap_apply({"image": jnp.stack([image] * batch_size)}, {})

        assert data["image"].shape == (batch_size, 32, 32, 3)
        assert not jnp.allclose(data["image"][0], data["image"][1])


class TestContrastOperatorJIT:
    """Test JAX JIT compatibility."""

    def test_jit_compatibility(self):
        config = ContrastOperatorConfig(
            field_key="image",
            contrast_factor=1.2,
            stochastic=False,
        )
        rngs = nnx.Rngs(0)
        operator = ContrastOperator(config, rngs=rngs)

        @nnx.jit
        def apply_jit(op, data):
            return op.apply(data, {}, {})

        data = {"image": jnp.ones((1, 4, 4, 3)) * 0.5}
        result = apply_jit(operator, data)
        assert result[0]["image"].shape == (1, 4, 4, 3)
