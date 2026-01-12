"""Tests for ContrastOperator.

Tests cover:
- Configuration validation and initialization
- Basic contrast transformations
- Stochastic mode
- JAX transformation compatibility
"""

import jax
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
        """Test basic config creation with required parameters."""
        config = ContrastOperatorConfig(
            field_key="image",
            contrast_range=(0.8, 1.2),
        )

        assert config.field_key == "image"
        assert config.contrast_range == (0.8, 1.2)
        assert config.clip_range == (0.0, 1.0)  # Default value

    def test_invalid_contrast_range(self):
        """Test validation of contrast_range parameter."""
        with pytest.raises(ValueError, match="contrast_range must be.*with min <= max"):
            ContrastOperatorConfig(
                field_key="image",
                contrast_range=(1.2, 0.8),  # Invalid: min > max
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

    def test_generate_random_params(self):
        """Test random parameter generation."""
        config = ContrastOperatorConfig(
            field_key="image",
            contrast_range=(0.5, 1.5),
            stochastic=True,
            stream_name="augment",
        )
        rngs = nnx.Rngs(0, augment=1)
        operator = ContrastOperator(config, rngs=rngs)

        batch_size = 10
        data_shapes = {"image": (batch_size, 32, 32, 3)}
        rng = jax.random.PRNGKey(0)

        params = operator.generate_random_params(rng, data_shapes)
        assert params["contrast"].shape == (batch_size,)
        assert jnp.all(params["contrast"] >= 0.5)
        assert jnp.all(params["contrast"] <= 1.5)


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
