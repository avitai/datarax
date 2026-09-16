"""Tests for BrightnessOperator.

This test suite validates the BrightnessOperator implementation which provides
brightness adjustment for image data. Tests cover configuration validation,
deterministic and stochastic modes, batch processing, and JAX compatibility.

Example Usage:
    config = BrightnessOperatorConfig(
        field_key="image",
        brightness_range=(-0.2, 0.2),
        stochastic=True,
        stream_name="augment",
    )
    operator = BrightnessOperator(config, rngs=nnx.Rngs(0, augment=1))
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from datarax.core.element_batch import Batch, Element
from datarax.operators.modality.image.brightness_operator import (
    BrightnessOperator,
    BrightnessOperatorConfig,
)


class TestBrightnessOperatorConfig:
    """Tests for BrightnessOperatorConfig validation."""

    def test_deterministic_config_defaults_to_no_change(self):
        """A deterministic config without a delta adds 0.0 and carries no range."""
        config = BrightnessOperatorConfig(field_key="image")
        assert config.field_key == "image"
        assert config.brightness_delta == 0.0
        assert config.brightness_range is None
        assert config.clip_range == (0.0, 1.0)  # Default for images

    def test_stochastic_config_defaults_to_the_documented_range(self):
        """A stochastic config without a range draws deltas from (-0.2, 0.2)."""
        config = BrightnessOperatorConfig(field_key="image", stochastic=True, stream_name="augment")
        assert config.brightness_range == (-0.2, 0.2)
        assert config.brightness_delta is None

    def test_custom_params_config(self):
        """Test configuration with custom parameters in each mode."""
        stochastic = BrightnessOperatorConfig(
            field_key="custom_image",
            brightness_range=(-0.5, 0.5),
            clip_range=(0.0, 255.0),
            stochastic=True,
            stream_name="augment",
        )
        deterministic = BrightnessOperatorConfig(
            field_key="custom_image", brightness_delta=0.1, clip_range=(0.0, 255.0)
        )
        assert stochastic.field_key == "custom_image"
        assert stochastic.brightness_range == (-0.5, 0.5)
        assert stochastic.clip_range == (0.0, 255.0)
        assert deterministic.brightness_delta == 0.1
        assert deterministic.clip_range == (0.0, 255.0)

    def test_invalid_brightness_range(self):
        """Test validation of invalid brightness_range."""
        with pytest.raises(ValueError, match="brightness_range must be a tuple"):
            BrightnessOperatorConfig(
                field_key="image",
                brightness_range=(-0.2,),  # Wrong length  # type: ignore[reportArgumentType]
                stochastic=True,
                stream_name="augment",
            )

        with pytest.raises(ValueError, match="min <= max"):
            BrightnessOperatorConfig(
                field_key="image",
                brightness_range=(0.2, -0.2),  # min > max
                stochastic=True,
                stream_name="augment",
            )


class TestBrightnessOperatorInitialization:
    """Tests for BrightnessOperator initialization modes."""

    def test_init_deterministic_mode(self):
        """Test initialization in deterministic mode."""
        config = BrightnessOperatorConfig(
            field_key="image",
            brightness_delta=0.1,
            stochastic=False,
        )
        operator = BrightnessOperator(config, rngs=nnx.Rngs(0))

        assert operator.config.brightness_delta == 0.1
        assert operator.config.stochastic is False

    def test_init_stochastic_mode(self):
        """Test initialization in stochastic mode."""
        config = BrightnessOperatorConfig(
            field_key="image",
            brightness_range=(-0.3, 0.3),
            stochastic=True,
            stream_name="augment",
        )
        operator = BrightnessOperator(config, rngs=nnx.Rngs(0, augment=1))

        assert operator.config.brightness_range == (-0.3, 0.3)
        assert operator.config.stochastic is True
        assert operator.config.stream_name == "augment"

    def test_init_stochastic_missing_stream_name(self):
        """Test that stochastic mode requires stream_name."""
        with pytest.raises(ValueError, match="stream_name"):
            BrightnessOperatorConfig(
                field_key="image",
                brightness_range=(-0.2, 0.2),
                stochastic=True,
                # Missing stream_name
            )


class TestBrightnessOperatorTransformations:
    """Tests for basic brightness transformations."""

    def test_deterministic_apply_single_element(self):
        """Test deterministic brightness on single element."""
        config = BrightnessOperatorConfig(
            field_key="image",
            brightness_delta=0.1,
            stochastic=False,
        )
        operator = BrightnessOperator(config, rngs=nnx.Rngs(0))

        # Create single element
        image = jnp.ones((32, 32, 3)) * 0.5
        data = {"image": image}

        result, state, metadata = operator.apply(data, {}, {})

        # Should be brightened by exactly 0.1
        assert jnp.allclose(result["image"], image + 0.1)

    def test_deterministic_apply_batch(self):
        """Test deterministic brightness on batch."""
        config = BrightnessOperatorConfig(
            field_key="image",
            brightness_delta=0.1,
            stochastic=False,
        )
        operator = BrightnessOperator(config, rngs=nnx.Rngs(0))

        # Create batch of elements
        images = jnp.ones((4, 32, 32, 3)) * 0.5
        elements = [Element(data={"image": img}, state={}) for img in images]
        batch = Batch(elements=elements)

        result_batch = operator.apply_batch(batch)

        # All should be brightened by exactly 0.1
        # Access stacked data directly
        result_images = result_batch.data.get_value()["image"]
        expected_images = images + 0.1
        assert jnp.allclose(result_images, expected_images)

    def test_stochastic_apply_with_clipping(self):
        """Test stochastic brightness with clipping enabled."""
        config = BrightnessOperatorConfig(
            field_key="image",
            brightness_range=(-2.0, 2.0),  # Large range to test clipping
            stochastic=True,
            stream_name="augment",
            clip_range=(0.0, 1.0),
        )
        operator = BrightnessOperator(config, rngs=nnx.Rngs(42, augment=1))

        # Create batch with values that will exceed [0,1] after adjustment
        images = jnp.ones((2, 16, 16, 3)) * 0.9
        elements = [Element(data={"image": img}, state={}) for img in images]
        batch = Batch(elements=elements)

        result_batch = operator.apply_batch(batch)

        # Values should be clipped to [0, 1]
        result_images = result_batch.data.get_value()["image"]
        assert jnp.all(result_images >= 0.0)
        assert jnp.all(result_images <= 1.0)

    def test_apply_without_clipping_returns_the_raw_adjustment(self):
        """``clip_range=None`` means no clipping: a delta past 1.0 comes back past 1.0."""
        config = BrightnessOperatorConfig(
            field_key="image",
            brightness_delta=0.5,
            stochastic=False,
            clip_range=None,
        )
        operator = BrightnessOperator(config, rngs=nnx.Rngs(0))

        images = jnp.ones((2, 16, 16, 3)) * 0.8
        elements = [Element(data={"image": img}, state={}) for img in images]
        batch = Batch(elements=elements)

        result_batch = operator.apply_batch(batch)

        result_images = result_batch.data.get_value()["image"]
        assert result_images.shape == (2, 16, 16, 3)
        assert jnp.allclose(result_images, 1.3)


class TestBrightnessOperatorEdgeCases:
    """Tests for edge cases and error handling."""

    def test_apply_zero_brightness_delta(self):
        """Test that zero brightness_delta produces no change."""
        config = BrightnessOperatorConfig(
            field_key="image",
            brightness_delta=0.0,
            stochastic=False,
        )
        operator = BrightnessOperator(config, rngs=nnx.Rngs(0))

        image = jnp.ones((16, 16, 3)) * 0.5
        data = {"image": image}

        result, _, _ = operator.apply(data, {}, {})

        # Should be unchanged
        assert jnp.allclose(result["image"], image)

    def test_apply_custom_image_key(self):
        """Test operator with custom field key."""
        config = BrightnessOperatorConfig(
            field_key="custom_image",
            brightness_delta=0.1,
            stochastic=False,
        )
        operator = BrightnessOperator(config, rngs=nnx.Rngs(0))

        image = jnp.ones((16, 16, 3)) * 0.5
        data = {"custom_image": image, "other": jnp.zeros(10)}

        result, _, _ = operator.apply(data, {}, {})

        # Custom image should be transformed, other data unchanged
        assert jnp.allclose(result["custom_image"], image + 0.1)
        assert jnp.allclose(result["other"], jnp.zeros(10))

    def test_apply_missing_image_key(self):
        """Test operator handles missing field gracefully."""
        config = BrightnessOperatorConfig(
            field_key="image",
            brightness_delta=0.1,
            stochastic=False,
        )
        operator = BrightnessOperator(config, rngs=nnx.Rngs(0))

        data = {"other": jnp.ones(10)}

        # Should raise KeyError from _extract_field
        with pytest.raises(KeyError):
            operator.apply(data, {}, {})

    def test_apply_target_key(self):
        """Test operator with target_key different from field_key."""
        config = BrightnessOperatorConfig(
            field_key="image",
            target_key="brightened_image",
            brightness_delta=0.1,
            stochastic=False,
        )
        operator = BrightnessOperator(config, rngs=nnx.Rngs(0))

        image = jnp.ones((16, 16, 3)) * 0.5
        data = {"image": image}

        result, _, _ = operator.apply(data, {}, {})

        # Original image unchanged, new field created
        assert "image" in result
        assert "brightened_image" in result
        assert jnp.allclose(result["image"], image)
        assert jnp.allclose(result["brightened_image"], image + 0.1)


class TestBrightnessOperatorStochasticMode:
    """Tests for stochastic random parameter generation."""

    @staticmethod
    def _deltas(brightness_range: tuple[float, float], batch_size: int) -> jnp.ndarray:
        """Recover each record's drawn delta from a mid-grey batch, one delta per record."""
        config = BrightnessOperatorConfig(
            field_key="image",
            brightness_range=brightness_range,
            stochastic=True,
            stream_name="augment",
            clip_range=None,
        )
        operator = BrightnessOperator(config, rngs=nnx.Rngs(42, augment=1))

        batch = {"image": jnp.ones((batch_size, 8, 8, 3)) * 0.5}
        data, _ = operator._vmap_apply(batch, {})
        return jnp.mean(data["image"], axis=(1, 2, 3)) - 0.5

    def test_one_delta_is_drawn_per_record(self):
        """The batch path draws one brightness delta for each record."""
        deltas = self._deltas((-0.1, 0.1), 4)

        assert deltas.shape == (4,)
        assert not jnp.allclose(deltas[0], deltas[1])

    def test_drawn_deltas_are_within_the_configured_range(self):
        """Every record's delta falls inside brightness_range."""
        deltas = self._deltas((-0.3, 0.5), 100)

        assert jnp.all(deltas >= -0.3 - 1e-5)
        assert jnp.all(deltas <= 0.5 + 1e-5)

    def test_stochastic_produces_different_results(self):
        """Test stochastic mode produces different results per element."""
        config = BrightnessOperatorConfig(
            field_key="image",
            brightness_range=(-0.2, 0.2),
            stochastic=True,
            stream_name="augment",
        )
        operator = BrightnessOperator(config, rngs=nnx.Rngs(42, augment=1))

        # Create batch with identical images
        images = jnp.ones((4, 16, 16, 3)) * 0.5
        elements = [Element(data={"image": img}, state={}) for img in images]
        batch = Batch(elements=elements)

        result_batch = operator.apply_batch(batch)

        # Results should differ between elements (high probability with range)
        result_images = result_batch.data.get_value()["image"]
        # At least one pair should be different
        found_difference = False
        for i in range(len(result_images) - 1):
            if not jnp.allclose(result_images[i], result_images[i + 1]):
                found_difference = True
                break
        assert found_difference, "Stochastic mode should produce varying results"


class TestBrightnessOperatorJAXCompatibility:
    """Tests for JAX transformation compatibility."""

    def test_jit_compatibility_deterministic(self):
        """Test JIT compilation with deterministic operator."""
        config = BrightnessOperatorConfig(
            field_key="image",
            brightness_delta=0.1,
            stochastic=False,
        )
        operator = BrightnessOperator(config, rngs=nnx.Rngs(0))

        @nnx.jit
        def jit_apply(op, data):
            result, state, metadata = op.apply(data, {}, {})
            return result

        image = jnp.ones((16, 16, 3)) * 0.5
        data = {"image": image}

        result = jit_apply(operator, data)
        assert jnp.allclose(result["image"], image + 0.1)

    def test_jit_compatibility_stochastic(self):
        """Test JIT compilation with stochastic operator."""
        config = BrightnessOperatorConfig(
            field_key="image",
            brightness_range=(-0.2, 0.2),
            stochastic=True,
            stream_name="augment",
        )
        operator = BrightnessOperator(config, rngs=nnx.Rngs(42, augment=1))

        @nnx.jit
        def jit_apply_batch(op, batch):
            return op.apply_batch(batch)

        images = jnp.ones((2, 16, 16, 3)) * 0.5
        elements = [Element(data={"image": img}, state={}) for img in images]
        batch = Batch(elements=elements)

        result_batch = jit_apply_batch(operator, batch)
        assert result_batch.batch_size == 2

    def test_vmap_compatibility(self):
        """Test vmap compatibility via apply_batch."""
        config = BrightnessOperatorConfig(
            field_key="image",
            brightness_delta=0.1,
            stochastic=False,
        )
        operator = BrightnessOperator(config, rngs=nnx.Rngs(0))

        # apply_batch uses vmap internally
        images = jnp.ones((4, 32, 32, 3)) * 0.5
        elements = [Element(data={"image": img}, state={}) for img in images]
        batch = Batch(elements=elements)

        result_batch = operator.apply_batch(batch)

        # All elements should be transformed correctly
        assert result_batch.batch_size == 4
        result_images = result_batch.data.get_value()["image"]
        assert result_images.shape == (4, 32, 32, 3)


class TestBrightnessOperatorCommonPatterns:
    """Tests for common brightness adjustment patterns."""

    def test_symmetric_range_pattern(self):
        """Test symmetric brightness range pattern (e.g., ±0.2)."""
        max_delta = 0.2
        config = BrightnessOperatorConfig(
            field_key="image",
            brightness_range=(-max_delta, max_delta),
            stochastic=True,
            stream_name="augment",
        )
        config = BrightnessOperatorConfig(
            field_key="image",
            brightness_range=(-max_delta, max_delta),
            stochastic=True,
            stream_name="augment",
            clip_range=None,
        )
        operator = BrightnessOperator(config, rngs=nnx.Rngs(42, augment=1))

        # Many records, to check the distribution the draws span
        batch = {"image": jnp.ones((1000, 4, 4, 3)) * 0.5}
        data, _ = operator._vmap_apply(batch, {})
        brightness_values = jnp.mean(data["image"], axis=(1, 2, 3)) - 0.5

        # Mean should be near 0, values should span the range
        assert jnp.abs(jnp.mean(brightness_values)) < 0.05  # Near zero mean
        assert jnp.min(brightness_values) < -0.1  # Uses negative range
        assert jnp.max(brightness_values) > 0.1  # Uses positive range

    def test_clip_behavior(self):
        """Test clipping behavior with clip_range enabled."""
        config = BrightnessOperatorConfig(
            field_key="image",
            brightness_delta=2.0,  # Large delta to force clipping
            stochastic=False,
            clip_range=(0.0, 1.0),
        )
        operator = BrightnessOperator(config, rngs=nnx.Rngs(0))

        image = jnp.ones((16, 16, 3)) * 0.9
        data = {"image": image}

        result, _, _ = operator.apply(data, {}, {})

        # Should be clipped to [0, 1]
        assert jnp.all(result["image"] >= 0.0)
        assert jnp.all(result["image"] <= 1.0)
        assert jnp.allclose(result["image"], 1.0)  # Clipped to max
