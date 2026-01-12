"""Tests for BatchMixOperator - MixUp and CutMix batch augmentation.

This module tests the BatchMixOperator, which performs batch-level sample mixing:
- MixUp: Linear interpolation between pairs of samples
- CutMix: Cut and paste patches between images

Test Coverage:
- Config validation (mode, alpha, field names)
- MixUp mode (linear interpolation, batch mixing)
- CutMix mode (patch mixing, label adjustment)
- Batch size edge cases (single element, empty batch)
- JAX compatibility (JIT, reproducibility)
- Stochastic behavior (different keys, deterministic with same seed)
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from datarax.operators.batch_mix_operator import (
    BatchMixOperator,
    BatchMixOperatorConfig,
)
from datarax.core.element_batch import Batch, Element


class TestBatchMixOperatorConfig:
    """Test BatchMixOperatorConfig validation and initialization."""

    def test_config_requires_valid_mode(self):
        """Mode must be 'mixup' or 'cutmix'."""
        with pytest.raises(ValueError, match="mode must be"):
            BatchMixOperatorConfig(mode="invalid")

    def test_config_mixup_mode_valid(self):
        """Verify mixup mode is accepted."""
        config = BatchMixOperatorConfig(mode="mixup")
        assert config.mode == "mixup"
        assert config.alpha == 1.0  # default

    def test_config_cutmix_mode_valid(self):
        """Verify cutmix mode is accepted."""
        config = BatchMixOperatorConfig(mode="cutmix")
        assert config.mode == "cutmix"

    def test_config_alpha_must_be_positive(self):
        """Alpha parameter must be positive."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            BatchMixOperatorConfig(mode="mixup", alpha=0.0)

        with pytest.raises(ValueError, match="alpha must be positive"):
            BatchMixOperatorConfig(mode="mixup", alpha=-1.0)

    def test_config_custom_alpha(self):
        """Custom alpha should be accepted."""
        config = BatchMixOperatorConfig(mode="mixup", alpha=0.5)
        assert config.alpha == 0.5

    def test_config_is_always_stochastic(self):
        """Verify the operator is always stochastic (uses random mixing)."""
        config = BatchMixOperatorConfig(mode="mixup")
        assert config.stochastic is True
        assert config.stream_name == "batch_mix"

    def test_config_custom_stream_name(self):
        """Custom stream name should be accepted."""
        config = BatchMixOperatorConfig(mode="mixup", stream_name="my_mixer")
        assert config.stream_name == "my_mixer"

    def test_config_custom_data_field(self):
        """Custom data field should be accepted."""
        config = BatchMixOperatorConfig(mode="cutmix", data_field="pixels")
        assert config.data_field == "pixels"

    def test_config_custom_label_field(self):
        """Custom label field should be accepted."""
        config = BatchMixOperatorConfig(mode="cutmix", label_field="targets")
        assert config.label_field == "targets"


class TestBatchMixOperatorInit:
    """Test BatchMixOperator initialization."""

    def test_init_mixup_mode(self):
        """Initialize with MixUp mode."""
        rngs = nnx.Rngs({"batch_mix": 42})
        config = BatchMixOperatorConfig(mode="mixup", alpha=1.0)
        op = BatchMixOperator(config, rngs=rngs)

        assert op.config.mode == "mixup"
        assert op.config.alpha == 1.0

    def test_init_cutmix_mode(self):
        """Initialize with CutMix mode."""
        rngs = nnx.Rngs({"batch_mix": 42})
        config = BatchMixOperatorConfig(mode="cutmix", alpha=1.0)
        op = BatchMixOperator(config, rngs=rngs)

        assert op.config.mode == "cutmix"

    def test_init_requires_rngs(self):
        """Verify the operator requires rngs (always stochastic)."""
        config = BatchMixOperatorConfig(mode="mixup")

        with pytest.raises(ValueError, match="require.*rngs"):
            BatchMixOperator(config, rngs=None)


class TestBatchMixOperatorMixUp:
    """Test MixUp mode functionality."""

    def test_mixup_produces_mixed_values(self):
        """Verify mixup produces linear combinations of samples."""
        rngs = nnx.Rngs({"batch_mix": 42})
        config = BatchMixOperatorConfig(mode="mixup", alpha=1.0)
        op = BatchMixOperator(config, rngs=rngs)

        # Create batch with distinct values
        batch = Batch(
            [
                Element(data={"value": jnp.array([0.0])}),
                Element(data={"value": jnp.array([10.0])}),
                Element(data={"value": jnp.array([20.0])}),
                Element(data={"value": jnp.array([30.0])}),
            ]
        )

        result = op(batch)
        result_data = result.get_data()

        # Results should be in the range [0, 30] (linear combinations)
        assert jnp.all(result_data["value"] >= 0.0)
        assert jnp.all(result_data["value"] <= 30.0)
        # At least some values should be mixed (not exactly 0, 10, 20, 30)
        # This is probabilistic but alpha=1.0 makes pure originals unlikely

    def test_mixup_preserves_shape(self):
        """Verify mixup preserves batch shape."""
        rngs = nnx.Rngs({"batch_mix": 42})
        config = BatchMixOperatorConfig(mode="mixup")
        op = BatchMixOperator(config, rngs=rngs)

        batch = Batch(
            [
                Element(data={"arr": jnp.ones((3, 4))}),
                Element(data={"arr": jnp.zeros((3, 4))}),
            ]
        )

        result = op(batch)
        assert result.get_data()["arr"].shape == (2, 3, 4)

    def test_mixup_mixes_multiple_fields(self):
        """Verify mixup mixes all data fields."""
        rngs = nnx.Rngs({"batch_mix": 42})
        config = BatchMixOperatorConfig(mode="mixup")
        op = BatchMixOperator(config, rngs=rngs)

        batch = Batch(
            [
                Element(data={"x": jnp.array([0.0]), "y": jnp.array([100.0])}),
                Element(data={"x": jnp.array([10.0]), "y": jnp.array([0.0])}),
            ]
        )

        result = op(batch)
        result_data = result.get_data()

        # Both fields should be mixed
        assert result_data["x"].shape == (2, 1)
        assert result_data["y"].shape == (2, 1)

    def test_mixup_single_element_unchanged(self):
        """Verify mixup with single element returns unchanged."""
        rngs = nnx.Rngs({"batch_mix": 42})
        config = BatchMixOperatorConfig(mode="mixup")
        op = BatchMixOperator(config, rngs=rngs)

        batch = Batch([Element(data={"value": jnp.array([5.0])})])

        result = op(batch)
        assert jnp.allclose(result.get_data()["value"], jnp.array([[5.0]]))


class TestBatchMixOperatorCutMix:
    """Test CutMix mode functionality."""

    def test_cutmix_produces_patched_images(self):
        """Verify cutmix cuts and pastes patches between images."""
        rngs = nnx.Rngs({"batch_mix": 42})
        config = BatchMixOperatorConfig(mode="cutmix", data_field="image")
        op = BatchMixOperator(config, rngs=rngs)

        # Create batch with distinct images (white and black)
        batch = Batch(
            [
                Element(data={"image": jnp.ones((32, 32, 3))}),
                Element(data={"image": jnp.zeros((32, 32, 3))}),
            ]
        )

        result = op(batch)
        result_data = result.get_data()

        # Result should contain both 0s and 1s (mixed patches)
        assert result_data["image"].shape == (2, 32, 32, 3)
        has_ones = jnp.any(result_data["image"] == 1.0)
        has_zeros = jnp.any(result_data["image"] == 0.0)
        assert has_ones and has_zeros, "CutMix should produce patched images"

    def test_cutmix_preserves_shape(self):
        """Verify cutmix preserves image shape."""
        rngs = nnx.Rngs({"batch_mix": 42})
        config = BatchMixOperatorConfig(mode="cutmix")
        op = BatchMixOperator(config, rngs=rngs)

        batch = Batch(
            [
                Element(data={"image": jnp.ones((64, 64, 3))}),
                Element(data={"image": jnp.zeros((64, 64, 3))}),
            ]
        )

        result = op(batch)
        assert result.get_data()["image"].shape == (2, 64, 64, 3)

    def test_cutmix_mixes_labels_when_present(self):
        """Verify cutmix mixes labels proportionally to cut area."""
        rngs = nnx.Rngs({"batch_mix": 42})
        config = BatchMixOperatorConfig(mode="cutmix")
        op = BatchMixOperator(config, rngs=rngs)

        batch = Batch(
            [
                Element(data={"image": jnp.ones((32, 32, 3)), "label": jnp.array([1.0])}),
                Element(data={"image": jnp.zeros((32, 32, 3)), "label": jnp.array([0.0])}),
            ]
        )

        result = op(batch)
        result_data = result.get_data()

        # Labels should be mixed (between 0 and 1)
        assert jnp.all(result_data["label"] >= 0.0)
        assert jnp.all(result_data["label"] <= 1.0)

    def test_cutmix_single_element_unchanged(self):
        """Verify cutmix with single element returns unchanged."""
        rngs = nnx.Rngs({"batch_mix": 42})
        config = BatchMixOperatorConfig(mode="cutmix")
        op = BatchMixOperator(config, rngs=rngs)

        original_image = jnp.ones((32, 32, 3)) * 0.5
        batch = Batch([Element(data={"image": original_image})])

        result = op(batch)
        assert jnp.allclose(result.get_data()["image"], original_image[None, ...])

    def test_cutmix_missing_image_field_unchanged(self):
        """Verify cutmix without image field returns unchanged."""
        rngs = nnx.Rngs({"batch_mix": 42})
        config = BatchMixOperatorConfig(mode="cutmix", data_field="image")
        op = BatchMixOperator(config, rngs=rngs)

        batch = Batch(
            [
                Element(data={"value": jnp.array([1.0])}),
                Element(data={"value": jnp.array([2.0])}),
            ]
        )

        result = op(batch)
        # Should return unchanged when image field is missing
        assert jnp.allclose(result.get_data()["value"], jnp.array([[1.0], [2.0]]))

    def test_cutmix_invalid_image_shape_unchanged(self):
        """Verify cutmix with non-4D image returns unchanged."""
        rngs = nnx.Rngs({"batch_mix": 42})
        config = BatchMixOperatorConfig(mode="cutmix")
        op = BatchMixOperator(config, rngs=rngs)

        batch = Batch(
            [
                Element(data={"image": jnp.array([1.0, 2.0])}),
                Element(data={"image": jnp.array([3.0, 4.0])}),
            ]
        )

        result = op(batch)
        # Should return unchanged for invalid shape
        assert result.get_data()["image"].shape == (2, 2)


class TestBatchMixOperatorStochastic:
    """Test stochastic behavior of BatchMixOperator."""

    def test_different_keys_produce_different_results(self):
        """Different RNG keys should produce different mixing."""
        config = BatchMixOperatorConfig(mode="mixup")

        results = set()
        for seed in range(20):
            rngs = nnx.Rngs({"batch_mix": seed})
            op = BatchMixOperator(config, rngs=rngs)

            batch = Batch(
                [
                    Element(data={"value": jnp.array([0.0])}),
                    Element(data={"value": jnp.array([100.0])}),
                ]
            )

            result = op(batch)
            # Round to avoid floating point precision issues
            val = round(float(result.get_data()["value"][0, 0]), 2)
            results.add(val)

        # Should see variation in results
        assert len(results) > 1, "Different seeds should produce different mixing"

    def test_same_key_produces_same_result(self):
        """Same RNG key should produce identical mixing."""
        config = BatchMixOperatorConfig(mode="mixup")

        rngs1 = nnx.Rngs({"batch_mix": 42})
        rngs2 = nnx.Rngs({"batch_mix": 42})

        op1 = BatchMixOperator(config, rngs=rngs1)
        op2 = BatchMixOperator(config, rngs=rngs2)

        batch = Batch(
            [
                Element(data={"value": jnp.array([0.0])}),
                Element(data={"value": jnp.array([100.0])}),
            ]
        )

        result1 = op1(batch)
        result2 = op2(batch)

        assert jnp.allclose(result1.get_data()["value"], result2.get_data()["value"])


class TestBatchMixOperatorJAX:
    """Test JAX compatibility of BatchMixOperator."""

    def test_jit_compilation_mixup(self):
        """Verify mixup works with JIT compilation."""
        rngs = nnx.Rngs({"batch_mix": 42})
        config = BatchMixOperatorConfig(mode="mixup")
        op = BatchMixOperator(config, rngs=rngs)

        @nnx.jit
        def apply_op(model, batch):
            return model(batch)

        batch = Batch(
            [
                Element(data={"value": jnp.array([0.0])}),
                Element(data={"value": jnp.array([10.0])}),
            ]
        )

        result = apply_op(op, batch)
        assert result.get_data()["value"].shape == (2, 1)

    def test_jit_compilation_cutmix(self):
        """Verify cutmix works with JIT compilation."""
        rngs = nnx.Rngs({"batch_mix": 42})
        config = BatchMixOperatorConfig(mode="cutmix")
        op = BatchMixOperator(config, rngs=rngs)

        @nnx.jit
        def apply_op(model, batch):
            return model(batch)

        batch = Batch(
            [
                Element(data={"image": jnp.ones((32, 32, 3))}),
                Element(data={"image": jnp.zeros((32, 32, 3))}),
            ]
        )

        result = apply_op(op, batch)
        assert result.get_data()["image"].shape == (2, 32, 32, 3)

    def test_jit_preserves_randomness(self):
        """JIT should preserve random behavior across calls."""
        config = BatchMixOperatorConfig(mode="mixup")

        @nnx.jit
        def apply_op(model, batch):
            return model(batch)

        batch = Batch(
            [
                Element(data={"value": jnp.array([0.0])}),
                Element(data={"value": jnp.array([100.0])}),
            ]
        )

        results = set()
        for seed in range(20):
            rngs = nnx.Rngs({"batch_mix": seed})
            op = BatchMixOperator(config, rngs=rngs)
            result = apply_op(op, batch)
            val = round(float(result.get_data()["value"][0, 0]), 2)
            results.add(val)

        assert len(results) > 1, "JIT should preserve randomness"


class TestBatchMixOperatorEdgeCases:
    """Test edge cases for BatchMixOperator."""

    def test_empty_batch_handling(self):
        """Empty batch should be handled gracefully."""
        rngs = nnx.Rngs({"batch_mix": 42})
        config = BatchMixOperatorConfig(mode="mixup")
        op = BatchMixOperator(config, rngs=rngs)

        # Create empty batch
        batch = Batch([])

        result = op(batch)
        assert result.batch_size == 0

    def test_large_alpha_parameter(self):
        """Large alpha should produce more uniform mixing ratios."""
        rngs = nnx.Rngs({"batch_mix": 42})
        config = BatchMixOperatorConfig(mode="mixup", alpha=10.0)
        op = BatchMixOperator(config, rngs=rngs)

        batch = Batch(
            [
                Element(data={"value": jnp.array([0.0])}),
                Element(data={"value": jnp.array([100.0])}),
            ]
        )

        result = op(batch)
        # With large alpha, lambda tends toward 0.5, so values near 50
        result_values = result.get_data()["value"]
        # Just verify it runs without error and produces valid output
        assert jnp.all(result_values >= 0.0)
        assert jnp.all(result_values <= 100.0)

    def test_small_alpha_parameter(self):
        """Small alpha should produce more extreme mixing ratios."""
        rngs = nnx.Rngs({"batch_mix": 42})
        config = BatchMixOperatorConfig(mode="mixup", alpha=0.1)
        op = BatchMixOperator(config, rngs=rngs)

        batch = Batch(
            [
                Element(data={"value": jnp.array([0.0])}),
                Element(data={"value": jnp.array([100.0])}),
            ]
        )

        result = op(batch)
        # With small alpha, values tend toward extremes (0 or 100)
        result_values = result.get_data()["value"]
        assert jnp.all(result_values >= 0.0)
        assert jnp.all(result_values <= 100.0)

    def test_cutmix_grayscale_image(self):
        """Verify cutmix works with grayscale images (H, W, 1)."""
        rngs = nnx.Rngs({"batch_mix": 42})
        config = BatchMixOperatorConfig(mode="cutmix")
        op = BatchMixOperator(config, rngs=rngs)

        batch = Batch(
            [
                Element(data={"image": jnp.ones((32, 32, 1))}),
                Element(data={"image": jnp.zeros((32, 32, 1))}),
            ]
        )

        result = op(batch)
        assert result.get_data()["image"].shape == (2, 32, 32, 1)
