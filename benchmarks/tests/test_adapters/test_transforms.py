"""Tests for shared adapter transform utilities.

TDD: Write tests first per Core Principle #2.
Verifies array-level transforms used by Grain, Datarax, PyTorch, and SPDL adapters.
"""

import numpy as np
import pytest

from benchmarks.adapters._utils import (
    apply_to_dict,
    cast_to_float32,
    color_jitter,
    gaussian_blur_np,
    gaussian_noise,
    normalize_uint8,
    random_brightness,
    random_grayscale,
    random_horizontal_flip,
    random_resized_crop,
    random_scale,
    random_solarize,
    STANDARD_TRANSFORMS,
)


class TestNormalizeUint8:
    """Tests for normalize_uint8 array-level transform."""

    def test_converts_uint8_to_float32(self):
        arr = np.array([0, 128, 255], dtype=np.uint8)
        result = normalize_uint8(arr)
        assert result.dtype == np.float32

    def test_scales_to_zero_one_range(self):
        arr = np.array([0, 255], dtype=np.uint8)
        result = normalize_uint8(arr)
        np.testing.assert_allclose(result, [0.0, 1.0])

    def test_passthrough_non_uint8(self):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        result = normalize_uint8(arr)
        np.testing.assert_array_equal(result, arr)
        assert result.dtype == np.float32

    def test_works_with_jax_arrays(self):
        jnp = pytest.importorskip("jax.numpy")
        arr = jnp.array([0, 128, 255], dtype=jnp.uint8)
        result = normalize_uint8(arr)
        assert result.dtype == jnp.float32
        np.testing.assert_allclose(result, [0.0, 128 / 255.0, 1.0], atol=1e-6)

    def test_jit_safe_with_tree_map(self):
        """Verify normalize_uint8 works inside jax.jit via tree.map — zero overhead."""
        jax = pytest.importorskip("jax")
        jnp = jax.numpy

        @jax.jit
        def apply_normalize(x):
            return jax.tree.map(normalize_uint8, x)

        data = {"image": jnp.ones((4, 8, 8, 3), dtype=jnp.uint8) * 128}
        result = apply_normalize(data)
        assert result["image"].dtype == jnp.float32
        np.testing.assert_allclose(result["image"][0, 0, 0, 0], 128 / 255.0, atol=1e-6)


class TestCastToFloat32:
    """Tests for cast_to_float32 array-level transform."""

    def test_casts_int32(self):
        arr = np.array([1, 2, 3], dtype=np.int32)
        result = cast_to_float32(arr)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_noop_on_float32(self):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        result = cast_to_float32(arr)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, arr)

    def test_works_with_jax_arrays(self):
        jnp = pytest.importorskip("jax.numpy")
        arr = jnp.array([1, 2, 3], dtype=jnp.int32)
        result = cast_to_float32(arr)
        assert result.dtype == jnp.float32

    def test_jit_safe_with_tree_map(self):
        """Verify cast_to_float32 works inside jax.jit via tree.map — zero overhead."""
        jax = pytest.importorskip("jax")
        jnp = jax.numpy

        @jax.jit
        def apply_cast(x):
            return jax.tree.map(cast_to_float32, x)

        data = {"tokens": jnp.array([1, 2, 3], dtype=jnp.int32)}
        result = apply_cast(data)
        assert result["tokens"].dtype == jnp.float32


class TestApplyToDict:
    """Tests for apply_to_dict dict-level wrapper."""

    def test_applies_fn_to_all_values(self):
        d = {"a": np.array([1, 2], dtype=np.int32), "b": np.array([3, 4], dtype=np.int32)}
        result = apply_to_dict(cast_to_float32, d)
        assert all(v.dtype == np.float32 for v in result.values())

    def test_preserves_keys(self):
        d = {"x": np.array([1]), "y": np.array([2])}
        result = apply_to_dict(cast_to_float32, d)
        assert set(result.keys()) == {"x", "y"}

    def test_returns_new_dict(self):
        d = {"a": np.array([1])}
        result = apply_to_dict(cast_to_float32, d)
        assert result is not d


# ---------------------------------------------------------------------------
# Stochastic transforms (used by PyTorch, Grain, SPDL adapters)
# ---------------------------------------------------------------------------


class TestGaussianNoise:
    """Tests for gaussian_noise transform."""

    def test_output_dtype_matches_input(self):
        arr = np.ones((8, 8), dtype=np.float32)
        result = gaussian_noise(arr)
        assert result.dtype == np.float32

    def test_output_shape_matches_input(self):
        arr = np.ones((4, 8, 8, 3), dtype=np.float32)
        result = gaussian_noise(arr)
        assert result.shape == arr.shape

    def test_adds_noise(self):
        arr = np.zeros((100, 100), dtype=np.float32)
        result = gaussian_noise(arr, std=1.0)
        assert np.std(result) > 0.5  # Should have substantial noise


class TestRandomBrightness:
    """Tests for random_brightness transform."""

    def test_output_dtype_preserved(self):
        arr = np.ones((8, 8), dtype=np.float32) * 0.5
        result = random_brightness(arr)
        assert result.dtype == np.float32

    def test_shifts_values(self):
        arr = np.ones((8, 8), dtype=np.float32) * 0.5
        result = random_brightness(arr, low=-0.5, high=0.5, rng=np.random.default_rng(42))
        # Result should differ from input (with overwhelming probability)
        assert not np.allclose(result, arr)

    def test_deterministic_with_rng(self):
        arr = np.ones((8, 8), dtype=np.float32) * 0.5
        first = random_brightness(arr, rng=np.random.default_rng(7))
        second = random_brightness(arr, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(first, second)


class TestRandomScale:
    """Tests for random_scale transform."""

    def test_output_dtype_preserved(self):
        arr = np.ones((8, 8), dtype=np.float32)
        result = random_scale(arr)
        assert result.dtype == np.float32

    def test_scales_values(self):
        arr = np.ones((8, 8), dtype=np.float32) * 10.0
        result = random_scale(arr, low=0.5, high=0.5)  # Deterministic scale = 0.5
        np.testing.assert_allclose(result, 5.0)


# ---------------------------------------------------------------------------
# Compute-heavy transforms (HCV-1 / HPC-1 scenarios)
# ---------------------------------------------------------------------------


class TestRandomResizedCrop:
    """Tests for random_resized_crop (CPU numpy version)."""

    def test_output_shape_is_target(self):
        arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = random_resized_crop(arr, target_h=224, target_w=224)
        assert result.shape == (224, 224, 3)

    def test_output_dtype_preserved(self):
        arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        result = random_resized_crop(arr)
        assert result.dtype == np.uint8

    def test_works_on_float32(self):
        arr = np.random.randn(64, 64, 3).astype(np.float32)
        result = random_resized_crop(arr, target_h=32, target_w=32)
        assert result.shape == (32, 32, 3)
        assert result.dtype == np.float32


class TestRandomHorizontalFlip:
    """Tests for random_horizontal_flip."""

    def test_both_outcomes_occur_at_half_probability(self):
        arr = np.arange(12).reshape(3, 4).astype(np.float32)
        rng = np.random.default_rng(0)
        outcomes = {random_horizontal_flip(arr, p=0.5, rng=rng).tobytes() for _ in range(32)}
        # Over 32 draws, both the flipped and unflipped variants appear.
        assert len(outcomes) == 2

    def test_forced_flip(self):
        arr = np.arange(12).reshape(3, 4).astype(np.float32)
        result = random_horizontal_flip(arr, p=1.0)  # Always flip
        expected = arr[:, ::-1]
        np.testing.assert_array_equal(result, expected)

    def test_never_flip(self):
        arr = np.arange(12).reshape(3, 4).astype(np.float32)
        result = random_horizontal_flip(arr, p=0.0)  # Never flip
        np.testing.assert_array_equal(result, arr)


class TestColorJitter:
    """Tests for color_jitter transform."""

    def test_output_dtype_preserved(self):
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = color_jitter(arr)
        assert result.dtype == np.uint8

    def test_output_shape_preserved(self):
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = color_jitter(arr)
        assert result.shape == arr.shape

    def test_output_clipped_to_valid_range(self):
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = color_jitter(arr)
        assert result.min() >= 0
        assert result.max() <= 255


class TestGaussianBlurNp:
    """Tests for gaussian_blur_np (scipy-based)."""

    def test_output_shape_preserved(self):
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = gaussian_blur_np(arr)
        assert result.shape == arr.shape

    def test_output_dtype_preserved(self):
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = gaussian_blur_np(arr)
        assert result.dtype == np.uint8

    def test_smooths_noise(self):
        """Blurring should reduce variance of noisy input."""
        arr = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        result = gaussian_blur_np(arr, sigma=2.0)
        assert np.var(result.astype(np.float32)) < np.var(arr.astype(np.float32))

    def test_2d_input(self):
        arr = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        result = gaussian_blur_np(arr)
        assert result.shape == (32, 32)


class TestRandomSolarize:
    """Tests for random_solarize transform."""

    def test_output_shape_preserved(self):
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = random_solarize(arr)
        assert result.shape == arr.shape

    def test_forced_solarize(self):
        """With seed that triggers solarize, pixels >= 128 should be inverted."""
        arr = np.array([0, 127, 128, 255], dtype=np.uint8)
        # default_rng(2).random() = 0.26 < 0.5, so solarize triggers.
        result = random_solarize(arr, threshold=128, rng=np.random.default_rng(2))
        # Pixels >= 128 inverted: 128->127, 255->0
        np.testing.assert_array_equal(result, [0, 127, 127, 0])

    def test_no_solarize_on_high_draw(self):
        arr = np.array([0, 127, 128, 255], dtype=np.uint8)
        # default_rng(0).random() = 0.64 >= 0.5, so the array passes through.
        result = random_solarize(arr, threshold=128, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(result, arr)


class TestRandomGrayscale:
    """Tests for random_grayscale transform."""

    def test_non_rgb_passthrough(self):
        """Non-3-channel arrays should pass through unchanged."""
        arr = np.random.randint(0, 255, (32, 32, 1), dtype=np.uint8)
        result = random_grayscale(arr)
        np.testing.assert_array_equal(result, arr)

    def test_2d_passthrough(self):
        arr = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        result = random_grayscale(arr)
        np.testing.assert_array_equal(result, arr)

    def test_forced_grayscale(self):
        """With p=1.0, RGB should become grayscale (replicated across channels)."""
        arr = np.array([[[100, 150, 200]]], dtype=np.uint8)  # 1x1x3
        result = random_grayscale(arr, p=1.0)
        # All channels should be equal (the mean)
        assert result[0, 0, 0] == result[0, 0, 1] == result[0, 0, 2]

    def test_output_shape_preserved(self):
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = random_grayscale(arr, p=1.0)
        assert result.shape == arr.shape


class TestGlobalRngIsolation:
    """Stochastic transforms must not consume or mutate numpy's global RNG."""

    def test_transforms_leave_global_rng_untouched(self):
        arr = np.ones((8, 8, 3), dtype=np.uint8)
        np.random.seed(123)
        expected_next_draw = np.random.random()
        np.random.seed(123)
        gaussian_noise(arr)
        random_brightness(arr)
        random_scale(arr)
        random_horizontal_flip(arr)
        random_resized_crop(arr, target_h=4, target_w=4)
        color_jitter(arr)
        random_solarize(arr)
        random_grayscale(arr)
        assert np.random.random() == expected_next_draw


class TestRegistryContracts:
    """Every registry transform preserves its declared shape/dtype contract."""

    # name -> (expected shape for (16, 16, 3) uint8 input, expected dtype)
    _CONTRACTS = {
        "Normalize": ((16, 16, 3), np.float32),
        "CastToFloat32": ((16, 16, 3), np.float32),
        "GaussianNoise": ((16, 16, 3), np.uint8),
        "RandomBrightness": ((16, 16, 3), np.uint8),
        "RandomScale": ((16, 16, 3), np.uint8),
        "RandomResizedCrop": ((224, 224, 3), np.uint8),
        "RandomHorizontalFlip": ((16, 16, 3), np.uint8),
        "ColorJitter": ((16, 16, 3), np.uint8),
        "GaussianBlur": ((16, 16, 3), np.uint8),
        "RandomSolarize": ((16, 16, 3), np.uint8),
        "RandomGrayscale": ((16, 16, 3), np.uint8),
    }

    def test_registry_fully_covered(self):
        """Every image transform in the registry has a contract entry."""
        image_transforms = set(STANDARD_TRANSFORMS) - {
            "LogTransform",
            "CreateAttentionMask",
            "CausalMaskGeneration",
        }
        assert image_transforms == set(self._CONTRACTS)

    @pytest.mark.parametrize("name", sorted(_CONTRACTS))
    def test_image_transform_contract(self, name: str):
        """Image transforms keep the declared output shape and dtype."""
        expected_shape, expected_dtype = self._CONTRACTS[name]
        arr = np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)
        result = STANDARD_TRANSFORMS[name](arr)
        assert result.shape == expected_shape
        assert result.dtype == expected_dtype

    def test_normalize_range(self):
        """Normalize maps uint8 into [0, 1]."""
        arr = np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)
        result = STANDARD_TRANSFORMS["Normalize"](arr)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_token_transform_contracts(self):
        """NLP transforms produce masks of the declared shape and dtype."""
        tokens = np.array([5, 9, 0, 0], dtype=np.int32)
        attention = STANDARD_TRANSFORMS["CreateAttentionMask"](tokens)
        np.testing.assert_array_equal(attention, [1.0, 1.0, 0.0, 0.0])
        causal = STANDARD_TRANSFORMS["CausalMaskGeneration"](tokens)
        assert causal.shape == (4, 4)
        np.testing.assert_array_equal(causal, np.tril(np.ones((4, 4), dtype=np.float32)))

    def test_log_transform_contract(self):
        """LogTransform preserves shape/dtype and is non-negative."""
        arr = np.random.default_rng(0).standard_normal((32, 13)).astype(np.float32)
        result = STANDARD_TRANSFORMS["LogTransform"](arr)
        assert result.shape == arr.shape
        assert result.dtype == np.float32
        assert result.min() >= 0.0
