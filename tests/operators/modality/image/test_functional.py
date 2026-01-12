"""Tests for JAX-native image manipulation operations in functional module."""

import jax
import jax.numpy as jnp

from datarax.operators.modality.image import functional


class TestFunctionalOps:
    """Test individual functional operations."""

    def test_adjust_brightness(self):
        """Test multiplicative brightness adjustment."""
        image = jnp.ones((4, 4, 3)) * 0.5
        result = functional.adjust_brightness(image, 1.2)
        expected = jnp.ones((4, 4, 3)) * 0.6
        assert jnp.allclose(result, expected)

    def test_adjust_brightness_delta(self):
        """Test additive brightness adjustment."""
        image = jnp.ones((4, 4, 3)) * 0.5
        result = functional.adjust_brightness_delta(image, 0.1)
        expected = jnp.ones((4, 4, 3)) * 0.6
        assert jnp.allclose(result, expected)

    def test_adjust_contrast(self):
        """Test contrast adjustment."""
        # Use 2x2 image so spatial mean makes sense
        image = jnp.array([[0.4, 0.6], [0.4, 0.6]])
        image = image[..., None]  # (2, 2, 1)

        # Mean is 0.5
        # Diff: [-0.1, 0.1]
        # Factor 1.2 -> Diff: [-0.12, 0.12]
        # Expected: [0.38, 0.62]

        result = functional.adjust_contrast(image, 1.2)
        expected = jnp.array([[0.38, 0.62], [0.38, 0.62]])[..., None]
        assert jnp.allclose(result, expected, atol=1e-6)

    def test_rotate(self):
        """Test bilinear rotation."""
        image = jnp.zeros((10, 10, 1))
        # Draw a line
        image = image.at[5, :].set(1.0)

        # Simpler check: result should have same shape and reasonable values
        result = functional.rotate(image, jnp.pi / 2.0)
        assert result.shape == image.shape
        assert jnp.all(result >= 0.0) and jnp.all(result <= 1.0)

        # Let's test a simpler case: fill value
        result = functional.rotate(image, 0.0, fill_value=0.5)
        assert jnp.allclose(result, image)  # No rotation


class TestJITCompatibility:
    """Explicit JIT compatibility tests for all functional operations."""

    @staticmethod
    def _check_jit(func, *args, **kwargs):
        """Helper to verify function works under jit."""
        jitted_func = jax.jit(func)
        # Should execute without error
        return jitted_func(*args, **kwargs)

    def test_jit_resize_image(self):
        image = jnp.ones((32, 32, 3))
        output_size = (16, 16)
        # Test needs static argnum for string method?
        # jax.image.resize allows method to be string.
        # But we need to check if our wrapper handles it.
        # functional.resize_image has 'method' as string.
        # Strings typically need to be static in JIT.
        # Let's wrap it to make sure.

        @jax.jit
        def op(img):
            return functional.resize_image(img, output_size, method="bilinear")

        res = op(image)
        assert res.shape == (16, 16, 3)

    def test_jit_center_crop(self):
        image = jnp.ones((32, 32, 3))
        output_size = (16, 16)

        @jax.jit
        def op(img):
            return functional.center_crop(img, output_size)

        res = op(image)
        assert res.shape == (16, 16, 3)

    def test_jit_random_crop(self):
        image = jnp.ones((32, 32, 3))
        output_size = (16, 16)
        key = jax.random.PRNGKey(0)

        @jax.jit
        def op(img, k):
            return functional.random_crop(img, output_size, k)

        res = op(image, key)
        assert res.shape == (16, 16, 3)

    def test_jit_normalize(self):
        image = jnp.ones((32, 32, 3))
        mean = jnp.array([0.5, 0.5, 0.5])
        std = jnp.array([0.5, 0.5, 0.5])

        @jax.jit
        def op(img):
            return functional.normalize(img, mean, std)

        res = op(image)
        assert res.shape == image.shape

    def test_jit_random_flip_left_right(self):
        image = jnp.ones((32, 32, 3))
        key = jax.random.PRNGKey(0)

        @jax.jit
        def op(img, k):
            return functional.random_flip_left_right(img, k)

        res = op(image, key)
        assert res.shape == image.shape

    def test_jit_random_flip_up_down(self):
        image = jnp.ones((32, 32, 3))
        key = jax.random.PRNGKey(0)

        @jax.jit
        def op(img, k):
            return functional.random_flip_up_down(img, k)

        res = op(image, key)
        assert res.shape == image.shape

    def test_jit_adjust_brightness(self):
        image = jnp.ones((32, 32, 3))

        @jax.jit
        def op(img):
            return functional.adjust_brightness(img, 1.2)

        res = op(image)
        assert res.shape == image.shape

    def test_jit_adjust_brightness_delta(self):
        image = jnp.ones((32, 32, 3))

        @jax.jit
        def op(img):
            return functional.adjust_brightness_delta(img, 0.2)

        res = op(image)
        assert res.shape == image.shape

    def test_jit_adjust_contrast(self):
        image = jnp.ones((32, 32, 3))

        @jax.jit
        def op(img):
            return functional.adjust_contrast(img, 1.2)

        res = op(image)
        assert res.shape == image.shape

    def test_jit_rgb_to_hsv(self):
        image = jnp.ones((32, 32, 3))

        @jax.jit
        def op(img):
            return functional.rgb_to_hsv(img)

        res = op(image)
        assert res.shape == image.shape

    def test_jit_hsv_to_rgb(self):
        image = jnp.ones((32, 32, 3))

        @jax.jit
        def op(img):
            return functional.hsv_to_rgb(img)

        res = op(image)
        assert res.shape == image.shape

    def test_jit_adjust_saturation(self):
        image = jnp.ones((32, 32, 3))

        @jax.jit
        def op(img):
            return functional.adjust_saturation(img, 1.2)

        res = op(image)
        assert res.shape == image.shape

    def test_jit_adjust_hue(self):
        image = jnp.ones((32, 32, 3))

        @jax.jit
        def op(img):
            return functional.adjust_hue(img, 0.1)

        res = op(image)
        assert res.shape == image.shape

    def test_jit_color_jitter(self):
        image = jnp.ones((32, 32, 3))
        key = jax.random.PRNGKey(0)

        @jax.jit
        def op(img, k):
            return functional.color_jitter(img, brightness=0.1, key=k)

        res = op(image, key)
        assert res.shape == image.shape

    def test_jit_convert_rgb_to_grayscale(self):
        image = jnp.ones((32, 32, 3))

        @jax.jit
        def op(img):
            return functional.convert_rgb_to_grayscale(img)

        res = op(image)
        # Should lose channel dimension or result in 2D?
        # Checked impl: returns jnp.dot(image, weights), shape [H, W]
        assert res.shape == (32, 32)

    def test_jit_rotate(self):
        image = jnp.ones((32, 32, 3))

        @jax.jit
        def op(img):
            return functional.rotate(img, 0.1)

        res = op(image)
        assert res.shape == image.shape
