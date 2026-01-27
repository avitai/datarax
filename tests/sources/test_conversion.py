"""Tests for the DLPack conversion module.

This module contains tests for the zero-copy data conversion utilities
that convert TensorFlow and HuggingFace data to JAX arrays.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from datarax.sources._conversion import (
    tf_to_jax,
    hf_to_jax,
    convert_batch_to_jax,
    stack_batches,
)


# =============================================================================
# Tests for hf_to_jax (no TensorFlow dependency)
# =============================================================================


class TestHfToJax:
    """Tests for HuggingFace to JAX conversion."""

    @pytest.mark.unit
    def test_numpy_array_conversion(self):
        """Test conversion of numpy arrays to JAX arrays."""
        np_array = np.array([[1.0, 2.0], [3.0, 4.0]])
        jax_array = hf_to_jax(np_array)

        assert isinstance(jax_array, jax.Array)
        assert jax_array.shape == (2, 2)
        np.testing.assert_array_equal(np.array(jax_array), np_array)

    @pytest.mark.unit
    def test_list_conversion(self):
        """Test conversion of Python lists to JAX arrays."""
        py_list = [1, 2, 3, 4]
        jax_array = hf_to_jax(py_list)

        assert isinstance(jax_array, jax.Array)
        assert jax_array.shape == (4,)
        np.testing.assert_array_equal(np.array(jax_array), np.array(py_list))

    @pytest.mark.unit
    def test_nested_list_conversion(self):
        """Test conversion of nested lists to JAX arrays."""
        nested_list = [[1, 2], [3, 4]]
        jax_array = hf_to_jax(nested_list)

        assert isinstance(jax_array, jax.Array)
        assert jax_array.shape == (2, 2)

    @pytest.mark.unit
    def test_scalar_conversion(self):
        """Test conversion of scalars to JAX arrays."""
        scalar = 42
        jax_array = hf_to_jax(scalar)

        assert isinstance(jax_array, jax.Array)
        assert jax_array.shape == ()
        assert int(jax_array) == 42

    @pytest.mark.unit
    def test_float_scalar_conversion(self):
        """Test conversion of float scalars to JAX arrays."""
        scalar = 3.14
        jax_array = hf_to_jax(scalar)

        assert isinstance(jax_array, jax.Array)
        assert jax_array.shape == ()
        assert float(jax_array) == pytest.approx(3.14)

    @pytest.mark.unit
    def test_pil_image_conversion(self):
        """Test conversion of PIL images to JAX arrays."""
        pytest.importorskip("PIL")
        from PIL import Image

        # Create a simple RGB image
        img = Image.new("RGB", (28, 28), color=(255, 128, 64))
        jax_array = hf_to_jax(img)

        assert isinstance(jax_array, jax.Array)
        assert jax_array.shape == (28, 28, 3)
        # Check RGB values
        assert jax_array[0, 0, 0] == 255  # Red
        assert jax_array[0, 0, 1] == 128  # Green
        assert jax_array[0, 0, 2] == 64  # Blue

    @pytest.mark.unit
    def test_grayscale_pil_image_conversion(self):
        """Test conversion of grayscale PIL images."""
        pytest.importorskip("PIL")
        from PIL import Image

        # Create a simple grayscale image
        img = Image.new("L", (28, 28), color=128)
        jax_array = hf_to_jax(img)

        assert isinstance(jax_array, jax.Array)
        assert jax_array.shape == (28, 28)
        assert jax_array[0, 0] == 128

    @pytest.mark.unit
    def test_string_list_passthrough(self):
        """Test that string lists are passed through unchanged."""
        string_list = ["hello", "world"]
        result = hf_to_jax(string_list)

        # Strings can't be converted to JAX arrays, should be returned as-is
        assert result == string_list


# =============================================================================
# Tests for tf_to_jax (requires TensorFlow)
# =============================================================================


class TestTfToJax:
    """Tests for TensorFlow to JAX conversion."""

    @pytest.fixture(autouse=True)
    def skip_without_tf(self):
        """Skip tests if TensorFlow is not available."""
        pytest.importorskip("tensorflow")

    @pytest.mark.unit
    def test_float32_tensor_conversion(self):
        """Test conversion of float32 TensorFlow tensor."""
        import tensorflow as tf

        tf_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
        jax_array = tf_to_jax(tf_tensor)

        assert isinstance(jax_array, jax.Array)
        assert jax_array.shape == (2, 2)
        assert jax_array.dtype == jnp.float32
        np.testing.assert_array_almost_equal(
            np.array(jax_array), np.array([[1.0, 2.0], [3.0, 4.0]])
        )

    @pytest.mark.unit
    def test_int32_tensor_conversion(self):
        """Test conversion of int32 TensorFlow tensor."""
        import tensorflow as tf

        tf_tensor = tf.constant([1, 2, 3, 4], dtype=tf.int32)
        jax_array = tf_to_jax(tf_tensor)

        assert isinstance(jax_array, jax.Array)
        assert jax_array.shape == (4,)
        assert jax_array.dtype == jnp.int32

    @pytest.mark.unit
    def test_uint8_tensor_conversion(self):
        """Test conversion of uint8 TensorFlow tensor (common for images)."""
        import tensorflow as tf

        tf_tensor = tf.constant([[[255, 128, 64]]], dtype=tf.uint8)
        jax_array = tf_to_jax(tf_tensor)

        assert isinstance(jax_array, jax.Array)
        assert jax_array.dtype == jnp.uint8

    @pytest.mark.unit
    def test_large_tensor_conversion(self):
        """Test conversion of larger tensors."""
        import tensorflow as tf

        # Simulate MNIST-like image batch
        tf_tensor = tf.random.uniform((32, 28, 28, 1), dtype=tf.float32)
        jax_array = tf_to_jax(tf_tensor)

        assert isinstance(jax_array, jax.Array)
        assert jax_array.shape == (32, 28, 28, 1)

    @pytest.mark.unit
    def test_scalar_tensor_conversion(self):
        """Test conversion of scalar TensorFlow tensor."""
        import tensorflow as tf

        tf_tensor = tf.constant(42)
        jax_array = tf_to_jax(tf_tensor)

        assert isinstance(jax_array, jax.Array)
        assert jax_array.shape == ()


# =============================================================================
# Tests for convert_batch_to_jax
# =============================================================================


class TestConvertBatchToJax:
    """Tests for batch conversion utility."""

    @pytest.mark.unit
    def test_batch_conversion_with_hf_converter(self):
        """Test batch conversion using hf_to_jax converter."""
        batch = {
            "image": np.ones((28, 28)),
            "label": 5,
        }

        jax_batch = convert_batch_to_jax(batch, hf_to_jax)

        assert "image" in jax_batch
        assert "label" in jax_batch
        assert isinstance(jax_batch["image"], jax.Array)
        assert isinstance(jax_batch["label"], jax.Array)

    @pytest.mark.unit
    def test_empty_batch_conversion(self):
        """Test conversion of empty batch."""
        batch = {}
        jax_batch = convert_batch_to_jax(batch, hf_to_jax)
        assert jax_batch == {}

    @pytest.mark.unit
    @pytest.mark.skipif(
        not pytest.importorskip("tensorflow", reason="TensorFlow not available"),
        reason="TensorFlow not available",
    )
    def test_batch_conversion_with_tf_converter(self):
        """Test batch conversion using tf_to_jax converter."""
        import tensorflow as tf

        batch = {
            "image": tf.constant([[1.0, 2.0]]),
            "label": tf.constant(5),
        }

        jax_batch = convert_batch_to_jax(batch, tf_to_jax)

        assert "image" in jax_batch
        assert "label" in jax_batch
        assert isinstance(jax_batch["image"], jax.Array)
        assert isinstance(jax_batch["label"], jax.Array)


# =============================================================================
# Tests for stack_batches
# =============================================================================


class TestStackBatches:
    """Tests for batch stacking utility."""

    @pytest.mark.unit
    def test_stack_two_batches(self):
        """Test stacking two batches."""
        batch1 = {"image": jnp.ones((32, 28, 28)), "label": jnp.arange(32)}
        batch2 = {"image": jnp.zeros((32, 28, 28)), "label": jnp.arange(32, 64)}

        combined = stack_batches([batch1, batch2])

        assert combined["image"].shape == (64, 28, 28)
        assert combined["label"].shape == (64,)

    @pytest.mark.unit
    def test_stack_single_batch(self):
        """Test stacking a single batch (should work like identity)."""
        batch = {"data": jnp.array([1, 2, 3])}
        combined = stack_batches([batch])

        assert combined["data"].shape == (3,)
        np.testing.assert_array_equal(np.array(combined["data"]), [1, 2, 3])

    @pytest.mark.unit
    def test_stack_empty_raises_error(self):
        """Test that stacking empty list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot stack empty"):
            stack_batches([])

    @pytest.mark.unit
    def test_stack_multiple_keys(self):
        """Test stacking batches with multiple keys."""
        batches = [
            {"a": jnp.array([[1]]), "b": jnp.array([[2]])},
            {"a": jnp.array([[3]]), "b": jnp.array([[4]])},
            {"a": jnp.array([[5]]), "b": jnp.array([[6]])},
        ]

        combined = stack_batches(batches)

        assert combined["a"].shape == (3, 1)
        assert combined["b"].shape == (3, 1)
        np.testing.assert_array_equal(np.array(combined["a"]), [[1], [3], [5]])
        np.testing.assert_array_equal(np.array(combined["b"]), [[2], [4], [6]])
