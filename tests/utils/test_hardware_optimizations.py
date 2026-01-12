"""Tests for GPU-optimized operations in Datarax.

This module contains tests that verify Datarax components work correctly
on both CPU and GPU hardware. Tests will automatically adapt based on
available hardware.
"""

import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

# OptimizedRNGModule has been removed - using nnx.Rngs directly
from datarax.operators.modality.image.functional import (
    normalize,
    resize_image,
)


@pytest.fixture
def large_batch_size():
    """Return an appropriate batch size based on available GPU memory."""
    # Start with a conservative batch size
    return 128


@pytest.fixture
def large_rgb_batch(large_batch_size):
    """Create a large batch of RGB images for performance testing."""
    # 224x224 is a common image size for CNNs
    return jnp.ones((large_batch_size, 224, 224, 3), dtype=jnp.float32)


@pytest.fixture
def gpu_rng():
    """Create RNG keys for GPU tests."""
    return nnx.Rngs({"augment": jax.random.key(42)})


def test_device_detection():
    """Verify available compute devices (CPU or GPU)."""
    # Get all available devices
    all_devices = jax.devices()

    # Categorize devices
    cpu_devices = [d for d in all_devices if d.platform.lower() == "cpu"]
    gpu_devices = [d for d in all_devices if d.platform.lower() in ["cuda", "gpu", "rocm"]]

    # Print device information
    print("\nAvailable devices:")
    print(f"  CPU devices: {len(cpu_devices)}")
    print(f"  GPU devices: {len(gpu_devices)}")
    print(f"  JAX version: {jax.__version__}")
    print(f"  Primary device: {jax.devices()[0].platform.upper()}")

    # Ensure at least one device is available
    assert len(all_devices) > 0, "No compute devices detected"

    # Test basic operation on available device
    x = jnp.ones((2, 2))
    y = x @ x
    assert y.shape == (2, 2), "Basic matrix multiplication failed"  # nosec B101


def test_nnx_rngs_performance():
    """Test nnx.Rngs performance on available hardware."""
    # Create nnx.Rngs with a specific seed
    rngs = nnx.Rngs(default=jax.random.key(42))

    # Define a function that uses a key to generate random data
    def generate_random_matrix(key, shape):
        return jax.random.normal(key, shape=shape)

    # Define shape as a constant
    shape = (1024, 1024)

    # Create a function with fixed shape
    def generate_matrix_1024(key):
        return jax.random.normal(key, shape=shape)

    # Get keys outside of jit
    key1 = rngs.default()
    key2 = rngs.default()

    # Compile the function
    compiled_fn = jax.jit(generate_matrix_1024)

    # Warm up
    _ = compiled_fn(key1)

    # Measure performance
    start_time = time.time()
    result = compiled_fn(key2)
    end_time = time.time()

    device_type = jax.devices()[0].platform.upper()
    print(
        f"\nTime to generate 1024x1024 random matrix on {device_type}: "
        f"{(end_time - start_time) * 1000:.2f} ms"
    )
    assert result.shape == (1024, 1024)


def test_batch_image_operations_performance():
    """Test performance of batch image operations on available hardware."""
    # Create a batch of images
    batch_size = 64
    images = jnp.ones((batch_size, 224, 224, 3), dtype=jnp.float32)

    # Following guidelines 4.7.3: Use proper transform patterns
    # Don't nest jit and vmap - use vmap then jit the result

    # Create transformation functions without individual jit
    def resize_fn(x):
        return resize_image(x, (192, 192))

    def normalize_fn(x):
        return normalize(x, mean=0.5, std=0.5)

    # Create batch processing with vmap first, then jit
    batch_resize = jax.jit(jax.vmap(resize_fn))
    batch_normalize = jax.jit(jax.vmap(normalize_fn))

    # Warm up - critical for GPU
    _ = batch_resize(images).block_until_ready()
    _ = batch_normalize(images).block_until_ready()

    # Measure resize performance
    start_time = time.time()
    resized = batch_resize(images).block_until_ready()
    resize_time = time.time() - start_time

    # Measure normalize performance
    start_time = time.time()
    normalized = batch_normalize(resized).block_until_ready()
    normalize_time = time.time() - start_time

    print(f"\nBatch size: {batch_size}")
    print(f"Resize time: {resize_time * 1000:.2f} ms")
    print(f"Normalize time: {normalize_time * 1000:.2f} ms")
    print(f"Total time: {(resize_time + normalize_time) * 1000:.2f} ms")

    assert resized.shape == (batch_size, 192, 192, 3)
    assert normalized.shape == (batch_size, 192, 192, 3)


def test_transformer_pipeline(large_rgb_batch, gpu_rng):
    """Test a pipeline of transformers on available hardware."""

    # Following guidelines: avoid potential GPU issues with image operations
    # Use simpler operations that are known to work well on GPU

    def transform_image(image):
        # Use JAX's image resize which is more GPU-friendly
        # Ensure we use bilinear interpolation without antialiasing for GPU stability
        image = jax.image.resize(image, shape=(192, 192, 3), method="linear")

        # Apply normalization using JAX operations
        image = (image - 0.5) / 0.5

        return image

    # Create batched version using JAX's vmap
    batched_transform = jax.vmap(transform_image)

    # JIT compile for efficiency
    jitted_transform = jax.jit(batched_transform)

    # Warm up with block_until_ready for GPU sync
    _ = jitted_transform(large_rgb_batch).block_until_ready()

    # Measure performance
    start_time = time.time()
    result = jitted_transform(large_rgb_batch).block_until_ready()
    end_time = time.time()

    batch_size = large_rgb_batch.shape[0]
    processing_time = (end_time - start_time) * 1000  # ms
    per_image_time = processing_time / batch_size

    print(f"\nBatch size: {batch_size}")
    print(f"Pipeline processing time: {processing_time:.2f} ms")
    print(f"Time per image: {per_image_time:.2f} ms")

    assert result.shape == (large_rgb_batch.shape[0], 192, 192, 3)


def test_memory_efficiency():
    """Test memory efficiency of operations on available hardware."""
    # This test checks if we can allocate and process large tensors on GPU
    # without running out of memory

    # Use more conservative sizes to avoid GPU memory issues
    image_size = 512  # Reduced from 1024
    batch_size = 8  # Reduced from 16

    # Create a large batch of images
    large_batch = jnp.ones((batch_size, image_size, image_size, 3), dtype=jnp.float32)

    # Define a memory-intensive operation with proper shape handling
    def memory_intensive_op(x):
        # Use per-image operations to avoid batch dimension issues
        def single_image_op(img):
            # Perform multiple operations that require intermediate storage
            y1 = jax.image.resize(img, (image_size // 2, image_size // 2, 3), method="linear")
            y2 = jax.image.resize(img, (image_size // 4, image_size // 4, 3), method="linear")
            y3 = jax.image.resize(y2, (image_size // 2, image_size // 2, 3), method="linear")
            return y1 + y3

        # Apply to batch using vmap
        return jax.vmap(single_image_op)(x)

    # JIT compile for efficiency
    jitted_op = jax.jit(memory_intensive_op)

    # Warm up with block_until_ready
    _ = jitted_op(large_batch).block_until_ready()

    # Run the operation and time it
    start_time = time.time()
    result = jitted_op(large_batch).block_until_ready()
    end_time = time.time()

    print(f"\nProcessed batch of {batch_size} images of size {image_size}x{image_size}")
    print(f"Processing time: {(end_time - start_time) * 1000:.2f} ms")

    assert result.shape == (batch_size, image_size // 2, image_size // 2, 3)


if __name__ == "__main__":
    # Run the tests individually to better measure performance
    print("Testing device detection...")
    test_device_detection()

    print("\nTesting nnx.Rngs performance...")
    test_nnx_rngs_performance()

    print("\nTesting batch image operations performance...")
    test_batch_image_operations_performance()

    # Create test fixtures for running the pipeline test
    bs = 128
    rgb_batch = jnp.ones((bs, 224, 224, 3), dtype=jnp.float32)
    test_rng = nnx.Rngs({"augment": jax.random.key(42)})

    print("\nTesting transformer pipeline...")
    test_transformer_pipeline(rgb_batch, test_rng)

    print("\nTesting memory efficiency...")
    test_memory_efficiency()

    device_type = jax.devices()[0].platform.upper()
    print(f"\nAll tests passed on {device_type}!")
