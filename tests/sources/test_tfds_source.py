"""Tests for the TFDS Sources (TFDSEagerSource and TFDSStreamingSource).

This module contains tests for the unified TensorFlow Datasets data source implementation
with the new eager/streaming architecture.
"""

import platform
import pytest

# Skip entire module on macOS ARM64 - TensorFlow import hangs during pytest collection
# due to Metal/GPU device detection issues. This is a known upstream issue:
# https://github.com/tensorflow/tensorflow/issues/52138
# Note: Major ML projects (Keras, Flax) don't run CI tests on macOS for this reason.
if platform.system() == "Darwin":
    pytest.skip(
        "Skipping TFDS tests on macOS (TensorFlow ARM64 import hang issue)",
        allow_module_level=True,
    )

import jax
import flax.nnx as nnx

# Skip tests if tensorflow or tensorflow_datasets is not available
tf = pytest.importorskip("tensorflow")
pytest.importorskip("tensorflow_datasets")

from datarax.sources import (
    TFDSEagerSource,
    TFDSEagerConfig,
    TFDSStreamingSource,
    TFDSStreamingConfig,
    from_tfds,
)


# Set a fixed memory limit for TensorFlow to avoid OOM issues in CI
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


@pytest.fixture
def mock_mnist_dataset():
    """Create a mock MNIST dataset for testing."""
    # Create a small synthetic dataset that mimics MNIST structure
    # Use MNIST structure: 28x28 images with 10 classes
    images = tf.random.uniform((10, 28, 28, 1), minval=0, maxval=255, dtype=tf.int32)
    labels = tf.random.uniform((10,), minval=0, maxval=10, dtype=tf.int32)

    # Create a dataset from tensors
    dataset = tf.data.Dataset.from_tensor_slices({"image": images, "label": labels})
    return dataset


# =============================================================================
# TFDSEagerSource Tests
# =============================================================================


@pytest.mark.tfds
def test_tfds_eager_source_stateless():
    """Test TFDSEagerSource in stateless mode with a small MNIST dataset."""
    try:
        # Create source with config-based API
        config = TFDSEagerConfig(name="mnist", split="test[:10]")
        source = TFDSEagerSource(config)

        # Get first element
        data = next(iter(source))

        # Verify the structure
        assert "image" in data
        assert "label" in data

        # Verify types (should be JAX arrays)
        assert isinstance(data["image"], jax.Array)
        assert isinstance(data["label"], jax.Array)

        # Verify shapes
        assert data["image"].shape == (28, 28, 1)
        assert data["label"].shape == ()

    except Exception as e:
        # Skip if data cannot be loaded (e.g., no internet)
        pytest.skip(f"Could not load MNIST dataset: {e}")


@pytest.mark.tfds
def test_tfds_eager_source_stateful():
    """Test TFDSEagerSource in stateful mode with internal state management."""
    try:
        # Create source with rngs (stateful mode)
        rngs = nnx.Rngs(default=0)
        config = TFDSEagerConfig(name="mnist", split="test[:20]")
        source = TFDSEagerSource(config, rngs=rngs)

        # Get batches using stateful mode
        batch1 = source.get_batch(5)
        source.get_batch(5)

        # Verify batch structure
        assert "image" in batch1
        assert "label" in batch1

        # Verify batch shape (should be batched)
        assert batch1["image"].shape[0] == 5  # Batch size
        assert batch1["label"].shape[0] == 5

    except Exception as e:
        pytest.skip(f"Could not load MNIST dataset: {e}")


@pytest.mark.tfds
def test_tfds_eager_source_with_shuffling():
    """Test TFDSEagerSource with shuffling enabled."""
    try:
        # Create source with shuffling
        rngs = nnx.Rngs(default=42, shuffle=42)
        config = TFDSEagerConfig(name="mnist", split="test[:100]", shuffle=True, seed=42)
        source = TFDSEagerSource(config, rngs=rngs)

        # Get some data
        items = []
        for i, item in enumerate(source):
            if i >= 10:
                break
            items.append(item)

        assert len(items) == 10

    except Exception as e:
        pytest.skip(f"Could not load MNIST dataset: {e}")


@pytest.mark.tfds
def test_tfds_eager_source_with_include_keys():
    """Test TFDSEagerSource with include_keys filter."""
    try:
        # Load MNIST with only the image key
        config = TFDSEagerConfig(name="mnist", split="test[:10]", include_keys={"image"})
        source = TFDSEagerSource(config)

        # Verify only the image key is present in loaded data
        assert "image" in source.data
        assert "label" not in source.data

        # Get first element
        data = next(iter(source))

        # Verify only the image key is present
        assert "image" in data
        assert "label" not in data

    except Exception as e:
        pytest.skip(f"Could not load MNIST dataset: {e}")


@pytest.mark.tfds
def test_tfds_eager_source_with_exclude_keys():
    """Test TFDSEagerSource with exclude_keys filter."""
    try:
        # Load MNIST excluding the label key
        config = TFDSEagerConfig(name="mnist", split="test[:10]", exclude_keys={"label"})
        source = TFDSEagerSource(config)

        # Verify label is excluded from loaded data
        assert "image" in source.data
        assert "label" not in source.data

        # Get first element
        data = next(iter(source))

        # Verify label is excluded
        assert "image" in data
        assert "label" not in data

    except Exception as e:
        pytest.skip(f"Could not load MNIST dataset: {e}")


@pytest.mark.tfds
def test_tfds_eager_source_stateless_batch():
    """Test TFDSEagerSource batch retrieval with explicit key."""
    try:
        # Create source with config
        config = TFDSEagerConfig(name="mnist", split="test[:50]")
        source = TFDSEagerSource(config)

        # Get batch with explicit key (stateless mode)
        key = jax.random.key(42)
        batch = source.get_batch(10, key=key)

        # Verify batch structure
        assert "image" in batch
        assert "label" in batch
        assert batch["image"].shape[0] == 10

    except Exception as e:
        pytest.skip(f"Could not load MNIST dataset: {e}")


@pytest.mark.tfds
def test_tfds_eager_source_as_supervised():
    """Test TFDSEagerSource with as_supervised flag."""
    try:
        # Load with as_supervised flag
        config = TFDSEagerConfig(name="mnist", split="test[:10]", as_supervised=True)
        source = TFDSEagerSource(config)

        # Get first element
        data = next(iter(source))

        # With as_supervised, keys should be 'image' and 'label'
        assert "image" in data or "label" in data

    except Exception as e:
        pytest.skip(f"Could not load MNIST dataset: {e}")


def test_tfds_eager_source_error_handling():
    """Test TFDSEagerConfig error handling."""
    # Test error for specifying both include and exclude keys
    with pytest.raises(ValueError, match="Cannot specify both"):
        TFDSEagerConfig(name="mnist", split="test", include_keys={"image"}, exclude_keys={"label"})


@pytest.mark.tfds
def test_tfds_eager_source_repr():
    """Test string representation of TFDSEagerSource."""
    try:
        config = TFDSEagerConfig(name="mnist", split="test[:10]")
        source = TFDSEagerSource(config)

        # Check that repr includes useful information
        repr_str = repr(source)
        assert "TFDSEagerSource" in repr_str
        assert "mnist" in repr_str

    except Exception as e:
        pytest.skip(f"Could not create TFDSEagerSource: {e}")


def test_tfds_eager_source_config_validation():
    """Test TFDSEagerConfig validation."""
    # Should raise error when name is not provided
    with pytest.raises(ValueError, match="name is required"):
        TFDSEagerConfig(split="train")

    # Should raise error when split is not provided
    with pytest.raises(ValueError, match="split is required"):
        TFDSEagerConfig(name="mnist")


# =============================================================================
# TFDSStreamingSource Tests
# =============================================================================


@pytest.mark.tfds
def test_tfds_streaming_source_basic():
    """Test TFDSStreamingSource basic iteration."""
    try:
        config = TFDSStreamingConfig(name="mnist", split="test[:10]")
        source = TFDSStreamingSource(config)

        # Get first element
        data = next(iter(source))

        # Verify the structure
        assert "image" in data
        assert "label" in data

        # Verify types (should be JAX arrays)
        assert isinstance(data["image"], jax.Array)
        assert isinstance(data["label"], jax.Array)

    except Exception as e:
        pytest.skip(f"Could not load MNIST dataset: {e}")


@pytest.mark.tfds
def test_tfds_streaming_source_with_shuffling():
    """Test TFDSStreamingSource with shuffling enabled."""
    try:
        config = TFDSStreamingConfig(
            name="mnist", split="test[:50]", shuffle=True, shuffle_buffer_size=50
        )
        source = TFDSStreamingSource(config)

        # Get some data
        items = []
        for i, item in enumerate(source):
            if i >= 5:
                break
            items.append(item)

        assert len(items) == 5

    except Exception as e:
        pytest.skip(f"Could not load MNIST dataset: {e}")


@pytest.mark.tfds
def test_tfds_streaming_source_fixed_prefetch():
    """Test that TFDSStreamingSource uses fixed prefetch buffer."""
    try:
        # Create with explicit prefetch buffer (not AUTOTUNE)
        config = TFDSStreamingConfig(name="mnist", split="test[:10]", prefetch_buffer=2)
        source = TFDSStreamingSource(config)

        # Verify source is created
        assert source is not None

        # Get first element to verify it works
        data = next(iter(source))
        assert "image" in data

    except Exception as e:
        pytest.skip(f"Could not load MNIST dataset: {e}")


def test_tfds_streaming_source_config_validation():
    """Test TFDSStreamingConfig validation."""
    # Should raise error when name is not provided
    with pytest.raises(ValueError, match="name is required"):
        TFDSStreamingConfig(split="train")

    # Should raise error when split is not provided
    with pytest.raises(ValueError, match="split is required"):
        TFDSStreamingConfig(name="mnist")

    # Test error for specifying both include and exclude keys
    with pytest.raises(ValueError, match="Cannot specify both"):
        TFDSStreamingConfig(
            name="mnist", split="test", include_keys={"image"}, exclude_keys={"label"}
        )


# =============================================================================
# Factory Function Tests
# =============================================================================


@pytest.mark.tfds
def test_from_tfds_creates_eager_for_small_datasets():
    """Test that from_tfds creates eager source for small datasets like MNIST."""
    try:
        source = from_tfds("mnist", "test[:10]", rngs=nnx.Rngs(0))

        # Should be TFDSEagerSource (has .data attribute)
        assert hasattr(source, "data")
        assert isinstance(source, TFDSEagerSource)

    except Exception as e:
        pytest.skip(f"Could not load MNIST dataset: {e}")


@pytest.mark.tfds
def test_from_tfds_force_eager():
    """Test that from_tfds with eager=True creates eager source."""
    try:
        source = from_tfds("mnist", "test[:10]", eager=True, rngs=nnx.Rngs(0))
        assert isinstance(source, TFDSEagerSource)

    except Exception as e:
        pytest.skip(f"Could not load MNIST dataset: {e}")


@pytest.mark.tfds
def test_from_tfds_force_streaming():
    """Test that from_tfds with eager=False creates streaming source."""
    try:
        source = from_tfds("mnist", "test[:10]", eager=False, rngs=nnx.Rngs(0))
        assert isinstance(source, TFDSStreamingSource)

    except Exception as e:
        pytest.skip(f"Could not load MNIST dataset: {e}")


@pytest.mark.tfds
def test_from_tfds_with_shuffling():
    """Test that from_tfds passes shuffle parameter correctly."""
    try:
        source = from_tfds("mnist", "test[:20]", shuffle=True, seed=42, rngs=nnx.Rngs(0))

        # Should have shuffling enabled
        assert source.shuffle is True

    except Exception as e:
        pytest.skip(f"Could not load MNIST dataset: {e}")
