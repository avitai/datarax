"""Common test fixtures for Datarax tests."""

import tempfile

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def device_mesh():
    """Create a device mesh for testing distributed operations."""
    devices = jax.devices()
    if len(devices) >= 4:
        # Create a 2x2 mesh if enough devices
        return jax.sharding.Mesh(np.array(devices[:4]).reshape(2, 2), ("data", "model"))
    elif len(devices) >= 2:
        # Create a 2x1 mesh if at least 2 devices
        return jax.sharding.Mesh(np.array(devices[:2]).reshape(2, 1), ("data", "model"))
    else:
        # Create a 1x1 mesh with the single device
        return jax.sharding.Mesh(np.array(devices[:1]).reshape(1, 1), ("data", "model"))


@pytest.fixture
def sample_image_data(size: int = 100) -> dict[str, np.ndarray]:
    """Generate sample image data for testing.

    Args:
        size: Number of samples to generate.

    Returns:
        A dictionary with image data and labels.
    """
    np.random.seed(42)  # for reproducibility
    return {
        "image": np.random.rand(size, 32, 32, 3).astype(np.float32),
        "label": np.random.randint(0, 10, size=(size,)),
    }


@pytest.fixture
def sample_text_dataset(size: int = 100) -> dict[str, np.ndarray]:
    """Generate sample text dataset for testing.

    Args:
        size: Number of samples to generate.

    Returns:
        A dictionary with text data and labels.
    """
    np.random.seed(42)  # for reproducibility

    # Generate sample sentences of varying lengths
    vocabulary = [
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
        "hello",
        "world",
        "test",
        "data",
        "jax",
        "flow",
        "framework",
        "machine",
        "learning",
        "neural",
        "network",
        "transformer",
    ]

    # Generate random sequences of words
    max_seq_length = 20
    texts = []
    for _ in range(size):
        seq_length = np.random.randint(5, max_seq_length + 1)
        text = " ".join(np.random.choice(vocabulary, size=seq_length))
        texts.append(text)

    # Generate random labels (binary classification)
    labels = np.random.randint(0, 2, size=(size,))

    return {
        "text": np.array(texts),
        "label": labels,
    }


@pytest.fixture
def sample_tabular_dataset(size: int = 100, num_features: int = 10) -> dict[str, np.ndarray]:
    """Generate sample tabular dataset for testing.

    Args:
        size: Number of samples to generate.
        num_features: Number of features.

    Returns:
        A dictionary with numerical features, categorical features, and labels.
    """
    np.random.seed(42)  # for reproducibility

    # Generate numerical features
    numerical = np.random.randn(size, num_features).astype(np.float32)

    # Generate categorical features (3 categories)
    categorical = np.random.randint(0, 3, size=(size, 3))

    # Generate binary labels
    labels = np.random.randint(0, 2, size=(size,))

    return {
        "numerical": numerical,
        "categorical": categorical,
        "label": labels,
    }


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def nnx_module_fixture():
    """Create a sample NNX module for testing."""

    class SampleModule(nnx.Module):
        def __init__(self, features: int = 64):
            super().__init__()
            self.dense1 = nnx.Linear(in_features=32, out_features=features, rngs=nnx.Rngs(0))
            self.dense2 = nnx.Linear(in_features=features, out_features=1, rngs=nnx.Rngs(1))

        def __call__(self, x, training: bool = False):
            x = self.dense1(x)
            x = jax.nn.relu(x)
            x = self.dense2(x)
            return x

    # Create and initialize the module
    module = SampleModule()
    x = jnp.ones((1, 32))
    # Module is already initialized with rngs, just call it
    module(x)

    return module


@pytest.fixture
def image_processing_fixtures():
    """Create image processing test fixtures."""
    # Sample small image for fast processing tests
    small_image = np.random.rand(16, 16, 3).astype(np.float32)

    # Sample batch of images
    batch_images = np.random.rand(8, 32, 32, 3).astype(np.float32)

    # Sample image with mask for segmentation tests
    mask = np.zeros((32, 32, 1), dtype=np.float32)
    mask[10:20, 10:20] = 1.0  # Create a square mask
    image_with_mask = {"image": np.random.rand(32, 32, 3).astype(np.float32), "mask": mask}

    return {
        "small_image": small_image,
        "batch_images": batch_images,
        "image_with_mask": image_with_mask,
    }


@pytest.fixture
def device_agnostic_test():
    """Skip if no supported device is available."""
    if not jax.devices():
        pytest.skip("No JAX devices available")
    return True


@pytest.fixture
def gpu_test():
    """Skip test if GPU is not available."""
    if jax.local_device_count("gpu") == 0:
        print("INFO: Skipping test that requires GPU")
        pytest.skip("Test requires GPU")
    return True


@pytest.fixture
def tpu_test():
    """Skip test if TPU is not available."""
    try:
        if jax.local_device_count("tpu") == 0:
            print("INFO: Skipping test that requires TPU")
            pytest.skip("Test requires TPU")
    except RuntimeError:
        print("INFO: TPU backend failed to initialize. Skipping test that requires TPU.")
        pytest.skip("Test requires TPU")
    return True


@pytest.fixture
def multi_device_test():
    """Skip test if multiple devices are not available."""
    device_count = jax.local_device_count()
    if device_count < 2:
        print(
            f"INFO: Only {device_count} device(s) available. "
            f"Skipping test that requires multiple devices."
        )
        pytest.skip("Test requires at least 2 devices")
    return device_count
