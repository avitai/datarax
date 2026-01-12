"""Test configuration for Datarax."""

import os
import sys
from typing import Any

# Set TensorFlow environment variables before import to prevent segfaults
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
# Disable TensorFlow's memory preallocation
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import jax
import jax.numpy as jnp
import pytest

# Configure TensorFlow GPU memory growth before any TF imports
try:
    import tensorflow as tf

    # Disable all GPUs for TensorFlow to avoid conflicts with JAX
    tf.config.set_visible_devices([], "GPU")
    # Force TensorFlow to use CPU only (without affecting JAX)
    tf.config.experimental.set_visible_devices([], "GPU")
except ImportError:
    pass  # TensorFlow not installed
except Exception as e:
    print(f"Warning: Could not configure TensorFlow: {e}")


# Configure beartype for runtime type checking
try:
    import beartype
    from beartype import BeartypeConf, BeartypeStrategy

    # Apply beartype configuration to enable runtime type checking
    try:
        beartype.beartype(conf=BeartypeConf(strategy=BeartypeStrategy.On))
    except Exception as e:
        print(f"Warning: Could not apply beartype configuration: {e}")
except ImportError:
    # Beartype is not installed, skipping configuration
    pass

# Add the tests directory to the Python path for easy importing of test_common
tests_dir = os.path.dirname(os.path.abspath(__file__))
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)

# Add the src directory to the Python path so tests can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from test_common.device_detection import (
    has_multiple_devices,
    is_distributed_env,
)


# Register custom markers
def pytest_configure(config):
    """Register custom markers for pytest."""
    config.addinivalue_line("markers", "tfds: mark test as requiring tensorflow_datasets")
    config.addinivalue_line("markers", "hf: mark test as requiring huggingface_datasets")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "end_to_end: mark test as an end-to-end test")
    config.addinivalue_line("markers", "benchmark: mark test as a performance benchmark")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "tpu: mark test as requiring TPU")


# Add command-line options for different test types
def pytest_addoption(parser):
    """Add command-line options to pytest."""
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="run integration tests",
    )
    parser.addoption(
        "--end-to-end",
        action="store_true",
        default=False,
        help="run end-to-end tests",
    )
    parser.addoption(
        "--benchmark",
        action="store_true",
        default=False,
        help="run performance benchmark tests",
    )
    parser.addoption(
        "--no-integration",
        action="store_true",
        default=False,
        help="skip integration tests",
    )
    parser.addoption(
        "--no-end-to-end",
        action="store_true",
        default=False,
        help="skip end-to-end tests",
    )
    parser.addoption(
        "--device",
        choices=["cpu", "gpu", "tpu", "all"],
        default="all",
        help="select device type for tests (cpu, gpu, tpu, or all)",
    )


# Skip tests based on command-line options
def pytest_collection_modifyitems(config, items):
    """Skip tests based on command-line options."""
    run_integration = config.getoption("--integration")
    run_end_to_end = config.getoption("--end-to-end")
    run_benchmark = config.getoption("--benchmark")
    skip_integration = config.getoption("--no-integration")
    skip_end_to_end = config.getoption("--no-end-to-end")
    device_option = config.getoption("--device")

    # Skip appropriately based on options
    skip_int = pytest.mark.skip(reason="integration test not selected")
    skip_e2e = pytest.mark.skip(reason="end-to-end test not selected")
    pytest.mark.skip(reason="benchmark not selected")

    # Only run specified test types if explicitly requested
    if run_integration or run_end_to_end or run_benchmark:
        for item in items:
            if run_integration and "integration" not in item.keywords:
                if run_end_to_end and "end_to_end" not in item.keywords:
                    if run_benchmark and "benchmark" not in item.keywords:
                        item.add_marker(pytest.mark.skip(reason="test type not selected"))

    # Skip integration tests if requested
    if skip_integration:
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_int)

    # Skip end-to-end tests if requested
    if skip_end_to_end:
        for item in items:
            if "end_to_end" in item.keywords:
                item.add_marker(skip_e2e)

    # Note: Benchmarks are now run by default to ensure complete testing
    # They can still be explicitly excluded with pytest -m "not benchmark"

    # Handle device-specific test selection
    if device_option != "all":
        for item in items:
            if device_option == "cpu":
                if "gpu" in item.keywords or "tpu" in item.keywords:
                    item.add_marker(
                        pytest.mark.skip(reason=f"test not selected for {device_option}")
                    )
            elif device_option == "gpu":
                if "tpu" in item.keywords:
                    item.add_marker(
                        pytest.mark.skip(reason=f"test not selected for {device_option}")
                    )
            elif device_option == "tpu":
                if "gpu" in item.keywords:
                    item.add_marker(
                        pytest.mark.skip(reason=f"test not selected for {device_option}")
                    )
                # Skip TPU tests for safety
                if "tpu" in item.keywords:
                    item.add_marker(pytest.mark.skip(reason="TPU tests skipped for stability"))

    # Skip TPU tests unconditionally to avoid segmentation faults
    skip_tpu = pytest.mark.skip(reason="TPU tests skipped for stability")
    for item in items:
        if "tpu" in item.keywords:
            item.add_marker(skip_tpu)


# Define fixtures that can be reused across tests
@pytest.fixture
def random_seed() -> int:
    """Return a fixed random seed for reproducible tests."""
    return 42


@pytest.fixture
def rng_key(random_seed) -> jax.Array:
    """Return a JAX RNG key for testing."""
    return jax.random.PRNGKey(random_seed)


@pytest.fixture
def sample_data() -> list[dict[str, Any]]:
    """Generate sample data for testing."""
    return [{"image": jnp.ones((28, 28, 3)), "label": jnp.array(i % 10)} for i in range(100)]


@pytest.fixture
def sample_batch() -> dict[str, jax.Array]:
    """Generate a sample batch for testing."""
    return {"image": jnp.ones((16, 28, 28, 3)), "label": jnp.arange(16) % 10}


@pytest.fixture
def sample_text_data() -> list[dict[str, Any]]:
    """Generate sample text data for testing."""
    sentences = [
        "This is a positive review.",
        "I really enjoyed this product.",
        "The service was terrible.",
        "I would not recommend this restaurant.",
        "Neutral statement about the weather.",
    ]
    return [{"text": sentences[i % len(sentences)], "label": jnp.array(i % 2)} for i in range(100)]


@pytest.fixture
def sample_tabular_data() -> list[dict[str, Any]]:
    """Generate sample tabular data for testing."""
    return [
        {
            "numeric": jnp.array([float(i), float(i + 1), float(i + 2)]),
            "categorical": i % 3,
            "label": jnp.array(i % 2),
        }
        for i in range(100)
    ]


@pytest.fixture
def temp_checkpoint_dir(tmpdir) -> str:
    """Create a temporary directory for checkpoint testing."""
    checkpoint_dir = tmpdir.mkdir("checkpoints")
    return str(checkpoint_dir)


@pytest.fixture
def device_matrix(request):
    """Create a matrix of available devices for testing different configurations."""
    device_counts = {
        "cpu": jax.local_device_count("cpu"),
        "gpu": jax.local_device_count("gpu"),
        "tpu": jax.local_device_count("tpu"),
    }

    return {
        "counts": device_counts,
        "total": sum(device_counts.values()),
        "has_multiple_types": sum(1 for count in device_counts.values() if count > 0) > 1,
        "has_multiple_devices": has_multiple_devices(),
        "is_distributed": is_distributed_env(),
    }
