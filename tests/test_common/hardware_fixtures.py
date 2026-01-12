"""Standardized test fixtures for hardware detection and testing.

This module provides fixtures for hardware-dependent tests in Datarax, ensuring
consistent handling of hardware availability and test skipping.
"""

import logging
import os

import jax
import pytest


# Configure logging
logger = logging.getLogger(__name__)


def get_available_devices():
    """Get information about available JAX devices.

    Returns:
        dict: Information about available devices
    """
    devices = {
        "cpu_count": jax.local_device_count("cpu"),
        "gpu_count": jax.local_device_count("gpu"),
        "tpu_count": jax.local_device_count("tpu"),
        "total_count": jax.local_device_count(),
        "default_device": jax.default_backend(),
    }

    # Log device information for debugging
    logger.info(f"Available devices: {devices}")

    return devices


@pytest.fixture
def device_info():
    """Fixture providing information about available JAX devices.

    Returns:
        dict: Information about available devices
    """
    return get_available_devices()


@pytest.fixture
def cpu_test():
    """Fixture for tests that require CPU.

    This is mostly a passthrough fixture since CPU is always available,
    but included for consistency with other device fixtures.
    """
    devices = get_available_devices()
    if devices["cpu_count"] == 0:
        pytest.skip("Test requires CPU, but no CPU devices found.")
    return devices


@pytest.fixture
def gpu_test():
    """Fixture for tests that require GPU.

    Automatically skips tests when GPU is not available.

    Returns:
        dict: Information about available devices if GPU is available

    Raises:
        pytest.skip: If no GPU is available
    """
    devices = get_available_devices()

    # Check for explicitly disabled GPU tests
    env_var = "DATARAX_DISABLE_GPU_TESTS"
    if os.environ.get(env_var, "").lower() in ("1", "true", "yes"):
        pytest.skip(f"GPU tests disabled by {env_var} environment variable")

    # Check actual hardware availability
    if devices["gpu_count"] == 0:
        pytest.skip("Test requires GPU, but no GPU devices found")

    return devices


@pytest.fixture
def tpu_test():
    """Fixture for tests that require TPU.

    Automatically skips tests when TPU is not available.

    Returns:
        dict: Information about available devices if TPU is available

    Raises:
        pytest.skip: If no TPU is available
    """
    devices = get_available_devices()

    # Check for explicitly disabled TPU tests
    env_var = "DATARAX_DISABLE_TPU_TESTS"
    if os.environ.get(env_var, "").lower() in ("1", "true", "yes"):
        pytest.skip(f"TPU tests disabled by {env_var} environment variable")

    # Check actual hardware availability
    if devices["tpu_count"] == 0:
        pytest.skip("Test requires TPU, but no TPU devices found")

    return devices


@pytest.fixture
def multi_device_test(min_devices=2):
    """Fixture for tests that require multiple devices of any type.

    Args:
        min_devices: Minimum number of devices required (default: 2)

    Returns:
        dict: Information about available devices if enough are available

    Raises:
        pytest.skip: If not enough devices are available
    """
    devices = get_available_devices()

    if devices["total_count"] < min_devices:
        msg = f"Test requires at least {min_devices} devices, "
        msg += f"but only {devices['total_count']} available"
        pytest.skip(msg)

    return devices


@pytest.fixture
def multi_gpu_test(min_gpus=2):
    """Fixture for tests that require multiple GPUs.

    Args:
        min_gpus: Minimum number of GPUs required (default: 2)

    Returns:
        dict: Information about available devices if enough GPUs are available

    Raises:
        pytest.skip: If not enough GPUs are available
    """
    devices = get_available_devices()

    # Check for explicitly disabled GPU tests
    env_var = "DATARAX_DISABLE_GPU_TESTS"
    if os.environ.get(env_var, "").lower() in ("1", "true", "yes"):
        pytest.skip(f"GPU tests disabled by {env_var} environment variable")

    # Check actual hardware availability
    if devices["gpu_count"] < min_gpus:
        msg = f"Test requires at least {min_gpus} GPUs, "
        msg += f"but only {devices['gpu_count']} available"
        pytest.skip(msg)

    return devices


@pytest.fixture
def clear_device_memory():
    """Fixture to clear device memory between tests.

    This is especially useful for GPU tests to prevent memory leaks and OOM errors.
    """
    # Run the test
    yield

    # Clear memory after test
    if jax.default_backend() == "gpu":
        # Force garbage collection
        import gc

        gc.collect()

        # Clear JAX device memory if possible
        try:
            jax.clear_caches()
            # Additional GPU-specific cleanup could be added here
        except Exception as e:
            logger.warning(f"Failed to clear device memory: {e}")


# Define markers for hardware-dependent tests
def pytest_configure(config):
    """Configure pytest with custom markers for hardware-dependent tests."""
    config.addinivalue_line("markers", "cpu: mark test as requiring CPU")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "tpu: mark test as requiring TPU")
    config.addinivalue_line("markers", "multi_device: mark test as requiring multiple devices")
    config.addinivalue_line("markers", "multi_gpu: mark test as requiring multiple GPUs")
