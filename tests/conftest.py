"""Test configuration for Datarax."""

import importlib.util
import os
import platform
import sys
from pathlib import Path
from typing import Any


# Detect platform
IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

# Set TensorFlow environment variables BEFORE any TF import to prevent hangs/segfaults
# These must be set before TensorFlow is imported anywhere
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress all TF logs
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN (can cause hangs)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"  # Pure Python protobuf

if IS_MACOS:
    # macOS-specific settings to prevent TensorFlow import hang on ARM64
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # No CUDA on macOS
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"  # Limit threading
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"  # Limit threading
    # Disable Metal/GPU detection that can hang in CI
    os.environ["TF_METAL_DEVICE_SELECTOR"] = ""
    os.environ["TF_DISABLE_MLC_BRIDGE"] = "1"  # Disable Apple ML Compute bridge
elif IS_LINUX:
    # CUDA-specific settings (Linux only)
    os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


def _configure_test_jax_multi_device_emulation() -> None:
    """Enable JAX CPU multi-device emulation for tests that need it.

    Multi-device tests (data-parallel sharding, mesh transforms) call
    ``jax.device_count()`` at collection time and skip when fewer than 2
    devices exist. On a single-GPU or CPU host we get one device by
    default. JAX exposes ``--xla_force_host_platform_device_count=N`` to
    fan out a single CPU into N logical devices for emulation.

    The flag only affects the CPU backend, so when emulation is active
    we also force ``JAX_PLATFORMS=cpu`` (overriding any earlier CUDA
    selection) — otherwise CUDA would be picked first and expose its
    actual one-or-zero device count.

    Tests can override the device count via ``DATARAX_TEST_DEVICE_COUNT``
    or disable emulation entirely with ``DATARAX_TEST_DEVICE_COUNT=0``.
    """
    requested = os.environ.get("DATARAX_TEST_DEVICE_COUNT", "8")
    if requested == "0":
        return
    flag = f"--xla_force_host_platform_device_count={requested}"
    existing = os.environ.get("XLA_FLAGS", "")
    if "xla_force_host_platform_device_count" not in existing:
        os.environ["XLA_FLAGS"] = (f"{existing} {flag}").strip()
    # Multi-device emulation lives on the CPU backend; force CPU so it
    # is exposed even when a CUDA plugin is available.
    os.environ["JAX_PLATFORMS"] = "cpu"


def _configure_test_jax_platforms() -> None:
    """Set a safe JAX backend default for test runs.

    Tests should run on CPU by default, while still allowing explicit
    CUDA platform selection via environment override.
    """
    explicit_platforms = os.environ.get("DATARAX_TEST_JAX_PLATFORMS")
    if explicit_platforms:
        os.environ["JAX_PLATFORMS"] = explicit_platforms
        return

    requested = os.environ.get("JAX_PLATFORMS", "")
    if not requested:
        os.environ["JAX_PLATFORMS"] = "cpu"
        return

    # If CUDA is requested but CUDA plugin support is missing in this env,
    # force CPU to avoid backend initialization failures during test collection.
    if "cuda" in requested:
        has_cuda_plugin = (
            _module_exists("jax_cuda12_plugin")
            or _module_exists("jax_cuda13_plugin")
            or _module_exists("jax_plugins.xla_cuda12")
            or _module_exists("jax_plugins.xla_cuda13")
        )
        if not has_cuda_plugin:
            os.environ["JAX_PLATFORMS"] = "cpu"


def _module_exists(module_name: str) -> bool:
    """Return whether a module can be imported without importing it."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


_configure_test_jax_platforms()
_configure_test_jax_multi_device_emulation()

import jax
import jax.numpy as jnp
import pytest

from datarax.utils.console import emit


# Pre-import Deep Lake before TensorFlow to avoid fatal OpenSSL conflict.
# See benchmarks/adapters/_preload.py for the full explanation.
try:
    import benchmarks.adapters._preload  # noqa: F401
except ImportError:
    pass  # benchmarks package not on path (core tests only)

# Configure TensorFlow - only on Linux
# Note: TensorFlow import on macOS ARM64 can hang during pytest collection due to
# Metal/GPU device detection issues. This is a known upstream issue (tensorflow/tensorflow#52138).
# Major ML projects (Keras, Flax) don't test on macOS at all for this reason.
# We skip TensorFlow-dependent tests on macOS using module-level pytest.skip().
if not IS_MACOS:
    try:
        import tensorflow as tf

        if IS_LINUX:
            # Disable all GPUs for TensorFlow to avoid conflicts with JAX
            try:
                tf.config.set_visible_devices([], "GPU")
            except (RuntimeError, ValueError):
                pass
    except ImportError:
        pass  # TensorFlow not installed
    except (OSError, RuntimeError, ValueError) as e:
        emit(f"Warning: Could not configure TensorFlow: {e}")


# Configure beartype for runtime type checking
try:
    import beartype
    from beartype import BeartypeConf, BeartypeStrategy

    # Apply beartype configuration to enable runtime type checking
    try:
        beartype.beartype(conf=BeartypeConf(strategy=BeartypeStrategy.On))
    except (RuntimeError, TypeError, ValueError) as e:
        emit(f"Warning: Could not apply beartype configuration: {e}")
except ImportError:
    # Beartype is not installed, skipping configuration
    pass

# Add the tests directory to the Python path for easy importing of test_common
tests_dir = str(Path(__file__).resolve().parent)
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)

# Add the src directory to the Python path so tests can import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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


def _deselect_items(config: Any, items: list[Any], deselected: list[Any]) -> None:
    """Remove deselected items from collection and notify pytest."""
    if not deselected:
        return

    unique_deselected: list[Any] = []
    seen_ids: set[int] = set()
    for item in deselected:
        item_id = id(item)
        if item_id in seen_ids:
            continue
        seen_ids.add(item_id)
        unique_deselected.append(item)

    deselected_ids = {id(item) for item in unique_deselected}
    items[:] = [item for item in items if id(item) not in deselected_ids]
    config.hook.pytest_deselected(items=unique_deselected)


def _deselect_unselected_test_types(
    config: Any,
    items: list[Any],
    *,
    run_integration: bool,
    run_end_to_end: bool,
    run_benchmark: bool,
) -> None:
    """Deselect tests outside explicitly requested test categories."""
    requested_markers: set[str] = set()
    if run_integration:
        requested_markers.add("integration")
    if run_end_to_end:
        requested_markers.add("end_to_end")
    if run_benchmark:
        requested_markers.add("benchmark")

    if not requested_markers:
        return

    deselected = [
        item for item in items if not any(marker in item.keywords for marker in requested_markers)
    ]
    _deselect_items(config, items, deselected)


def _apply_explicit_deselect_flags(
    config: Any, items: list[Any], *, skip_integration: bool, skip_end_to_end: bool
) -> None:
    """Apply explicit command-line deselect flags."""
    deselected: list[Any] = []
    if skip_integration:
        deselected.extend(item for item in items if "integration" in item.keywords)

    if skip_end_to_end:
        deselected.extend(item for item in items if "end_to_end" in item.keywords)

    _deselect_items(config, items, deselected)


def _apply_device_filter(items: list[Any], *, device_option: str) -> None:
    """Filter tests by requested device type."""
    if device_option == "all":
        return

    for item in items:
        if device_option == "cpu" and ("gpu" in item.keywords or "tpu" in item.keywords):
            item.add_marker(pytest.mark.skip(reason="test not selected for cpu"))
        elif device_option == "gpu" and "tpu" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="test not selected for gpu"))
        elif device_option == "tpu":
            if "gpu" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="test not selected for tpu"))
            if "tpu" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="TPU tests skipped for stability"))


def _skip_tpu_tests_unconditionally(items: list[Any]) -> None:
    """Always skip TPU tests to avoid unstable runtime crashes."""
    skip_tpu = pytest.mark.skip(reason="TPU tests skipped for stability")
    for item in items:
        if "tpu" in item.keywords:
            item.add_marker(skip_tpu)


# Skip tests based on command-line options
def pytest_collection_modifyitems(config, items):
    """Skip tests based on command-line options."""
    run_integration = config.getoption("--integration")
    run_end_to_end = config.getoption("--end-to-end")
    run_benchmark = config.getoption("--benchmark")
    skip_integration = config.getoption("--no-integration")
    skip_end_to_end = config.getoption("--no-end-to-end")
    device_option = config.getoption("--device")

    _deselect_unselected_test_types(
        config,
        items,
        run_integration=run_integration,
        run_end_to_end=run_end_to_end,
        run_benchmark=run_benchmark,
    )
    _apply_explicit_deselect_flags(
        config,
        items,
        skip_integration=skip_integration,
        skip_end_to_end=skip_end_to_end,
    )

    # Note: Benchmarks are now run by default to ensure complete testing
    # They can still be explicitly excluded with pytest -m "not benchmark"
    _apply_device_filter(items, device_option=device_option)
    _skip_tpu_tests_unconditionally(items)


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
    del request
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
