"""Test configuration for Datarax."""

import os
import platform
import sys
from pathlib import Path
from typing import Any

from absl import flags


# absl refuses to read a flag until flags are parsed, and nothing under pytest ever parses
# them: absl.app.run() does it in normal execution, and pytest does not call it. grain reads
# its own flags at runtime — grain.DataLoader with worker_count > 0 reaches
# --grain_enable_multiprocess_worker_profiling from a worker thread — so the suite must mark
# them parsed itself rather than depend on some import happening to do it first.
#
# This surfaced on the jax 0.11.1 / flax 0.12.9 upgrade: the same test passed on jax 0.10.0 /
# flax 0.12.7 and raised UnparsedFlagAccessError afterwards, with no change to grain (0.2.18
# both sides) or absl-py (2.4.0 both sides). Which code path used to avoid the read is not
# established; marking the flags parsed is correct either way, because an unparsed-flag read
# is a latent failure that only luck was hiding.
flags.FLAGS.mark_as_parsed()

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


# JAX reads JAX_PLATFORMS when it is imported, so the backend is chosen first, by a
# module that imports nothing that loads JAX (test_common's package does). Test helpers
# import through the tests package: tests/ itself on sys.path would make every test
# subpackage a top-level module, and tests/benchmarks would shadow benchmarks.
from tests.jax_test_environment import has_cuda_plugin, resolve_test_environment


# Backend and device emulation must be decided before JAX is imported.
os.environ.update(resolve_test_environment(os.environ, cuda_plugin_available=has_cuda_plugin()))

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

# Add the src directory to the Python path so tests can import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# Register custom markers
def pytest_configure(config):
    """Register custom markers for pytest."""
    config.addinivalue_line("markers", "tfds: mark test as requiring tensorflow_datasets")
    config.addinivalue_line("markers", "hf: mark test as requiring huggingface_datasets")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "end_to_end: mark test as an end-to-end test")
    config.addinivalue_line("markers", "benchmark: mark test as a performance benchmark")


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


# Skip tests based on command-line options
def pytest_collection_modifyitems(config, items):
    """Skip tests based on command-line options."""
    run_integration = config.getoption("--integration")
    run_end_to_end = config.getoption("--end-to-end")
    run_benchmark = config.getoption("--benchmark")
    skip_integration = config.getoption("--no-integration")
    skip_end_to_end = config.getoption("--no-end-to-end")

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
