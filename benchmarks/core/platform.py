"""JAX platform initialization and resource detection utilities.

Provides device counting, memory querying, and scenario memory gating
for both CPU (system RAM) and GPU (device VRAM) backends.

Design ref: Section 6.4.2 of the benchmark report.
"""

from __future__ import annotations

import functools
import math
import subprocess

import jax


# ---------------------------------------------------------------------------
# Device utilities
# ---------------------------------------------------------------------------


def init_platform(backend: str | None = None) -> str:
    """Initialize JAX with the specified backend.

    Args:
        backend: JAX backend to use ('cpu', 'gpu', 'tpu').
            If None, uses JAX's default selection.

    Returns:
        The active backend name.
    """
    if backend is not None:
        jax.config.update("jax_platform_name", backend)
    return jax.default_backend()


def get_device_count() -> int:
    """Return the number of available JAX devices."""
    return jax.device_count()


def get_local_device_count() -> int:
    """Return the number of local JAX devices."""
    return jax.local_device_count()


def required_devices(min_count: int):
    """Skip test/scenario if fewer than min_count devices available.

    Args:
        min_count: Minimum number of JAX devices required.

    Returns:
        Decorator that skips the test with pytest.skip if insufficient devices.
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if get_device_count() < min_count:
                import pytest

                pytest.skip(f"Requires {min_count} devices, found {get_device_count()}")
            return fn(*args, **kwargs)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Memory detection
# ---------------------------------------------------------------------------


def _get_system_ram_mb() -> float:
    """Query available system RAM in MB via /proc/meminfo or psutil."""
    # Try /proc/meminfo first (Linux, no deps)
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024  # kB → MB
    except (FileNotFoundError, OSError):
        pass

    # Fall back to psutil
    try:
        import psutil

        return psutil.virtual_memory().available / (1024**2)
    except ImportError:
        pass

    # Last resort: total memory from os.sysconf (POSIX)
    try:
        import os

        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return (pages * page_size) / (1024**2)
    except (AttributeError, ValueError, OSError):
        pass

    # Cannot determine — return a conservative 4 GB
    return 4096.0


def _get_gpu_memory_mb() -> float | None:
    """Query available GPU memory in MB via nvidia-smi.

    Returns:
        Available GPU memory in MB, or None if no GPU / nvidia-smi unavailable.
    """
    try:
        output = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        # May return multiple lines for multiple GPUs — use the first
        free_mb = float(output.splitlines()[0])
        return free_mb
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


def get_available_memory_mb(backend: str | None = None) -> float:
    """Query available memory in MB for the active or specified backend.

    For GPU backend, returns the minimum of system RAM and GPU VRAM
    (since data is generated on CPU then transferred). For CPU/TPU,
    returns system RAM.

    Args:
        backend: Override backend ('cpu', 'gpu', 'tpu').
            If None, uses the current JAX default.

    Returns:
        Available memory in MB.
    """
    active_backend = backend or jax.default_backend()

    if active_backend == "gpu":
        gpu_mb = _get_gpu_memory_mb()
        if gpu_mb is not None:
            # Data lives on both CPU (generation) and GPU (compute),
            # so the bottleneck is whichever is smaller
            sys_mb = _get_system_ram_mb()
            return min(sys_mb, gpu_mb)

    return _get_system_ram_mb()


# ---------------------------------------------------------------------------
# Memory estimation and gating
# ---------------------------------------------------------------------------


def estimate_scenario_memory_mb(
    dataset_size: int,
    element_shape: tuple[int, ...],
    dtype_bytes: int = 4,
) -> float:
    """Estimate dataset memory in MB.

    Computes: ``dataset_size * prod(element_shape) * dtype_bytes / 1024^2``.

    Args:
        dataset_size: Number of elements in the dataset.
        element_shape: Shape of each element.
        dtype_bytes: Bytes per scalar (default 4 for float32).

    Returns:
        Estimated memory in MB.
    """
    elements_per_sample = math.prod(element_shape)
    total_bytes = dataset_size * elements_per_sample * dtype_bytes
    return total_bytes / (1024**2)


def can_run_scenario(
    variant: ScenarioVariant,  # noqa: F821 — forward ref to avoid circular import
    safety_factor: float = 2.5,
    backend: str | None = None,
) -> bool:
    """Check if a scenario variant can run given available memory.

    The safety factor accounts for JAX internal copies, pipeline buffers,
    and batch materialization during execution.

    Args:
        variant: ScenarioVariant to check.
        safety_factor: Multiplier on estimated memory (default 2.5).
            E.g., a 400 MB dataset with factor 2.5 needs 1000 MB available.
        backend: Override backend for memory query.

    Returns:
        True if the variant's estimated memory fits within available resources.
    """
    estimated_mb = estimate_scenario_memory_mb(
        dataset_size=variant.config.dataset_size,
        element_shape=variant.config.element_shape,
    )
    available_mb = get_available_memory_mb(backend=backend)
    return estimated_mb * safety_factor < available_mb
