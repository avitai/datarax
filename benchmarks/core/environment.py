"""System fingerprint capture for benchmark reproducibility.

Based on JAX CI's version printing pattern and MLPerf's
system description requirements.

Design ref: Section 6.4.8 of the benchmark report.
"""

from __future__ import annotations

import platform

# Controlled local system introspection commands (git/nvidia-smi).
import subprocess  # nosec B404
import sys
from datetime import datetime, UTC
from typing import Any

import jax


def capture_environment() -> dict[str, Any]:
    """Capture complete environment for benchmark reproducibility.

    Returns:
        Dictionary containing Python version, JAX version, backend,
        device info, OS info, git commit, and optional GPU/TPU details.
    """
    env: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": _get_git_commit(),
        "python_version": sys.version,
        "jax_version": jax.__version__,
        "platform": {
            "backend": jax.default_backend(),
            "device_count": jax.device_count(),
            "local_device_count": jax.local_device_count(),
            "process_count": jax.process_count(),
            "devices": [str(d) for d in jax.devices()],
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
    }

    # GPU-specific info
    if jax.default_backend() == "gpu":
        gpu_info = _get_gpu_info()
        if gpu_info is not None:
            env["gpu"] = gpu_info

    # TPU-specific info
    if jax.default_backend() == "tpu":
        env["tpu_type"] = str(jax.devices()[0])
        try:
            env["libtpu_version"] = jax.extend.backend.get_backend().platform_version  # type: ignore[reportAttributeAccessIssue]
        except (ImportError, AttributeError):
            pass

    return env


def _get_git_commit() -> str:
    """Get current git commit hash, or 'unknown' if not in a repo."""
    try:
        # Static git command; no user-controlled input.
        return (
            subprocess.check_output(  # nosec B603 B607
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _get_gpu_info() -> str | None:
    """Query nvidia-smi for GPU information."""
    try:
        # Static nvidia-smi query; no user-controlled input.
        return (
            subprocess.check_output(  # nosec B603 B607
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
