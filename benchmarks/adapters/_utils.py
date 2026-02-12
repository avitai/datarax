"""Shared utilities for benchmark adapters.

Extracts common patterns used across multiple adapters:
- Array-level transforms (Grain, Datarax — both use numpy-compatible dtypes)
- Temp directory setup (FFCV, MosaicML, WebDataset, LitData, Energon, Deep Lake)
- TAR shard writing with numpy serialization (WebDataset, Energon)
"""

from __future__ import annotations

import io
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.adapters.base import ScenarioConfig


# ---------------------------------------------------------------------------
# Array-level transform functions (DRY: shared by Grain + Datarax adapters)
# ---------------------------------------------------------------------------
# These work on both numpy and JAX arrays since jnp.uint8 == np.uint8
# and both support .astype(). TF/DALI adapters use framework-specific ops.
#
# JIT compatibility: These functions are safe inside jax.jit / nnx.jit
# because they only use .dtype (a static property known at trace time)
# and .astype() (available on both numpy and JAX arrays). However,
# apply_to_dict uses a Python dict comprehension and is NOT JIT-safe.
# Inside JIT, use jax.tree.map(normalize_uint8, pytree) instead.


def normalize_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize uint8 array to [0, 1] float32. Non-uint8 arrays pass through.

    Safe inside ``jax.jit``: ``.dtype`` is static and ``.astype()`` traces.
    """
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return arr


def cast_to_float32(arr: np.ndarray) -> np.ndarray:
    """Cast array to float32.

    Safe inside ``jax.jit``: ``.astype()`` traces on JAX arrays.
    """
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Stochastic array-level transforms (DRY: shared by numpy-based adapters)
# ---------------------------------------------------------------------------
# Used by PyTorch, SPDL, Grain adapters. Each uses numpy's global RNG
# for per-batch randomness (idiomatic for numpy-based frameworks).


def gaussian_noise(arr: np.ndarray, std: float = 0.05) -> np.ndarray:
    """Add Gaussian noise with given standard deviation."""
    return (arr + np.random.normal(0, std, arr.shape)).astype(arr.dtype)


def random_brightness(
    arr: np.ndarray,
    low: float = -0.2,
    high: float = 0.2,
) -> np.ndarray:
    """Adjust brightness by a random uniform delta."""
    return (arr + np.random.uniform(low, high)).astype(arr.dtype)


def random_scale(
    arr: np.ndarray,
    low: float = 0.8,
    high: float = 1.2,
) -> np.ndarray:
    """Scale values by a random uniform factor."""
    return (arr * np.random.uniform(low, high)).astype(arr.dtype)


def apply_to_dict(
    fn: Any,
    element: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Apply a per-array transform to all values in a dict.

    NOT safe inside ``jax.jit`` — use ``jax.tree.map(fn, pytree)`` instead.
    """
    return {k: fn(v) for k, v in element.items()}


def setup_temp_dir(config: ScenarioConfig, subdir: str) -> tuple[Path, Any]:
    """Get or create a temp directory for adapter-specific data.

    If ``config.extra["tmp_dir"]`` is set, uses that path (no cleanup
    needed).  Otherwise creates a ``TemporaryDirectory`` that the caller
    must clean up via ``cleanup_temp_dir()``.

    Returns:
        ``(data_path, tmp_dir_obj)`` where ``tmp_dir_obj`` is ``None``
        when using a user-supplied path.
    """
    tmp_dir = config.extra.get("tmp_dir")
    if tmp_dir:
        path = Path(tmp_dir) / subdir
        path.mkdir(parents=True, exist_ok=True)
        return path, None
    tmp_obj = tempfile.TemporaryDirectory()
    path = Path(tmp_obj.name) / subdir
    path.mkdir()
    return path, tmp_obj


def cleanup_temp_dir(tmp_dir: Any) -> None:
    """Clean up a TemporaryDirectory if not None."""
    if tmp_dir is not None:
        tmp_dir.cleanup()


def write_numpy_tar(data: dict[str, np.ndarray], tar_path: Path) -> None:
    """Write a dict of numpy arrays to a TAR file with ``.npy`` serialization.

    Each sample ``i`` produces one TAR member per key:
    ``{i:06d}.{key}.npy`` containing ``np.save(arr[i])``.

    Used by WebDataset and Energon adapters for TAR shard conversion.
    """
    primary_key = next(iter(data))
    n = len(data[primary_key])

    with tarfile.open(str(tar_path), "w") as tar:
        for i in range(n):
            for key, arr in data.items():
                buf = io.BytesIO()
                np.save(buf, arr[i])
                buf.seek(0)
                info = tarfile.TarInfo(name=f"{i:06d}.{key}.npy")
                info.size = buf.getbuffer().nbytes
                tar.addfile(info, buf)
