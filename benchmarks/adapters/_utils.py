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


# ---------------------------------------------------------------------------
# Compute-heavy transforms for production-realistic benchmarks (HCV-1, HPC-1)
# ---------------------------------------------------------------------------
# These implement the standard SSL/ImageNet augmentation pipeline in pure
# numpy (CPU). Datarax uses GPU-JIT-compiled JAX versions. The performance
# gap between numpy-CPU and JAX-GPU is what the heavy benchmarks measure.


def random_resized_crop(
    arr: np.ndarray,
    target_h: int = 224,
    target_w: int = 224,
    scale_low: float = 0.08,
    scale_high: float = 1.0,
) -> np.ndarray:
    """Random crop + bilinear resize to target size (numpy CPU)."""
    h, w = arr.shape[0], arr.shape[1]
    area = h * w
    scale = np.random.uniform(scale_low, scale_high)
    crop_area = int(area * scale)
    aspect = np.exp(np.random.uniform(-np.log(4 / 3), np.log(4 / 3)))
    crop_w = min(int(np.sqrt(crop_area * aspect)), w)
    crop_h = min(int(np.sqrt(crop_area / aspect)), h)
    x0 = np.random.randint(0, max(w - crop_w, 1) + 1)
    y0 = np.random.randint(0, max(h - crop_h, 1) + 1)
    cropped = arr[y0 : y0 + crop_h, x0 : x0 + crop_w]

    # Bilinear resize via nearest-neighbor (fast numpy approximation)
    row_idx = (np.arange(target_h) * crop_h / target_h).astype(int)
    col_idx = (np.arange(target_w) * crop_w / target_w).astype(int)
    row_idx = np.clip(row_idx, 0, crop_h - 1)
    col_idx = np.clip(col_idx, 0, crop_w - 1)
    return cropped[np.ix_(row_idx, col_idx)].astype(arr.dtype)


def random_horizontal_flip(arr: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Flip image horizontally with probability p."""
    if np.random.random() < p:
        return np.ascontiguousarray(arr[:, ::-1])
    return arr


def color_jitter(
    arr: np.ndarray,
    brightness: float = 0.4,
    contrast: float = 0.4,
    saturation: float = 0.4,
) -> np.ndarray:
    """Per-channel brightness/contrast/saturation jitter."""
    result = arr.astype(np.float32)
    # Brightness
    result = result + np.random.uniform(-brightness, brightness)
    # Contrast
    mean = result.mean()
    factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
    result = (result - mean) * factor + mean
    # Saturation (approximate: blend with grayscale)
    if result.ndim == 3 and result.shape[-1] == 3:
        gray = result.mean(axis=-1, keepdims=True)
        sat_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
        result = gray + (result - gray) * sat_factor
    return np.clip(result, 0, 255).astype(arr.dtype)


def gaussian_blur_np(
    arr: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> np.ndarray:
    """5x5 Gaussian blur via separable convolution (numpy CPU)."""
    del kernel_size
    from scipy.ndimage import gaussian_filter

    if arr.ndim == 3:
        # Apply per-channel
        return gaussian_filter(arr.astype(np.float32), sigma=(sigma, sigma, 0)).astype(arr.dtype)
    return gaussian_filter(arr.astype(np.float32), sigma=sigma).astype(arr.dtype)


def random_solarize(arr: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Invert pixels above threshold."""
    if np.random.random() < 0.5:
        mask = arr >= threshold
        result = arr.copy()
        result[mask] = 255 - result[mask]
        return result
    return arr


def random_grayscale(arr: np.ndarray, p: float = 0.2) -> np.ndarray:
    """Convert to grayscale with probability p (replicate across channels)."""
    if arr.ndim == 3 and arr.shape[-1] == 3 and np.random.random() < p:
        gray = arr.mean(axis=-1, keepdims=True).astype(arr.dtype)
        return np.broadcast_to(gray, arr.shape).copy()
    return arr


# ---------------------------------------------------------------------------
# NLP / Tabular transforms for heavy benchmark scenarios (HNLP-1, HTAB-1)
# ---------------------------------------------------------------------------


def log_transform(arr: np.ndarray) -> np.ndarray:
    """Element-wise log1p transform for dense features."""
    return np.log1p(np.abs(arr.astype(np.float32))).astype(arr.dtype)


def hash_embedding_index(
    arr: np.ndarray,
    num_buckets: int = 10000,
) -> np.ndarray:
    """Hash integers into embedding bucket indices."""
    return (np.abs(arr) % num_buckets).astype(np.int32)


def create_causal_mask(arr: np.ndarray) -> np.ndarray:
    """Create a lower-triangular causal attention mask.

    Input shape: (seq_len,) token IDs -> output shape: (seq_len, seq_len).
    """
    seq_len = arr.shape[-1] if arr.ndim > 1 else arr.shape[0]
    return np.tril(np.ones((seq_len, seq_len), dtype=np.float32))


def create_attention_mask(arr: np.ndarray, pad_id: int = 0) -> np.ndarray:
    """Create binary attention mask (1 for real tokens, 0 for padding)."""
    return (arr != pad_id).astype(np.float32)


def pad_to_max_len(arr: np.ndarray, max_len: int = 512, pad_id: int = 0) -> np.ndarray:
    """Pad a 1-D token sequence to max_len."""
    if arr.shape[0] >= max_len:
        return arr[:max_len]
    pad_width = max_len - arr.shape[0]
    return np.pad(arr, (0, pad_width), constant_values=pad_id)


# ---------------------------------------------------------------------------
# Standard transform registry (DRY: shared by numpy-based adapters)
# ---------------------------------------------------------------------------
# Maps transform names to array-level functions. Adapters that support the
# full set import this directly; adapters supporting a subset can pick keys.
# Framework-specific adapters (tf.data, DALI, Datarax) define their own.

STANDARD_TRANSFORMS: dict[str, Any] = {
    "Normalize": normalize_uint8,
    "CastToFloat32": cast_to_float32,
    "GaussianNoise": gaussian_noise,
    "RandomBrightness": random_brightness,
    "RandomScale": random_scale,
    "RandomResizedCrop": random_resized_crop,
    "RandomHorizontalFlip": random_horizontal_flip,
    "ColorJitter": color_jitter,
    "GaussianBlur": gaussian_blur_np,
    "RandomSolarize": random_solarize,
    "RandomGrayscale": random_grayscale,
    "LogTransform": log_transform,
    "CreateAttentionMask": create_attention_mask,
    "CausalMaskGeneration": create_causal_mask,
}

# Minimal subset used by adapters that only support basic transforms
BASIC_TRANSFORMS: dict[str, Any] = {
    "Normalize": normalize_uint8,
    "CastToFloat32": cast_to_float32,
}


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
