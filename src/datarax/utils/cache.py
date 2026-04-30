"""Two-tier cache helpers for dataset adapters.

Datasets that download external archives (HuggingFace, TFDS, ArrayRecord,
wearable physiology, etc.) all share the same cache lifecycle: download a raw
archive once, extract or preprocess it into a stable layout, then re-use the
processed artifacts on every subsequent run. Centralizing the layout in this
module prevents each adapter from inventing its own subtly-different cache
directory layout.

Layout
------

::

    <base>/
      raw/        ← downloaded archives, large opaque files
      processed/  ← stable per-source preprocessed outputs (.npy, sharded files,
                    intermediate caches)

The split allows the ``raw/`` archive to be deleted after preprocessing
without invalidating the cached training artifacts in ``processed/``.
"""

from __future__ import annotations

import os
import shutil
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CacheLayout:
    """Two-tier cache structure: raw archives + processed artifacts."""

    base: Path
    raw: Path
    processed: Path


def make_layout(base: Path) -> CacheLayout:
    """Construct a ``CacheLayout`` rooted at ``base``.

    The two subdirectories are not created eagerly; callers create them
    on first write so a read-only audit (e.g., ``processed_cache_ready``)
    does not leave empty directories behind.
    """
    base = Path(base)
    return CacheLayout(base=base, raw=base / "raw", processed=base / "processed")


def resolve_cache_dir(user_path: str | Path | None, *, default_name: str) -> Path:
    """Resolve a user-provided cache dir or fall back to a default.

    Default falls under ``$XDG_CACHE_HOME/<default_name>`` when
    ``XDG_CACHE_HOME`` is set, otherwise ``~/.cache/<default_name>``. This
    matches the convention used by Hugging Face Datasets and TFDS.

    Args:
        user_path: Explicit cache directory, or ``None`` to use the default.
        default_name: Subdirectory name to append to the default root.

    Returns:
        Absolute ``Path`` to the cache directory (does not create it).
    """
    if user_path is not None:
        return Path(user_path)

    xdg = os.environ.get("XDG_CACHE_HOME")
    root = Path(xdg) if xdg else Path.home() / ".cache"
    return root / default_name


def processed_cache_ready(processed: Path, required_files: Iterable[str]) -> bool:
    """Return ``True`` only if every required file exists under ``processed``.

    Used by dataset constructors to short-circuit re-preprocessing when a
    previous run has already populated the processed cache.

    Args:
        processed: Path to the ``processed/`` subdirectory.
        required_files: Names (or relative paths) that must all exist.

    Returns:
        ``True`` iff every file in ``required_files`` exists under ``processed``.
    """
    return all((processed / name).exists() for name in required_files)


def ensure_zip_member_extracted(
    archive: Path,
    extract_root: Path,
    member_name: str,
    *,
    target_name: str | None = None,
) -> Path:
    """Idempotently extract a single zip member to ``extract_root``.

    First call extracts the member to ``extract_root / (target_name or
    Path(member_name).name)``. Subsequent calls are no-ops (returns the
    existing path without re-extraction), so this helper is safe to call
    on every dataset construction.

    Args:
        archive: Path to the source ``.zip`` file.
        extract_root: Directory under which the member will be placed.
        member_name: Path of the member inside the archive.
        target_name: Optional override for the on-disk filename.

    Returns:
        Path to the extracted file.
    """
    target_basename = target_name or Path(member_name).name
    target_path = extract_root / target_basename
    if target_path.exists():
        return target_path

    extract_root.mkdir(parents=True, exist_ok=True)
    with (
        zipfile.ZipFile(archive, "r") as zf,
        zf.open(member_name, "r") as src,
        target_path.open("wb") as dst,
    ):
        shutil.copyfileobj(src, dst)
    return target_path


__all__ = [
    "CacheLayout",
    "ensure_zip_member_extracted",
    "make_layout",
    "processed_cache_ready",
    "resolve_cache_dir",
]
