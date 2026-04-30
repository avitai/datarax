"""Tests for ``datarax.utils.cache`` — two-tier raw/+processed/ cache helpers.

These helpers centralize the cache-resolution logic that every dataset adapter
(HuggingFace, TFDS, ArrayRecord, future wearable datasets) needs, so each
source doesn't reinvent download/extract/cache-ready logic.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from datarax.utils.cache import (
    CacheLayout,
    ensure_zip_member_extracted,
    make_layout,
    processed_cache_ready,
    resolve_cache_dir,
)


def test_cache_layout_creates_raw_and_processed(tmp_path: Path) -> None:
    """``make_layout`` returns a ``CacheLayout`` with raw/ and processed/ subpaths."""
    layout = make_layout(tmp_path)

    assert isinstance(layout, CacheLayout)
    assert layout.base == tmp_path
    assert layout.raw == tmp_path / "raw"
    assert layout.processed == tmp_path / "processed"


def test_resolve_cache_dir_uses_default_when_none(tmp_path: Path, monkeypatch) -> None:
    """When user passes ``None``, resolver picks a default location under XDG_CACHE_HOME."""
    fake_xdg = tmp_path / "xdg_cache"
    monkeypatch.setenv("XDG_CACHE_HOME", str(fake_xdg))

    resolved = resolve_cache_dir(None, default_name="my_dataset")

    assert resolved == fake_xdg / "my_dataset"


def test_resolve_cache_dir_honors_user_path(tmp_path: Path) -> None:
    """When user passes a path, the resolver uses it verbatim (after Path conversion)."""
    user_dir = tmp_path / "my_custom_cache"

    resolved = resolve_cache_dir(user_dir, default_name="ignored")

    assert resolved == user_dir
    assert isinstance(resolved, Path)


def test_processed_cache_ready_returns_false_when_missing(tmp_path: Path) -> None:
    """``processed_cache_ready`` returns False when any required file is missing."""
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "X_train.npy").touch()
    # y_train.npy is missing

    ready = processed_cache_ready(processed, ["X_train.npy", "y_train.npy"])

    assert ready is False


def test_processed_cache_ready_returns_true_when_all_files_present(tmp_path: Path) -> None:
    """``processed_cache_ready`` returns True only when every required file exists."""
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "X_train.npy").touch()
    (processed / "y_train.npy").touch()
    (processed / "X_test.npy").touch()

    ready = processed_cache_ready(processed, ["X_train.npy", "y_train.npy", "X_test.npy"])

    assert ready is True


def test_ensure_zip_member_extracted_extracts_then_skips(tmp_path: Path) -> None:
    """First call extracts the member; second call is a no-op."""
    archive = tmp_path / "data.zip"
    extract_root = tmp_path / "out"

    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("inner/file.txt", "payload")

    target = ensure_zip_member_extracted(
        archive, extract_root, "inner/file.txt", target_name="extracted.txt"
    )
    assert target == extract_root / "extracted.txt"
    assert target.read_text() == "payload"

    # Second call: mtime should not change (idempotent skip).
    first_mtime = target.stat().st_mtime_ns
    target2 = ensure_zip_member_extracted(
        archive, extract_root, "inner/file.txt", target_name="extracted.txt"
    )
    assert target2 == target
    assert target.stat().st_mtime_ns == first_mtime
