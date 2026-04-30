"""Tests for ``LocalFilesOnlyMixin`` — air-gapped / no-network source loading.

Sources that download external archives (HuggingFace, TFDS, ArrayRecord)
must support a uniform ``local_files_only`` flag that, when True, refuses to
hit the network and instead either loads from the local cache or raises a
clear ``FileNotFoundError`` with the resolved cache path so the user can
populate it offline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from datarax.core.data_source import LocalFilesOnlyMixin


class _FakeSource(LocalFilesOnlyMixin):
    """Minimal subclass that exercises the mixin's contract."""

    def __init__(self, *, local_files_only: bool) -> None:
        self.local_files_only = local_files_only


def test_local_files_only_raises_when_archive_missing(tmp_path: Path) -> None:
    """With ``local_files_only=True`` and no cache, ``_check_local_cache`` raises.

    The error must mention the resolved path so the user knows where to drop
    the archive (instead of a generic "file not found").
    """
    source = _FakeSource(local_files_only=True)
    expected = tmp_path / "missing_archive.zip"

    with pytest.raises(FileNotFoundError) as exc_info:
        source._check_local_cache([expected], dataset_name="my_dataset")

    msg = str(exc_info.value)
    assert "my_dataset" in msg
    assert str(expected) in msg
    assert "local_files_only" in msg


def test_local_files_only_passes_when_cache_present(tmp_path: Path) -> None:
    """With ``local_files_only=True`` and a populated cache, no exception is raised."""
    source = _FakeSource(local_files_only=True)
    archive = tmp_path / "data.zip"
    archive.touch()

    # Should not raise.
    source._check_local_cache([archive], dataset_name="my_dataset")


def test_local_files_only_false_skips_check(tmp_path: Path) -> None:
    """With ``local_files_only=False``, the mixin does NOT raise on missing cache.

    The source is allowed to fall through to its normal download path.
    """
    source = _FakeSource(local_files_only=False)
    missing = tmp_path / "nonexistent.zip"

    # Should not raise — the source decides for itself whether to download.
    source._check_local_cache([missing], dataset_name="my_dataset")
