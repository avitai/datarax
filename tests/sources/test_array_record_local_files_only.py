"""Tests for ``local_files_only`` on ``ArrayRecordSourceModule``.

ArrayRecord sources are intrinsically local-only (no network downloads). The
``local_files_only`` flag therefore behaves as a path-existence pre-check:
when True and any path is missing, the source raises a clear
``FileNotFoundError`` instead of falling through to Grain's lower-level error.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from flax import nnx

from datarax.sources.array_record_source import (
    ArrayRecordSourceConfig,
    ArrayRecordSourceModule,
)


def _stub_grain_source() -> MagicMock:
    """Return a minimal Grain ``ArrayRecordDataSource`` substitute."""
    source = MagicMock()
    source.__len__ = MagicMock(return_value=8)
    return source


def test_array_record_local_files_only_raises_when_path_missing(tmp_path: Path) -> None:
    """Pre-checks every path; missing path → ``FileNotFoundError`` with path context."""
    missing = tmp_path / "missing.arrayrecord"

    config = ArrayRecordSourceConfig(local_files_only=True)
    with pytest.raises(FileNotFoundError) as exc_info:
        ArrayRecordSourceModule(config, paths=str(missing), rngs=nnx.Rngs(0))

    msg = str(exc_info.value)
    assert str(missing) in msg
    assert "local_files_only" in msg


def test_array_record_local_files_only_passes_when_paths_present(tmp_path: Path) -> None:
    """When the path exists, the pre-check passes and Grain is invoked."""
    archive = tmp_path / "data.arrayrecord"
    archive.touch()

    with patch(
        "datarax.sources.array_record_source.grain.sources.ArrayRecordDataSource",
        return_value=_stub_grain_source(),
    ):
        config = ArrayRecordSourceConfig(local_files_only=True)
        # Should not raise.
        ArrayRecordSourceModule(config, paths=str(archive), rngs=nnx.Rngs(0))


def test_array_record_local_files_only_default_skips_check(tmp_path: Path) -> None:
    """Default ``local_files_only=False`` does NOT pre-check (defers to Grain)."""
    missing = tmp_path / "nonexistent.arrayrecord"

    with patch(
        "datarax.sources.array_record_source.grain.sources.ArrayRecordDataSource",
        return_value=_stub_grain_source(),
    ):
        config = ArrayRecordSourceConfig()  # local_files_only=False by default
        # Should not raise — Grain itself handles the missing-file path
        # (which may surface a different error, but that is Grain's contract).
        ArrayRecordSourceModule(config, paths=str(missing), rngs=nnx.Rngs(0))
