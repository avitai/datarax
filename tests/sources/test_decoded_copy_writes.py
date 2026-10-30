"""How a decoded copy reaches disk: reserved files, the durability order, and a reader-safe clear.

Every ``.partial`` file is created fresh and reserved to its full size with
``os.posix_fallocate`` before a row is written, so a full disk is refused in the writing process
instead of killing a writer with ``SIGBUS`` later. A finished copy's files are fsynced once each,
renamed, the directory fsynced, and the manifest written last through a fsynced temporary file;
clearing an old copy unlinks its manifest first, so a reader starting meanwhile is refused.
These tests check that order; power-loss safety is not claimed.
"""

from __future__ import annotations

import contextlib
import errno
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import tests.test_common.decoded_copies as decoded_copies
from datarax.sources import open_decoded_copy
from datarax.sources.decoded_copy import _files
from datarax.sources.decoded_copy._files import begin_copy, clear_copy
from datarax.sources.decoded_copy._layout import LayoutColumn
from datarax.sources.decoded_copy._units import source_layout
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from tests.test_common.decoded_copies import IDENTITY, write_copy
from tests.test_common.small_filesystem import run_on_small_filesystem


_UNIT = 4


def _source(records: int = 13) -> MemorySource:
    return MemorySource(
        MemorySourceConfig(),
        [
            {"x": np.full(3, i, np.float32), "y": np.int64(i), "name": f"r{i}"}
            for i in range(records)
        ],
    )


def _path_of(fd: int) -> str:
    return os.readlink(f"/proc/self/fd/{fd}")


class _Events:
    """The copy's file operations in order, each with whether a manifest existed at that moment."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.log: list[tuple[str, str, bool]] = []

    def add(self, kind: str, path: str) -> None:
        name = Path(path).name if Path(path) != self.directory else "<dir>"
        self.log.append((kind, name, (self.directory / "manifest.json").exists()))

    def index(self, kind: str, name: str) -> int:
        return next(i for i, (k, n, _) in enumerate(self.log) if (k, n) == (kind, name))

    def all(self, kind: str) -> list[str]:
        return [n for k, n, _ in self.log if k == kind]


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Events:
    """Records fsync, rename, replace, unlink and each finished unit in the writing process."""
    if not Path("/proc/self/fd").is_dir():
        pytest.skip("names an fsynced descriptor through /proc, which this platform lacks")
    found = _Events(tmp_path / "copy")
    fsync, rename, replace, unlink = os.fsync, os.rename, os.replace, os.unlink

    def spy(kind: str, call: Callable[..., Any], path_of: Callable[..., str]) -> Callable[..., Any]:
        def recorded(*args: Any, **kwargs: Any) -> Any:
            found.add(kind, path_of(*args))
            return call(*args, **kwargs)

        return recorded

    monkeypatch.setattr(os, "fsync", spy("fsync", fsync, _path_of))
    monkeypatch.setattr(os, "rename", spy("rename", rename, lambda src, dst: str(src)))
    monkeypatch.setattr(os, "replace", spy("replace", replace, lambda src, dst: str(dst)))
    monkeypatch.setattr(os, "unlink", spy("unlink", unlink, lambda path, **_: str(path)))
    write_unit = decoded_copies.write_unit

    def unit(*args: Any, **kwargs: Any) -> Any:
        written = write_unit(*args, **kwargs)
        found.add("unit", str(found.directory))
        return written

    monkeypatch.setattr(decoded_copies, "write_unit", unit)
    return found


class TestTheDurabilityOrder:
    def test_each_partial_is_fsynced_once_after_the_last_unit_then_renamed(
        self, events: _Events
    ) -> None:
        write_copy(_source(), events.directory, identity=IDENTITY, unit_records=_UNIT)

        last_unit = max(i for i, (k, _, _) in enumerate(events.log) if k == "unit")
        partials = sorted(n for n in events.all("fsync") if n.endswith(".partial"))
        assert sorted(events.all("rename")) == partials
        assert len(partials) == len(set(partials)) == 3, "two columns and the provenance"
        for name in partials:
            assert last_unit < events.index("fsync", name) < events.index("rename", name)

    def test_the_manifest_is_written_last_through_a_fsynced_temporary_file(
        self, events: _Events
    ) -> None:
        write_copy(_source(), events.directory, identity=IDENTITY, unit_records=_UNIT)

        last_rename = max(i for i, (k, _, _) in enumerate(events.log) if k == "rename")
        dir_syncs = [i for i, (k, n, _) in enumerate(events.log) if (k, n) == ("fsync", "<dir>")]
        tmp_sync = events.index("fsync", "manifest.json.tmp")
        replaced = events.index("replace", "manifest.json")
        assert dir_syncs[0] > last_rename
        assert dir_syncs[0] < tmp_sync < replaced < dir_syncs[-1]
        assert dir_syncs[-1] == len(events.log) - 1, "the directory is fsynced last"
        assert not any(had for _, _, had in events.log[:replaced]), "no manifest before the end"

    def test_clearing_a_copy_unlinks_the_manifest_and_syncs_before_any_data_file(
        self, events: _Events
    ) -> None:
        write_copy(_source(), events.directory, identity=IDENTITY, unit_records=_UNIT)
        events.log.clear()

        clear_copy(events.directory)

        manifest = events.index("unlink", "manifest.json")
        synced = events.index("fsync", "<dir>")
        data = [
            i for i, (k, n, _) in enumerate(events.log) if k == "unlink" and n != "manifest.json"
        ]
        assert len(data) == 3, "two columns and the provenance"
        assert manifest < synced < min(data)
        assert {p.name for p in events.directory.iterdir()} <= {".lock"}

    def test_clearing_refuses_a_directory_holding_a_file_the_format_does_not_name(
        self, tmp_path: Path
    ) -> None:
        directory = tmp_path / "copy"
        write_copy(_source(), directory, identity=IDENTITY, unit_records=_UNIT)
        (directory / "notes.txt").write_text("mine")

        with pytest.raises(ValueError, match=r"notes\.txt"):
            clear_copy(directory)
        assert (directory / "manifest.json").exists(), "nothing is unlinked before the refusal"

    def test_a_copy_is_not_begun_over_a_finished_one(self, tmp_path: Path) -> None:
        directory = tmp_path / "copy"
        layout = write_copy(_source(), directory, identity=IDENTITY, unit_records=_UNIT)

        with pytest.raises(FileExistsError, match="manifest"):
            with begin_copy(directory, layout, unit_records=_UNIT):
                pass


class TestTheReservation:
    def test_reserves_each_partial_to_its_full_size(self, tmp_path: Path) -> None:
        layout = source_layout(_source(4096))

        with begin_copy(tmp_path / "copy", layout, unit_records=_UNIT) as files:
            for name in files.column_files():
                stat = (tmp_path / "copy" / f"{name}.partial").stat()
                assert stat.st_blocks * 512 >= stat.st_size > 4096 * 8 - 1

    def test_a_full_disk_is_refused_naming_its_terms_and_leaves_no_partial(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        calls: list[int] = []
        reserve = os.posix_fallocate

        def full(fd: int, offset: int, length: int) -> None:
            calls.append(length)
            if len(calls) == 2:
                raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
            reserve(fd, offset, length)

        monkeypatch.setattr(os, "posix_fallocate", full)
        layout = source_layout(_source())
        directory = tmp_path / "copy"

        with pytest.raises(OSError, match=r"copy.*y.*needs \d+ bytes.*\d+ bytes free") as raised:
            with begin_copy(directory, layout, unit_records=_UNIT):
                pytest.fail("the body runs only once every column is reserved")
        assert raised.value.errno == errno.ENOSPC
        assert len(calls) == 2
        assert not list(directory.glob("*.partial"))

    def test_without_posix_fallocate_a_copy_is_refused_naming_the_platform(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delattr(os, "posix_fallocate")

        with pytest.raises(OSError, match=sys.platform):
            with begin_copy(tmp_path / "copy", source_layout(_source()), unit_records=4):
                pass

    @pytest.mark.slow
    def test_on_a_real_small_filesystem_the_refusal_frees_what_it_reserved(
        self, tmp_path: Path
    ) -> None:
        """A tmpfs of 1 MiB in a user namespace: free space after the refusal equals before."""
        mount = tmp_path / "small"
        mount.mkdir()
        run = run_on_small_filesystem(
            mount,
            f"""
            import os
            from datarax.sources.decoded_copy._files import begin_copy
            from datarax.sources.decoded_copy._layout import ColumnLayout, LayoutColumn
            layout = ColumnLayout(
                columns=(LayoutColumn(name="x", dtype="<f4", shape=(1024,)),), length=1024
            )
            def free():
                s = os.statvfs({str(mount)!r})
                return s.f_bavail * s.f_frsize
            before = free()
            try:
                with begin_copy({str(mount / "copy")!r}, layout, unit_records=256):
                    print("written")
            except OSError as error:
                print("refused", error.errno, before, free(), str(error)[:200])
            """,
        )
        words = run.stdout.split()
        assert words[:2] == ["refused", str(errno.ENOSPC)], run.stdout + run.stderr
        assert words[2] == words[3], "the refused reservation's blocks are freed"


def test_a_reader_started_before_the_manifest_is_refused(tmp_path: Path) -> None:
    layout = source_layout(_source())
    directory = tmp_path / "copy"

    with begin_copy(directory, layout, unit_records=_UNIT):
        with pytest.raises(FileNotFoundError, match="manifest"):
            open_decoded_copy(directory)


def test_the_format_names_its_lock_file() -> None:
    assert _files.is_copy_file(".lock")
    assert _files.is_copy_file("manifest.json")
    assert not _files.is_copy_file("notes.txt")
    assert LayoutColumn(name="a/b", dtype="<f4", shape=(2,)).row_bytes == 8


class TestClosingAfterAFailure:
    """A writer unwinding a failed build keeps the build's error; its close's error is a note."""

    @staticmethod
    def _failing_close() -> contextlib.ExitStack:
        stack = contextlib.ExitStack()

        def close() -> None:
            raise RuntimeError("write() failed: No space left on device")

        stack.callback(close)
        return stack

    def test_the_build_s_error_propagates_with_the_close_s_as_a_note(self) -> None:
        cause = OSError(errno.ENOSPC, "no room for the values")

        with pytest.raises(OSError, match="no room for the values") as raised:
            try:
                raise cause
            except OSError as error:
                _files.close_files(self._failing_close(), (type(error), error, error.__traceback__))
                raise

        assert raised.value is cause
        assert any("No space left on device" in note for note in raised.value.__notes__)

    def test_with_no_error_propagating_a_failing_close_raises(self) -> None:
        with pytest.raises(RuntimeError, match="No space left on device"):
            _files.close_files(self._failing_close(), (None, None, None))
