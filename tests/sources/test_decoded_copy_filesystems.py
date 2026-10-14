"""Where a decoded copy may be written, and one writer per copy.

Workers write a copy's rows in place through shared memory maps, which is safe on a local disk
and unverified on a networked or FUSE filesystem, so writing there is refused by the target's
filesystem type (``psutil.disk_partitions(all=True)``, the mount whose mountpoint is the longest
prefix of the target's real path); reading a finished copy from such a mount is allowed. One
process writes a copy at a time: :class:`~datarax.sources.decoded_copy._files.CopyLock` holds an
exclusive ``flock`` on ``<copy>/.lock``, which the kernel frees when its holder dies.
"""

from __future__ import annotations

import os
import signal
import subprocess  # nosec B404 - the tests start their own writer processes
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import pytest

from datarax.sources import open_decoded_copy
from datarax.sources.decoded_copy import _files
from datarax.sources.decoded_copy._files import CopyLock, refuse_unsupported_filesystem
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from tests.test_common.decoded_copies import IDENTITY, write_copy


_PARTITION = type(psutil.disk_partitions(all=True)[0])
"""psutil's own record of a mount (``sdiskpart``), as ``disk_partitions`` returns it."""


def _mounts(*entries: tuple[str, str]) -> list[Any]:
    return [_PARTITION(f"dev{i}", point, kind, "rw") for i, (point, kind) in enumerate(entries)]


@pytest.fixture
def mounts(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    found = _mounts(
        ("/", "ext4"),
        ("/data/nfs", "nfs4"),
        ("/data/bucket", "fuse.gcsfuse"),
        ("/data/bucket/local", "ext4"),
        ("/data/plain", "fuse"),
        ("/data/my", "nfs"),
        ("/data/my disk", "ext4"),
    )
    monkeypatch.setattr(_files.psutil, "disk_partitions", lambda all=False: found)  # noqa: A002
    return found


class TestTheFilesystemRefusal:
    @pytest.mark.parametrize(
        ("target", "kind"),
        [
            ("/data/nfs/copy", "nfs4"),
            ("/data/bucket/copy", "fuse.gcsfuse"),
            ("/data/plain/a/b", "fuse"),
            ("/data/my/copy", "nfs"),
        ],
    )
    def test_a_networked_or_fuse_target_is_refused_naming_its_type(
        self, mounts: list[Any], target: str, kind: str
    ) -> None:
        with pytest.raises(ValueError, match=rf"{kind}.*local disk"):
            refuse_unsupported_filesystem(target)

    @pytest.mark.parametrize(
        ("target", "kind"),
        [
            ("/home/user/copy", "ext4"),
            ("/data/bucket/local/copy", "ext4"),
            ("/data/my disk/copy", "ext4"),
            ("/data/nfsish/copy", "ext4"),
        ],
        ids=["root", "longest-prefix-wins", "space-in-mountpoint", "whole-components-only"],
    )
    def test_a_local_target_is_written(self, mounts: list[Any], target: str, kind: str) -> None:
        assert refuse_unsupported_filesystem(target) == kind

    def test_a_finished_copy_on_a_refused_mount_still_opens(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        directory = tmp_path / "copy"
        source = MemorySource(MemorySourceConfig(), {"x": np.arange(10, dtype=np.int32)})
        write_copy(source, directory, identity=IDENTITY, unit_records=4)
        found = _mounts(("/", "ext4"), (str(tmp_path.resolve()), "nfs4"))
        monkeypatch.setattr(_files.psutil, "disk_partitions", lambda all=False: found)  # noqa: A002

        with pytest.raises(ValueError, match="nfs4"):
            refuse_unsupported_filesystem(directory)
        assert len(open_decoded_copy(directory)) == 10

    def test_the_real_mount_table_classifies_dev_shm_as_tmpfs(self) -> None:
        if not Path("/dev/shm").is_dir():  # nosec B108 - only classified
            pytest.skip("this platform has no /dev/shm")
        assert refuse_unsupported_filesystem("/dev/shm/a-copy") == "tmpfs"  # nosec B108


_HOLD = """
import sys, time
from datarax.sources.decoded_copy._files import CopyLock
with CopyLock(sys.argv[1]):
    print("held", flush=True)
    time.sleep(float(sys.argv[2]))
print("released", flush=True)
"""


def _writer(directory: Path, seconds: float) -> subprocess.Popen[str]:
    return subprocess.Popen(  # nosec B603 - fixed arguments
        [sys.executable, "-c", _HOLD, str(directory), str(seconds)],
        stdout=subprocess.PIPE,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )


class TestOneWriter:
    def test_a_second_writer_waits_while_the_first_holds_the_lock(self, tmp_path: Path) -> None:
        directory = tmp_path / "copy"
        with CopyLock(directory):
            second = _writer(directory, 0)
            time.sleep(1.5)
            waiting = second.poll() is None
        out, _ = second.communicate(timeout=60)
        assert waiting, "the second writer waits on the lock"
        assert out.split() == ["held", "released"]

    def test_a_killed_holder_frees_the_lock_at_once(self, tmp_path: Path) -> None:
        directory = tmp_path / "copy"
        holder = _writer(directory, 600)
        try:
            assert holder.stdout is not None
            assert holder.stdout.readline().strip() == "held"
            os.kill(holder.pid, signal.SIGKILL)
            holder.wait(timeout=30)
        finally:
            if holder.poll() is None:
                os.kill(holder.pid, signal.SIGKILL)
                holder.wait(timeout=30)
        acquired = threading.Event()

        def take() -> None:
            with CopyLock(directory):
                acquired.set()

        start = time.perf_counter()
        thread = threading.Thread(target=take, daemon=True)
        thread.start()
        thread.join(timeout=10)
        assert acquired.is_set(), "the lock of a killed holder is free"
        assert time.perf_counter() - start < 5
        assert (directory / ".lock").exists(), (
            "the lock file stays: no process locks an unlinked one"
        )
