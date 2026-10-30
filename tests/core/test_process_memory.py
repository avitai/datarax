"""The memory a process occupies, now and at its peak, as a RAM budget counts it.

Linux reads the resident size. macOS reads the physical footprint, which also counts the pages
its memory compressor holds; its resident size leaves them out. The macOS path runs anywhere:
the platform is set to ``darwin`` and ``proc_pid_rusage`` is replaced by a recorder, checked
against xnu's ``rusage_info_v4`` (``bsd/sys/resource.h``). A macOS run checks the call itself.
"""

from __future__ import annotations

import ctypes
import errno
import os
import sys
from typing import Any

import psutil
import pytest

from datarax.core import process_memory


class TestOnLinux:
    @pytest.mark.skipif(sys.platform != "linux", reason="reads Linux's resident size")
    def test_now_is_the_resident_size(self) -> None:
        before = psutil.Process().memory_info().rss
        now = process_memory.current_bytes()
        after = psutil.Process().memory_info().rss

        assert min(before, after) - (16 << 20) <= now <= max(before, after) + (16 << 20)

    @pytest.mark.skipif(sys.platform != "linux", reason="reads Linux's ru_maxrss, in KiB")
    def test_the_peak_is_never_below_now(self) -> None:
        assert process_memory.peak_bytes() >= process_memory.current_bytes() - (16 << 20)


class TestOnMacOS:
    @pytest.fixture
    def on_darwin(self, monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
        calls: list[tuple[int, int]] = []

        def recorded(pid: int, flavor: int, buffer: Any) -> int:
            calls.append((pid, flavor))
            info = buffer._obj
            info.ri_resident_size = 100
            info.ri_phys_footprint = 300
            info.ri_lifetime_max_phys_footprint = 500
            return 0

        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(process_memory, "_proc_pid_rusage", lambda: recorded)
        return calls

    def test_now_is_the_physical_footprint(self, on_darwin: list[tuple[int, int]]) -> None:
        assert process_memory.current_bytes() == 300
        assert on_darwin == [(os.getpid(), 4)]  # RUSAGE_INFO_V4

    def test_the_peak_is_the_lifetime_maximum_footprint(
        self, on_darwin: list[tuple[int, int]]
    ) -> None:
        assert process_memory.peak_bytes() == 500

    def test_a_failed_read_is_refused_naming_the_call(
        self, monkeypatch: pytest.MonkeyPatch, on_darwin: list[tuple[int, int]]
    ) -> None:
        def refused(pid: int, flavor: int, buffer: Any) -> int:
            ctypes.set_errno(errno.EPERM)
            return -1

        monkeypatch.setattr(process_memory, "_proc_pid_rusage", lambda: refused)

        with pytest.raises(OSError, match="proc_pid_rusage") as raised:
            process_memory.current_bytes()
        assert raised.value.errno == errno.EPERM


def test_the_record_has_rusage_info_v4_s_layout() -> None:
    """A 16-byte uuid and 35 uint64 fields, the footprints at xnu's offsets."""
    record = process_memory._RusageInfoV4

    assert ctypes.sizeof(record) == 16 + 35 * 8
    assert record.ri_phys_footprint.offset == 16 + 7 * 8
    assert record.ri_lifetime_max_phys_footprint.offset == 16 + 28 * 8
