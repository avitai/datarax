"""The memory this process occupies, now and at its peak, as a RAM budget counts it.

Linux reads the resident size. On macOS the resident size leaves out the pages the memory
compressor holds, which still occupy RAM, so a budget would count less than the process uses;
macOS reads the physical footprint instead, Apple's own accounting of a process's memory
(``ri_phys_footprint`` and ``ri_lifetime_max_phys_footprint`` from libproc's
``proc_pid_rusage``). No released psutil reads it: psutil adds it in 8.0 (``memory_extras()``),
whose ``proc.c`` this reading follows, so it goes through ctypes with xnu's ``rusage_info_v4``
(``bsd/sys/resource.h``) until then.
"""

from __future__ import annotations

import ctypes
import functools
import os
import resource
import sys
from collections.abc import Callable
from typing import Any

import psutil


_RUSAGE_INFO_V4 = 4

_V4_FIELDS = (
    "ri_user_time",
    "ri_system_time",
    "ri_pkg_idle_wkups",
    "ri_interrupt_wkups",
    "ri_pageins",
    "ri_wired_size",
    "ri_resident_size",
    "ri_phys_footprint",
    "ri_proc_start_abstime",
    "ri_proc_exit_abstime",
    "ri_child_user_time",
    "ri_child_system_time",
    "ri_child_pkg_idle_wkups",
    "ri_child_interrupt_wkups",
    "ri_child_pageins",
    "ri_child_elapsed_abstime",
    "ri_diskio_bytesread",
    "ri_diskio_byteswritten",
    "ri_cpu_time_qos_default",
    "ri_cpu_time_qos_maintenance",
    "ri_cpu_time_qos_background",
    "ri_cpu_time_qos_utility",
    "ri_cpu_time_qos_legacy",
    "ri_cpu_time_qos_user_initiated",
    "ri_cpu_time_qos_user_interactive",
    "ri_billed_system_time",
    "ri_serviced_system_time",
    "ri_logical_writes",
    "ri_lifetime_max_phys_footprint",
    "ri_instructions",
    "ri_cycles",
    "ri_billed_energy",
    "ri_serviced_energy",
    "ri_interval_max_phys_footprint",
    "ri_runnable_time",
)


class _RusageInfoV4(ctypes.Structure):
    """xnu's ``struct rusage_info_v4``: a 16-byte uuid, then 35 ``uint64_t``."""

    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in _V4_FIELDS]


def current_bytes() -> int:
    """The memory this process occupies now.

    Returns:
        The resident size; on macOS the physical footprint, compressed pages included.
    """
    if sys.platform == "darwin":
        return int(_rusage().ri_phys_footprint)
    return psutil.Process().memory_info().rss


def peak_bytes() -> int:
    """The most memory this process has occupied.

    Returns:
        The peak resident size (``ru_maxrss``, which Linux reports in KiB); on macOS the lifetime
        maximum of the physical footprint.
    """
    if sys.platform == "darwin":
        return int(_rusage().ri_lifetime_max_phys_footprint)
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _rusage() -> _RusageInfoV4:
    """This process's ``rusage_info_v4``.

    Returns:
        The record ``proc_pid_rusage`` filled for this process.

    Raises:
        OSError: If ``proc_pid_rusage`` refuses, naming it.
    """
    info = _RusageInfoV4()
    if _proc_pid_rusage()(os.getpid(), _RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
        code = ctypes.get_errno()
        raise OSError(code, f"proc_pid_rusage(RUSAGE_INFO_V4) failed: {os.strerror(code)}")
    return info


@functools.cache
def _proc_pid_rusage() -> Callable[[int, int, Any], int]:
    """The ``int proc_pid_rusage(int pid, int flavor, rusage_info_t *buffer)`` of libproc."""
    call = ctypes.CDLL(None, use_errno=True).proc_pid_rusage
    call.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(_RusageInfoV4)]
    call.restype = ctypes.c_int
    return call
