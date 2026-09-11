"""Idle notices of the command monitor, driven by an injected clock.

The monitor polls a running command about once a second. These tests step a fake clock
instead of sleeping, so the notice schedule is checked exactly rather than raced against
the runner's timer resolution. They record the monitor's status messages rather than the
console text, which wraps at the terminal width.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from benchmarks.automation import _subprocess as sp


IDLE_NOTICE = "no new command output for"
SETUP_FINISHED = "Dependency installation finished"


class FakeClock:
    """Monotonic time source the test advances by hand."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def status_messages(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every status message the monitor emits."""
    messages: list[str] = []

    def record(message: str, *, level: str = "INFO", transient: bool = False) -> None:
        messages.append(message)

    monkeypatch.setattr(sp, "_status", record)
    return messages


def _monitor(tmp_path: Path, clock: FakeClock) -> sp._CommandMonitor:
    runner = sp._CommandRunner(
        args=["long-running-command"],
        log_path=tmp_path / "command.log",
        check=True,
        cwd=None,
        env=None,
        timeout_sec=None,
        live_peek=False,
        peek_interval_sec=1,
        heartbeat_interval_sec=1,
        idle_notice_sec=2,
        stall_timeout_sec=None,
        stall_diagnostics=None,
        capture=sp._OutputCapture(None),
        clock=clock,
    )
    return sp._CommandMonitor(runner, MagicMock(), io.BytesIO(), io.BytesIO())


def _tick_at(monitor: sp._CommandMonitor, clock: FakeClock, now: float) -> None:
    clock.now = now
    monitor._handle_running_tick(now - monitor.runner.start)


def _output_at(monitor: sp._CommandMonitor, clock: FakeClock, now: float, line: str) -> None:
    clock.now = now
    monitor.consume_chunk("stdout", f"{line}\n")


def _idle_notices(messages: list[str]) -> list[str]:
    return [m for m in messages if m.startswith((IDLE_NOTICE, "no command output observed"))]


def test_no_idle_notice_before_the_threshold(tmp_path: Path, status_messages: list[str]) -> None:
    clock = FakeClock()
    monitor = _monitor(tmp_path, clock)
    _output_at(monitor, clock, 0.0, "warmup-line")

    _tick_at(monitor, clock, 1.9)

    assert _idle_notices(status_messages) == []


def test_idle_notice_at_the_threshold_names_the_last_line(
    tmp_path: Path, status_messages: list[str]
) -> None:
    clock = FakeClock()
    monitor = _monitor(tmp_path, clock)
    _output_at(monitor, clock, 0.0, "warmup-line")

    _tick_at(monitor, clock, 2.0)

    assert _idle_notices(status_messages) == [
        f"{IDLE_NOTICE} 2s; last line: warmup-line | phase: starting"
    ]


def test_idle_notice_repeats_once_per_threshold(tmp_path: Path, status_messages: list[str]) -> None:
    clock = FakeClock()
    monitor = _monitor(tmp_path, clock)
    _output_at(monitor, clock, 0.0, "warmup-line")
    _tick_at(monitor, clock, 2.0)

    _tick_at(monitor, clock, 3.9)
    assert len(_idle_notices(status_messages)) == 1

    _tick_at(monitor, clock, 4.0)
    assert len(_idle_notices(status_messages)) == 2
    assert _idle_notices(status_messages)[-1].startswith(f"{IDLE_NOTICE} 4s;")


def test_new_output_restarts_the_idle_schedule(tmp_path: Path, status_messages: list[str]) -> None:
    clock = FakeClock()
    monitor = _monitor(tmp_path, clock)
    _output_at(monitor, clock, 0.0, "warmup-line")
    _tick_at(monitor, clock, 2.0)
    _output_at(monitor, clock, 3.0, "second-line")

    _tick_at(monitor, clock, 4.9)
    assert len(_idle_notices(status_messages)) == 1

    _tick_at(monitor, clock, 5.0)
    assert _idle_notices(status_messages)[-1] == (
        f"{IDLE_NOTICE} 2s; last line: second-line | phase: starting"
    )


def test_idle_notice_before_any_output_reports_the_starting_phase(
    tmp_path: Path, status_messages: list[str]
) -> None:
    clock = FakeClock()
    monitor = _monitor(tmp_path, clock)

    _tick_at(monitor, clock, 2.0)

    assert _idle_notices(status_messages) == ["no command output observed for 2s | phase: starting"]


def test_idle_notice_after_setup_carries_the_phase_and_warns_once(
    tmp_path: Path, status_messages: list[str]
) -> None:
    clock = FakeClock()
    monitor = _monitor(tmp_path, clock)
    _output_at(monitor, clock, 0.0, "(setup pid=1) Installed 226 packages in 23.74s")

    _tick_at(monitor, clock, 2.0)
    _tick_at(monitor, clock, 4.0)

    notices = _idle_notices(status_messages)
    assert len(notices) == 2
    assert all(notice.endswith("| phase: setup_complete") for notice in notices)
    assert sum(message.startswith(SETUP_FINISHED) for message in status_messages) == 1
