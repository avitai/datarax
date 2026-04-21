"""Logged subprocess execution with progress, stall detection, and capture limits.

Provides the core ``_run_logged_command`` function used by the orchestrator
to run local and remote commands while streaming output to log files and
providing real-time progress feedback.
"""

from __future__ import annotations

import codecs
import json
import shlex
import shutil
import subprocess  # nosec B404
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import BinaryIO, TextIO

from benchmarks.automation._orchestrator_utils import (
    _activity_bar,
    _clear_transient_status_line,
    _shorten,
    _status,
    CommandResult,
    DEFAULT_CAPTURE_LIMIT_CHARS,
    OrchestrationError,
)


def _detect_phase(line: str) -> str | None:
    """Infer a coarse remote-execution phase from one output line."""
    lowered = line.lower()
    if "installing collected packages:" in lowered:
        return "setup_installing"
    if "downloading " in lowered or " downloaded " in f" {lowered}":
        return "setup_downloading"
    if "prepared " in lowered and " packages " in f" {lowered}":
        return "setup_prepared"
    if "installed " in lowered and " packages " in f" {lowered}":
        return "setup_complete"
    if "job started. streaming logs" in lowered:
        return "run_streaming"
    if "cluster-ready" in lowered:
        return "run_ready"
    return None


@dataclass(slots=True)
class _OutputCapture:
    """Bounded stdout/stderr capture that keeps the newest output."""

    capture_limit_chars: int | None
    captured: dict[str, str] = field(default_factory=lambda: {"stdout": "", "stderr": ""})
    dropped: dict[str, int] = field(default_factory=lambda: {"stdout": 0, "stderr": 0})

    @property
    def limit(self) -> int | None:
        """Return normalized capture limit."""
        if self.capture_limit_chars is None:
            return None
        return max(self.capture_limit_chars, 1)

    def append(self, stream: str, chunk: str) -> None:
        """Append decoded output to the bounded capture buffer."""
        if not chunk:
            return
        if self.limit is None:
            self.captured[stream] += chunk
            return
        combined = self.captured[stream] + chunk
        self.dropped[stream] += max(len(combined) - self.limit, 0)
        self.captured[stream] = combined[-self.limit :]

    def format(self, stream: str) -> str:
        """Return captured output with a truncation marker when needed."""
        text = self.captured[stream]
        dropped_chars = self.dropped[stream]
        if dropped_chars <= 0:
            return text
        marker = (
            f"[output truncated: dropped {dropped_chars} chars, showing last {len(text)} chars]\n"
        )
        return marker + text


@dataclass(slots=True)
class _CommandRunner:
    """Stateful command runner used by `_run_logged_command`."""

    args: list[str]
    log_path: Path
    check: bool
    cwd: Path | None
    env: dict[str, str] | None
    timeout_sec: int | None
    live_peek: bool
    peek_interval_sec: int
    heartbeat_interval_sec: int
    idle_notice_sec: int
    stall_timeout_sec: int | None
    stall_diagnostics: Callable[[], dict[str, str]] | None
    capture: _OutputCapture
    start: float = 0.0

    @property
    def rendered(self) -> str:
        """Return a shell-rendered command for logs."""
        return " ".join(shlex.quote(a) for a in self.args)

    @property
    def stdout_spool_path(self) -> Path:
        """Return temporary stdout spool path."""
        return self.log_path.with_suffix(self.log_path.suffix + ".stdout.tmp")

    @property
    def stderr_spool_path(self) -> Path:
        """Return temporary stderr spool path."""
        return self.log_path.with_suffix(self.log_path.suffix + ".stderr.tmp")

    @property
    def effective_stall_timeout_sec(self) -> int | None:
        """Return normalized stall timeout."""
        if self.stall_timeout_sec is None or self.stall_timeout_sec <= 0:
            return None
        return max(self.stall_timeout_sec, 1)

    def run(self) -> CommandResult:
        """Execute the command and return captured output."""
        self._prepare_log()
        try:
            return self._run_with_spools()
        finally:
            self._cleanup_spools()

    def _prepare_log(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        _clear_transient_status_line()
        _status(f"Running command: {self.rendered}")
        _status(f"Command log: {self.log_path}")
        self.start = time.monotonic()
        with self.log_path.open("w", encoding="utf-8") as handle:
            handle.write(
                "\n".join(
                    [
                        f"$ {self.rendered}",
                        "",
                        f"started_at: {datetime.now(UTC).isoformat()}",
                        "",
                    ],
                )
                + "\n",
            )

    def _run_with_spools(self) -> CommandResult:
        with (
            self.stdout_spool_path.open("wb") as stdout_sink,
            self.stderr_spool_path.open("wb") as stderr_sink,
            self.stdout_spool_path.open("rb") as stdout_reader,
            self.stderr_spool_path.open("rb") as stderr_reader,
        ):
            proc = self._start_process(stdout_sink, stderr_sink)
            monitor = _CommandMonitor(self, proc, stdout_reader, stderr_reader)
            monitor.wait()
            monitor.flush_decoders()
            self.write_final_log()
            return self._build_result(proc)

    def _start_process(
        self,
        stdout_sink: BinaryIO,
        stderr_sink: BinaryIO,
    ) -> subprocess.Popen[bytes]:
        try:
            return subprocess.Popen(  # nosec B603
                self.args,
                stdout=stdout_sink,
                stderr=stderr_sink,
                text=False,
                cwd=self.cwd,
                env=self.env,
            )
        except OSError as exc:
            self.write_final_log(stderr_suffix_lines=[f"failed_to_start: {exc!r}"])
            _status(f"Failed to execute command: {self.rendered}", level="ERROR")
            _clear_transient_status_line()
            raise OrchestrationError(
                f"Failed to execute command: {self.rendered}\nSee log: {self.log_path}\n{exc}",
            ) from exc

    def _build_result(self, proc: subprocess.Popen[bytes]) -> CommandResult:
        result = CommandResult(
            args=self.args,
            returncode=proc.returncode or 0,
            stdout=self.capture.format("stdout"),
            stderr=self.capture.format("stderr"),
        )
        elapsed = time.monotonic() - self.start
        self._raise_for_failed_result(proc, result, elapsed)
        _clear_transient_status_line()
        _status(f"Command completed in {elapsed:.1f}s")
        return result

    def _raise_for_failed_result(
        self,
        proc: subprocess.Popen[bytes],
        result: CommandResult,
        elapsed: float,
    ) -> None:
        if not self.check or proc.returncode == 0:
            return
        tail = "\n".join(result.combined.splitlines()[-30:])
        _clear_transient_status_line()
        _status(
            f"Command failed with exit code {proc.returncode} after {elapsed:.1f}s",
            level="ERROR",
        )
        raise OrchestrationError(
            f"Command failed ({proc.returncode}): {self.rendered}\n"
            f"See log: {self.log_path}\n{tail}",
        )

    def write_final_log(
        self,
        *,
        stderr_suffix_lines: list[str] | None = None,
        diagnostics: dict[str, str] | None = None,
    ) -> None:
        """Persist stdout/stderr spools and optional failure metadata."""
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write("STDOUT:\n")
            with self.stdout_spool_path.open("r", encoding="utf-8", errors="replace") as src:
                shutil.copyfileobj(src, handle)
            handle.write("\n\nSTDERR:\n")
            with self.stderr_spool_path.open("r", encoding="utf-8", errors="replace") as src:
                shutil.copyfileobj(src, handle)
            self._write_optional_log_sections(handle, stderr_suffix_lines, diagnostics)

    def _write_optional_log_sections(
        self,
        handle: TextIO,
        stderr_suffix_lines: list[str] | None,
        diagnostics: dict[str, str] | None,
    ) -> None:
        if stderr_suffix_lines:
            handle.write("\n")
            handle.write("\n".join(stderr_suffix_lines))
            handle.write("\n")
        if diagnostics is not None:
            handle.write("\nSTALL_DIAGNOSTICS:\n")
            handle.write(json.dumps(diagnostics, indent=2))
            handle.write("\n")

    def _cleanup_spools(self) -> None:
        for path in (self.stdout_spool_path, self.stderr_spool_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


@dataclass(slots=True)
class _CommandMonitor:
    """Poll a running process while consuming output and reporting progress."""

    runner: _CommandRunner
    proc: subprocess.Popen[bytes]
    stdout_reader: BinaryIO
    stderr_reader: BinaryIO
    last_progress_line: str | None = None
    last_peek_line: str | None = None
    next_peek_emit: float = 0.0
    phase_hint: str = "starting"
    setup_complete_idle_noted: bool = False
    heartbeat_interval_sec: int = field(init=False)
    idle_notice_sec: int = field(init=False)
    next_heartbeat: int = field(init=False)
    next_idle_notice: int = field(init=False)
    last_output_at: float = field(init=False)
    decoders: dict[str, codecs.IncrementalDecoder] = field(
        default_factory=lambda: {
            "stdout": codecs.getincrementaldecoder("utf-8")(errors="replace"),
            "stderr": codecs.getincrementaldecoder("utf-8")(errors="replace"),
        },
    )

    def __post_init__(self) -> None:
        self.heartbeat_interval_sec = max(self.runner.heartbeat_interval_sec, 1)
        self.idle_notice_sec = max(self.runner.idle_notice_sec, 1)
        self.next_heartbeat = self.heartbeat_interval_sec
        self.next_idle_notice = self.idle_notice_sec
        self.last_output_at = self.runner.start

    def wait(self) -> None:
        """Wait for process completion with timeout and stall handling."""
        try:
            while not self._poll_once():
                pass
        except KeyboardInterrupt as exc:
            self._handle_keyboard_interrupt(exc)

    def _poll_once(self) -> bool:
        try:
            self.proc.wait(timeout=1)
            self.drain_until_stable(max_wait_sec=1.0)
            return True
        except subprocess.TimeoutExpired:
            self.consume_available_output()
            self._handle_running_tick(time.monotonic() - self.runner.start)
            return False

    def _handle_running_tick(self, elapsed: float) -> None:
        self._raise_on_timeout(elapsed)
        self._emit_heartbeat(elapsed)
        silence_for = time.monotonic() - self.last_output_at
        self._raise_on_stall(silence_for)
        self._emit_idle_notice(silence_for)

    def _raise_on_timeout(self, elapsed: float) -> None:
        if self.runner.timeout_sec is None or elapsed < self.runner.timeout_sec:
            return
        self.terminate_process()
        self.drain_until_stable(max_wait_sec=1.0)
        self.flush_decoders()
        self.runner.write_final_log(
            stderr_suffix_lines=[f"timed_out: command exceeded {self.runner.timeout_sec}s"],
        )
        _status(
            f"Command timed out after {elapsed:.0f}s: {_shorten(self.runner.rendered)}",
            level="ERROR",
        )
        _clear_transient_status_line()
        raise OrchestrationError(
            f"Command timed out after {self.runner.timeout_sec}s: {self.runner.rendered}\n"
            f"See log: {self.runner.log_path}",
        ) from None

    def _emit_heartbeat(self, elapsed: float) -> None:
        if elapsed < self.next_heartbeat:
            return
        tick = int(elapsed // self.heartbeat_interval_sec)
        bar = _activity_bar(tick)
        _status(
            f"{bar} {elapsed:.0f}s elapsed | running: {_shorten(self.runner.rendered)}",
            transient=True,
        )
        self.next_heartbeat += self.heartbeat_interval_sec

    def _raise_on_stall(self, silence_for: float) -> None:
        timeout = self.runner.effective_stall_timeout_sec
        if timeout is None or silence_for < timeout:
            return
        diagnostics = self._collect_stall_diagnostics(silence_for, timeout)
        self.terminate_process()
        self.drain_until_stable(max_wait_sec=1.0)
        self.flush_decoders()
        self.runner.write_final_log(
            stderr_suffix_lines=[
                f"stalled: no output for {silence_for:.0f}s (threshold: {timeout}s)",
            ],
            diagnostics=diagnostics,
        )
        _clear_transient_status_line()
        raise OrchestrationError(
            (
                "Stall detected: command produced no output for "
                f"{silence_for:.0f}s (threshold: {timeout}s)\n"
                f"Command: {self.runner.rendered}\n"
                f"See log: {self.runner.log_path}{self._diagnostics_summary(diagnostics)}"
            ),
        ) from None

    def _collect_stall_diagnostics(self, silence_for: float, timeout: int) -> dict[str, str]:
        _status(
            (
                "Stall detector triggered after "
                f"{silence_for:.0f}s of no output (threshold: {timeout}s). "
                "Running diagnostics"
            ),
            level="ERROR",
        )
        if self.runner.stall_diagnostics is None:
            return {}
        try:
            return self.runner.stall_diagnostics() or {}
        except (RuntimeError, OSError, ValueError) as diag_exc:
            return {"diagnostics_error": f"Failed to run stall diagnostics: {diag_exc}"}

    @staticmethod
    def _diagnostics_summary(diagnostics: dict[str, str]) -> str:
        parts = []
        if diagnostics.get("queue"):
            parts.append("\nqueue:\n" + "\n".join(diagnostics["queue"].splitlines()[-12:]))
        if diagnostics.get("logs_tail"):
            parts.append("\nlogs_tail:\n" + "\n".join(diagnostics["logs_tail"].splitlines()[-20:]))
        if diagnostics.get("diagnostics_error"):
            parts.append("\ndiagnostics_error: " + diagnostics["diagnostics_error"])
        return "".join(parts)

    def _emit_idle_notice(self, silence_for: float) -> None:
        if silence_for < self.next_idle_notice:
            return
        phase_suffix = f" | phase: {self.phase_hint}" if self.phase_hint else ""
        self._emit_idle_status(silence_for, phase_suffix)
        if self.phase_hint == "setup_complete" and not self.setup_complete_idle_noted:
            _status(
                (
                    "Dependency installation finished; benchmark pipeline has moved past "
                    "setup and is waiting for the next launch/output event"
                ),
                level="WARN",
            )
            self.setup_complete_idle_noted = True
        self.next_idle_notice += self.idle_notice_sec

    def _emit_idle_status(self, silence_for: float, phase_suffix: str) -> None:
        if self.last_progress_line:
            _status(
                (
                    f"no new command output for {silence_for:.0f}s; "
                    f"last line: {self.last_progress_line}{phase_suffix}"
                ),
                level="WARN",
            )
            return
        _status(
            f"no command output observed for {silence_for:.0f}s{phase_suffix}",
            level="WARN",
        )

    def consume_available_output(self) -> None:
        """Consume stdout and stderr bytes currently visible in spool files."""
        self._consume_reader_delta("stdout")
        self._consume_reader_delta("stderr")

    def _consume_reader_delta(self, stream: str) -> None:
        reader = self.stdout_reader if stream == "stdout" else self.stderr_reader
        data = reader.read()
        if not data:
            return
        self.consume_chunk(stream, self.decoders[stream].decode(data))

    def consume_chunk(self, stream: str, chunk: str) -> None:
        """Record decoded output and update progress hints."""
        if not chunk:
            return
        self.runner.capture.append(stream, chunk)
        self.last_output_at = time.monotonic()
        self.next_idle_notice = self.idle_notice_sec
        self._update_progress_line(chunk)
        self._update_phase_hint(chunk)
        self._emit_live_peek_if_ready()

    def _update_progress_line(self, chunk: str) -> None:
        for raw_line in chunk.splitlines():
            line = raw_line.strip()
            if line:
                self.last_progress_line = line

    def _update_phase_hint(self, chunk: str) -> None:
        for raw_line in chunk.splitlines():
            detected = _detect_phase(raw_line.strip())
            if detected is None or detected == self.phase_hint:
                continue
            self.phase_hint = detected
            self._emit_setup_complete_notice(detected)

    def _emit_setup_complete_notice(self, detected: str) -> None:
        if detected != "setup_complete":
            return
        self.setup_complete_idle_noted = False
        _status("Remote dependency install appears complete; waiting for Sky launch finalization")

    def _emit_live_peek_if_ready(self) -> None:
        if not self._should_emit_live_peek():
            return
        _status(f"peek: {self.last_progress_line}")
        self.last_peek_line = self.last_progress_line
        interval = max(self.runner.peek_interval_sec, 1)
        self.next_peek_emit = (time.monotonic() - self.runner.start) + interval

    def _should_emit_live_peek(self) -> bool:
        return bool(
            self.runner.live_peek
            and self.last_progress_line
            and self.last_progress_line != self.last_peek_line
            and (time.monotonic() - self.runner.start) >= self.next_peek_emit
        )

    def flush_decoders(self) -> None:
        """Flush incremental decoders into capture buffers."""
        for stream in ("stdout", "stderr"):
            self.consume_chunk(stream, self.decoders[stream].decode(b"", final=True))

    def drain_until_stable(self, max_wait_sec: float = 1.0) -> None:
        """Drain spool readers until offsets stabilize or timeout expires."""
        deadline = time.monotonic() + max_wait_sec
        previous = (-1, -1)
        while True:
            self.consume_available_output()
            current = (self.stdout_reader.tell(), self.stderr_reader.tell())
            if current == previous or time.monotonic() >= deadline:
                break
            previous = current
            time.sleep(0.05)

    def terminate_process(self) -> None:
        """Terminate the child process, escalating to kill if needed."""
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)

    def _handle_keyboard_interrupt(self, exc: KeyboardInterrupt) -> None:
        self.terminate_process()
        self.drain_until_stable(max_wait_sec=1.0)
        self.flush_decoders()
        self.runner.write_final_log(stderr_suffix_lines=["interrupted: KeyboardInterrupt"])
        _status(f"Command interrupted by user: {self.runner.rendered}", level="ERROR")
        _clear_transient_status_line()
        raise OrchestrationError("Interrupted while executing command") from exc


def _run_logged_command(
    args: list[str],
    log_path: Path,
    check: bool = True,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout_sec: int | None = None,
    live_peek: bool = True,
    peek_interval_sec: int = 5,
    _heartbeat_interval_sec: int = 10,
    _idle_notice_sec: int = 60,
    stall_timeout_sec: int | None = None,
    stall_diagnostics: Callable[[], dict[str, str]] | None = None,
    capture_limit_chars: int | None = DEFAULT_CAPTURE_LIMIT_CHARS,
) -> CommandResult:
    """Execute a command and persist stdout/stderr to a log file."""
    runner = _CommandRunner(
        args=args,
        log_path=log_path,
        check=check,
        cwd=cwd,
        env=env,
        timeout_sec=timeout_sec,
        live_peek=live_peek,
        peek_interval_sec=peek_interval_sec,
        heartbeat_interval_sec=_heartbeat_interval_sec,
        idle_notice_sec=_idle_notice_sec,
        stall_timeout_sec=stall_timeout_sec,
        stall_diagnostics=stall_diagnostics,
        capture=_OutputCapture(capture_limit_chars),
    )
    return runner.run()
