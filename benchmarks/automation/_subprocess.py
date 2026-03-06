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
from datetime import datetime, UTC
from pathlib import Path

from benchmarks.automation._orchestrator_utils import (
    _activity_bar,
    _clear_transient_status_line,
    _shorten,
    _status,
    CommandResult,
    DEFAULT_CAPTURE_LIMIT_CHARS,
    OrchestrationError,
)


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
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = " ".join(shlex.quote(a) for a in args)
    _clear_transient_status_line()

    capture_limit = None
    if capture_limit_chars is not None:
        capture_limit = max(capture_limit_chars, 1)

    captured = {"stdout": "", "stderr": ""}
    dropped = {"stdout": 0, "stderr": 0}

    stdout_spool_path = log_path.with_suffix(log_path.suffix + ".stdout.tmp")
    stderr_spool_path = log_path.with_suffix(log_path.suffix + ".stderr.tmp")

    def detect_phase(line: str) -> str | None:
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

    def _append_capture(stream: str, chunk: str) -> None:
        if not chunk:
            return
        current = captured[stream]
        if capture_limit is None:
            captured[stream] = current + chunk
            return
        combined = current + chunk
        if len(combined) <= capture_limit:
            captured[stream] = combined
            return
        over = len(combined) - capture_limit
        dropped[stream] += over
        captured[stream] = combined[over:]

    def _format_capture(stream: str) -> str:
        text = captured[stream]
        dropped_chars = dropped[stream]
        if dropped_chars <= 0:
            return text
        marker = (
            f"[output truncated: dropped {dropped_chars} chars, showing last {len(text)} chars]\n"
        )
        return marker + text

    def _write_final_log(
        *,
        stderr_suffix_lines: list[str] | None = None,
        diagnostics: dict[str, str] | None = None,
    ) -> None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("STDOUT:\n")
            with stdout_spool_path.open("r", encoding="utf-8", errors="replace") as src:
                shutil.copyfileobj(src, handle)
            handle.write("\n\nSTDERR:\n")
            with stderr_spool_path.open("r", encoding="utf-8", errors="replace") as src:
                shutil.copyfileobj(src, handle)
            if stderr_suffix_lines:
                handle.write("\n")
                handle.write("\n".join(stderr_suffix_lines))
                handle.write("\n")
            if diagnostics is not None:
                handle.write("\nSTALL_DIAGNOSTICS:\n")
                handle.write(json.dumps(diagnostics, indent=2))
                handle.write("\n")

    _status(f"Running command: {rendered}")
    _status(f"Command log: {log_path}")
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "\n".join(
                [
                    f"$ {rendered}",
                    "",
                    f"started_at: {datetime.now(UTC).isoformat()}",
                    "",
                ],
            )
            + "\n",
        )

    heartbeat_interval_sec = max(_heartbeat_interval_sec, 1)
    idle_notice_sec = max(_idle_notice_sec, 1)
    effective_stall_timeout_sec = None
    if stall_timeout_sec is not None and stall_timeout_sec > 0:
        effective_stall_timeout_sec = max(stall_timeout_sec, 1)
    next_heartbeat = heartbeat_interval_sec
    next_idle_notice = idle_notice_sec
    last_progress_line: str | None = None
    last_peek_line: str | None = None
    next_peek_emit = 0.0
    last_output_at = start
    phase_hint = "starting"
    setup_complete_idle_noted = False

    decoders = {
        "stdout": codecs.getincrementaldecoder("utf-8")(errors="replace"),
        "stderr": codecs.getincrementaldecoder("utf-8")(errors="replace"),
    }

    try:
        with (
            stdout_spool_path.open("wb") as stdout_sink,
            stderr_spool_path.open("wb") as stderr_sink,
            stdout_spool_path.open("rb") as stdout_reader,
            stderr_spool_path.open("rb") as stderr_reader,
        ):
            try:
                proc = subprocess.Popen(  # nosec B603
                    args,
                    stdout=stdout_sink,
                    stderr=stderr_sink,
                    text=False,
                    cwd=cwd,
                    env=env,
                )
            except OSError as exc:
                _write_final_log(stderr_suffix_lines=[f"failed_to_start: {exc!r}"])
                _status(f"Failed to execute command: {rendered}", level="ERROR")
                _clear_transient_status_line()
                raise OrchestrationError(
                    f"Failed to execute command: {rendered}\nSee log: {log_path}\n{exc}",
                ) from exc

            def _update_progress_line(chunk: str) -> None:
                nonlocal last_progress_line
                if not chunk:
                    return
                for raw_line in chunk.splitlines():
                    line = raw_line.strip()
                    if line:
                        last_progress_line = line

            def _consume_chunk(stream: str, chunk: str) -> None:
                nonlocal last_output_at, next_idle_notice, phase_hint
                nonlocal setup_complete_idle_noted, next_peek_emit, last_peek_line
                if not chunk:
                    return
                _append_capture(stream, chunk)
                last_output_at = time.monotonic()
                next_idle_notice = idle_notice_sec

                _update_progress_line(chunk)
                for raw_line in chunk.splitlines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    detected = detect_phase(line)
                    if detected is None or detected == phase_hint:
                        continue
                    phase_hint = detected
                    if detected == "setup_complete":
                        setup_complete_idle_noted = False
                        _status(
                            (
                                "Remote dependency install appears complete; waiting "
                                "for Sky launch finalization"
                            ),
                        )

                if (
                    live_peek
                    and last_progress_line
                    and last_progress_line != last_peek_line
                    and (time.monotonic() - start) >= next_peek_emit
                ):
                    _status(f"peek: {last_progress_line}")
                    last_peek_line = last_progress_line
                    interval = max(peek_interval_sec, 1)
                    next_peek_emit = (time.monotonic() - start) + interval

            def _consume_reader_delta(stream: str) -> None:
                reader = stdout_reader if stream == "stdout" else stderr_reader
                data = reader.read()
                if not data:
                    return
                text = decoders[stream].decode(data)
                _consume_chunk(stream, text)

            def _consume_available_output() -> None:
                _consume_reader_delta("stdout")
                _consume_reader_delta("stderr")

            def _flush_decoders() -> None:
                for stream in ("stdout", "stderr"):
                    remaining = decoders[stream].decode(b"", final=True)
                    _consume_chunk(stream, remaining)

            def _drain_until_stable(max_wait_sec: float = 1.0) -> None:
                deadline = time.monotonic() + max_wait_sec
                previous = (-1, -1)
                while True:
                    _consume_available_output()
                    current = (stdout_reader.tell(), stderr_reader.tell())
                    if current == previous:
                        break
                    previous = current
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(0.05)

            def _terminate_process() -> None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)

            try:
                while True:
                    try:
                        proc.wait(timeout=1)
                        _drain_until_stable(max_wait_sec=1.0)
                        break
                    except subprocess.TimeoutExpired:
                        _consume_available_output()
                        elapsed = time.monotonic() - start
                        if timeout_sec is not None and elapsed >= timeout_sec:
                            _terminate_process()
                            _drain_until_stable(max_wait_sec=1.0)
                            _flush_decoders()
                            _write_final_log(
                                stderr_suffix_lines=[
                                    f"timed_out: command exceeded {timeout_sec}s",
                                ],
                            )
                            _status(
                                f"Command timed out after {elapsed:.0f}s: {_shorten(rendered)}",
                                level="ERROR",
                            )
                            _clear_transient_status_line()
                            raise OrchestrationError(
                                f"Command timed out after {timeout_sec}s: {rendered}\n"
                                f"See log: {log_path}",
                            ) from None

                        if elapsed >= next_heartbeat:
                            tick = int(elapsed // heartbeat_interval_sec)
                            bar = _activity_bar(tick)
                            _status(
                                f"{bar} {elapsed:.0f}s elapsed | running: {_shorten(rendered)}",
                                transient=True,
                            )
                            next_heartbeat += heartbeat_interval_sec

                        silence_for = elapsed - (last_output_at - start)
                        if (
                            effective_stall_timeout_sec is not None
                            and silence_for >= effective_stall_timeout_sec
                        ):
                            diagnostics: dict[str, str] = {}
                            _status(
                                (
                                    "Stall detector triggered after "
                                    f"{silence_for:.0f}s of no output "
                                    f"(threshold: {effective_stall_timeout_sec}s). "
                                    "Running diagnostics"
                                ),
                                level="ERROR",
                            )
                            if stall_diagnostics is not None:
                                try:
                                    diagnostics = stall_diagnostics() or {}
                                except Exception as diag_exc:  # noqa: BLE001
                                    diagnostics = {
                                        "diagnostics_error": (
                                            f"Failed to run stall diagnostics: {diag_exc}"
                                        ),
                                    }
                            _terminate_process()
                            _drain_until_stable(max_wait_sec=1.0)
                            _flush_decoders()
                            _write_final_log(
                                stderr_suffix_lines=[
                                    (
                                        "stalled: no output for "
                                        f"{silence_for:.0f}s "
                                        f"(threshold: {effective_stall_timeout_sec}s)"
                                    ),
                                ],
                                diagnostics=diagnostics,
                            )

                            diagnostics_summary = ""
                            if diagnostics:
                                if diagnostics.get("queue"):
                                    diagnostics_summary += "\nqueue:\n" + "\n".join(
                                        diagnostics["queue"].splitlines()[-12:]
                                    )
                                if diagnostics.get("logs_tail"):
                                    diagnostics_summary += "\nlogs_tail:\n" + "\n".join(
                                        diagnostics["logs_tail"].splitlines()[-20:]
                                    )
                                if diagnostics.get("diagnostics_error"):
                                    diagnostics_summary += (
                                        "\ndiagnostics_error: " + diagnostics["diagnostics_error"]
                                    )
                            _clear_transient_status_line()
                            raise OrchestrationError(
                                (
                                    "Stall detected: command produced no output for "
                                    f"{silence_for:.0f}s "
                                    f"(threshold: {effective_stall_timeout_sec}s)\n"
                                    f"Command: {rendered}\n"
                                    f"See log: {log_path}{diagnostics_summary}"
                                ),
                            ) from None

                        if silence_for >= next_idle_notice:
                            phase_suffix = ""
                            if phase_hint:
                                phase_suffix = f" | phase: {phase_hint}"
                            if last_progress_line:
                                _status(
                                    (
                                        f"no new command output for {silence_for:.0f}s; "
                                        f"last line: {last_progress_line}{phase_suffix}"
                                    ),
                                    level="WARN",
                                )
                            else:
                                _status(
                                    (
                                        f"no command output observed for {silence_for:.0f}s"
                                        f"{phase_suffix}"
                                    ),
                                    level="WARN",
                                )
                            if phase_hint == "setup_complete" and not setup_complete_idle_noted:
                                _status(
                                    (
                                        "Dependency installation finished; benchmark "
                                        "pipeline has moved past setup and is waiting "
                                        "for the next launch/output event"
                                    ),
                                    level="WARN",
                                )
                                setup_complete_idle_noted = True
                            next_idle_notice += idle_notice_sec
            except KeyboardInterrupt as exc:
                _terminate_process()
                _drain_until_stable(max_wait_sec=1.0)
                _flush_decoders()
                _write_final_log(stderr_suffix_lines=["interrupted: KeyboardInterrupt"])
                _status(f"Command interrupted by user: {rendered}", level="ERROR")
                _clear_transient_status_line()
                raise OrchestrationError("Interrupted while executing command") from exc

            _flush_decoders()
            _write_final_log()

            result = CommandResult(
                args=args,
                returncode=proc.returncode or 0,
                stdout=_format_capture("stdout"),
                stderr=_format_capture("stderr"),
            )
            elapsed = time.monotonic() - start
            if check and proc.returncode != 0:
                tail = "\n".join(result.combined.splitlines()[-30:])
                _clear_transient_status_line()
                _status(
                    f"Command failed with exit code {proc.returncode} after {elapsed:.1f}s",
                    level="ERROR",
                )
                raise OrchestrationError(
                    f"Command failed ({proc.returncode}): {rendered}\nSee log: {log_path}\n{tail}",
                )
            _clear_transient_status_line()
            _status(f"Command completed in {elapsed:.1f}s")
            return result
    finally:
        try:
            stdout_spool_path.unlink(missing_ok=True)
            stderr_spool_path.unlink(missing_ok=True)
        except OSError:
            pass
