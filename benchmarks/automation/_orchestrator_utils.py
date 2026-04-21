"""Utility functions, constants, and data classes for the benchmark orchestrator.

Extracted from vast_orchestrator to keep the main module under the file-length
limit while preserving a clean separation between orchestration logic and
supporting helpers.
"""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess  # nosec B404
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from datarax.utils.console import emit


# ── Constants ──────────────────────────────────────────────────────────

DEFAULT_SUBSET_SCENARIOS = ["CV-1", "NLP-1", "TAB-1", "PC-1", "IO-1", "PR-2"]
DEFAULT_CAPTURE_LIMIT_CHARS = 1_000_000
DEFAULT_ADAPTERS = [
    "Datarax",
    "Google Grain",
    "jax-dataloader",
    "tf.data",
    "PyTorch DataLoader",
    "NVIDIA DALI",
    "SPDL",
    "MosaicML Streaming",
    "WebDataset",
    "HuggingFace Datasets",
    "LitData",
    "Deep Lake",
]
REMOTE_GPU_ENV_EXPORTS = [
    "export JAX_PLATFORMS=cuda,cpu",
    "export JAX_PLATFORM_NAME=gpu",
    "export XLA_PYTHON_CLIENT_PREALLOCATE=false",
]
REMOTE_RESULTS_ROOT = "/root/results"
REMOTE_DATA_ROOT = "/root/benchmark-data"

# ── Compiled patterns ─────────────────────────────────────────────────

_NVIDIA_SMI_ROW_RE = re.compile(
    (
        r"(?P<name>(?:NVIDIA|Tesla|GeForce|Quadro|AMD)[^,]*),\s*"
        r"(?P<memory>\d+\s*(?:MiB|GiB|MB|GB)),\s*"
        r"(?P<driver>[0-9][0-9A-Za-z.\-]+)"
    ),
)
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_SKY_RSYNC_COMMAND_RE = re.compile(r"^\s*rsync\b", re.MULTILINE)

# ── Mutable terminal state ────────────────────────────────────────────

_TRANSIENT_ACTIVE = False
_TRANSIENT_LINE = ""


# ── Data classes ──────────────────────────────────────────────────────


class OrchestrationError(RuntimeError):
    """Raised for orchestration workflow failures."""


@dataclass
class CommandResult:
    """Captured subprocess output."""

    args: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def combined(self) -> str:
        """Return stdout and stderr joined as a single string."""
        return f"{self.stdout}\n{self.stderr}".strip()


# ── Display helpers ───────────────────────────────────────────────────


def _progress_bar(completed: int, total: int, width: int = 24) -> str:
    """Render a bounded progress bar."""
    if total <= 0:
        total = 1
    ratio = min(max(completed / total, 0.0), 1.0)
    filled = int(round(width * ratio))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _activity_bar(tick: int, width: int = 24) -> str:
    """Render an indeterminate bar for long-running commands."""
    if width < 3:
        width = 3
    pos = tick % width
    cells = ["-"] * width
    cells[pos] = "#"
    return "[" + "".join(cells) + "]"


def _shorten(text: str, *, limit: int = 96) -> str:
    """Shorten very long command strings for status output."""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _terminal_width() -> int:
    """Return a bounded terminal width used for status rendering."""
    return max(shutil.get_terminal_size((120, 24)).columns, 40)


def _clear_transient_status_line() -> None:
    """Erase transient progress state so future logs don't redraw stale lines."""
    global _TRANSIENT_ACTIVE, _TRANSIENT_LINE  # noqa: PLW0603

    if _TRANSIENT_ACTIVE and sys.stdout.isatty():
        width = _terminal_width()
        sys.stdout.write("\r" + (" " * width) + "\r")
        sys.stdout.flush()
    _TRANSIENT_ACTIVE = False
    _TRANSIENT_LINE = ""


def _status(message: str, *, level: str = "INFO", transient: bool = False) -> None:
    """Emit timestamped CLI status messages with immediate flush.

    When ``transient=True`` and stdout is a TTY, render as an in-place status
    line.
    """
    global _TRANSIENT_ACTIVE, _TRANSIENT_LINE  # noqa: PLW0603

    ts = datetime.now().strftime("%H:%M:%S")
    prefix = f"[{ts}] [{level}] "
    terminal_width = _terminal_width()

    if transient and sys.stdout.isatty():
        content_width = max(terminal_width - len(prefix), 8)
        single = _shorten(str(message), limit=content_width)
        line = f"{prefix}{single}"
        emit("\r" + line.ljust(terminal_width), end="", flush=True)
        _TRANSIENT_ACTIVE = True
        _TRANSIENT_LINE = line
        return

    redraw_transient = _TRANSIENT_ACTIVE and sys.stdout.isatty()
    if redraw_transient:
        sys.stdout.write("\r" + (" " * terminal_width) + "\r")
        sys.stdout.flush()

    content_width = max(terminal_width - len(prefix), 20)
    raw_lines = str(message).splitlines() or [""]
    for raw_line in raw_lines:
        wrapped = textwrap.wrap(
            raw_line,
            width=content_width,
            break_long_words=True,
            break_on_hyphens=False,
        )
        if not wrapped:
            emit(prefix.rstrip(), flush=True)
            continue
        for part in wrapped:
            emit(f"{prefix}{part}", flush=True)

    if redraw_transient:
        transient_line = _shorten(_TRANSIENT_LINE, limit=terminal_width)
        sys.stdout.write("\r" + transient_line.ljust(terminal_width))
        sys.stdout.flush()


# ── Interactive helpers ───────────────────────────────────────────────


def _confirm(prompt: str, *, assume_yes: bool) -> bool:
    """Prompt for user confirmation unless auto-confirm is enabled."""
    if assume_yes:
        _status(f"{prompt} [auto-confirmed]")
        return True

    try:
        answer = input(f"{prompt} [y/N]: ").strip().lower()
    except EOFError:
        _status("No interactive input available for confirmation", level="ERROR")
        return False
    return answer in {"y", "yes"}


def _timestamp_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


# ── GPU / config helpers ─────────────────────────────────────────────


def _effective_allowed_gpu_tokens(tokens: list[str] | None) -> list[str]:
    """Normalize allowed GPU token list, falling back to the default token."""
    cleaned = [token.strip() for token in (tokens or []) if token and token.strip()]
    return cleaned or ["A100"]


def _primary_gpu_resource(tokens: list[str]) -> str:
    """Convert the first allowed GPU token to a SkyPilot ``--gpus`` resource string."""
    return f"{tokens[0]}:1"


def _normalize_capture_limit_chars(value: int | None) -> int | None:
    """Normalize capture limit; non-positive values disable in-memory capture cap."""
    if value is None:
        return DEFAULT_CAPTURE_LIMIT_CHARS
    if value <= 0:
        return None
    return value


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# ── Version / capability probing ──────────────────────────────────────


def _build_sky_upgrade_hint(*, repo_root: Path, infra: str) -> str:
    """Return a concrete local command to upgrade SkyPilot for this infra."""
    pip_exe = repo_root / ".venv" / "bin" / "pip"
    pip_cmd = str(pip_exe) if pip_exe.exists() else "pip"
    extra = "vast" if infra.lower() == "vast" else "all"
    return f'{shlex.quote(pip_cmd)} install -U "skypilot[{extra}]"'


def _build_click_pin_hint(*, repo_root: Path) -> str:
    """Return a concrete local command to pin Click to a compatible range."""
    pip_exe = repo_root / ".venv" / "bin" / "pip"
    pip_cmd = str(pip_exe) if pip_exe.exists() else "pip"
    return f'{shlex.quote(pip_cmd)} install "click<8.3"'


def _extract_semver(value: str | None) -> tuple[int, int, int] | None:
    """Extract MAJOR.MINOR.PATCH from arbitrary text."""
    if not value:
        return None
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", value)
    if match is None:
        return None
    parts = match.groups()
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def _probe_python_package_version(*, python_executable: Path, package: str) -> str:
    """Probe installed package version using the target python interpreter."""
    if not python_executable.exists():
        return "unknown"
    script = f"import importlib.metadata as m; print(m.version({package!r}))"
    try:
        proc = subprocess.run(  # nosec B603
            [str(python_executable), "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unknown"

    text = (proc.stdout or proc.stderr).strip()
    if proc.returncode != 0 or not text:
        return "unknown"
    first_line = text.splitlines()[0].strip()
    return first_line or "unknown"


def _is_known_skypilot_click_backend_bug(*, sky_version: str, click_version: str) -> bool:
    """Detect known ``False backend is not supported`` launch incompatibility."""
    sky_semver = _extract_semver(sky_version)
    click_semver = _extract_semver(click_version)
    if sky_semver is None or click_semver is None:
        return False
    # SkyPilot 0.11.x + Click 8.3.x is known to break `sky launch`.
    return sky_semver < (0, 12, 0) and click_semver >= (8, 3, 0)


def _probe_skypilot_capabilities(sky_executable: str) -> dict[str, Any]:
    """Inspect SkyPilot CLI capabilities needed by the orchestrator."""
    version_proc = subprocess.run(  # nosec B603
        [sky_executable, "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    help_proc = subprocess.run(  # nosec B603
        [sky_executable, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    version_text = (f"{version_proc.stdout}\n{version_proc.stderr}").strip()
    help_text = (f"{help_proc.stdout}\n{help_proc.stderr}").strip()
    supports_rsync = bool(_SKY_RSYNC_COMMAND_RE.search(help_text.lower()))

    version_line = "unknown"
    for line in version_text.splitlines():
        line = line.strip()
        if line:
            version_line = line
            break

    return {
        "version": version_line,
        "supports_rsync": supports_rsync,
    }


def _remote_shell_prefix() -> list[str]:
    """Common remote shell setup used by all benchmark commands."""
    return [
        "set -euo pipefail",
        "cd ~/datarax",
        "unset RAY_ADDRESS",
        'export PYTHONPATH="${PWD}:${PYTHONPATH:-}"',
        *REMOTE_GPU_ENV_EXPORTS,
    ]


# ── Output parsing ────────────────────────────────────────────────────


def _extract_json_line(text: str) -> dict[str, Any]:
    """Extract the last JSON object from multi-line command output."""
    decoder = json.JSONDecoder()
    for line in reversed(text.splitlines()):
        line = _ANSI_ESCAPE_RE.sub("", line).strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
        # Sky often prefixes logs: "(sky-cmd, pid=...) {...json...}".
        for idx, char in enumerate(line):
            if char != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(line[idx:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    raise OrchestrationError("Unable to parse JSON payload from command output.")


def _extract_nvidia_smi_rows(text: str) -> list[str]:
    """Extract parseable nvidia-smi CSV rows from wrapped Sky command output."""
    rows: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("command to run:"):
            continue
        match = _NVIDIA_SMI_ROW_RE.search(line)
        if match is None:
            continue
        rows.append(
            (
                f"{match.group('name').strip()}, "
                f"{match.group('memory').strip()}, "
                f"{match.group('driver').strip()}"
            ),
        )
    return rows
