"""Tests for scripts/distributed_test_runner.py, the entry point of distributed test containers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from substrax.testing import run_python

from tests.scripts.script_loader import load_script, REPO_ROOT


_LOGGING_PROBE = (
    "import json, logging; "
    "from tests.scripts.script_loader import load_script; "
    "before = len(logging.getLogger().handlers); "
    "load_script('distributed_test_runner'); "
    "print(json.dumps([before, len(logging.getLogger().handlers)]))"
)


def test_importing_the_runner_leaves_the_root_logger_alone() -> None:
    """Logging is configured by the entry point, not as a side effect of importing the module."""
    result = run_python(_LOGGING_PROBE, timeout=180.0, cwd=REPO_ROOT)

    before, after = result.check().last_json()
    assert after == before


def test_the_test_suite_runs_in_this_interpreter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """pytest starts through ``sys.executable``, not whatever ``python`` PATH resolves."""
    runner: ModuleType = load_script("distributed_test_runner")
    calls: list[list[str]] = []

    def record(command: list[str], **kwargs: Any) -> None:
        calls.append(command)

    monkeypatch.setattr(runner, "setup_distributed_environment", lambda: None)
    monkeypatch.setattr(runner.subprocess, "run", record)
    monkeypatch.setattr(sys, "argv", ["distributed_test_runner.py", "tests/core"])
    monkeypatch.setenv("PATH", str(tmp_path / "no-interpreter-here"))

    runner.main()

    assert calls == [[sys.executable, "-m", "pytest", "tests/core"]]
