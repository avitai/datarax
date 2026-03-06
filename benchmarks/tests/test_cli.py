"""Tests for the benchmark CLI module entrypoint behavior."""

from __future__ import annotations

import runpy
import sys

import pytest


def test_python_module_invocation_executes_click_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`python -m benchmarks.cli --help` must execute Click and print help."""
    monkeypatch.setattr(sys, "argv", ["benchmarks.cli", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("benchmarks.cli", run_name="__main__")

    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "Usage:" in out
    assert "run" in out
