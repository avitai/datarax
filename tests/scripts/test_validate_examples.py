"""Tests for scripts/validate_examples.py's execution check."""

from __future__ import annotations

import textwrap
from pathlib import Path
from types import ModuleType

import pytest

from tests.scripts.script_loader import load_script


@pytest.fixture(scope="module")
def validate_examples() -> ModuleType:
    return load_script("validate_examples")


def _example(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "01_example.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


def test_a_running_example_passes_without_python_on_the_path(
    validate_examples: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The example runs in this interpreter, whatever PATH holds."""
    example = _example(tmp_path, 'print("ran")\n')
    result = validate_examples.ValidationResult(file_path=example)
    monkeypatch.setenv("PATH", str(tmp_path / "no-interpreter-here"))

    validate_examples.validate_execution(example, result)

    assert result.errors == []
    assert "Execution passed" in result.info


def test_a_failing_example_reports_its_error(validate_examples: ModuleType, tmp_path: Path) -> None:
    example = _example(
        tmp_path,
        """
        import sys

        sys.stderr.write("the pipeline broke\\n")
        raise SystemExit(3)
        """,
    )
    result = validate_examples.ValidationResult(file_path=example)

    validate_examples.validate_execution(example, result)

    assert result.passed is False
    assert len(result.errors) == 1
    assert "exit code 3" in result.errors[0]
    assert "the pipeline broke" in result.errors[0]


def test_an_example_past_its_budget_reports_the_timeout(
    validate_examples: ModuleType, tmp_path: Path
) -> None:
    example = _example(tmp_path, "import time\n\ntime.sleep(60)\n")
    result = validate_examples.ValidationResult(file_path=example)

    validate_examples.validate_execution(example, result, timeout=2)

    assert result.errors == ["Execution timed out after 2s"]
