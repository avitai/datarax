"""Tests for Datarax's example scripts until they follow substrax's example guide.

The numbered examples under ``examples/`` write their outputs through ``substrax.artifacts``, the
docs pages show their scripts' own code, and every example runs to completion as a script. The
guide's checks and its shared example tests take the examples over once they are converted to it.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
from pathlib import Path

import pytest
from substrax.artifacts import OUTPUT_DIR_ENV
from substrax.examples import discover_examples
from substrax.testing import run_python

from tests.jax_test_environment import forwarded_jax_environment
from tests.scripts.script_loader import load_script


# Detect macOS - TensorFlow crashes on macOS ARM64
IS_MACOS = platform.system() == "Darwin"

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"
EXAMPLE_TIMEOUT_SECONDS = int(os.environ.get("DATARAX_EXAMPLE_TIMEOUT_SECONDS", "120"))

# Substrings that identify a transient dataset-download / network failure (e.g.
# HuggingFace Hub rate-limiting unauthenticated CI runners). Examples that hit
# these are skipped rather than failed: they exercise external services, not
# Datarax code. Real code errors (AttributeError, shape mismatch, …) do not
# match and still fail the test.
_NETWORK_FAILURE_SIGNATURES = (
    "couldn't reach",
    "connectionerror",
    "connectionreseterror",
    "max retries exceeded",
    "too many requests",
    "429 client error",
    "hfhubhttperror",
    "readtimeout",
    "temporary failure in name resolution",
    "network is unreachable",
    "failed to resolve",
    "name or service not known",
    "sslerror",
    "rate limit",
    "no such file or directory: '/home/",  # missing local dataset cache dir
)


def is_numbered_example(path: Path) -> bool:
    """Whether a script is a numbered example rather than a helper or a test of one."""
    return re.match(r"^\d+_", path.name) is not None and "test" not in path.name.lower()


def find_example_files(examples_dir: Path) -> list[Path]:
    """The numbered examples under ``examples_dir``; substrax's discovery skips private names."""
    return [path for path in discover_examples(examples_dir) if is_numbered_example(path)]


def example_id(path: Path) -> str:
    """Generate a short test ID from file path."""
    return str(path.relative_to(EXAMPLES_DIR))


EXAMPLE_FILES = find_example_files(EXAMPLES_DIR)

# scripts/check_docs_pairs.py owns the rule for a docs page's code that shows output.
check_docs_pairs = load_script("check_docs_pairs")
DOCS_PAGES = check_docs_pairs.docs_pages(REPO_ROOT)


class TestExampleDiscovery:
    """The numbered examples are the ones these tests check and run."""

    def test_numbered_examples_are_found_and_everything_else_is_not(self, tmp_path: Path) -> None:
        for relative in (
            "core/01_quickstart.py",
            "advanced/02_guide.py",
            "core/03_test_helpers.py",
            "comparison/04_versus.py",
            "_templates/05_template.py",
            "core/notes.py",
            "core/__init__.py",
        ):
            (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / relative).write_text("", encoding="utf-8")

        found = [path.relative_to(tmp_path).as_posix() for path in find_example_files(tmp_path)]

        assert found == ["advanced/02_guide.py", "comparison/04_versus.py", "core/01_quickstart.py"]

    def test_the_repository_has_examples_to_check(self) -> None:
        """A rule that matched no file would pass every check while checking nothing."""
        assert EXAMPLE_FILES


class TestExampleOutputs:
    """Examples write outputs through substrax.artifacts, so a run never rewrites tracked files."""

    @pytest.mark.parametrize("example_path", EXAMPLE_FILES, ids=example_id)
    def test_outputs_are_resolved_through_substrax(self, example_path: Path) -> None:
        content = example_path.read_text()

        assert "DATARAX_EXAMPLES_OUTPUT_DIR" not in content
        assert "docs/assets" not in content


class TestDocsPagesShowTheScriptsCode:
    """Code a docs page shows with output is its script's code (``scripts/check_docs_pairs.py``)."""

    @pytest.mark.parametrize(
        "page", DOCS_PAGES, ids=lambda page: str(page.relative_to(REPO_ROOT / "docs" / "examples"))
    )
    def test_code_shown_with_output_is_a_run_of_the_script(self, page: Path) -> None:
        assert check_docs_pairs.check_page(page, REPO_ROOT) == []

    def test_the_pages_hold_code_to_check(self) -> None:
        """A rule that matched no pair would pass every page while checking nothing."""
        pairs = sum(len(check_docs_pairs.PAIR.findall(page.read_text())) for page in DOCS_PAGES)

        assert pairs > 0


def unavailable_reason(returncode: int, stderr: str) -> str | None:
    """The first network-failure signature in a failed run's standard error, ignoring case."""
    if returncode == 0:
        return None
    lowered = stderr.lower()
    return next(
        (signature for signature in _NETWORK_FAILURE_SIGNATURES if signature in lowered), None
    )


class TestUnavailableReason:
    """Only a failed run that names an unreachable service is skipped."""

    def test_a_failed_run_naming_a_network_failure_is_unavailable(self) -> None:
        assert (
            unavailable_reason(1, "requests.exceptions.ConnectionError: ...") == "connectionerror"
        )

    def test_a_failed_run_with_a_code_error_is_not(self) -> None:
        assert unavailable_reason(1, "AttributeError: 'Batch' object has no attribute 'x'") is None

    def test_a_run_that_succeeded_is_never_unavailable(self) -> None:
        assert unavailable_reason(0, "429 Client Error: retried and succeeded") is None


@pytest.mark.slow
class TestExampleExecution:
    """Tests that execute example files.

    These tests are marked as slow and run in the long-running CI tier.
    Run with: pytest -m slow
    """

    @pytest.mark.parametrize("example_path", EXAMPLE_FILES, ids=example_id)
    def test_example_executes(self, example_path: Path, output_dir: Path) -> None:
        """Each example runs to completion in its own interpreter, with its outputs redirected.

        The child runs from the repository root, which relative paths in examples assume, and gets
        the test process's backend and emulated devices, so multi-device examples keep running on
        several devices. An example that needs an unreachable dataset or service is skipped,
        because it exercises that service rather than Datarax; one that runs past its budget
        fails, because a timeout is a finding.
        """
        # Skip TFDS examples on macOS - TensorFlow hangs on ARM64
        # https://github.com/tensorflow/tensorflow/issues/52138
        if IS_MACOS and "tfds" in str(example_path).lower():
            pytest.skip("TFDS examples skipped on macOS (TensorFlow ARM64 issue)")

        try:
            result = run_python(
                example_path,
                timeout=EXAMPLE_TIMEOUT_SECONDS,
                env={
                    **forwarded_jax_environment(os.environ),
                    "TFDS_DISABLE_PROGRESS_BAR": "1",
                    OUTPUT_DIR_ENV: str(output_dir),
                },
                cwd=REPO_ROOT,
            )
        except subprocess.TimeoutExpired:
            pytest.fail(
                f"{example_id(example_path)} ran past its {EXAMPLE_TIMEOUT_SECONDS} s budget"
            )

        reason = unavailable_reason(result.returncode, result.stderr)
        if reason is not None:
            pytest.skip(
                f"Example needs an external dataset or network, unavailable here ({reason!r})."
            )
        result.check()
