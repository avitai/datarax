"""Tests for Datarax example files.

These tests validate that example files:
1. Follow the 7-part documentation structure
2. Have proper metadata
3. Are synchronized with their notebook counterparts
4. Execute without errors (optional, slow tests)
"""

from __future__ import annotations

import importlib.util
import os
import platform
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest
from substrax.testing import run_example, unavailable_reason

from tests.jax_test_environment import forwarded_jax_environment


# Detect macOS - TensorFlow crashes on macOS ARM64
IS_MACOS = platform.system() == "Darwin"

# Path to runnable examples directory
REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_DIR = REPO_ROOT / "examples"
EXAMPLE_TIMEOUT_SECONDS = int(os.environ.get("DATARAX_EXAMPLE_TIMEOUT_SECONDS", "120"))

# Required sections for validation
REQUIRED_SECTIONS = ["Overview", "Setup"]

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
RECOMMENDED_SECTIONS = ["Learning Goals", "Next Steps"]


def _load_validate_examples() -> ModuleType:
    """Load ``scripts/validate_examples.py``, which owns the rule for what counts as an example."""
    spec = importlib.util.spec_from_file_location(
        "validate_examples", REPO_ROOT / "scripts" / "validate_examples.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["validate_examples"] = module
    spec.loader.exec_module(module)
    return module


find_example_files = _load_validate_examples().find_example_files


# Generate test IDs from file paths
def example_id(path: Path) -> str:
    """Generate a short test ID from file path."""
    return str(path.relative_to(EXAMPLES_DIR))


EXAMPLE_FILES = find_example_files(EXAMPLES_DIR)


class TestExampleStructure:
    """Tests for example file structure and content."""

    @pytest.mark.parametrize("example_path", EXAMPLE_FILES, ids=example_id)
    def test_has_cell_markers(self, example_path: Path) -> None:
        """Each example should have proper Jupytext cell markers."""
        content = example_path.read_text()
        markers = re.findall(r"^# %%", content, re.MULTILINE)
        assert len(markers) >= 3, f"Expected at least 3 cell markers, found {len(markers)}"

    @pytest.mark.parametrize("example_path", EXAMPLE_FILES, ids=example_id)
    def test_has_markdown_cells(self, example_path: Path) -> None:
        """Each example should have markdown documentation cells."""
        content = example_path.read_text()
        markdown_cells = re.findall(r"^# %% \[markdown\]", content, re.MULTILINE)
        assert len(markdown_cells) >= 2, (
            f"Expected at least 2 markdown cells, found {len(markdown_cells)}"
        )

    @pytest.mark.parametrize("example_path", EXAMPLE_FILES, ids=example_id)
    def test_has_required_sections(self, example_path: Path) -> None:
        """Each example should have required documentation sections."""
        content = example_path.read_text()
        for section in REQUIRED_SECTIONS:
            pattern = re.compile(rf"##\s+{section}", re.IGNORECASE)
            assert pattern.search(content), f"Missing required section: {section}"

    @pytest.mark.parametrize("example_path", EXAMPLE_FILES, ids=example_id)
    def test_has_metadata_table(self, example_path: Path) -> None:
        """Each example should have a metadata table with Level and Runtime."""
        content = example_path.read_text()
        assert re.search(r"\*\*Level\*\*", content), "Missing Level in metadata"
        assert re.search(r"\*\*Runtime\*\*", content), "Missing Runtime in metadata"

    @pytest.mark.parametrize("example_path", EXAMPLE_FILES, ids=example_id)
    def test_no_star_imports(self, example_path: Path) -> None:
        """Examples should not use star imports."""
        content = example_path.read_text()
        star_imports = re.findall(r"^\s*from\s+\S+\s+import\s+\*", content, re.MULTILINE)
        assert not star_imports, f"Found star imports: {star_imports}"

    @pytest.mark.parametrize("example_path", EXAMPLE_FILES, ids=example_id)
    def test_has_learning_goals(self, example_path: Path) -> None:
        """Each example should have learning goals or objectives."""
        content = example_path.read_text()
        has_goals = re.search(r"Learning (?:Goals|Objectives)", content, re.IGNORECASE)
        assert has_goals, "Missing 'Learning Goals' or 'Learning Objectives' section"


class TestExampleSync:
    """Tests for example file synchronization."""

    @pytest.mark.parametrize("example_path", EXAMPLE_FILES, ids=example_id)
    def test_notebook_exists(self, example_path: Path) -> None:
        """Each Python example should have a corresponding notebook."""
        notebook_path = example_path.with_suffix(".ipynb")
        assert notebook_path.exists(), f"Missing notebook: {notebook_path.name}"

    @pytest.mark.parametrize("example_path", EXAMPLE_FILES, ids=example_id)
    def test_notebook_not_stale(self, example_path: Path) -> None:
        """Notebook should not be significantly older than Python file."""
        notebook_path = example_path.with_suffix(".ipynb")
        if not notebook_path.exists():
            pytest.skip("Notebook does not exist")

        py_mtime = example_path.stat().st_mtime
        nb_mtime = notebook_path.stat().st_mtime

        # Allow 5 minute tolerance
        tolerance = 300
        if py_mtime - nb_mtime > tolerance:
            pytest.fail(f"Notebook may be out of sync (py newer by {py_mtime - nb_mtime:.0f}s)")


class TestExampleDiscovery:
    """``scripts/validate_examples.py`` decides which files are runnable examples."""

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

        assert found == ["advanced/02_guide.py", "core/01_quickstart.py"]

    def test_a_single_example_file_is_returned_as_given(self, tmp_path: Path) -> None:
        example = tmp_path / "01_quickstart.py"
        example.write_text("", encoding="utf-8")

        assert find_example_files(example) == [example]


class TestExampleOutputs:
    """Examples write outputs through substrax.artifacts, so a run never rewrites tracked files."""

    @pytest.mark.parametrize("example_path", EXAMPLE_FILES, ids=example_id)
    def test_outputs_are_resolved_through_substrax(self, example_path: Path) -> None:
        content = example_path.read_text()

        assert "DATARAX_EXAMPLES_OUTPUT_DIR" not in content
        assert "docs/assets" not in content


@pytest.mark.slow
class TestExampleExecution:
    """Tests that execute example files.

    These tests are marked as slow and run in the long-running CI tier.
    Run with: pytest -m slow
    """

    @pytest.mark.parametrize("example_path", EXAMPLE_FILES, ids=example_id)
    def test_example_executes(self, example_path: Path, output_dir: Path) -> None:
        """Each example runs to completion in its own interpreter, with its outputs redirected.

        The child gets the test process's backend and emulated devices, so multi-device examples
        keep running on several devices. An example that needs an unreachable dataset or service
        is skipped, because it exercises that service rather than Datarax; one that runs past its
        budget fails, because a timeout is a finding.
        """
        # Skip TFDS examples on macOS - TensorFlow hangs on ARM64
        # https://github.com/tensorflow/tensorflow/issues/52138
        if IS_MACOS and "tfds" in str(example_path).lower():
            pytest.skip("TFDS examples skipped on macOS (TensorFlow ARM64 issue)")

        run = run_example(
            example_path,
            repo_root=REPO_ROOT,
            output_dir=output_dir,
            timeout=EXAMPLE_TIMEOUT_SECONDS,
            call_main=False,
            env={**forwarded_jax_environment(os.environ), "TFDS_DISABLE_PROGRESS_BAR": "1"},
        )

        reason = unavailable_reason(run, _NETWORK_FAILURE_SIGNATURES)
        if reason is not None:
            pytest.skip(
                f"Example needs an external dataset or network, unavailable here ({reason!r})."
            )
        run.result.check()
