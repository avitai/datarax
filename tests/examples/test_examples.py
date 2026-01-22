"""Tests for Datarax example files.

These tests validate that example files:
1. Follow the 7-part documentation structure
2. Have proper metadata
3. Are synchronized with their notebook counterparts
4. Execute without errors (optional, slow tests)
"""

from __future__ import annotations

import platform
import re
import subprocess
import sys
from pathlib import Path

import pytest

# Detect macOS - TensorFlow crashes on macOS ARM64
IS_MACOS = platform.system() == "Darwin"

# Path to examples directory
EXAMPLES_DIR = Path(__file__).parent.parent.parent / "docs" / "examples"

# Required sections for validation
REQUIRED_SECTIONS = ["Overview", "Setup"]
RECOMMENDED_SECTIONS = ["Learning Goals", "Next Steps"]


def find_example_files() -> list[Path]:
    """Find all Python example files."""
    if not EXAMPLES_DIR.exists():
        return []

    examples = []
    for py_file in EXAMPLES_DIR.rglob("*.py"):
        if py_file.name.startswith("_"):
            continue
        if "test" in py_file.name.lower():
            continue
        if re.match(r"^\d+_", py_file.name):
            examples.append(py_file)

    return sorted(examples)


# Generate test IDs from file paths
def example_id(path: Path) -> str:
    """Generate a short test ID from file path."""
    return str(path.relative_to(EXAMPLES_DIR))


EXAMPLE_FILES = find_example_files()


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


@pytest.mark.slow
class TestExampleExecution:
    """Tests that execute example files.

    These tests are marked as slow and skipped by default.
    Run with: pytest -m slow
    """

    @pytest.mark.parametrize("example_path", EXAMPLE_FILES, ids=example_id)
    def test_example_executes(self, example_path: Path) -> None:
        """Each example should execute without errors."""
        # Skip TFDS examples on macOS - TensorFlow hangs on ARM64
        # https://github.com/tensorflow/tensorflow/issues/52138
        if IS_MACOS and "tfds" in str(example_path).lower():
            pytest.skip("TFDS examples skipped on macOS (TensorFlow ARM64 issue)")

        result = subprocess.run(
            [sys.executable, str(example_path)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=example_path.parent.parent.parent.parent,  # Repo root
        )
        if result.returncode != 0:
            # Truncate long error messages
            stderr = result.stderr[:1000] if len(result.stderr) > 1000 else result.stderr
            pytest.fail(f"Example failed with exit code {result.returncode}:\n{stderr}")
