#!/usr/bin/env python
"""Check synchronization between Python scripts and Jupyter notebooks.

This script verifies that .py and .ipynb file pairs are properly synchronized
using jupytext's py:percent format. It compares the actual content, not just
modification times.

Usage:
    python scripts/check_sync.py                     # Check all examples
    python scripts/check_sync.py --path examples/core/
    python scripts/check_sync.py --fix               # Auto-regenerate out-of-sync notebooks
    python scripts/check_sync.py --verbose           # Show detailed output

Exit codes:
    0 - All files in sync
    1 - Some files out of sync or missing pairs
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# Allow importing sibling scripts (validate_examples lives in the same directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_examples import find_example_files


def extract_code_from_py(py_path: Path) -> list[str]:
    """Extract code cells from a Python percent-format file.

    Args:
        py_path: Path to the Python file.

    Returns:
        List of code cell contents (stripped).
    """
    content = py_path.read_text()
    cells = []

    # Split on cell markers
    parts = re.split(r"^# %%.*$", content, flags=re.MULTILINE)

    for part in parts[1:]:  # Skip content before first cell marker
        # Skip markdown cells (they start with triple quotes after the marker)
        # Also handle raw strings (r""") used for markdown with special characters
        stripped = part.strip()
        if stripped.startswith(('"""', "'''", 'r"""', "r'''")):
            continue
        if stripped:
            cells.append(stripped)

    return cells


def extract_code_from_ipynb(ipynb_path: Path) -> list[str]:
    """Extract code cells from a Jupyter notebook.

    Args:
        ipynb_path: Path to the notebook file.

    Returns:
        List of code cell contents (stripped).
    """
    with open(ipynb_path) as f:
        notebook = json.load(f)

    cells = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            if isinstance(source, list):
                content = "".join(source).strip()
            else:
                content = source.strip()
            if content:
                cells.append(content)

    return cells


def normalize_code(code: str) -> str:
    """Normalize code for comparison.

    Removes whitespace differences that don't affect execution.

    Args:
        code: Code string to normalize.

    Returns:
        Normalized code string.
    """
    # Remove trailing whitespace from lines
    lines = [line.rstrip() for line in code.split("\n")]
    # Remove empty lines at start/end
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    # Collapse consecutive blank lines (ruff format adds PEP 8 double blanks
    # but jupytext collapses them in notebooks)
    collapsed: list[str] = []
    prev_blank = False
    for line in lines:
        is_blank = not line
        if is_blank and prev_blank:
            continue
        collapsed.append(line)
        prev_blank = is_blank
    return "\n".join(collapsed)


def compare_files(py_path: Path, ipynb_path: Path) -> tuple[bool, str]:
    """Compare a Python file with its corresponding notebook.

    Args:
        py_path: Path to the Python file.
        ipynb_path: Path to the notebook file.

    Returns:
        Tuple of (is_synced, message).
    """
    if not ipynb_path.exists():
        return False, "notebook missing"

    try:
        py_cells = extract_code_from_py(py_path)
        nb_cells = extract_code_from_ipynb(ipynb_path)
    except Exception as e:
        return False, f"parse error: {e}"

    # Normalize and compare
    py_normalized = [normalize_code(c) for c in py_cells]
    nb_normalized = [normalize_code(c) for c in nb_cells]

    if len(py_normalized) != len(nb_normalized):
        return False, f"cell count mismatch (py: {len(py_normalized)}, nb: {len(nb_normalized)})"

    for i, (py_cell, nb_cell) in enumerate(zip(py_normalized, nb_normalized)):
        if py_cell != nb_cell:
            # Find first difference
            py_lines = py_cell.split("\n")
            nb_lines = nb_cell.split("\n")
            for j, (pl, nl) in enumerate(zip(py_lines, nb_lines)):
                if pl != nl:
                    return False, f"content differs at cell {i + 1}, line {j + 1}"
            if len(py_lines) != len(nb_lines):
                return False, f"line count differs at cell {i + 1}"
            return False, f"content differs at cell {i + 1}"

    return True, "synced"


def regenerate_notebook(py_path: Path, verbose: bool = False) -> bool:
    """Regenerate notebook from Python file using jupytext.

    Args:
        py_path: Path to the Python file.
        verbose: Show command output.

    Returns:
        True if successful.
    """
    try:
        result = subprocess.run(
            [
                "python",
                "scripts/jupytext_converter.py",
                "py-to-nb",
                str(py_path),
            ],
            capture_output=not verbose,
            text=True,
            cwd=py_path.parent.parent.parent.parent,  # Repo root
        )
        return result.returncode == 0
    except Exception:
        return False


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check synchronization between .py and .ipynb files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("examples"),
        help="Path to check (file or directory)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Regenerate out-of-sync notebooks",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Notebook Sync Checker")
    print("=" * 60)
    print()

    # Find files
    examples = find_example_files(args.path)

    if not examples:
        print(f"No example files found in {args.path}")
        return 1

    print(f"Checking {len(examples)} file(s)")
    if args.fix:
        print("  (auto-fix mode enabled)")
    print()

    # Check each file
    synced = 0
    out_of_sync = 0
    fixed = 0

    for py_path in examples:
        ipynb_path = py_path.with_suffix(".ipynb")
        is_synced, message = compare_files(py_path, ipynb_path)

        rel_path = (
            py_path.relative_to(Path.cwd()) if py_path.is_relative_to(Path.cwd()) else py_path
        )

        if is_synced:
            synced += 1
            if args.verbose:
                print(f"✅ {rel_path}")
        else:
            if args.fix and message != "notebook missing":
                if regenerate_notebook(py_path, args.verbose):
                    fixed += 1
                    print(f"🔧 {rel_path} ({message}) -> fixed")
                else:
                    out_of_sync += 1
                    print(f"❌ {rel_path} ({message}) -> fix failed")
            else:
                out_of_sync += 1
                print(f"❌ {rel_path} ({message})")

    # Summary
    print()
    print("-" * 60)
    print(f"Results: {synced} synced, {out_of_sync} out of sync")
    if fixed:
        print(f"  Fixed: {fixed}")

    return 0 if out_of_sync == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
