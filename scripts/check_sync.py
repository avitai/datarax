#!/usr/bin/env python
"""Check synchronization between Python scripts and Jupyter notebooks.

This script verifies that .py and .ipynb file pairs are properly synchronized
using jupytext's py:percent format. It compares every code and markdown cell's
type and text, not modification times.

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
import sys
from pathlib import Path

import jupytext


# Allow importing sibling scripts (validate_examples lives in the same directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jupytext_converter import convert_py_to_nb
from validate_examples import find_example_files


type Cell = tuple[str, str]
"""A notebook cell as its type (``code`` or ``markdown``) and its text."""


def cells_from_py(py_path: Path) -> list[Cell]:
    """The cells of a percent-format example, as jupytext reads them for the notebook.

    jupytext writes the pairs, so its reading is the one that counts: it parses the
    ``# %% [markdown]`` cells written as string literals, and follows ``.jupytext.toml``.

    Args:
        py_path: Path to the Python file.

    Returns:
        The non-empty cells, in order.
    """
    return [
        (cell.cell_type, cell.source.strip())
        for cell in jupytext.read(py_path).cells
        if cell.source.strip()
    ]


def cells_from_ipynb(ipynb_path: Path) -> list[Cell]:
    """The code and markdown cells of a notebook.

    Read as JSON: the notebooks carry no cell ids (the converter strips them), which
    jupytext's reader would warn about through nbformat.

    Args:
        ipynb_path: Path to the notebook file.

    Returns:
        The non-empty cells, in order.
    """
    with ipynb_path.open() as f:
        notebook = json.load(f)

    cells = []
    for cell in notebook.get("cells", []):
        source = cell.get("source", [])
        content = ("".join(source) if isinstance(source, list) else source).strip()
        if cell.get("cell_type") in {"code", "markdown"} and content:
            cells.append((cell["cell_type"], content))
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
        py_cells = cells_from_py(py_path)
        nb_cells = cells_from_ipynb(ipynb_path)
    except Exception as e:
        return False, f"parse error: {e}"

    if len(py_cells) != len(nb_cells):
        return False, f"cell count mismatch (py: {len(py_cells)}, nb: {len(nb_cells)})"

    for i, ((py_type, py_text), (nb_type, nb_text)) in enumerate(
        zip(py_cells, nb_cells, strict=True), start=1
    ):
        if py_type != nb_type:
            return False, f"cell {i} is {py_type} in the script and {nb_type} in the notebook"
        py_lines = normalize_code(py_text).split("\n")
        nb_lines = normalize_code(nb_text).split("\n")
        if py_lines != nb_lines:
            line = next(
                (j for j, (a, b) in enumerate(zip(py_lines, nb_lines), start=1) if a != b),
                min(len(py_lines), len(nb_lines)) + 1,
            )
            return False, f"{py_type} cell {i} differs at line {line}"

    return True, "synced"


def regenerate_notebook(py_path: Path, verbose: bool = False) -> bool:
    """Regenerate the notebook paired with a Python file through ``jupytext_converter``.

    The converter runs jupytext with the interpreter running this script, so the result does
    not depend on which ``python`` the shell resolves, and it reports its own failures.

    Args:
        py_path: Path to the Python file.
        verbose: Show the converter's detailed output.

    Returns:
        True if the notebook was regenerated.
    """
    return convert_py_to_nb(py_path, verbose=verbose)


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
