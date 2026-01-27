#!/usr/bin/env python
"""Validate example files for structure, quality, and execution.

This script validates that example files follow the 7-part documentation structure
and meet quality standards for the Datarax examples system.

Usage:
    python scripts/validate_examples.py                    # Validate all examples
    python scripts/validate_examples.py --path docs/examples/core/
    python scripts/validate_examples.py --execute          # Also run examples
    python scripts/validate_examples.py --verbose          # Show detailed output

Validation checks:
    1. 7-part structure (Header, Overview, Setup, Implementation, Results, Next Steps)
    2. Metadata table presence (Level, Runtime, Prerequisites)
    3. Learning objectives (numbered list)
    4. Proper cell markers (# %%)
    5. No star imports
    6. Notebook synchronization (.py matches .ipynb)
    7. Optional: Execution test
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Required sections in order of appearance
REQUIRED_SECTIONS = [
    "Overview",
    "Setup",
]

# Recommended sections (warn if missing)
RECOMMENDED_SECTIONS = [
    "Learning Goals",
    "Next Steps",
]

# Metadata fields that should be present
REQUIRED_METADATA = ["Level", "Runtime"]

# Patterns to detect issues
STAR_IMPORT_PATTERN = re.compile(r"^\s*from\s+\S+\s+import\s+\*", re.MULTILINE)
CELL_MARKER_PATTERN = re.compile(r"^# %%", re.MULTILINE)
MARKDOWN_CELL_PATTERN = re.compile(r"^# %% \[markdown\]", re.MULTILINE)
METADATA_TABLE_PATTERN = re.compile(r"\|\s*\*\*Level\*\*\s*\|")
LEARNING_GOALS_PATTERN = re.compile(r"(?:Learning Goals|Learning Objectives)", re.IGNORECASE)


@dataclass
class ValidationResult:
    """Result of validating a single example file."""

    file_path: Path
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        """Add an error (causes validation to fail)."""
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str) -> None:
        """Add a warning (doesn't fail validation)."""
        self.warnings.append(msg)

    def add_info(self, msg: str) -> None:
        """Add informational message."""
        self.info.append(msg)


def validate_structure(content: str, result: ValidationResult) -> None:
    """Validate the 7-part documentation structure."""
    # Check for cell markers
    cell_markers = CELL_MARKER_PATTERN.findall(content)
    if len(cell_markers) < 3:
        result.add_error(f"Too few cell markers (# %%): found {len(cell_markers)}, need at least 3")

    # Check for markdown cells
    markdown_cells = MARKDOWN_CELL_PATTERN.findall(content)
    if len(markdown_cells) < 2:
        result.add_error(f"Too few markdown cells: found {len(markdown_cells)}, need at least 2")

    # Check for required sections
    for section in REQUIRED_SECTIONS:
        pattern = re.compile(rf"##\s+{section}", re.IGNORECASE)
        if not pattern.search(content):
            result.add_error(f"Missing required section: {section}")

    # Check for recommended sections
    for section in RECOMMENDED_SECTIONS:
        pattern = re.compile(rf"##\s+.*{section}", re.IGNORECASE)
        if not pattern.search(content):
            result.add_warning(f"Missing recommended section: {section}")


def validate_metadata(content: str, result: ValidationResult) -> None:
    """Validate presence of metadata table."""
    if not METADATA_TABLE_PATTERN.search(content):
        result.add_warning("Missing metadata table (| **Level** |)")

    for field_name in REQUIRED_METADATA:
        pattern = re.compile(rf"\*\*{field_name}\*\*", re.IGNORECASE)
        if not pattern.search(content):
            result.add_warning(f"Missing metadata field: {field_name}")


def validate_learning_goals(content: str, result: ValidationResult) -> None:
    """Validate presence of learning goals/objectives."""
    if not LEARNING_GOALS_PATTERN.search(content):
        result.add_warning("Missing 'Learning Goals' or 'Learning Objectives' section")
    else:
        # Check for numbered list near learning goals section
        # Look for the section and then check for numbered items within 500 chars
        goals_match = re.search(
            r"Learning (?:Goals|Objectives)(.{0,500})",
            content,
            re.IGNORECASE | re.DOTALL,
        )
        if goals_match:
            goals_text = goals_match.group(1)
            # Look for numbered list items (1., 2., etc.)
            if not re.search(r"\d+\.\s+\w", goals_text):
                result.add_warning("Learning goals should use numbered list (1., 2., etc.)")


def validate_imports(content: str, result: ValidationResult) -> None:
    """Validate import statements."""
    if STAR_IMPORT_PATTERN.search(content):
        result.add_error("Star imports (from X import *) are not allowed")


def validate_notebook_sync(py_path: Path, result: ValidationResult) -> None:
    """Validate that .py and .ipynb files are synchronized."""
    ipynb_path = py_path.with_suffix(".ipynb")
    if not ipynb_path.exists():
        result.add_error(f"Missing notebook file: {ipynb_path.name}")
        return

    # Check modification times
    py_mtime = py_path.stat().st_mtime
    ipynb_mtime = ipynb_path.stat().st_mtime

    # Allow 60 second tolerance for sync operations
    if abs(py_mtime - ipynb_mtime) > 60:
        if py_mtime > ipynb_mtime:
            result.add_warning(
                f"Notebook may be out of sync (py newer by {py_mtime - ipynb_mtime:.0f}s)"
            )
        else:
            result.add_warning(
                f"Python file may be out of sync (ipynb newer by {ipynb_mtime - py_mtime:.0f}s)"
            )


def validate_execution(py_path: Path, result: ValidationResult, timeout: int = 120) -> None:
    """Validate that the example executes without errors."""
    try:
        proc = subprocess.run(
            ["python", str(py_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=py_path.parent.parent.parent.parent,  # Run from repo root
        )
        if proc.returncode != 0:
            # Truncate long error messages
            stderr = proc.stderr[:500] if len(proc.stderr) > 500 else proc.stderr
            result.add_error(f"Execution failed (exit code {proc.returncode}): {stderr}")
        else:
            result.add_info("Execution passed")
    except subprocess.TimeoutExpired:
        result.add_error(f"Execution timed out after {timeout}s")
    except Exception as e:
        result.add_error(f"Execution error: {e}")


def validate_file(
    py_path: Path,
    check_execution: bool = False,
) -> ValidationResult:
    """Validate a single example file."""
    result = ValidationResult(file_path=py_path)

    if not py_path.exists():
        result.add_error("File does not exist")
        return result

    content = py_path.read_text()

    # Run all validations
    validate_structure(content, result)
    validate_metadata(content, result)
    validate_learning_goals(content, result)
    validate_imports(content, result)
    validate_notebook_sync(py_path, result)

    if check_execution:
        validate_execution(py_path, result)

    return result


def find_example_files(base_path: Path) -> list[Path]:
    """Find all Python example files in the given path."""
    if base_path.is_file():
        return [base_path] if base_path.suffix == ".py" else []

    examples = []
    for py_file in base_path.rglob("*.py"):
        # Skip __init__.py, test files, and non-example directories
        if py_file.name.startswith("_"):
            continue
        if "test" in py_file.name.lower():
            continue
        # Skip comparison directory (benchmark scripts, not tutorials)
        if "comparison" in py_file.parts:
            continue
        # Only include files with numbered prefixes (01_, 02_, etc.)
        if re.match(r"^\d+_", py_file.name):
            examples.append(py_file)

    return sorted(examples)


def print_result(result: ValidationResult, verbose: bool = False) -> None:
    """Print validation result for a single file."""
    status = "✅ PASS" if result.passed else "❌ FAIL"
    rel_path = (
        result.file_path.relative_to(Path.cwd())
        if result.file_path.is_relative_to(Path.cwd())
        else result.file_path
    )

    print(f"{status} {rel_path}")

    if result.errors:
        for error in result.errors:
            print(f"    ❌ ERROR: {error}")

    if verbose or not result.passed:
        for warning in result.warnings:
            print(f"    ⚠️  WARN: {warning}")

    if verbose:
        for info in result.info:
            print(f"    ℹ️  INFO: {info}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate Datarax example files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("examples"),
        help="Path to validate (file or directory)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Also execute examples to verify they run",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output including warnings and info",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Datarax Examples Validator")
    print("=" * 60)
    print()

    # Find example files
    examples = find_example_files(args.path)

    if not examples:
        print(f"No example files found in {args.path}")
        return 1

    print(f"Found {len(examples)} example file(s)")
    if args.execute:
        print("  (with execution testing enabled)")
    print()

    # Validate each file
    results = []
    for py_path in examples:
        result = validate_file(py_path, check_execution=args.execute)
        results.append(result)
        print_result(result, verbose=args.verbose)

    # Summary
    print()
    print("-" * 60)
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    total_errors = sum(len(r.errors) for r in results)
    total_warnings = sum(len(r.warnings) for r in results)

    print(f"Results: {passed}/{len(results)} passed")
    if total_errors:
        print(f"  Errors: {total_errors}")
    if total_warnings:
        print(f"  Warnings: {total_warnings}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
