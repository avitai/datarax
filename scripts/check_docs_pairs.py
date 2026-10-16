#!/usr/bin/env python
"""Check that a docs example page's code that shows output is the code of the script it documents.

A *pair* on a page under ``docs/examples`` is a python block followed by a
``**Terminal Output:**`` block. The output came from running the example script, so the code
above it must be that script's code: each pair's lines (trailing whitespace ignored) are one
contiguous run of the script's lines. A python block without output is an illustration and is not
checked. A page holding a pair names exactly one script under ``examples/``, so the script a pair
comes from is never ambiguous; such a page naming none or several is a finding, never skipped.

Usage:
    python scripts/check_docs_pairs.py            # check every page under docs/examples
    python scripts/check_docs_pairs.py --root DIR # check the repository at DIR

Exit codes:
    0 - every pair is its script's code
    1 - a pair is not, or a page with pairs names no single script
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


PAIR = re.compile(
    r"```python\n((?:(?!```).)*?)\n```[ \t]*\n\s*\*\*Terminal Output[^*\n]*:?\*\*[^\n]*\n"
    r"\s*```[^\n]*\n(?:(?!```).)*?\n```",
    re.DOTALL,
)
"""A python block and the output block after it; group 1 is the code."""

SCRIPT = re.compile(r"examples/(?:[a-z_]+/)*[0-9a-z_]+\.py")
"""A path to an example script, as a page names it (a repository path or inside a URL)."""


def docs_pages(root: Path) -> list[Path]:
    """The markdown pages under ``docs/examples``, sorted.

    Args:
        root: The repository root.

    Returns:
        Every ``.md`` file under ``root/docs/examples``.
    """
    return sorted((root / "docs" / "examples").rglob("*.md"))


def _lines(text: str) -> list[str]:
    return [line.rstrip() for line in text.splitlines()]


def _is_contiguous_run(needle: list[str], haystack: list[str]) -> bool:
    width = len(needle)
    return any(
        haystack[start : start + width] == needle for start in range(len(haystack) - width + 1)
    )


def check_page(page: Path, root: Path) -> list[str]:
    """The findings for one page: each pair whose code is not its script's code.

    Args:
        page: A markdown page.
        root: The repository root, which the page's script paths are relative to.

    Returns:
        One message per finding, empty when every pair holds.
    """
    text = page.read_text(encoding="utf-8")
    pairs = [_lines(code) for code in PAIR.findall(text)]
    if not pairs:
        return []
    scripts = sorted({path for path in SCRIPT.findall(text) if (root / path).is_file()})
    if len(scripts) != 1:
        return [
            f"{page}: shows output, so it must name exactly one example script; "
            f"it names {len(scripts)} ({', '.join(scripts) or 'none'})"
        ]
    source = _lines((root / scripts[0]).read_text(encoding="utf-8"))
    return [
        f"{page}: code is not a contiguous run of {scripts[0]} "
        f"(first line not found in order: {next((line for line in code if line not in source), code[0])!r})"
        for code in pairs
        if not _is_contiguous_run(code, source)
    ]


def main(argv: list[str] | None = None) -> int:
    """Check every docs example page and print each finding.

    Args:
        argv: Command-line arguments, ``sys.argv[1:]`` when ``None``.

    Returns:
        The exit code: 0 when every pair holds, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        description="Check that docs example code shown with output is the script's code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    findings = [
        finding for page in docs_pages(args.root) for finding in check_page(page, args.root)
    ]
    for finding in findings:
        print(finding)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
