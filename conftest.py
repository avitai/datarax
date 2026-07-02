"""Root conftest: adds --all-suites flag to run all test directories.

By default, pytest only runs tests/ (configured in pyproject.toml testpaths).
Use --all-suites to also collect benchmarks/tests/.

Usage:
    uv run pytest                    # core tests only (tests/)
    uv run pytest --all-suites       # tests/ + benchmarks/tests/
"""

from __future__ import annotations

from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--all-suites`` command-line flag."""
    parser.addoption(
        "--all-suites",
        action="store_true",
        default=False,
        help="Run all test suites: tests/, benchmarks/tests/",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Append extra suite directories to collection when ``--all-suites`` is set."""
    if not config.getoption("--all-suites", default=False):
        return

    root = Path(config.rootpath)
    extra_dirs = [
        root / "benchmarks" / "tests",
    ]

    for d in extra_dirs:
        if d.is_dir():
            config.args.append(str(d))
