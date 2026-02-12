"""Root conftest: adds --all-suites flag to run all test directories.

By default, pytest only runs tests/ (configured in pyproject.toml testpaths).
Use --all-suites to also collect benchmarks/tests/ and tools/benchkit/tests/.

Usage:
    uv run pytest                    # core tests only (tests/)
    uv run pytest --all-suites       # all test suites
"""

from __future__ import annotations

from pathlib import Path


def pytest_addoption(parser):
    parser.addoption(
        "--all-suites",
        action="store_true",
        default=False,
        help="Run all test suites: tests/, benchmarks/tests/, tools/benchkit/tests/",
    )


def pytest_configure(config):
    if not config.getoption("--all-suites", default=False):
        return

    root = Path(config.rootpath)
    extra_dirs = [
        root / "benchmarks" / "tests",
        root / "tools" / "benchkit" / "tests",
    ]

    for d in extra_dirs:
        if d.is_dir():
            config.args.append(str(d))
