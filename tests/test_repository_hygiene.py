"""Repository hygiene tests.

Enforces structural invariants that protect the source tree:
- Removed packages stay removed: worker processes are Grain's, read through the host stage
  (``datarax.pipeline.host_workers``), so no ``datarax.workers`` placeholder or
  ``datarax.memory`` shared-memory manager exists beside them.
- Test subpackages must not be importable as top-level modules, where they would
  shadow the repository's own ``benchmarks``, ``examples`` and ``scripts``.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest


TESTS_DIR = Path(__file__).resolve().parent
TEST_SUBPACKAGES = sorted(path.parent.name for path in TESTS_DIR.glob("*/__init__.py"))


@pytest.mark.parametrize("name", ["datarax.workers", "datarax.memory"])
def test_removed_packages_do_not_import(name: str) -> None:
    """Grain's worker processes replace the reserved ``workers`` namespace and the manager."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(name)


@pytest.mark.parametrize("name", TEST_SUBPACKAGES)
def test_test_subpackages_are_not_top_level_modules(name: str) -> None:
    """A ``tests`` subpackage is reachable only as ``tests.<name>``.

    ``tests/benchmarks`` shares its name with the repository's ``benchmarks``
    package. With ``tests/`` itself on ``sys.path``, ``import benchmarks`` finds the
    test package, ``benchmarks.core`` stops importing and the Deep Lake preload in
    ``tests/conftest.py`` is skipped without an error.
    """
    spec = importlib.util.find_spec(name)
    locations = [] if spec is None else [spec.origin, *(spec.submodule_search_locations or [])]

    assert not [
        location
        for location in locations
        if location is not None and Path(location).resolve().is_relative_to(TESTS_DIR)
    ]
