"""Repository hygiene tests.

Enforces structural invariants that protect the source tree:
- Reserved namespaces (``datarax.workers``) must be self-documenting
  rather than empty placeholders.
- Documented public APIs (``datarax.memory.SharedMemoryManager``) must remain
  importable at the advertised path.
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


def test_workers_module_imports_cleanly() -> None:
    """The reserved-namespace placeholder must import without side effects."""
    module = importlib.import_module("datarax.workers")
    assert module is not None


def test_workers_module_init_documents_reservation() -> None:
    """The placeholder docstring must explicitly state the reservation.

    An empty 1-line docstring would read as dead code. The docstring must
    name the reserved purpose (multiprocessing) and at least one existing
    alternative so contributors don't reinvent worker logic in other modules.
    """
    module = importlib.import_module("datarax.workers")
    docstring = module.__doc__ or ""
    assert "multiprocessing" in docstring.lower(), (
        "datarax.workers is reserved for the multiprocessing backend; "
        "the module docstring must say so explicitly."
    )
    assert "MemorySource" in docstring or "prefetcher" in docstring, (
        "The docstring must list at least one existing parallel-worker "
        "concept (MemorySource(num_workers=...) or datarax.control.prefetcher) "
        "so contributors don't reinvent worker logic."
    )


def test_workers_module_raises_on_unknown_attr() -> None:
    """Accessing a non-existent symbol must raise ``NotImplementedError``.

    This prevents ``AttributeError`` ambiguity (is the symbol typo'd, missing,
    or reserved?) and makes the reservation discoverable at the access site.
    """
    module = importlib.import_module("datarax.workers")
    with pytest.raises(NotImplementedError, match="multiprocessing"):
        _ = module.WorkerPoolModule


def test_memory_module_exports_documented_public_api() -> None:
    """The documented import ``from datarax.memory import SharedMemoryManager`` must work.

    The public docs advertise this exact path; tests and benchmarks import the
    submodule directly only as an implementation detail. The package-level
    re-export must keep the public API intact.
    """
    from datarax.memory import SharedMemoryManager  # noqa: PLC0415 — runtime import is the contract

    assert SharedMemoryManager is not None


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
