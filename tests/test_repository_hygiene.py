"""Repository hygiene tests.

Enforces structural invariants that protect the source tree:
- Removed packages stay removed: worker processes are Grain's, read through the host stage
  (``datarax.pipeline.host_workers``), so no ``datarax.workers`` placeholder or
  ``datarax.memory`` shared-memory manager exists beside them.
- Test subpackages must not be importable as top-level modules, where they would
  shadow the repository's own ``benchmarks``, ``examples`` and ``scripts``.
- The decoded copy's files, locks and writers are used through ``with`` only: no ``.close()``
  call is written by hand in the modules that write a copy.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path

import pytest


TESTS_DIR = Path(__file__).resolve().parent
REPOSITORY = TESTS_DIR.parent
TEST_SUBPACKAGES = sorted(path.parent.name for path in TESTS_DIR.glob("*/__init__.py"))
COPY_WRITERS = (
    "src/datarax/sources/decoded_copy/_files.py",
    "src/datarax/sources/decoded_copy/_units.py",
)
"""The modules that write a decoded copy: their resources are released by ``with`` alone."""


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


def _close_calls(source: str) -> list[int]:
    """The lines of every ``<object>.close()`` call in ``source``.

    ``os.close(descriptor)`` closes a raw descriptor, which no ``with`` holds; it is the body of
    the context manager that holds one, so it is not counted.
    """
    return [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "close"
        and not (isinstance(node.func.value, ast.Name) and node.func.value.id == "os")
    ]


@pytest.mark.parametrize("module", COPY_WRITERS)
def test_a_copy_writer_closes_nothing_by_hand(module: str) -> None:
    """Files, locks, maps and the provenance writer are context managers (global rule 7)."""
    assert _close_calls((REPOSITORY / module).read_text()) == []


def test_positive_control_a_hand_written_close_is_found() -> None:
    assert _close_calls("handle = open('x')\nhandle.close()\nos.close(3)\n") == [2]
