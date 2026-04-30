"""Repository hygiene tests.

Enforces structural invariants that protect the source tree:
- Reserved namespaces (``datarax.workers``) must be self-documenting
  rather than empty placeholders.
- Documented public APIs (``datarax.memory.SharedMemoryManager``) must remain
  importable at the advertised path.
"""

from __future__ import annotations

import importlib

import pytest


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
