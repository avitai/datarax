"""Load a repository script by path; ``scripts/`` is not an importable package."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str) -> ModuleType:
    """Execute ``scripts/<name>.py`` and return it as the module ``name``.

    The module is registered in ``sys.modules`` before it runs, as an import would register it;
    ``dataclasses`` looks a class's module up there.

    Args:
        name: The script's file name without ``.py``.

    Returns:
        The executed module.
    """
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
