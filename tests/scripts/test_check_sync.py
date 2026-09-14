"""Tests for scripts/check_sync.py, which checks and regenerates example notebooks."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest

from tests.scripts.script_loader import load_script


_EXAMPLE = '''# %% [markdown]
"""
# A small example
"""

# %%
value = 1 + 1
print(value)
'''


@pytest.fixture(scope="module")
def check_sync() -> ModuleType:
    return load_script("check_sync")


def test_a_notebook_is_regenerated_without_python_on_the_path(
    check_sync: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The interpreter running the script regenerates the notebook, whatever PATH holds."""
    example = tmp_path / "01_small_example.py"
    example.write_text(_EXAMPLE, encoding="utf-8")
    monkeypatch.setenv("PATH", str(tmp_path / "no-interpreter-here"))

    assert check_sync.regenerate_notebook(example) is True
    assert example.with_suffix(".ipynb").is_file()
