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


@pytest.fixture
def paired_example(check_sync: ModuleType, tmp_path: Path) -> Path:
    """An example script with its notebook regenerated from it, so the two are in sync."""
    example = tmp_path / "01_small_example.py"
    example.write_text(_EXAMPLE, encoding="utf-8")
    assert check_sync.regenerate_notebook(example) is True
    return example


def test_a_regenerated_pair_is_in_sync(check_sync: ModuleType, paired_example: Path) -> None:
    assert check_sync.compare_files(paired_example, paired_example.with_suffix(".ipynb")) == (
        True,
        "synced",
    )


def test_a_markdown_only_edit_is_out_of_sync(check_sync: ModuleType, paired_example: Path) -> None:
    """Prose is part of the pair: a notebook with stale markdown is out of sync."""
    paired_example.write_text(
        _EXAMPLE.replace("# A small example", "# A renamed example"), encoding="utf-8"
    )

    is_synced, message = check_sync.compare_files(
        paired_example, paired_example.with_suffix(".ipynb")
    )

    assert is_synced is False
    assert "markdown cell 1" in message


def test_a_code_edit_is_out_of_sync(check_sync: ModuleType, paired_example: Path) -> None:
    paired_example.write_text(_EXAMPLE.replace("1 + 1", "2 + 2"), encoding="utf-8")

    is_synced, message = check_sync.compare_files(
        paired_example, paired_example.with_suffix(".ipynb")
    )

    assert is_synced is False
    assert "code cell 2" in message


def test_fix_regenerates_a_notebook_with_stale_markdown(
    check_sync: ModuleType, paired_example: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paired_example.write_text(
        _EXAMPLE.replace("# A small example", "# A renamed example"), encoding="utf-8"
    )
    monkeypatch.setattr(
        "sys.argv", ["check_sync.py", "--path", str(paired_example.parent), "--fix"]
    )

    assert check_sync.main() == 0
    notebook = paired_example.with_suffix(".ipynb").read_text(encoding="utf-8")
    assert "# A renamed example" in notebook
    assert "# A small example" not in notebook
