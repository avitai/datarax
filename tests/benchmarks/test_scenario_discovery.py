"""The benchmark scenario catalogue is complete and fails loudly when broken."""

import ast
import importlib
from pathlib import Path

import pytest

from benchmarks.scenarios import _CATEGORY_PACKAGES, discover_scenarios


REPO_ROOT = Path(__file__).resolve().parents[2]


def _scenario_ids_on_disk() -> set[str]:
    """Read SCENARIO_ID from every module under the category packages without importing."""
    ids: set[str] = set()
    for package in _CATEGORY_PACKAGES:
        directory = REPO_ROOT / Path(*package.split("."))
        for path in directory.glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in tree.body:
                targets = (
                    node.targets
                    if isinstance(node, ast.Assign)
                    else [node.target]
                    if isinstance(node, ast.AnnAssign) and node.value is not None
                    else []
                )
                if any(
                    isinstance(target, ast.Name) and target.id == "SCENARIO_ID"
                    for target in targets
                ):
                    ids.add(ast.literal_eval(node.value))  # type: ignore[union-attr]
    return ids


def test_discovery_finds_every_scenario_module_on_disk():
    discovered = {module.SCENARIO_ID for module in discover_scenarios()}

    assert discovered == _scenario_ids_on_disk()
    assert len(discovered) > 0


def test_a_scenario_that_fails_to_import_fails_discovery(monkeypatch: pytest.MonkeyPatch):
    real_import = importlib.import_module

    def broken(name: str, *args, **kwargs):
        if name.endswith("cv1_image_classification"):
            raise ImportError("No module named 'PIL'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", broken)

    with pytest.raises(ImportError, match="PIL"):
        discover_scenarios()
