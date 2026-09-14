"""Meshes that datarax documents use Auto axes.

``jax.make_mesh`` builds ``AxisType.Explicit`` axes unless it is given ``axis_types`` (jax 0.11),
and under Explicit axes the backward pass of a batch-sharded step raises. Every mesh a reader
copies, from a docstring, the README, a docs page or an example script, therefore names its
``axis_types``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CALL = "make_mesh("
DOCUMENTED_NODES = ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef

pytestmark = pytest.mark.contract


def unguarded_make_mesh_calls(text: str) -> list[str]:
    """Return each ``make_mesh(...)`` call in ``text`` whose arguments do not name ``axis_types``.

    A call whose parentheses never close is returned as well, from its name to the end of the text.
    """
    calls = []
    start = text.find(CALL)
    while start != -1:
        depth = 0
        end = len(text)
        for index in range(start + len(CALL) - 1, len(text)):
            depth += {"(": 1, ")": -1}.get(text[index], 0)
            if depth == 0:
                end = index + 1
                break
        call = text[start:end]
        if depth != 0 or "axis_types" not in call:
            calls.append(call)
        start = text.find(CALL, start + len(CALL))
    return calls


def documented_texts() -> dict[str, str]:
    """Map every docstring under src/datarax, the README, docs pages and examples to its text."""
    texts: dict[str, str] = {}
    for path in sorted((ROOT / "src" / "datarax").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, DOCUMENTED_NODES) and (docstring := ast.get_docstring(node)):
                texts[f"{path.relative_to(ROOT)}:{getattr(node, 'lineno', 0)}"] = docstring
    pages = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    scripts = sorted((ROOT / "examples").rglob("*.py"))
    for path in [*pages, *scripts]:
        texts[str(path.relative_to(ROOT))] = path.read_text(encoding="utf-8")
    return texts


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('mesh = jax.make_mesh((4,), ("data",))', ['make_mesh((4,), ("data",))']),
        ('jax.make_mesh((4, 2), ("data", "model"), axis_types=(AxisType.Auto,) * 2)', []),
        ('jax.make_mesh(\n    (4, 2),\n    ("data", "model"),\n    axis_types=types,\n)', []),
        ('jax.sharding.Mesh(devices, ("data",))', []),
        (
            'a = jax.make_mesh((1,), ("x",), axis_types=t)\nb = jax.make_mesh((2,), ("y",))',
            ['make_mesh((2,), ("y",))'],
        ),
        ('jax.make_mesh((4,), ("data",)\n', ['make_mesh((4,), ("data",)\n']),
    ],
)
def test_scanner_returns_only_calls_without_axis_types(text: str, expected: list[str]) -> None:
    assert unguarded_make_mesh_calls(text) == expected


def test_corpus_reaches_docstrings_readme_docs_pages_and_examples() -> None:
    texts = documented_texts()

    assert any(name.startswith("src/datarax/") for name in texts)
    assert "README.md" in texts
    assert "docs/sharding/index.md" in texts
    assert "examples/advanced/distributed/02_sharding_guide.py" in texts


def test_documented_meshes_name_their_axis_types() -> None:
    violations = {
        name: calls
        for name, text in documented_texts().items()
        if (calls := unguarded_make_mesh_calls(text))
    }

    assert violations == {}
