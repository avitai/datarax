"""Meshes in datarax examples and pages come from substrax and are entered with ``jax.set_mesh``.

jax deprecates entering a ``Mesh`` as a context manager and names ``jax.set_mesh(mesh)`` as its
replacement, the form jax's and Flax NNX's docs and substrax use. ``jax.set_mesh`` sets the concrete
and abstract mesh, so APIs that take a sharding also accept a bare ``PartitionSpec`` inside it.

Examples, benchmarks and docs pages build meshes with ``substrax.mesh.DeviceMeshManager``, whose
axes are Auto, and shard batches with ``substrax.spmd``, so a reader copies one pattern. Tests
build meshes directly, because they exercise the sharding mechanics themselves.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve()
MESH_CONTEXT = re.compile(r"\bwith\s+(?:(?:[A-Za-z_]\w*\.)*\w*mesh\s*:|(?:[A-Za-z_]\w*\.)*Mesh\()")
MESH_CONSTRUCTION = re.compile(r"\bMesh\(|\bmake_mesh\(")

pytestmark = pytest.mark.contract


def matching_lines(pattern: re.Pattern[str], text: str) -> list[int]:
    """Return the 1-based numbers of the lines in ``text`` that ``pattern`` matches."""
    return [
        number for number, line in enumerate(text.splitlines(), start=1) if pattern.search(line)
    ]


def context_corpus() -> list[Path]:
    """The README, docs pages, and the Python files and notebooks a reader or test runs."""
    paths = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    for root in ("src", "tests", "benchmarks", "examples"):
        paths += sorted((ROOT / root).rglob("*.py"))
    paths += sorted((ROOT / "examples").rglob("*.ipynb"))
    return [path for path in paths if path.resolve() != THIS_FILE]


def construction_corpus() -> list[Path]:
    """The README, docs pages, examples with their notebooks, and benchmarks outside their tests."""
    paths = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    paths += sorted((ROOT / "examples").rglob("*.py"))
    paths += sorted((ROOT / "examples").rglob("*.ipynb"))
    tests = ROOT / "benchmarks" / "tests"
    paths += [
        path for path in sorted((ROOT / "benchmarks").rglob("*.py")) if tests not in path.parents
    ]
    return paths


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("with mesh:\n    pass", [1]),
        ("x = 1\n        with self.mesh:", [2]),
        ('    "    with data_mesh:\\n",', [1]),
        ('with Mesh(devices, axis_names=("data",)):', [1]),
        ('with jax.sharding.Mesh(devices, ("x",)):', [1]),
        ("with jax.set_mesh(mesh):", []),
        ('mesh = jax.sharding.Mesh(devices, ("data",))', []),
    ],
)
def test_context_matcher_finds_only_direct_mesh_contexts(text: str, expected: list[int]) -> None:
    assert matching_lines(MESH_CONTEXT, text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('mesh = Mesh(devices, axis_names=("data",))', [1]),
        ('mesh = jax.sharding.Mesh(devices, ("x",))', [1]),
        ('mesh = jax.make_mesh((4,), ("data",), axis_types=types)', [1]),
        ("mesh = DeviceMeshManager.create_data_parallel_mesh()", []),
        ('abstract = AbstractMesh((2,), ("x",))', []),
    ],
)
def test_construction_matcher_finds_raw_mesh_constructors(text: str, expected: list[int]) -> None:
    assert matching_lines(MESH_CONSTRUCTION, text) == expected


def test_corpora_reach_what_their_rules_cover() -> None:
    contexts = {str(path.relative_to(ROOT)) for path in context_corpus()}
    constructions = {str(path.relative_to(ROOT)) for path in construction_corpus()}

    assert {
        "README.md",
        "src/datarax/sharding/jax_process_sharder.py",
        "tests/sharding/test_direct_sharding.py",
        "benchmarks/distributed_scaling_benchmark.py",
        "examples/advanced/distributed/02_sharding_guide.ipynb",
        "docs/examples/advanced/distributed/sharding-guide.md",
    } <= contexts
    assert {
        "README.md",
        "docs/sharding/index.md",
        "examples/advanced/distributed/01_sharding_quickref.py",
        "examples/advanced/distributed/01_sharding_quickref.ipynb",
        "benchmarks/distributed_scaling_benchmark.py",
    } <= constructions
    assert not any(name.startswith("benchmarks/tests/") for name in constructions)
    assert not any(name.startswith("tests/") for name in constructions)


def test_no_code_or_page_uses_the_deprecated_mesh_context() -> None:
    violations = {
        str(path.relative_to(ROOT)): lines
        for path in context_corpus()
        if (lines := matching_lines(MESH_CONTEXT, path.read_text(encoding="utf-8")))
    }

    assert violations == {}


def test_examples_benchmarks_and_pages_build_meshes_through_substrax() -> None:
    violations = {
        str(path.relative_to(ROOT)): lines
        for path in construction_corpus()
        if (lines := matching_lines(MESH_CONSTRUCTION, path.read_text(encoding="utf-8")))
    }

    assert violations == {}
