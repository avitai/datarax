"""Device requirements go through the substrax pytest plugin's markers.

A test that needs a GPU backend declares ``accelerator(kind="gpu")``, and one that needs several
devices declares ``devices(count)``. The plugin skips them from the backend and devices the run
selected, which ``DATARAX_TEST_JAX_PLATFORMS`` chooses. These markers are the only device
mechanism: no other device marker is registered or used, no pytest command passes a ``--device``
option, and no skip condition probes the devices JAX sees.
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path
from typing import TypeGuard

import pytest


ROOT = Path(__file__).resolve().parents[1]
OTHER_DEVICE_MARKERS = frozenset({"gpu", "tpu", "gpu_required", "cuda", "cpu"})
MARKER_USE = re.compile(r"pytest\.mark\.(?:gpu|tpu|gpu_required|cuda|cpu)\b")
DEVICE_OPTION = re.compile(r"\bpytest\b.*\s--device\b")
DEVICE_PROBES = frozenset({"device_count", "local_device_count", "devices"})

pytestmark = pytest.mark.contract


def python_modules() -> list[Path]:
    """Return every Python module under ``tests`` and ``benchmarks``."""
    return [*sorted((ROOT / "tests").rglob("*.py")), *sorted((ROOT / "benchmarks").rglob("*.py"))]


def pytest_device_option_lines(text: str) -> list[str]:
    """Return each pytest command in ``text`` that passes ``--device``, joining continued lines."""
    return [line for line in text.replace("\\\n", " ").splitlines() if DEVICE_OPTION.search(line)]


def _calls_pytest(node: ast.AST, name: str) -> TypeGuard[ast.Call]:
    """Whether ``node`` calls ``pytest.<name>`` or ``pytest.mark.<name>``."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
        and ast.unparse(node.func.value) in {"pytest", "pytest.mark"}
    )


def _probes_devices(condition: ast.AST) -> bool:
    """Whether ``condition`` calls ``device_count``, ``local_device_count`` or ``devices``."""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in DEVICE_PROBES
        for node in ast.walk(condition)
    )


def device_probing_skip_lines(source: str) -> list[int]:
    """Return the line of each ``skipif`` or guarded ``pytest.skip`` that probes devices."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if _calls_pytest(node, "skipif") and node.args and _probes_devices(node.args[0]):
            lines.append(node.lineno)
        elif (
            isinstance(node, ast.If)
            and _probes_devices(node.test)
            and any(
                isinstance(statement, ast.Expr) and _calls_pytest(statement.value, "skip")
                for statement in node.body
            )
        ):
            lines.append(node.lineno)
    return sorted(lines)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("uv run pytest tests --device=cpu -v", ["uv run pytest tests --device=cpu -v"]),
        ("./run_tests.sh --device=cpu", []),
        ("uv run pytest tests \\\n  --device=gpu", ["uv run pytest tests    --device=gpu"]),
    ],
)
def test_matcher_finds_pytest_commands_passing_the_device_option(
    text: str, expected: list[str]
) -> None:
    assert pytest_device_option_lines(text) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('@pytest.mark.skipif(jax.device_count() < 2, reason="two")\ndef test(): ...', [1]),
        ('x = 1\nif len(jax.devices()) < 2:\n    pytest.skip("two", allow_module_level=True)', [2]),
        ('@pytest.mark.skipif(sys.platform == "darwin", reason="linux")\ndef test(): ...', []),
        ("mesh = jax.make_mesh((jax.device_count(),), ('data',))", []),
    ],
)
def test_matcher_finds_skips_that_probe_devices(source: str, expected: list[int]) -> None:
    assert device_probing_skip_lines(source) == expected


def test_substrax_markers_are_the_only_device_markers() -> None:
    ini = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["pytest"]
    registered = {line.split(":")[0].strip() for line in ini["ini_options"]["markers"]}
    conftest = (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    users = [
        str(path.relative_to(ROOT))
        for path in python_modules()
        if MARKER_USE.search(path.read_text(encoding="utf-8"))
    ]

    control = (ROOT / "tests" / "examples" / "test_examples.py").read_text(encoding="utf-8")
    assert "slow" in registered
    assert re.search(r"pytest\.mark\.slow\b", control)
    assert registered & OTHER_DEVICE_MARKERS == set()
    assert [name for name in OTHER_DEVICE_MARKERS if f'"markers", "{name}:' in conftest] == []
    assert users == []


def test_no_skip_condition_probes_the_devices() -> None:
    offenders = {
        str(path.relative_to(ROOT)): lines
        for path in python_modules()
        if (lines := device_probing_skip_lines(path.read_text(encoding="utf-8")))
    }

    assert offenders == {}


def test_no_pytest_command_passes_a_device_option() -> None:
    conftest = (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    corpus = [
        *sorted((ROOT / ".github" / "workflows").glob("*.yml")),
        *sorted((ROOT / "scripts").glob("*.sh")),
        *sorted((ROOT / "docs").rglob("*.md")),
        ROOT / "tests" / "README.md",
    ]
    offenders = {
        str(path.relative_to(ROOT)): lines
        for path in corpus
        if (lines := pytest_device_option_lines(path.read_text(encoding="utf-8")))
    }

    assert '"--device"' not in conftest
    assert offenders == {}


@pytest.mark.parametrize("script", ["run_tests.sh", "run_gpu_tests.sh"])
def test_gpu_test_runners_select_cuda_through_the_test_variable(script: str) -> None:
    """Tests ignore an exported JAX_PLATFORMS, so a GPU run names the test variable."""
    text = (ROOT / "scripts" / script).read_text(encoding="utf-8")

    assert 'DATARAX_TEST_JAX_PLATFORMS="cuda"' in text
    assert 'JAX_PLATFORMS="cuda"' not in text.replace('DATARAX_TEST_JAX_PLATFORMS="cuda"', "")
