"""Device requirements go through the substrax pytest plugin's markers.

A test that needs a GPU backend declares ``accelerator(kind="gpu")``, and one that needs several
devices declares ``devices(count)``. The plugin skips them from the backend and devices the run
selected, which ``DATARAX_TEST_JAX_PLATFORMS`` chooses. Markers that only a keyword filter behind a
pytest ``--device`` option acted on, and tests probing ``jax.devices()`` to skip themselves, are
gone.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REMOVED_MARKERS = frozenset({"gpu", "tpu", "gpu_required", "cuda", "cpu"})
MARKER_USE = re.compile(r"pytest\.mark\.(?:gpu|tpu|gpu_required|cuda|cpu)\b")
DEVICE_OPTION = re.compile(r"\bpytest\b.*\s--device\b")

pytestmark = pytest.mark.contract


def pytest_device_option_lines(text: str) -> list[str]:
    """Return each pytest command in ``text`` that passes ``--device``, joining continued lines."""
    return [line for line in text.replace("\\\n", " ").splitlines() if DEVICE_OPTION.search(line)]


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


def test_no_removed_device_marker_is_registered_or_used() -> None:
    ini = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["pytest"]
    registered = {line.split(":")[0].strip() for line in ini["ini_options"]["markers"]}
    conftest = (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    modules = [
        *sorted((ROOT / "tests").rglob("*.py")),
        *sorted((ROOT / "benchmarks").rglob("*.py")),
    ]
    users = [
        str(path.relative_to(ROOT))
        for path in modules
        if MARKER_USE.search(path.read_text(encoding="utf-8"))
    ]

    control = (ROOT / "tests" / "examples" / "test_examples.py").read_text(encoding="utf-8")
    assert "slow" in registered
    assert re.search(r"pytest\.mark\.slow\b", control)
    assert registered & REMOVED_MARKERS == set()
    assert [name for name in REMOVED_MARKERS if f'"markers", "{name}:' in conftest] == []
    assert users == []


def test_no_hardware_helper_modules_remain() -> None:
    assert not (ROOT / "tests" / "test_common" / "hardware_fixtures.py").exists()
    assert not (ROOT / "tests" / "test_common" / "device_detection.py").exists()


def test_no_pytest_command_passes_the_removed_device_option() -> None:
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
