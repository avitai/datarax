"""``scripts/require_python.py`` fails a CI leg whose interpreter is not its matrix version."""

from __future__ import annotations

import subprocess  # nosec B404
import sys
from types import ModuleType

import pytest

from tests.scripts.script_loader import load_script, REPO_ROOT


SCRIPT = REPO_ROOT / "scripts" / "require_python.py"
RUNNING = f"{sys.version_info.major}.{sys.version_info.minor}"
OTHER = "3.12" if RUNNING != "3.12" else "3.13"


@pytest.fixture(scope="module")
def require_python() -> ModuleType:
    return load_script("require_python")


def test_the_running_version_is_accepted(require_python: ModuleType) -> None:
    assert require_python.main([RUNNING]) == 0


def test_another_version_is_refused_naming_both(require_python: ModuleType) -> None:
    with pytest.raises(SystemExit, match=rf"runs Python {RUNNING}, not {OTHER}"):
        require_python.main([OTHER])


@pytest.mark.parametrize("argument", ["3", "3.13.1", "three.thirteen"])
def test_a_version_that_is_not_major_minor_is_refused(
    require_python: ModuleType, argument: str
) -> None:
    with pytest.raises(SystemExit, match="major.minor"):
        require_python.main([argument])


@pytest.mark.parametrize(("version", "fails"), [(RUNNING, False), (OTHER, True)])
def test_the_script_s_exit_status_is_the_verdict(version: str, fails: bool) -> None:
    result = subprocess.run(  # nosec B603
        [sys.executable, str(SCRIPT), version], capture_output=True, text=True, check=False
    )

    assert (result.returncode != 0) is fails, result.stderr
    assert RUNNING in result.stdout + result.stderr
