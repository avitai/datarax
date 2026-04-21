"""Tooling and dependency contracts for audit remediation."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml
from packaging.requirements import Requirement


REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
PRE_COMMIT = REPO_ROOT / ".pre-commit-config.yaml"
IMPORTLINTER = REPO_ROOT / ".importlinter"


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text())


def _dependency_names(dependencies: list[str]) -> list[str]:
    names = []
    for dependency in dependencies:
        # Handles version specs, extras, direct references, and environment markers.
        name = re.split(r"[<>=!~;\\[ @]", dependency, maxsplit=1)[0]
        names.append(name)
    return names


def test_grain_dependency_declares_single_package_with_current_api() -> None:
    """Datarax should target the package exposing the current Grain APIs."""
    config = _pyproject()
    base_dependencies = config["project"]["dependencies"]
    optional_dependencies = config["project"]["optional-dependencies"]
    all_dependencies = base_dependencies + [
        dependency for dependencies in optional_dependencies.values() for dependency in dependencies
    ]

    dependency_names = _dependency_names(all_dependencies)

    assert "grain-nightly>=0.2.16.dev20260112" in base_dependencies
    assert "grain" not in dependency_names
    assert _dependency_names(base_dependencies).count("grain-nightly") == 1


def test_tfds_dependency_set_is_warning_clean() -> None:
    """TFDS tests should not rely on third-party deprecation filters."""
    dependencies = {
        Requirement(dependency).name: Requirement(dependency)
        for dependency in _pyproject()["project"]["dependencies"]
    }

    assert dependencies["tensorflow"].specifier.contains("2.20.0")
    assert not dependencies["tensorflow"].specifier.contains("2.21.0")
    assert dependencies["protobuf"].specifier.contains("5.29.6")
    assert not dependencies["protobuf"].specifier.contains("6.0.0")


def test_python_version_range_matches_backend_support() -> None:
    """Do not advertise Python versions unsupported by JAX/TFDS backends."""
    requires_python = _pyproject()["project"]["requires-python"]

    assert requires_python == ">=3.11,<3.14"


def test_required_quality_tools_are_in_dev_extra() -> None:
    """The SWE guide gates must be installable from the dev extra."""
    dev_dependencies = _dependency_names(_pyproject()["project"]["optional-dependencies"]["dev"])

    required = {
        "bandit",
        "deadcode",
        "deptry",
        "flake8",
        "flake8-functions-names",
        "import-linter",
        "interrogate",
        "pylint",
        "pyright",
        "pytest-cov",
        "radon",
        "ruff",
        "vulture",
        "wemake-python-styleguide",
        "xenon",
    }

    assert required <= set(dev_dependencies)


def test_full_test_collection_dependencies_are_in_test_extra() -> None:
    """The default test extra must include packages imported during collection."""
    test_dependencies = _dependency_names(_pyproject()["project"]["optional-dependencies"]["test"])

    assert "matplotlib" in test_dependencies


def test_importlinter_contract_exists_for_datarax_layers() -> None:
    """Architecture checks should have a concrete Import Linter contract."""
    content = IMPORTLINTER.read_text()

    assert "root_package = datarax" in content
    assert "type = layers" in content
    for layer in (
        "datarax.cli",
        "datarax.monitoring",
        "datarax.dag",
        "datarax.operators",
        "datarax.sources",
        "datarax.distributed",
        "datarax.sharding",
        "datarax.control",
        "datarax.samplers",
        "datarax.batching",
        "datarax.core",
        "datarax.utils",
    ):
        assert layer in content


def test_protected_directories_are_excluded_from_tooling() -> None:
    """Protected generated/data directories should be excluded consistently."""
    config = _pyproject()
    pre_commit = yaml.safe_load(PRE_COMMIT.read_text())
    protected = "example_data"

    assert protected in config["tool"]["pyright"]["exclude"]
    assert protected in config["tool"]["bandit"]["exclude_dirs"]
    assert config["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]

    hook_excludes = [
        hook.get("exclude", "")
        for repo in pre_commit["repos"]
        for hook in repo.get("hooks", [])
        if hook.get("exclude")
    ]
    assert hook_excludes
    assert all(protected in exclude for exclude in hook_excludes)


def test_no_deferred_modernization_ignores_remain() -> None:
    """Modernization rules covered by the audit should be active."""
    ruff_ignores = set(_pyproject()["tool"]["ruff"]["lint"].get("ignore", []))

    assert {"UP006", "UP007", "UP015", "UP024", "E731"}.isdisjoint(ruff_ignores)
    assert "can be fixed later" not in PYPROJECT.read_text()
    assert "Fix Flax NNX deprecation" not in PYPROJECT.read_text()


def test_ruff_argument_limit_is_explicit_and_bounded() -> None:
    """Configuration-heavy public APIs need a documented, bounded PLR0913 limit."""
    pylint_config = _pyproject()["tool"]["ruff"]["lint"]["pylint"]

    assert pylint_config["max-args"] == 13
