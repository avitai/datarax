"""Tooling and dependency contracts for audit remediation."""

from __future__ import annotations

import ast
import re
import tomllib
from collections.abc import Callable
from pathlib import Path

import yaml
from packaging.requirements import Requirement


REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
PRE_COMMIT = REPO_ROOT / ".pre-commit-config.yaml"
IMPORTLINTER = REPO_ROOT / ".importlinter"
SETUP_SCRIPT = REPO_ROOT / "setup.sh"
# An extra setup.sh documents (`name` extra) or syncs (--extra name).
_EXTRA_REFERENCE = re.compile(r"`([a-z0-9][a-z0-9_-]*)` extra|--extra ([a-z0-9][a-z0-9_-]*)")


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

    assert "grain>=0.2.18" in base_dependencies
    assert "grain-nightly" not in dependency_names
    assert _dependency_names(base_dependencies).count("grain") == 1


def test_tfds_dependency_set_is_warning_clean() -> None:
    """TFDS tests should not rely on third-party deprecation filters."""
    config = _pyproject()
    base = {
        Requirement(dependency).name: Requirement(dependency)
        for dependency in config["project"]["dependencies"]
    }
    # tensorflow is deliberately not a base dependency: a JAX-native package should not
    # pull TensorFlow on a plain install, so it lives in the tfds extra beside
    # tensorflow-datasets. The ceiling is what this test exists to hold, and it moves with
    # the declaration rather than staying pinned to where the declaration used to be.
    tfds = {
        Requirement(dependency).name: Requirement(dependency)
        for dependency in config["project"]["optional-dependencies"]["tfds"]
    }

    assert "tensorflow" not in base, "tensorflow must stay out of the base install"
    assert tfds["tensorflow"].specifier.contains("2.20.0")
    assert not tfds["tensorflow"].specifier.contains("2.21.0")
    assert base["protobuf"].specifier.contains("5.29.6")
    assert not base["protobuf"].specifier.contains("6.0.0")


def test_python_version_range_matches_backend_support() -> None:
    """Do not advertise Python versions unsupported by JAX/TFDS backends."""
    requires_python = _pyproject()["project"]["requires-python"]

    assert requires_python == ">=3.12,<3.14"


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
        "datarax.pipeline",
        "datarax.operators",
        "datarax.sources",
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


def test_setup_script_names_only_declared_extras() -> None:
    """Every extra setup.sh documents or syncs is declared in pyproject.toml."""
    named = {
        documented or synced
        for documented, synced in _EXTRA_REFERENCE.findall(SETUP_SCRIPT.read_text())
    }
    declared = set(_pyproject()["project"]["optional-dependencies"])

    assert {"dev", "test", "cuda12"} <= named
    assert named - declared == set()


CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
# Test module roots CI must cover: the core suite and the benchmark harness suite.
TEST_ROOTS = ("tests", "benchmarks/tests")
CAP_OVERRIDES = ("--fail-under=0", "--no-cov", "-o addopts", "--override-ini")


def _is_under(path: str, root: str) -> bool:
    return path == root or path.startswith(f"{root.rstrip('/')}/")


def _pytest_roots(command: str) -> list[str]:
    """Return the path arguments of the pytest invocations in a workflow ``run`` script."""
    joined = command.replace("\\\n", " ")
    roots = []
    for line in joined.splitlines():
        if "pytest" not in line:
            continue
        arguments = line.split("pytest", 1)[1].split()
        roots += [
            a.rstrip("/")
            for a in arguments
            if not a.startswith("-") and a.split("/")[0] in {"tests", "benchmarks"}
        ]
    return roots


def test_every_test_module_is_selected_by_a_ci_job() -> None:
    """Each ``test_*.py`` under the test roots is on the path of some CI pytest job."""
    workflow = yaml.safe_load(CI_WORKFLOW.read_text())
    roots = [
        root
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        for root in _pytest_roots(str(step.get("run", "")))
    ]
    modules = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for test_root in TEST_ROOTS
        for path in (REPO_ROOT / test_root).rglob("test_*.py")
    )

    assert len(modules) > 200
    assert [
        module for module in modules if not any(_is_under(module, root) for root in roots)
    ] == []


def coverage_floor_violations(workflow: dict, pyproject: dict) -> list[str]:
    """Return why CI would not fail below the coverage floor, if it would not."""
    # PyYAML reads the `on:` key as the boolean True.
    triggers = workflow.get("on") or workflow.get(True) or {}
    job = workflow["jobs"]["coverage"]
    commands = "\n".join(str(step.get("run", "")) for step in job["steps"])
    floor = pyproject["tool"]["coverage"]["report"].get("fail_under")

    problems = []
    if floor is None or float(floor) < 80:
        problems.append(f"[tool.coverage.report] fail_under is {floor}, not at least 80")
    if not {"push", "pull_request"} <= set(triggers):
        problems.append(f"CI runs on {sorted(triggers)}, not on both push and pull_request")
    if "if" in job:
        problems.append(f"the coverage job only runs when {job['if']}")
    if "coverage report" not in commands:
        problems.append("the coverage job does not run coverage report")
    problems += [
        f"the coverage job overrides the floor with {o}" for o in CAP_OVERRIDES if o in commands
    ]
    return problems


def test_ci_fails_below_the_coverage_floor() -> None:
    """The combined coverage report applies pyproject's floor on every push and pull request."""
    assert coverage_floor_violations(yaml.safe_load(CI_WORKFLOW.read_text()), _pyproject()) == []


def test_ci_uploads_no_coverage_to_codecov() -> None:
    """coverage.py in CI is the coverage gate; no workflow uploads to Codecov."""
    uses = [
        str(step.get("uses", ""))
        for path in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml"))
        for job in (yaml.safe_load(path.read_text()) or {}).get("jobs", {}).values()
        for step in job.get("steps", [])
    ]

    assert any(action.startswith("actions/checkout@") for action in uses)
    assert [action for action in uses if action.startswith("codecov/")] == []


def test_readme_and_docs_show_no_codecov_badge() -> None:
    """Coverage is not published to Codecov, so no page links a Codecov badge."""
    pages = ("README.md", "docs/index.md", "scripts/generate_docs.py")

    assert [page for page in pages if "codecov.io" in (REPO_ROOT / page).read_text()] == []


_ENVIRONMENT_METHODS = frozenset({"update", "setdefault", "pop", "popitem", "clear"})
_TEST_MODULE_ROOTS = ("tests", "benchmarks/tests")


def _is_os_environ(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "environ"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def _writes_environment(node: ast.AST) -> bool:
    """Whether one statement or expression changes the process environment."""
    if isinstance(node, (ast.Assign, ast.AugAssign, ast.Delete)):
        targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
        return any(isinstance(t, ast.Subscript) and _is_os_environ(t.value) for t in targets)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        function = node.func
        if _is_os_environ(function.value) and function.attr in _ENVIRONMENT_METHODS:
            return True
        return (
            isinstance(function.value, ast.Name)
            and function.value.id == "os"
            and function.attr in {"putenv", "unsetenv"}
        )
    return False


def _is_main_guard(node: ast.AST) -> bool:
    """Whether ``node`` is ``if __name__ == "__main__":``, which runs only as a script."""
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "__main__"
    )


def _module_level_nodes(path: Path, matches: Callable[[ast.AST], bool]) -> list[int]:
    """Line numbers of nodes that ``matches`` and that run when the module is imported.

    Function and class bodies and the ``__main__`` guard run only when called or run as a script,
    so they are not searched.
    """
    lines: list[int] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return
        if _is_main_guard(node):
            return
        if isinstance(node, (ast.stmt, ast.expr)) and matches(node):
            lines.append(node.lineno)
        for child in ast.iter_child_nodes(node):
            visit(child)

    for statement in ast.parse(path.read_text(encoding="utf-8")).body:
        visit(statement)
    return lines


def _module_level_environment_writes(path: Path) -> list[int]:
    """Line numbers of environment writes that run when the module is imported."""
    return _module_level_nodes(path, _writes_environment)


def _configures_logging(node: ast.AST) -> bool:
    """Whether a node calls ``logging.basicConfig``."""
    if not isinstance(node, ast.Call):
        return False
    function = node.func
    if isinstance(function, ast.Attribute):
        return (
            function.attr == "basicConfig"
            and isinstance(function.value, ast.Name)
            and function.value.id == "logging"
        )
    return isinstance(function, ast.Name) and function.id == "basicConfig"


def test_no_test_module_writes_the_environment_at_import() -> None:
    """A test module that changes os.environ at import changes it for every test collected after it.

    Environment a test needs belongs in that test (``monkeypatch.setenv``); the process-wide
    choices live in ``tests/conftest.py``, which runs before any test module is imported.
    """
    modules = [
        path
        for root in _TEST_MODULE_ROOTS
        for path in sorted((REPO_ROOT / root).rglob("test_*.py"))
    ]
    writes = [
        f"{path.relative_to(REPO_ROOT)}:{line}"
        for path in modules
        for line in _module_level_environment_writes(path)
    ]

    assert len(modules) > 100
    assert writes == []


_IMPORTABLE_ROOTS = ("src", "scripts", "examples", "tests", "benchmarks")


def test_no_module_configures_logging_at_import() -> None:
    """Importing a module must leave the root logger alone; an entry point configures it in main().

    A module that calls ``logging.basicConfig`` at import changes logging for everything imported
    after it, test collection included.
    """
    modules = [
        path
        for root in _IMPORTABLE_ROOTS
        for path in sorted((REPO_ROOT / root).rglob("*.py"))
        if "example_data" not in path.parts
    ]
    configured = [
        f"{path.relative_to(REPO_ROOT)}:{line}"
        for path in modules
        for line in _module_level_nodes(path, _configures_logging)
    ]

    assert len(modules) > 200
    assert configured == []
