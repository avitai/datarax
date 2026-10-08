"""Tooling and dependency contracts for audit remediation."""

from __future__ import annotations

import ast
import configparser
import fnmatch
import re
import shlex
import subprocess  # nosec B404
import tomllib
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement

from tests.test_common.protobuf_runtime import RUNTIME_VARIABLE


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


_PIPELINE_INTERNAL_LAYERS = (
    "datarax.pipeline.pipeline",
    "datarax.pipeline.host_stage | datarax.pipeline.iteration",
    "datarax.pipeline.run_configuration | datarax.pipeline.run_units",
    "datarax.pipeline.epochs | datarax.pipeline.dag_call | datarax.pipeline.read_plan",
    "datarax.pipeline.host_workers",
)
"""The pipeline's internal layers, top first; ``|`` joins independent siblings."""


def _layers(lines: list[str]) -> list[frozenset[str]]:
    """Each layer as the set of its modules, top first."""
    return [frozenset(part.strip() for part in line.split("|")) for line in lines if line.strip()]


def _contract_layers(path: Path, contract: str) -> list[frozenset[str]]:
    """The layers an Import Linter file declares for ``contract``, parsed."""
    parser = configparser.ConfigParser()
    parser.read_string(path.read_text())
    return _layers(parser[f"importlinter:contract:{contract}"]["layers"].splitlines())


def test_the_pipeline_internal_contract_declares_its_layers_exactly() -> None:
    """The host stage and the iteration sit above the run configuration, the plan and naming,
    and the worker machinery at the bottom, which every reader of a run imports."""
    assert _contract_layers(IMPORTLINTER, "pipeline-internal") == _layers(
        list(_PIPELINE_INTERNAL_LAYERS)
    )


def test_positive_control_a_layer_missing_from_the_contract_is_seen(tmp_path: Path) -> None:
    declared = IMPORTLINTER.read_text()
    missing = declared.replace(" | datarax.pipeline.read_plan", "")
    assert missing != declared, "the control removes nothing: read_plan is not declared"
    copy = tmp_path / ".importlinter"
    copy.write_text(missing)
    assert _contract_layers(copy, "pipeline-internal") != _layers(list(_PIPELINE_INTERNAL_LAYERS))


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


def test_type_checking_runs_in_the_locked_environment() -> None:
    """The pyright hook checks against the lock, not an environment resolved from PyPI.

    A hook environment of its own installs the sibling packages at whatever version shipped
    last, so a release of one of them turns the gate red on a tree the lock still pins to the
    previous version.
    """
    pre_commit = yaml.safe_load(PRE_COMMIT.read_text())
    hooks = [
        hook
        for repo in pre_commit["repos"]
        for hook in repo.get("hooks", [])
        if hook["id"] == "pyright"
    ]

    assert len(hooks) == 1
    (hook,) = hooks
    assert hook.get("language") == "system"
    assert hook.get("entry", "").startswith("uv run --no-sync pyright")
    assert "additional_dependencies" not in hook


def test_lint_job_type_checks_in_the_environment_setup_installs() -> None:
    """The lint job syncs at least the extras setup.sh gives every developer.

    pyright checks ``src``, ``tests`` and ``examples`` against the installed packages, and
    ``datasets`` (the ``data`` extra) is imported by a source module; an environment without
    it type-checks that module against Unknown and reports a defect the code does not have.
    """
    lint = yaml.safe_load(CI_WORKFLOW.read_text())["jobs"]["lint"]
    install = next(step for step in lint["steps"] if step.get("name") == "Install dependencies")
    synced = set(re.findall(r"--extra ([a-z0-9_-]+)", install["run"]))
    setup_synced = {
        synced_extra
        for _, synced_extra in _EXTRA_REFERENCE.findall(SETUP_SCRIPT.read_text())
        if synced_extra
    }

    assert {"dev", "test", "data"} <= setup_synced
    assert {"dev", "test", "data"} <= synced


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


DATASET_JOB = "prepare_example_datasets"
DATASET_ARTIFACT = "example-datasets"


def _step_using(job: dict, action: str) -> dict:
    """The one step of ``job`` that runs ``action`` (``owner/name@``)."""
    (step,) = [step for step in job["steps"] if str(step.get("uses", "")).startswith(action)]
    return step


def test_long_running_examples_never_download_the_datasets() -> None:
    """The example tier reads the prepared datasets: the cache, else this run's artifact.

    CIFAR-10 comes from a host the runners fetch at 13-16 s per MiB, so an example must not
    download it inside its time budget. A cache save can fail ("Unable to reserve cache"),
    so on a miss the dataset job also hands its result to the example tier as an artifact.
    """
    jobs = _jobs()
    prepare = jobs[DATASET_JOB]
    long_running = jobs["long_running_tests"]
    needs = long_running["needs"]
    saved = _step_using(prepare, "actions/cache@")
    uploaded = _step_using(prepare, "actions/upload-artifact@")
    restored = _step_using(long_running, "actions/cache/restore@")
    downloaded = _step_using(long_running, "actions/download-artifact@")
    prepare_commands = "\n".join(str(step.get("run", "")) for step in prepare["steps"])
    long_running_commands = "\n".join(str(step.get("run", "")) for step in long_running["steps"])

    assert DATASET_JOB in ([needs] if isinstance(needs, str) else needs)
    assert restored["with"]["key"] == saved["with"]["key"]
    assert restored["with"]["path"] == saved["with"]["path"]
    assert "scripts/prepare_example_datasets.py" in prepare_commands
    assert "scripts/prepare_example_datasets.py" not in long_running_commands
    assert uploaded["if"] == f"steps.{saved['id']}.outputs.cache-hit != 'true'"
    assert uploaded["with"]["path"] == saved["with"]["path"]
    assert uploaded["with"]["name"] == downloaded["with"]["name"] == DATASET_ARTIFACT
    assert downloaded["if"] == f"steps.{restored['id']}.outputs.cache-hit != 'true'"


def test_main_keeps_the_dataset_cache_pull_requests_restore() -> None:
    """The dataset job runs on every push to main, even one that repeats a pull request.

    A pull request run restores caches saved by its own ref or by main, never by another pull
    request, so main must hold the prepared datasets. Gated, the job skipped every merge, and
    once main's entry was evicted no run rebuilt it until the nightly schedule.
    """
    assert not _consults_the_gate(_jobs()[DATASET_JOB])


def test_every_uv_cache_is_pruned_before_it_is_saved() -> None:
    """A saved uv cache holds only what uv built, not the wheels it downloaded.

    Unpruned, a cache of this repository's extras (TensorFlow, PyTorch) is about 3.9 GB, and a
    handful of them exceed the repository's Actions cache budget; GitHub then evicts the least
    recently used entry, which is the prepared dataset cache the example tier fails without.
    setup-uv prunes only when asked (``prune-cache`` defaults to false from v9).
    """
    unpruned = []
    for workflow_path in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")):
        for job_name, job in yaml.safe_load(workflow_path.read_text())["jobs"].items():
            for step in job.get("steps", []):
                if not str(step.get("uses", "")).startswith("astral-sh/setup-uv@"):
                    continue
                if step.get("with", {}).get("prune-cache") is not True:
                    unpruned.append(f"{workflow_path.name}:{job_name}")

    assert unpruned == [], f"setup-uv steps saving an unpruned cache: {unpruned}"


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
    if job.get("if") not in (None, GATE_CONDITION):
        problems.append(f"the coverage job only runs when {job['if']}")
    if "coverage report" not in commands:
        problems.append("the coverage job does not run coverage report")
    problems += [
        f"the coverage job overrides the floor with {o}" for o in CAP_OVERRIDES if o in commands
    ]
    return problems


def test_ci_fails_below_the_coverage_floor() -> None:
    """The combined coverage report applies pyproject's floor on every push and pull request.

    The one condition the job may carry is the already-tested gate, which stands down only
    where the same tree already reported coverage on the pull request that produced it.
    """
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


GATE_JOB = "already_tested"
# performance_tests runs only on main and waits on lint, and GitHub skips a job whose
# dependency skipped, so gating lint would take the one job that never repeats with it.
# prepare_example_datasets keeps main's dataset cache (see its own test).
UNGATED_JOBS = frozenset({GATE_JOB, "lint", DATASET_JOB})
GATE_CONDITION = f"needs.{GATE_JOB}.outputs.skip != 'true'"


def _jobs() -> dict[str, dict]:
    return yaml.safe_load(CI_WORKFLOW.read_text())["jobs"]


def _runs_only_on_main(job: dict) -> bool:
    """Whether the job's own condition confines it to a push to main."""
    condition = str(job.get("if", ""))
    return "refs/heads/main" in condition


def _consults_the_gate(job: dict) -> bool:
    return f"needs.{GATE_JOB}.outputs.skip" in yaml.safe_dump(job)


def _transitive_needs(name: str, jobs: dict[str, dict]) -> set[str]:
    """Every job ``name`` waits on, directly or through another."""
    pending, seen = [name], set()
    while pending:
        current = pending.pop()
        needs = jobs.get(current, {}).get("needs", [])
        for dependency in [needs] if isinstance(needs, str) else needs:
            if dependency not in seen:
                seen.add(dependency)
                pending.append(dependency)
    return seen


def test_the_gate_reports_a_skip_only_for_a_push() -> None:
    """A schedule or a manual run re-measures the tree on purpose and must not be skipped.

    The daily run exists to catch a dependency moving under an unchanged tree, and a
    ``workflow_dispatch`` is usually asked for because someone wants the run.
    """
    gate = _jobs()[GATE_JOB]
    reported_by = gate["outputs"]["skip"]
    writers = [step for step in gate["steps"] if f"steps.{step.get('id')}.outputs" in reported_by]

    assert [step.get("if") for step in writers] == ["github.event_name == 'push'"]


def test_a_job_that_repeats_the_pull_request_consults_the_gate() -> None:
    """Work a pull request already did over the same tree does not run again on the merge."""
    jobs = _jobs()
    repeated = {
        name
        for name, job in jobs.items()
        if name not in UNGATED_JOBS and not _runs_only_on_main(job)
    }

    ungated = sorted(name for name in repeated if not _consults_the_gate(jobs[name]))

    assert ungated == [], f"these repeat the pull request without consulting the gate: {ungated}"


def test_nothing_a_main_only_job_waits_on_consults_the_gate() -> None:
    """A job confined to main must not be skipped because something it needs was.

    GitHub skips a job whose dependency skipped, so gating a shared job would silently take
    the main-only ones with it.
    """
    jobs = _jobs()
    for name, job in jobs.items():
        if not _runs_only_on_main(job):
            continue
        gated = sorted(
            dependency
            for dependency in _transitive_needs(name, jobs)
            if dependency != GATE_JOB and _consults_the_gate(jobs[dependency])
        )
        assert gated == [], f"{name} runs only on main but waits on gated {gated}"


def test_an_unanswered_gate_leaves_the_work_running() -> None:
    """The gate fails safe: where it answers nothing, every job runs as it would without it.

    The compare step does not run for a schedule or a manual run, and the lookups inside it
    answer ``unknown`` rather than failing, so an empty output is an ordinary outcome. Every
    consumer must read it as "test this tree": it may only stand work down on ``'true'``.
    """
    for name, job in _jobs().items():
        if name == GATE_JOB or not _consults_the_gate(job):
            continue
        condition = job.get("if")
        assert condition in (None, GATE_CONDITION), f"{name} runs only when {condition}"

        pattern = rf"needs\.{GATE_JOB}\.outputs\.skip\s*(==|!=)\s*'([a-z]+)'"
        compared = set(re.findall(pattern, yaml.safe_dump(job)))
        assert {value for _, value in compared} == {"true"}, f"{name} compares against {compared}"


def test_the_version_is_the_one_pyproject_declares() -> None:
    """``pyproject.toml`` is the version's one source; the package reads it once installed."""
    import datarax

    project = _pyproject()["project"]

    assert "version" not in project.get("dynamic", [])
    assert datarax.__version__ == project["version"]


def test_ruff_hooks_run_the_locked_ruff() -> None:
    """One ruff version: the hooks run the lock's ruff, not a separately pinned hook repository."""
    pre_commit = yaml.safe_load(PRE_COMMIT.read_text())
    hooks = [
        hook
        for repo in pre_commit["repos"]
        for hook in repo.get("hooks", [])
        if hook["id"] in {"ruff", "ruff-format"}
    ]

    assert {hook["id"] for hook in hooks} == {"ruff", "ruff-format"}
    assert [repo["repo"] for repo in pre_commit["repos"] if "ruff-pre-commit" in repo["repo"]] == []
    for hook in hooks:
        assert hook.get("language") == "system", hook["id"]
        assert hook.get("entry", "").startswith("uv run --no-sync ruff"), hook["id"]


def test_coverage_measures_pass_statements() -> None:
    """A bare ``pass`` exclusion drops executable statements from the measurement."""
    exclusions = _pyproject()["tool"]["coverage"]["report"]["exclude_lines"]

    assert "pass" not in exclusions


def test_the_global_pre_commit_exclude_names_tracked_files() -> None:
    """An exclude that matches no tracked file is stale."""
    exclude = yaml.safe_load(PRE_COMMIT.read_text()).get("exclude")
    if exclude is None:
        return
    tracked = subprocess.run(  # noqa: S603  # nosec B603 B607 - git on this repository
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()

    assert any(re.search(exclude, path) for path in tracked), f"{exclude} matches no tracked file"


def _tracked_files_naming(name: str) -> set[str]:
    listing = subprocess.run(  # noqa: S603  # nosec B603 B607 - git on this repository
        ["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    needle = name.encode()
    return {
        path
        for path in listing.split("\0")
        if path and (REPO_ROOT / path).is_file() and needle in (REPO_ROOT / path).read_bytes()
    }


def test_no_tracked_file_chooses_the_protobuf_runtime() -> None:
    """The runtime TFDS records are parsed on is the one a plain install selects (upb).

    Pure-Python protobuf doubles the per-record TFDS decode (132 vs 71 us on CIFAR-10), and a
    variable exported by the managed env file, the test conftest or a source module reaches every
    process started from there.
    """
    # The control: the same scan sees a variable the env generator and the macOS workflow name.
    assert {"scripts/setup_env.py", ".github/workflows/macos.yml"} <= _tracked_files_naming(
        "JAX_PLATFORMS"
    )

    assert _tracked_files_naming(RUNTIME_VARIABLE) == set()


def test_pull_requests_gate_coverage_on_changed_lines() -> None:
    """The combined coverage report checks the lines a pull request changes at 80%."""
    job = yaml.safe_load(CI_WORKFLOW.read_text())["jobs"]["coverage"]
    checkout = next(
        s for s in job["steps"] if str(s.get("uses", "")).startswith("actions/checkout@")
    )
    diff_cover = [s for s in job["steps"] if "diff-cover" in str(s.get("run", ""))]
    test_extra = _dependency_names(_pyproject()["project"]["optional-dependencies"]["test"])

    assert checkout.get("with", {}).get("fetch-depth") == 0
    assert len(diff_cover) == 1
    assert diff_cover[0].get("if") == "github.event_name == 'pull_request'"
    assert "--compare-branch=origin/main --fail-under=80" in diff_cover[0]["run"]
    assert "diff-cover" in test_extra


# The stack's gate lives in substrax, the base layer, pinned by commit: one implementation, which
# finds the merged pull request (squash or rebase) through GitHub's commit-to-pull-request
# association and treats a pending or cancelled check as unproven.
GATE_ACTION = (
    "avitai/substrax/.github/actions/already-tested@ec43d80bbae0040375f746cd1bb07dddb0a0b169"
)
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
MACOS_WORKFLOW = WORKFLOWS / "macos.yml"


def _on_macos(job: dict) -> bool:
    """Whether the job runs on macOS: its runner, the matrix its runner reads, or the runner it
    passes a called workflow."""
    runner = [
        job.get("runs-on", ""),
        job.get("strategy", {}).get("matrix", {}),
        job.get("with", {}),
    ]
    return "macos" in yaml.safe_dump(runner).lower()


def _workflow_triggers(workflow: dict) -> set[str]:
    # PyYAML reads the `on:` key as the boolean True.
    on = workflow.get("on") or workflow.get(True) or {}
    return {on} if isinstance(on, str) else set(on)


def test_the_gate_uses_the_stacks_released_action() -> None:
    gate = _jobs()[GATE_JOB]
    compare = next(step for step in gate["steps"] if step.get("id") == "compare")

    assert compare.get("uses") == GATE_ACTION
    assert "run" not in compare


def test_no_workflow_run_by_a_push_or_pull_request_uses_macos() -> None:
    for path in sorted(WORKFLOWS.glob("*.yml")):
        workflow = yaml.safe_load(path.read_text())
        if not _workflow_triggers(workflow) & {"push", "pull_request"}:
            continue
        on_macos = sorted(name for name, job in workflow["jobs"].items() if _on_macos(job))
        assert on_macos == [], f"{path.name} runs macOS on a push or pull request: {on_macos}"


def _run_jobs(job: dict) -> list[dict]:
    """The jobs that run for ``job``: itself, or those of the workflow it calls."""
    called = job.get("uses")
    if called is None:
        return [job]
    return list(
        yaml.safe_load((REPO_ROOT / called.removeprefix("./")).read_text())["jobs"].values()
    )


def test_the_macos_workflow_runs_nightly_on_demand_and_under_a_runner_cap() -> None:
    workflow = yaml.safe_load(MACOS_WORKFLOW.read_text())
    macos_jobs = {name: job for name, job in workflow["jobs"].items() if _on_macos(job)}
    running = [run for job in macos_jobs.values() for run in _run_jobs(job)]
    commands = "\n".join(str(step.get("run", "")) for job in running for step in job["steps"])

    assert _workflow_triggers(workflow) == {"schedule", "workflow_dispatch"}
    assert set(macos_jobs) == {"unit_tests", "build"}
    for job in running:
        assert int(job["strategy"]["max-parallel"]) <= 2, "a macOS job without a runner cap"
    assert "pytest tests benchmarks/tests" in commands
    assert "python -m build" in commands


def test_a_quiet_night_runs_no_macos_job() -> None:
    """A scheduled run stands down when main has not moved since the last green scheduled run;
    a manual run, such as the release checklist's, always runs."""
    workflow = yaml.safe_load(MACOS_WORKFLOW.read_text())
    compare = next(s for s in workflow["jobs"]["main_moved"]["steps"] if s.get("id") == "compare")

    assert compare.get("if") == "github.event_name == 'schedule'"
    assert "--status success" in compare["run"]
    assert "--event schedule" in compare["run"]
    for name, job in workflow["jobs"].items():
        if name != "main_moved":
            assert job.get("if") == "needs.main_moved.outputs.unchanged != 'true'", name


def test_the_release_checklist_runs_macos_before_the_tag() -> None:
    releasing = (REPO_ROOT / "RELEASING.md").read_text()

    assert "gh workflow run macos.yml" in releasing
    assert releasing.index("gh workflow run macos.yml") < releasing.index("git tag -a")


# The lockfile audit also lives in substrax, pinned by commit: it exports every extra the lock
# resolves and runs a pinned pip-audit through uvx, isolated from the project, on each export.
AUDIT_ACTION = "avitai/substrax/.github/actions/audit-lock@13e98b78127606be04437bd7c5de894fb765a28e"


def test_the_security_audit_reads_every_extra_in_the_lock() -> None:
    """The dependency audit covers the whole lockfile, not the extras the test suite installs.

    Audited from ``uv sync --extra dev --extra test --extra data``, the job never saw the
    benchmark and automation extras, where Dependabot reported a critical PyJWT advisory. The
    stack's ``audit-lock`` action reads the ignored advisories from ``pyproject.toml`` and
    refuses an empty reason or an entry no advisory matches.
    """
    security = yaml.safe_load((WORKFLOWS / "security.yml").read_text())
    steps = [step for job in security["jobs"].values() for step in job["steps"]]
    uses = [str(step.get("uses", "")) for step in steps]
    commands = "\n".join(str(step.get("run", "")) for step in steps)
    setup_uv = next(i for i, used in enumerate(uses) if used.startswith("astral-sh/setup-uv@"))

    assert uses.count(AUDIT_ACTION) == 1
    assert setup_uv < uses.index(AUDIT_ACTION)
    assert "pip-audit" not in commands
    assert "pip_audit" not in commands
    assert "audit_lock" not in commands
    assert not (REPO_ROOT / "scripts" / "audit_lock.py").exists()
    assert _pyproject()["tool"]["substrax"]["audit-lock"]["ignore"]


UNIT_JOB = "unit_tests"
SHARD_COUNT = "UNIT_TEST_SHARDS"


def _unit_test_step() -> dict:
    (step,) = [step for step in _jobs()[UNIT_JOB]["steps"] if "pytest" in str(step.get("run", ""))]
    return step


def _pytest_arguments(command: str) -> list[str]:
    """The arguments of the one pytest invocation in a workflow ``run`` script.

    An expression keeps its inner spaces out, so ``${{ matrix.group }}`` is one argument,
    ``${{matrix.group}}``.
    """
    lines = command.replace("\\\n", " ").splitlines()
    (invocation,) = [
        line for line in lines if "pytest" in line and not line.lstrip().startswith("#")
    ]
    invocation = re.sub(r"\$\{\{\s*([^}]*?)\s*\}\}", r"${{\1}}", invocation)
    return invocation.split("pytest", 1)[1].split()


def _option(arguments: list[str], name: str) -> str:
    """The value given to ``name``, as ``name value`` or ``name=value``."""
    for index, argument in enumerate(arguments):
        if argument == name:
            return arguments[index + 1]
        if argument.startswith(f"{name}="):
            return argument.split("=", 1)[1]
    raise AssertionError(f"pytest is not given {name}: {arguments}")


def test_the_unit_suite_runs_as_one_collection_split_into_shards() -> None:
    """Each matrix leg runs one pytest-split group of the same collection.

    The group list is exactly 1..N for the N shards the job declares, so no group is left
    unrun, and every leg reports on its own (``fail-fast`` off) so one red shard does not
    cancel the others' coverage.
    """
    job = _jobs()[UNIT_JOB]
    shards = int(job["env"][SHARD_COUNT])
    matrix = job["strategy"]["matrix"]
    arguments = _pytest_arguments(_unit_test_step()["run"])

    assert shards >= 2
    assert matrix["group"] == list(range(1, shards + 1))
    assert job["strategy"]["fail-fast"] is False
    assert "${{ matrix.group }}" in job["name"]
    assert _option(arguments, "--splits") == f'"${SHARD_COUNT}"'
    assert _option(arguments, "--group") == "${{matrix.group}}"
    assert _option(arguments, "--splitting-algorithm") == "least_duration"
    assert "pytest-split" in _dependency_names(
        _pyproject()["project"]["optional-dependencies"]["test"]
    )
    assert "--extra test" in str(
        next(s for s in job["steps"] if s.get("name") == "Install dependencies")["run"]
    )


def test_every_shard_orders_the_collection_with_one_seed() -> None:
    """The shards of a run share pytest-randomly's seed, so each sees the same item order.

    pytest-randomly reorders the collection before pytest-split assigns it, and pytest-split
    breaks ties between equally long tests of the same name by that order. A seed drawn per
    process gives each shard its own order, and then a test can land in two shards while
    another lands in none. The run id is one value across the run, and a new one per run, so
    the order still changes from run to run.
    """
    job = _jobs()[UNIT_JOB]
    arguments = _pytest_arguments(_unit_test_step()["run"])
    install = next(s for s in job["steps"] if s.get("name") == "Install dependencies")["run"]

    assert _option(arguments, "--randomly-seed") == "${{github.run_id}}"
    assert "pytest-randomly" in _dependency_names(
        _pyproject()["project"]["optional-dependencies"]["dev"]
    )
    assert "--extra dev" in install


def _expand_matrix(template: str, matrix: dict) -> list[str]:
    """``template`` with each ``${{ matrix.key }}`` replaced, for every combination."""
    names = [template]
    for key, values in matrix.items():
        placeholder = f"${{{{ matrix.{key} }}}}"
        if placeholder in template:
            names = [name.replace(placeholder, str(value)) for name in names for value in values]
    return names


def test_the_coverage_job_combines_every_shard() -> None:
    """Each shard uploads its own coverage data, and the coverage job reads all of them.

    The artifact name carries the Python version and the group, so no two legs write the same
    artifact; the coverage job waits on the whole matrix, downloads every ``coverage-*``
    artifact into a directory of its own (not merged, where same-named ``.coverage`` files
    would overwrite each other) and combines what it finds.
    """
    jobs = _jobs()
    unit = jobs[UNIT_JOB]
    upload = _step_using(unit, "actions/upload-artifact@")
    coverage = jobs["coverage"]
    download = _step_using(coverage, "actions/download-artifact@")
    combine = next(s for s in coverage["steps"] if "coverage combine" in str(s.get("run", "")))
    names = _expand_matrix(upload["with"]["name"], unit["strategy"]["matrix"])
    legs = len(unit["strategy"]["matrix"]["python-version"]) * len(
        unit["strategy"]["matrix"]["group"]
    )

    assert len(set(names)) == len(names) == legs
    assert UNIT_JOB in coverage["needs"]
    assert all(fnmatch.fnmatchcase(name, download["with"]["pattern"]) for name in names)
    assert download["with"].get("merge-multiple") is not True
    assert f'find {download["with"]["path"]} -name ".coverage"' in combine["run"]


def test_no_extra_installs_pip_audit() -> None:
    """The audit runs its own pinned pip-audit in isolation; no extra needs to carry one."""
    extras = _pyproject()["project"]["optional-dependencies"]

    for name, dependencies in extras.items():
        assert "pip-audit" not in _dependency_names(dependencies), name


TFDS_JOB = "long_running_tests"
TFDS_FIXTURE_MODULE = "tests.test_common.tfds_fixture"
_MARKER_EXPRESSION = re.compile(r"""-m\s+(?:"([^"]*)"|'([^']*)'|(\S+))""")
_PYTEST_COMMAND = re.compile(r"(?:^|[\s/])pytest\s")


def _pytest_lines(job: dict) -> list[str]:
    """Each pytest invocation in the job's ``run`` steps, continuation lines joined."""
    commands = "\n".join(str(step.get("run", "")) for step in job.get("steps", []))
    lines = commands.replace("\\\n", " ").splitlines()
    return [
        line for line in lines if _PYTEST_COMMAND.search(line) and not line.strip().startswith("#")
    ]


def _selects(line: str, markers: set[str]) -> bool:
    """Whether the pytest command line selects a test carrying exactly ``markers``."""
    from _pytest.mark.expression import Expression  # noqa: PLC0415 - pytest's own -m parser

    expressions = ["".join(groups) for groups in _MARKER_EXPRESSION.findall(line)]
    return all(
        Expression.compile(expression).evaluate(lambda name, **_: name in markers)
        for expression in expressions
    )


def _workflow_jobs() -> dict[str, dict]:
    return {
        f"{path.name}:{name}": job
        for path in sorted(WORKFLOWS.glob("*.yml"))
        for name, job in yaml.safe_load(path.read_text())["jobs"].items()
    }


def test_the_tfds_tests_run_in_the_long_running_job_after_the_fixture_step() -> None:
    """The job that installs TensorFlow prepares the offline fixture, then runs the TFDS tests.

    Preparing a TFDS dataset imports TensorFlow, which only the long-running job installs (the
    ``tfds`` extra), so the fixture is prepared there in a step of its own and named in
    ``DATARAX_TFDS_FIXTURE_DIR``; a missing fixture then fails the tests rather than skipping them.
    """
    job = _jobs()[TFDS_JOB]
    runs = [str(step.get("run", "")) for step in job["steps"]]
    fixture = next(k for k, run in enumerate(runs) if TFDS_FIXTURE_MODULE in run)
    tests = next(k for k, run in enumerate(runs) if _PYTEST_COMMAND.search(run))

    assert "--extra tfds" in "\n".join(runs)
    assert fixture < tests
    assert "DATARAX_TFDS_FIXTURE_DIR" in runs[fixture]
    assert "$GITHUB_ENV" in runs[fixture]
    (line,) = _pytest_lines(job)
    assert _selects(line, {"tfds"})
    assert _selects(line, {"tfds", "slow"})


def test_every_other_lane_deselects_the_tfds_tests_explicitly() -> None:
    """No other lane installs TensorFlow, so each states that it leaves the TFDS tests out."""
    lanes = [
        (name, line)
        for name, job in _workflow_jobs().items()
        if name != f"ci.yml:{TFDS_JOB}"
        for line in _pytest_lines(job)
    ]

    assert len(lanes) >= 6
    assert [name for name, line in lanes if _selects(line, {"tfds"})] == []
    assert [name for name, line in lanes if "not tfds" not in line] == []


# Events on which a workflow measures the tree unattended; a skip there is read from the log.
MEASURING_EVENTS = frozenset({"push", "pull_request", "schedule"})
# `pytest -r` report characters that list each skip with its reason: `s` itself, or `a`/`A`
# (all outcomes but passes, and all outcomes).
SKIP_REPORT_CHARACTERS = frozenset("saA")


def _reports_skip_reasons(line: str) -> bool:
    """Whether the pytest command line asks for a ``SKIPPED [n] file: reason`` summary line."""
    arguments = shlex.split(line.split("pytest", 1)[1])
    characters = [
        argument[2:] or (arguments[index + 1] if index + 1 < len(arguments) else "")
        for index, argument in enumerate(arguments)
        if argument.startswith("-r") and not argument.startswith("--")
    ]
    return any(SKIP_REPORT_CHARACTERS & set(chars) for chars in characters)


def _measuring_pytest_lines() -> list[tuple[str, str]]:
    """``(workflow:job, command)`` for each pytest run by a push, pull request or schedule."""
    lines = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        workflow = yaml.safe_load(path.read_text())
        # PyYAML reads the `on:` key as the boolean True.
        triggers = workflow.get("on") or workflow.get(True) or {}
        if not MEASURING_EVENTS & set(triggers):
            continue
        lines += [
            (f"{path.name}:{name}", line)
            for name, job in workflow["jobs"].items()
            for line in _pytest_lines(job)
        ]
    return lines


@pytest.mark.parametrize(
    ("line", "reports"),
    [
        ("uv run pytest tests -rs -v", True),
        ("uv run pytest tests -r s", True),
        ("uv run pytest tests -ra", True),
        ("uv run pytest tests -rfEs", True),
        ("uv run pytest tests -v -m 'not slow'", False),
        ("uv run pytest tests -rfE", False),
        ("uv run pytest tests --rootdir=s", False),
    ],
)
def test_skip_reason_detection(line: str, reports: bool) -> None:
    """The check reads pytest's ``-r`` option itself, not any argument containing ``-r``."""
    assert _reports_skip_reasons(line) is reports


def test_every_measuring_pytest_command_explains_its_skips() -> None:
    """A skip count in a CI log comes with each skip's reason.

    Without ``-r`` naming skips, pytest reports how many tests skipped but not why, and a
    module-level skip prints no per-test line at all, so a lane can lose a dependency and
    still read as green.
    """
    lines = _measuring_pytest_lines()

    assert len(lines) >= 8
    assert [name for name, line in lines if not _reports_skip_reasons(line)] == []
