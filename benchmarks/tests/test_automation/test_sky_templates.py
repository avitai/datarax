"""Regression tests for Sky benchmark templates."""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[3]

TEMPLATE_PATHS = [
    Path("benchmarks/sky/cpu-benchmark.yaml"),
    Path("benchmarks/sky/gpu-benchmark.yaml"),
    Path("benchmarks/sky/tpu-benchmark.yaml"),
]


def test_templates_do_not_install_uv_with_root_pip() -> None:
    for template in TEMPLATE_PATHS:
        text = template.read_text()
        assert "pip install uv" not in text
        assert "python -m ensurepip" not in text
        # The gpu template delegates venv creation to setup.sh --python 3.11.
        assert "uv venv --python 3.11 .venv" in text or "./setup.sh" in text
        assert "uv pip install --python .venv/bin/python" in text
        assert ".venv/bin/datarax-bench run" not in text
        assert ".venv/bin/python -m benchmarks.cli run" in text


def test_setup_script_supports_benchmark_extra() -> None:
    """setup.sh --with-benchmarks adds the competitor-framework extra.

    uv sync is declarative, so the remote bootstrap must request every extra
    in a single sync; a second sync would remove previously installed ones.
    """
    result = subprocess.run(
        ["./setup.sh", "--dry-run", "--backend", "cuda12", "--with-benchmarks"],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=60,
        check=True,
    )
    sync_line = next(line for line in result.stdout.splitlines() if line.startswith("uv sync"))
    assert "--extra benchmark" in sync_line
    assert "--extra gpu" in sync_line
    assert "--extra data" in sync_line


def test_gpu_template_delegates_setup_to_canonical_script() -> None:
    """The sky template must bootstrap via setup.sh, not a hand-rolled sync.

    The template's environment drifted from setup.sh once already; a single
    canonical bootstrap prevents that class of failure.
    """
    template = yaml.safe_load((_REPO_ROOT / "benchmarks/sky/gpu-benchmark.yaml").read_text())
    setup = template["setup"]
    assert "./setup.sh" in setup
    assert "--with-benchmarks" in setup
    assert "uv sync" not in setup
