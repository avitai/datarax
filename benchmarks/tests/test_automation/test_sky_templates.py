"""Regression tests for Sky benchmark templates."""

from __future__ import annotations

from pathlib import Path


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
        assert "uv venv --python 3.11 .venv" in text
        assert "uv pip install --python .venv/bin/python" in text
        assert ".venv/bin/datarax-bench run" not in text
        assert ".venv/bin/python -m benchmarks.cli run" in text
