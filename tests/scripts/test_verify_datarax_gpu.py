"""Tests for the Datarax backend verification helper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_verify_module():
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "verify_datarax_gpu.py"
    spec = importlib.util.spec_from_file_location("datarax_verify_gpu", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_verifier_fails_when_backend_query_errors(monkeypatch) -> None:
    """A backend initialization error must produce a non-zero verifier exit."""
    verify_gpu = _load_verify_module()
    report = verify_gpu.VerificationReport(
        datarax_backend="cuda12",
        jax_platforms="cuda,cpu",
        platform="Linux",
        python="3.11",
        jax_import_ok=True,
        jax_version="0.0",
        default_backend=None,
        gpu_device_count=0,
        devices=[],
        error="backend error",
    )

    monkeypatch.setattr(
        verify_gpu, "parse_args", lambda: SimpleNamespace(json=False, require_gpu=False)
    )
    monkeypatch.setattr(verify_gpu, "collect_report", lambda: report)

    assert verify_gpu.main() == 1
