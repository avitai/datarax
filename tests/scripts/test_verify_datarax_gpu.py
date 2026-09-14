"""Tests for the Datarax backend verification helper."""

from __future__ import annotations

from types import SimpleNamespace

from tests.scripts.script_loader import load_script


def _load_verify_module():
    return load_script("verify_datarax_gpu")


def test_verifier_fails_when_backend_query_errors(monkeypatch) -> None:
    """A backend initialization error must produce a non-zero verifier exit."""
    verify_gpu = _load_verify_module()
    report = verify_gpu.VerificationReport(
        datarax_backend="cuda12",
        jax_platforms="cuda,cpu",
        platform="Linux",
        python="3.12",
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
