"""Tests for the setup environment helper scripts."""

from __future__ import annotations

from pathlib import Path

from tests.scripts.script_loader import load_script


def _load_setup_env_module():
    return load_script("setup_env")


def test_cpu_backend_sets_cpu_platform_without_cuda_paths(tmp_path: Path) -> None:
    """CPU backend writes deterministic managed env without CUDA path injection."""
    setup_env = _load_setup_env_module()

    contents = setup_env.build_env_contents(tmp_path, "cpu")

    assert "export DATARAX_BACKEND=cpu" in contents
    assert "export DATARAX_ENV_ROOT=" in contents
    assert "export JAX_PLATFORMS=cpu" in contents
    assert "LD_LIBRARY_PATH" not in contents
    assert "CUDA_HOME" not in contents
    assert "XLA_FLAGS" not in contents


def test_accelerator_backends_leave_platform_selection_to_jax(tmp_path: Path) -> None:
    """CUDA and Metal backends let JAX select the available accelerator."""
    setup_env = _load_setup_env_module()

    cuda_contents = setup_env.build_env_contents(tmp_path, "cuda12")
    metal_contents = setup_env.build_env_contents(tmp_path, "metal")

    assert "export DATARAX_BACKEND=cuda12" in cuda_contents
    assert "unset JAX_PLATFORMS" in cuda_contents
    assert "export DATARAX_BACKEND=metal" in metal_contents
    assert "unset JAX_PLATFORMS" in metal_contents


def test_show_status_reports_managed_and_user_env_layers(tmp_path: Path, capsys) -> None:
    """Status output reports generated and user-owned env files separately."""
    setup_env = _load_setup_env_module()
    managed_env = tmp_path / ".datarax.env"
    managed_env.write_text("export DATARAX_BACKEND=cpu\n")
    (tmp_path / ".env.local").write_text("export EXAMPLE=1\n")

    result = setup_env.show_status(tmp_path, managed_env)

    assert result == 0
    output = capsys.readouterr().out
    assert ".datarax.env present: True" in output
    assert ".env present: False" in output
    assert ".env.local present: True" in output
    assert "Configured backend: cpu" in output


def test_the_memory_fraction_uses_the_name_jaxlib_reads(tmp_path: Path) -> None:
    """jaxlib reads XLA_CLIENT_MEM_FRACTION and refuses it beside the deprecated name.

    activate.sh unsets every variable of the previous managed list before sourcing the new file,
    so a shell activated with the old name loses it on the next activation.
    """
    setup_env = _load_setup_env_module()

    contents = setup_env.build_env_contents(tmp_path, "cuda12")

    assert "export XLA_CLIENT_MEM_FRACTION=0.75" in contents
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION" not in contents
