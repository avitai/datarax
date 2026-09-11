"""The JAX environment a test run configures: backend choice and CPU device emulation.

Tests run on CPU with eight emulated devices unless they ask otherwise, so
multi-device tests run on any host. A run that asks for an accelerator with
``DATARAX_TEST_JAX_PLATFORMS`` gets that accelerator and no emulation; asking
for CUDA without JAX's CUDA plugin fails instead of silently running on CPU.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.jax_test_environment import resolve_test_jax_environment


_EMULATION = "--xla_force_host_platform_device_count"


def test_tests_default_to_eight_emulated_cpu_devices() -> None:
    env = resolve_test_jax_environment({}, cuda_plugin_available=True)

    assert env["JAX_PLATFORMS"] == "cpu"
    assert f"{_EMULATION}=8" in env["XLA_FLAGS"]


def test_an_inherited_cuda_selection_still_runs_tests_on_emulated_cpus() -> None:
    env = resolve_test_jax_environment({"JAX_PLATFORMS": "cuda,cpu"}, cuda_plugin_available=True)

    assert env["JAX_PLATFORMS"] == "cpu"
    assert f"{_EMULATION}=8" in env["XLA_FLAGS"]


def test_an_explicit_accelerator_request_disables_cpu_emulation() -> None:
    env = resolve_test_jax_environment(
        {"DATARAX_TEST_JAX_PLATFORMS": "cuda,cpu", "XLA_FLAGS": "--xla_gpu_autotune_level=2"},
        cuda_plugin_available=True,
    )

    assert env["JAX_PLATFORMS"] == "cuda,cpu"
    assert env["XLA_FLAGS"] == "--xla_gpu_autotune_level=2"


def test_an_explicit_cpu_request_keeps_emulation() -> None:
    env = resolve_test_jax_environment(
        {"DATARAX_TEST_JAX_PLATFORMS": "cpu"}, cuda_plugin_available=True
    )

    assert env["JAX_PLATFORMS"] == "cpu"
    assert f"{_EMULATION}=8" in env["XLA_FLAGS"]


def test_a_device_count_of_zero_disables_emulation() -> None:
    env = resolve_test_jax_environment(
        {"DATARAX_TEST_DEVICE_COUNT": "0"}, cuda_plugin_available=False
    )

    assert env["JAX_PLATFORMS"] == "cpu"
    assert _EMULATION not in env["XLA_FLAGS"]


def test_a_requested_device_count_is_used_and_existing_flags_are_kept() -> None:
    env = resolve_test_jax_environment(
        {"DATARAX_TEST_DEVICE_COUNT": "4", "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false"},
        cuda_plugin_available=False,
    )

    assert env["XLA_FLAGS"] == f"--xla_cpu_multi_thread_eigen=false {_EMULATION}=4"


def test_a_device_count_already_in_the_flags_is_not_repeated() -> None:
    env = resolve_test_jax_environment(
        {"XLA_FLAGS": f"{_EMULATION}=2"}, cuda_plugin_available=False
    )

    assert env["XLA_FLAGS"] == f"{_EMULATION}=2"


def test_asking_for_cuda_without_the_plugin_fails() -> None:
    with pytest.raises(RuntimeError, match="CUDA plugin"):
        resolve_test_jax_environment(
            {"DATARAX_TEST_JAX_PLATFORMS": "cuda"}, cuda_plugin_available=False
        )


def test_importing_the_resolver_does_not_import_jax() -> None:
    """JAX reads JAX_PLATFORMS at import, so choosing the backend must not load it."""
    repo_root = str(Path(__file__).resolve().parents[1])
    code = (
        f"import sys; sys.path.insert(0, {repo_root!r}); "
        "import tests.jax_test_environment; print('jax' in sys.modules)"
    )
    result = subprocess.run(  # noqa: S603 - fixed interpreter and code
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "False"
