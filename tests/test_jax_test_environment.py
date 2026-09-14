"""The JAX environment a test run starts with: backend choice and CPU device emulation.

Tests run on CPU with eight emulated devices unless they ask otherwise, so multi-device tests run
on any host. A run that asks for an accelerator with ``DATARAX_TEST_JAX_PLATFORMS`` gets that
accelerator and no emulation; asking for CUDA without JAX's CUDA plugin fails instead of silently
running on CPU. substrax resolves the environment; these tests pin datarax's choices and check that
a fresh interpreter started with the result sees the devices it names.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from substrax.testing import run_python

from tests.jax_test_environment import resolve_test_environment


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEVICE_COUNT_FLAG = "--xla_force_host_platform_device_count"
_IMPORT_PROBE = (
    "import json, sys; import tests.jax_test_environment; print(json.dumps('jax' in sys.modules))"
)


def test_tests_default_to_eight_emulated_cpu_devices() -> None:
    env = resolve_test_environment({}, cuda_plugin_available=True)

    assert env == {"JAX_PLATFORMS": "cpu", "JAX_NUM_CPU_DEVICES": "8"}


def test_an_inherited_cuda_selection_still_runs_tests_on_emulated_cpus() -> None:
    env = resolve_test_environment({"JAX_PLATFORMS": "cuda,cpu"}, cuda_plugin_available=True)

    assert env == {"JAX_PLATFORMS": "cpu", "JAX_NUM_CPU_DEVICES": "8"}


def test_an_explicit_accelerator_request_disables_cpu_emulation_and_keeps_the_flags() -> None:
    env = resolve_test_environment(
        {"DATARAX_TEST_JAX_PLATFORMS": "cuda,cpu", "XLA_FLAGS": "--xla_gpu_autotune_level=2"},
        cuda_plugin_available=True,
    )

    assert env == {"JAX_PLATFORMS": "cuda,cpu"}


def test_an_explicit_cpu_request_keeps_emulation() -> None:
    env = resolve_test_environment(
        {"DATARAX_TEST_JAX_PLATFORMS": "cpu"}, cuda_plugin_available=True
    )

    assert env == {"JAX_PLATFORMS": "cpu", "JAX_NUM_CPU_DEVICES": "8"}


def test_a_device_count_of_zero_disables_emulation() -> None:
    env = resolve_test_environment({"DATARAX_TEST_DEVICE_COUNT": "0"}, cuda_plugin_available=False)

    assert env == {"JAX_PLATFORMS": "cpu"}


def test_a_requested_device_count_is_used_and_the_inherited_flags_are_left_alone() -> None:
    env = resolve_test_environment(
        {"DATARAX_TEST_DEVICE_COUNT": "4", "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false"},
        cuda_plugin_available=False,
    )

    assert env == {"JAX_PLATFORMS": "cpu", "JAX_NUM_CPU_DEVICES": "4"}


def test_a_device_count_already_in_the_flags_is_kept() -> None:
    env = resolve_test_environment(
        {"XLA_FLAGS": f"{_DEVICE_COUNT_FLAG}=2"}, cuda_plugin_available=False
    )

    assert env == {"JAX_PLATFORMS": "cpu"}


def test_asking_for_cuda_without_the_plugin_fails() -> None:
    with pytest.raises(RuntimeError, match="CUDA plugin"):
        resolve_test_environment(
            {"DATARAX_TEST_JAX_PLATFORMS": "cuda"}, cuda_plugin_available=False
        )


def test_importing_the_environment_module_does_not_import_jax() -> None:
    """JAX reads JAX_PLATFORMS at import, so choosing the backend must not load it."""
    result = run_python(
        _IMPORT_PROBE,
        timeout=120.0,
        cwd=_REPO_ROOT,
    )

    assert result.check().last_json() is False


@pytest.mark.parametrize(("requested", "devices"), [("0", 1), ("2", 2), (None, 8)])
def test_a_fresh_interpreter_started_with_the_environment_sees_its_cpu_devices(
    requested: str | None, devices: int
) -> None:
    base = {} if requested is None else {"DATARAX_TEST_DEVICE_COUNT": requested}
    env = resolve_test_environment(base, cuda_plugin_available=False)

    result = run_python(
        "import json, jax; print(json.dumps([jax.default_backend(), jax.device_count()]))",
        env=env,
        timeout=180.0,
    )

    assert result.check().last_json() == ["cpu", devices]


def test_the_substrax_pytest_plugin_is_enabled(pytestconfig: pytest.Config) -> None:
    """It fails a test that changes global jax configuration and adds the device markers."""
    assert pytestconfig.pluginmanager.hasplugin("substrax.testing.pytest_plugin")
