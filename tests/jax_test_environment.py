"""JAX backend selection and CPU device emulation for test runs.

Tests run on CPU with emulated devices unless ``DATARAX_TEST_JAX_PLATFORMS``
asks for an accelerator. An inherited ``JAX_PLATFORMS`` (for example the
``cuda,cpu`` a developer shell exports) does not move tests onto a GPU, so a
local run matches CI unless the accelerator is requested for the test run.
"""

from __future__ import annotations

from collections.abc import Mapping


_EMULATION_FLAG = "--xla_force_host_platform_device_count"
_DEFAULT_DEVICE_COUNT = "8"


def resolve_test_jax_environment(
    env: Mapping[str, str], *, cuda_plugin_available: bool
) -> dict[str, str]:
    """Return the ``JAX_PLATFORMS`` and ``XLA_FLAGS`` a test run uses.

    ``DATARAX_TEST_JAX_PLATFORMS`` naming a backend other than ``cpu`` selects
    it and disables emulation, which only affects the CPU backend. Otherwise the
    run uses the CPU with ``DATARAX_TEST_DEVICE_COUNT`` emulated devices (eight
    by default, none when it is ``0``), keeping any device count already in
    ``XLA_FLAGS``.

    Args:
        env: The process environment.
        cuda_plugin_available: Whether a JAX CUDA plugin is installed.

    Returns:
        Values for ``JAX_PLATFORMS`` and ``XLA_FLAGS``.

    Raises:
        RuntimeError: If CUDA is requested and no JAX CUDA plugin is installed.
    """
    flags = env.get("XLA_FLAGS", "")
    requested = env.get("DATARAX_TEST_JAX_PLATFORMS", "")
    accelerators = [
        name.strip() for name in requested.split(",") if name.strip() not in ("", "cpu")
    ]
    if any(name.startswith("cuda") for name in accelerators) and not cuda_plugin_available:
        raise RuntimeError(
            f"DATARAX_TEST_JAX_PLATFORMS={requested!r} asks for CUDA, but no JAX CUDA plugin "
            "is installed; run ./setup.sh --backend cuda12."
        )
    if accelerators:
        return {"JAX_PLATFORMS": requested, "XLA_FLAGS": flags}
    count = env.get("DATARAX_TEST_DEVICE_COUNT", _DEFAULT_DEVICE_COUNT)
    if count != "0" and _EMULATION_FLAG not in flags:
        flags = f"{flags} {_EMULATION_FLAG}={count}".strip()
    return {"JAX_PLATFORMS": "cpu", "XLA_FLAGS": flags}
