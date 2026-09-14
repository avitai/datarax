"""The JAX environment a datarax test run starts with: backend choice and CPU device emulation.

``substrax.runtime.resolve_test_runtime`` decides it; this module holds datarax's part, the
``DATARAX_TEST_`` variable prefix and how an installed JAX CUDA plugin is detected. Tests run on
CPU with eight emulated devices unless ``DATARAX_TEST_JAX_PLATFORMS`` asks for an accelerator, and
an inherited ``JAX_PLATFORMS`` does not move them onto a GPU, so a local run matches CI. Importing
this module imports no jax, because jax reads these variables when it is imported.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping

from substrax.runtime import resolve_test_runtime, runtime_environment


TEST_VARIABLE_PREFIX = "DATARAX_TEST_"
_CUDA_PLUGINS = ("jax_cuda12_plugin", "jax_cuda13_plugin")


def has_cuda_plugin() -> bool:
    """Whether a JAX CUDA plugin can be imported, checked without importing it."""
    return any(importlib.util.find_spec(name) is not None for name in _CUDA_PLUGINS)


def resolve_test_environment(
    env: Mapping[str, str], *, cuda_plugin_available: bool
) -> dict[str, str]:
    """Return the variables a test run writes into its environment before importing jax.

    Args:
        env: The process environment.
        cuda_plugin_available: Whether a JAX CUDA plugin is installed.

    Returns:
        The variables to set. An inherited variable the run leaves as it is, such as
        ``XLA_FLAGS``, is absent.

    Raises:
        RuntimeError: If CUDA is requested and no JAX CUDA plugin is installed.
    """
    runtime = resolve_test_runtime(
        env, prefix=TEST_VARIABLE_PREFIX, cuda_plugin_available=cuda_plugin_available
    )
    return runtime_environment(runtime, env)
