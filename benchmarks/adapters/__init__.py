"""Benchmark adapter registry.

Provides a decorator-based registry for BenchmarkAdapter implementations
and functions to discover available adapters.

Design ref: Section 7.4 of the benchmark report.
"""

from __future__ import annotations

import importlib

from benchmarks.adapters.base import BenchmarkAdapter

_ADAPTER_REGISTRY: dict[str, type[BenchmarkAdapter]] = {}


def register(cls: type[BenchmarkAdapter]) -> type[BenchmarkAdapter]:
    """Register a BenchmarkAdapter subclass in the global registry.

    Usage::

        @register
        class MyAdapter(BenchmarkAdapter):
            ...
    """
    # Use the adapter's name property to key the registry
    name = cls._get_registry_name()
    _ADAPTER_REGISTRY[name] = cls
    return cls


def get_available_adapters() -> dict[str, type[BenchmarkAdapter]]:
    """Return all registered adapters that are currently available."""
    return {
        name: adapter_cls
        for name, adapter_cls in _ADAPTER_REGISTRY.items()
        if adapter_cls._check_available()
    }


def get_adapters_for_scenario(scenario_id: str) -> dict[str, type[BenchmarkAdapter]]:
    """Return registered adapters that support a given scenario."""
    return {
        name: adapter_cls
        for name, adapter_cls in _ADAPTER_REGISTRY.items()
        if adapter_cls._check_available() and scenario_id in adapter_cls._supported()
    }


# ---------------------------------------------------------------------------
# Auto-register all available adapters
# ---------------------------------------------------------------------------
# Importing an adapter module triggers the @register decorator.
# ImportError means the framework isn't installed — silently skip.

# Pre-import Deep Lake before any adapter that might pull in TensorFlow.
# See benchmarks/adapters/_preload.py for the full explanation.
import benchmarks.adapters._preload  # noqa: F401


def _try_import(module_path: str) -> None:
    """Attempt to import an adapter module; swallow ImportError."""
    try:
        importlib.import_module(module_path)
    except ImportError:
        pass


# Datarax adapter — always available (core framework)
# Import triggers @register on DataraxAdapter
from benchmarks.adapters.datarax_adapter import DataraxAdapter  # noqa: F401, E402

# Peer framework adapters — import triggers @register
_PEER_MODULES = [
    "benchmarks.adapters.grain_adapter",
    "benchmarks.adapters.jax_dl_adapter",
    "benchmarks.adapters.tfdata_adapter",
    "benchmarks.adapters.pytorch_dl_adapter",
    "benchmarks.adapters.dali_adapter",
    "benchmarks.adapters.ffcv_adapter",
    "benchmarks.adapters.spdl_adapter",
    "benchmarks.adapters.mosaic_adapter",
    "benchmarks.adapters.webdataset_adapter",
    "benchmarks.adapters.hf_datasets_adapter",
    "benchmarks.adapters.ray_data_adapter",
    "benchmarks.adapters.litdata_adapter",
    "benchmarks.adapters.energon_adapter",
    "benchmarks.adapters.deep_lake_adapter",
]

for _mod in _PEER_MODULES:
    _try_import(_mod)
