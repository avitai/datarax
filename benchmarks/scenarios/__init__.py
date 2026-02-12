"""Benchmark scenario registry.

Provides discover_scenarios() to find all scenario modules and
their variants. Scenario modules are organized by category
(vision, nlp, tabular, etc.) and each exports:

    SCENARIO_ID: str
    VARIANTS: dict[str, ScenarioVariant]
    TIER1_VARIANT: str | None
    get_variant(name: str) -> ScenarioVariant
"""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType
from benchmarks.scenarios.base import ScenarioVariant  # noqa: F401

# Category -> subpackage mapping
_CATEGORY_PACKAGES = [
    "benchmarks.scenarios.vision",
    "benchmarks.scenarios.nlp",
    "benchmarks.scenarios.tabular",
    "benchmarks.scenarios.multimodal",
    "benchmarks.scenarios.pipeline_complexity",
    "benchmarks.scenarios.io",
    "benchmarks.scenarios.distributed",
    "benchmarks.scenarios.production",
    "benchmarks.scenarios.augmentation",
    "benchmarks.scenarios.datarax_unique",
]


def _is_scenario_module(module: ModuleType) -> bool:
    """Check if a module is a valid scenario module."""
    return (
        hasattr(module, "SCENARIO_ID")
        and hasattr(module, "VARIANTS")
        and hasattr(module, "get_variant")
    )


def discover_scenarios(
    tier: int | None = None,
) -> list[ModuleType]:
    """Discover all scenario modules, optionally filtered by tier.

    Args:
        tier: If 1, only return scenarios with TIER1_VARIANT defined.
              If 2 or None, return all scenarios.

    Returns:
        List of scenario modules.
    """
    modules: list[ModuleType] = []

    for pkg_name in _CATEGORY_PACKAGES:
        try:
            pkg = importlib.import_module(pkg_name)
        except ImportError:
            continue

        for importer, mod_name, is_pkg in pkgutil.iter_modules(pkg.__path__, prefix=f"{pkg_name}."):
            if is_pkg:
                continue
            try:
                mod = importlib.import_module(mod_name)
            except ImportError:
                continue
            if _is_scenario_module(mod):
                if tier == 1:
                    if getattr(mod, "TIER1_VARIANT", None) is not None:
                        modules.append(mod)
                else:
                    modules.append(mod)

    return modules


def get_scenario_by_id(scenario_id: str) -> ModuleType | None:
    """Find a specific scenario module by its SCENARIO_ID.

    Args:
        scenario_id: e.g. "CV-1", "NLP-1".

    Returns:
        The scenario module, or None if not found.
    """
    for mod in discover_scenarios():
        if mod.SCENARIO_ID == scenario_id:
            return mod
    return None
