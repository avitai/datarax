"""TOML configuration loading utility.

Loads scenario definitions and hardware profiles from TOML files
in the ``benchmarks/config/`` directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def load_scenarios(config_path: Path | None = None) -> dict[str, Any]:
    """Load scenario definitions from TOML.

    Args:
        config_path: Path to scenarios TOML file. Defaults to
            ``benchmarks/config/scenarios.toml``.

    Returns:
        Dictionary keyed by scenario ID.
    """
    path = config_path or (_CONFIG_DIR / "scenarios.toml")
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return data.get("scenario", {})


def load_hardware_profile(profile_name: str) -> dict[str, Any]:
    """Load a hardware profile from TOML.

    Args:
        profile_name: Profile name (e.g., 'ci_cpu'). Loaded from
            ``benchmarks/config/hardware_profiles/<name>.toml``.

    Returns:
        Hardware profile configuration dictionary.
    """
    path = _CONFIG_DIR / "hardware_profiles" / f"{profile_name}.toml"
    with open(path, "rb") as f:
        return tomllib.load(f)
