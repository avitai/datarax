"""TOML configuration loading and saving utilities.

This module provides functions for loading and saving TOML configuration files
for Datarax pipelines and components.
"""

from pathlib import Path
from typing import Any, Union

import tomli_w  # type: ignore

try:
    import tomllib  # type: ignore
except ImportError:
    import tomli as tomllib  # type: ignore


def load_toml(config_path: Union[str, Path], encoding: str = "utf-8") -> dict[str, Any]:
    """Load a TOML configuration file.

    Args:
        config_path: Path to the TOML configuration file
        encoding: Character encoding to use (default: utf-8)

    Returns:
        Dictionary containing the parsed TOML configuration

    Raises:
        FileNotFoundError: If the configuration file does not exist
        tomllib.TOMLDecodeError: If the configuration file is invalid TOML
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("rb") as f:
        return tomllib.load(f)


def save_toml(
    config: dict[str, Any], config_path: Union[str, Path], encoding: str = "utf-8"
) -> None:
    """Save a configuration dictionary to a TOML file.

    Args:
        config: Dictionary containing the configuration to save
        config_path: Path to save the TOML configuration file
        encoding: Character encoding to use (default: utf-8)

    Raises:
        OSError: If the file cannot be written
    """
    config_path = Path(config_path)

    # Create parent directories if they don't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with config_path.open("wb") as f:
        tomli_w.dump(config, f)


def deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries.

    This function performs a deep merge of two dictionaries, with values from
    `override` taking precedence. For nested dictionaries, the merge is recursive.

    Args:
        base: Base dictionary to merge into
        override: Dictionary with values that override the base

    Returns:
        A new dictionary containing the merged values
    """
    result = base.copy()

    for key, override_value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(override_value, dict):
            # Recursively merge nested dictionaries
            result[key] = deep_merge_dict(result[key], override_value)
        else:
            # Override or add the value
            result[key] = override_value

    return result


def load_config_with_includes(
    config_path: Union[str, Path],
    encoding: str = "utf-8",
    include_key: str = "include",
    processed_paths: set[Path] | None = None,
) -> dict[str, Any]:
    """Load a TOML configuration file with support for including other files.

    This function loads a TOML configuration file and processes any include
    directives to merge configurations from multiple files. Included files can
    themselves include other files, up to a reasonable depth.

    Args:
        config_path: Path to the TOML configuration file
        encoding: Character encoding to use (default: utf-8)
        include_key: Key that specifies included config files (default: "include")
        processed_paths: Set of already processed paths to prevent cycles

    Returns:
        Dictionary containing the merged TOML configuration

    Raises:
        FileNotFoundError: If any configuration file does not exist
        RecursionError: If circular includes are detected
        tomllib.TOMLDecodeError: If any configuration file is invalid TOML
    """
    config_path = Path(config_path).resolve()

    if processed_paths is None:
        processed_paths = set()

    if config_path in processed_paths:
        raise RecursionError(f"Circular include detected: {config_path}")

    processed_paths.add(config_path)

    # Load the base configuration
    config = load_toml(config_path, encoding)

    # Process includes if present
    includes = config.pop(include_key, [])
    if isinstance(includes, str):
        includes = [includes]

    base_dir = config_path.parent

    # Start with an empty configuration that will be updated with each include
    merged_config = {}

    # Process each included file
    for include_path in includes:
        include_full_path = (base_dir / include_path).resolve()

        # Load and merge the included config (recursive)
        included_config = load_config_with_includes(
            include_full_path,
            encoding=encoding,
            include_key=include_key,
            processed_paths=processed_paths,
        )

        # Merge with the result so far
        merged_config = deep_merge_dict(merged_config, included_config)

    # Finally, merge with the current config (which takes precedence)
    merged_config = deep_merge_dict(merged_config, config)

    return merged_config
