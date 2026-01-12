"""Environment variable integration for configuration.

This module provides utilities for integrating environment variables with
configuration files, allowing for environment-specific overrides.
"""

import os
from typing import Any


def get_env_value(env_var: str, default: Any = None, prefix: str = "DATARAX_") -> str | None:
    """Get a value from an environment variable.

    Args:
        env_var: The name of the environment variable (without prefix)
        default: Default value to return if the environment variable is not set
        prefix: Prefix to apply to the environment variable name

    Returns:
        The environment variable value, or the default if not set
    """
    full_name = f"{prefix}{env_var}"
    return os.environ.get(full_name, default)


def apply_environment_overrides(
    config: dict[str, Any], prefix: str = "DATARAX_", separator: str = "__"
) -> dict[str, Any]:
    """Apply environment variable overrides to a configuration dictionary.

    Environment variables can override configuration values using a naming
    convention. For example, to override `config.database.host`, the environment
    variable would be `DATARAX_DATABASE__HOST`.

    Args:
        config: The configuration dictionary to apply overrides to
        prefix: Prefix for environment variables to consider
        separator: Separator used to indicate nested keys

    Returns:
        Configuration dictionary with environment overrides applied
    """
    result = config.copy()

    # Get all environment variables with the given prefix
    env_vars = {k: v for k, v in os.environ.items() if k.startswith(prefix)}

    for env_name, env_value in env_vars.items():
        # Remove prefix
        config_path = env_name[len(prefix) :]

        # Skip empty paths
        if not config_path:
            continue

        # Split path by separator to get nested keys
        keys = config_path.split(separator)

        # Convert to lowercase for case insensitivity
        keys = [k.lower() for k in keys]

        # Apply the override
        _set_nested_value(result, keys, _convert_value(env_value))

    return result


def _convert_value(value: str) -> Any:
    """Convert a string value to an appropriate type.

    This function attempts to convert string values from environment variables
    to appropriate Python types (bool, int, float, or string).

    Args:
        value: The string value to convert

    Returns:
        The converted value
    """
    # Handle boolean values
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False

    # Handle numeric values
    try:
        # Try to convert to integer
        return int(value)
    except ValueError:
        try:
            # Try to convert to float
            return float(value)
        except ValueError:
            # Keep as string
            return value


def _set_nested_value(config: dict[str, Any], keys: list[str], value: Any) -> None:
    """Set a value in a nested dictionary using a list of keys.

    Args:
        config: The dictionary to modify
        keys: List of keys indicating the path to the value
        value: The value to set
    """
    # Handle single-level key
    if len(keys) == 1:
        config[keys[0]] = value
        return

    # Handle nested keys
    current_key = keys[0]

    # Create nested dictionary if it doesn't exist
    if current_key not in config or not isinstance(config[current_key], dict):
        config[current_key] = {}

    # Recursively set the value in the nested dictionary
    _set_nested_value(config[current_key], keys[1:], value)
