"""Configuration management for Datarax.

This package provides utilities for loading, validating, and managing
configuration for Datarax pipelines and components.
"""

from datarax.config.environment import (
    apply_environment_overrides,
    get_env_value,
)
from datarax.config.loaders import load_toml_from_path, save_toml_to_path
from datarax.config.registry import (
    create_component_from_config,
    get_component_constructor,
    is_component_registered,
    list_registered_components,
    register_component,
)
from datarax.config.schema import ConfigSchema, is_schema_type_valid


__all__ = [
    # Loaders
    "load_toml_from_path",
    "save_toml_to_path",
    # Schema
    "ConfigSchema",
    "is_schema_type_valid",
    # Environment
    "apply_environment_overrides",
    "get_env_value",
    # Registry
    "register_component",
    "get_component_constructor",
    "is_component_registered",
    "list_registered_components",
    "create_component_from_config",
]
