"""Configuration management for Datarax.

This package provides utilities for loading, validating, and managing
configuration for Datarax pipelines and components.
"""

from datarax.config.environment import (
    apply_environment_overrides,
    get_env_value,
)
from datarax.config.loaders import load_toml, save_toml
from datarax.config.registry import (
    create_component_from_config,
    get_component_constructor,
    is_component_registered,
    list_registered_components,
    register_component,
)
from datarax.config.schema import ConfigSchema, SchemaValidator


__all__ = [
    # Loaders
    "load_toml",
    "save_toml",
    # Schema
    "ConfigSchema",
    "SchemaValidator",
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
