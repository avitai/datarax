"""Configuration for Datarax: composable TOML parameters, validated and overridden.

Configurations are small TOML files that include and override one another
(:func:`load_config_from_path_with_includes`), take environment overrides for values they
already have (:func:`apply_environment_overrides`), and are checked by a user-defined
:class:`ConfigSchema`. Pipelines are built in Python from the validated values.
"""

from datarax.config.environment import (
    apply_environment_overrides,
    CONFIG_ENV_PREFIX,
    get_env_value,
)
from datarax.config.loaders import (
    deep_merge_dict,
    load_config_from_path_with_includes,
    load_toml_from_path,
    save_toml_to_path,
)
from datarax.config.registry import (
    create_component_from_config,
    get_component_constructor,
    is_component_registered,
    list_registered_components,
    register_component,
)
from datarax.config.schema import ConfigSchema, is_schema_type_valid, SchemaField, ValidationError


__all__ = [
    # Loaders
    "load_toml_from_path",
    "load_config_from_path_with_includes",
    "deep_merge_dict",
    "save_toml_to_path",
    # Schema
    "ConfigSchema",
    "SchemaField",
    "ValidationError",
    "is_schema_type_valid",
    # Environment
    "CONFIG_ENV_PREFIX",
    "apply_environment_overrides",
    "get_env_value",
    # Registry
    "register_component",
    "get_component_constructor",
    "is_component_registered",
    "list_registered_components",
    "create_component_from_config",
]
