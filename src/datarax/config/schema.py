"""Configuration schema definition and validation.

This module provides classes and utilities for defining and validating
configuration schemas for Datarax pipelines and components.

The Datarax configuration system uses TOML for defining pipeline configurations
and supports configuration of NNX-specific features, including:

- RNG streams for stochastic operations
- State persistence for stateful components
- NNX module configuration options
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Type, Union


# Type alias for schema type definitions
SchemaType = Union[
    Type[str],
    Type[int],
    Type[float],
    Type[bool],
    Type[list],
    Type[dict],
    "ConfigSchema",
    Type["ConfigSchema"],
]


class ValidationError(Exception):
    """Exception raised when configuration validation fails."""

    pass


@dataclass
class SchemaField:
    """Definition of a field in a configuration schema.

    Attributes:
        type: The expected type of the field
        required: Whether the field is required
        default: Default value for the field if not specified
        validator: Optional function to validate the field value
        description: Optional description of the field
    """

    type: SchemaType
    required: bool = True
    default: Any = None
    validator: Callable[[Any], bool] | None = None
    description: str | None = None


class ConfigSchema:
    """Base class for configuration schemas.

    Subclasses should define schema fields as class variables using SchemaField.

    Examples:
        ```python
        class MyConfigSchema(ConfigSchema):
            name: SchemaField = SchemaField(str, required=True)
            count: SchemaField = SchemaField(int, required=False, default=0)
        ```
    """

    @classmethod
    def get_schema_fields(cls) -> dict[str, SchemaField]:
        """Get all schema fields defined in the class.

        Returns:
            Dictionary mapping field names to SchemaField instances
        """
        fields = {}

        for name, field_type in cls.__annotations__.items():
            if hasattr(cls, name):
                attr = getattr(cls, name)
                if isinstance(attr, SchemaField):
                    fields[name] = attr

        return fields

    @classmethod
    def validate(cls, config: dict[str, Any]) -> dict[str, Any]:
        """Validate a configuration dictionary against the schema.

        Args:
            config: The configuration dictionary to validate

        Returns:
            A validated configuration dictionary with defaults applied

        Raises:
            ValidationError: If the configuration fails validation
        """
        schema_fields = cls.get_schema_fields()
        validated = {}

        # Check for required fields and apply defaults
        for name, schema_field in schema_fields.items():
            if name in config:
                value = config[name]

                # Validate type
                if not SchemaValidator.validate_type(value, schema_field.type):
                    raise ValidationError(f"Field '{name}' should be of type {schema_field.type}")

                # Apply custom validator if provided
                if schema_field.validator is not None and not schema_field.validator(value):
                    raise ValidationError(f"Field '{name}' failed custom validation")

                validated[name] = value
            elif schema_field.required:
                raise ValidationError(f"Required field '{name}' is missing")
            else:
                # Use default value for non-required fields
                validated[name] = schema_field.default

        # Check for unknown fields
        unknown_fields = set(config.keys()) - set(schema_fields.keys())
        if unknown_fields:
            raise ValidationError(f"Unknown fields in configuration: {', '.join(unknown_fields)}")

        return validated

    @classmethod
    def create(cls, config: dict[str, Any]) -> dict[str, Any]:
        """Create a validated configuration dictionary from the schema.

        Args:
            config: The configuration dictionary to validate

        Returns:
            A validated configuration dictionary with defaults applied

        Raises:
            ValidationError: If the configuration fails validation
        """
        return cls.validate(config)


class SchemaValidator:
    """Utility class for validating configuration values against schema types."""

    @staticmethod
    def validate_type(value: Any, expected_type: SchemaType) -> bool:
        """Validate that a value matches an expected type.

        Args:
            value: The value to validate
            expected_type: The expected type of the value

        Returns:
            True if the value matches the expected type, False otherwise
        """
        # Handle primitive types
        if expected_type in (str, int, float, bool):
            return isinstance(value, expected_type)

        # Handle lists
        if expected_type == list or expected_type == Type[list]:
            return isinstance(value, list)

        # Handle dictionaries
        if expected_type == dict or expected_type == Type[dict]:
            return isinstance(value, dict)

        # Handle ConfigSchema classes
        if isinstance(expected_type, type) and issubclass(expected_type, ConfigSchema):
            if not isinstance(value, dict):
                return False

            try:
                expected_type.validate(value)
                return True
            except ValidationError:
                return False

        # Handle ConfigSchema instances
        if isinstance(expected_type, ConfigSchema):
            if not isinstance(value, dict):
                return False

            try:
                expected_type.validate(value)
                return True
            except ValidationError:
                return False

        # Unknown type
        return False


@dataclass
class PipelineSchema(ConfigSchema):
    """Schema for pipeline configuration.

    This schema defines the structure of a pipeline configuration file,
    including data sources, transformers, augmenters, and other components.
    """

    name: SchemaField = field(
        default_factory=lambda: SchemaField(str, required=True, description="Name of the pipeline")
    )

    description: SchemaField = field(
        default_factory=lambda: SchemaField(
            str, required=False, default="", description="Description of the pipeline"
        )
    )

    version: SchemaField = field(
        default_factory=lambda: SchemaField(
            str,
            required=False,
            default="0.1.0",
            description="Version of the pipeline configuration",
        )
    )

    sources: SchemaField = field(
        default_factory=lambda: SchemaField(
            dict, required=True, description="Data source configurations"
        )
    )

    transforms: SchemaField = field(
        default_factory=lambda: SchemaField(
            dict, required=False, default={}, description="Data transformation configurations"
        )
    )

    augmenters: SchemaField = field(
        default_factory=lambda: SchemaField(
            dict, required=False, default={}, description="Data augmentation configurations"
        )
    )

    samplers: SchemaField = field(
        default_factory=lambda: SchemaField(
            dict, required=False, default={}, description="Data sampling configurations"
        )
    )

    batch_size: SchemaField = field(
        default_factory=lambda: SchemaField(
            int, required=False, default=32, description="Default batch size for the pipeline"
        )
    )

    random_seed: SchemaField = field(
        default_factory=lambda: SchemaField(
            int, required=False, default=42, description="Random seed for reproducibility"
        )
    )

    rng_streams: SchemaField = field(
        default_factory=lambda: SchemaField(
            dict,
            required=False,
            default={"default": 42, "augment": 43, "dropout": 44},
            description="RNG streams for NNX components with their seed values",
        )
    )

    checkpointing: SchemaField = field(
        default_factory=lambda: SchemaField(
            dict,
            required=False,
            default={"enabled": False, "directory": "checkpoints", "frequency": 1000},
            description="Checkpointing configuration for saving pipeline state",
        )
    )

    device_mesh: SchemaField = field(
        default_factory=lambda: SchemaField(
            dict,
            required=False,
            default={},
            description="JAX device mesh configuration for distributed training",
        )
    )


@dataclass
class NNXComponentSchema(ConfigSchema):
    """Schema for configuring components that use NNX modules.

    This schema defines the structure for components that leverage
    Flax NNX for state management and computation.
    """

    type: SchemaField = field(
        default_factory=lambda: SchemaField(
            str, required=True, description="The type of NNX component to create"
        )
    )

    params: SchemaField = field(
        default_factory=lambda: SchemaField(
            dict,
            required=False,
            default={},
            description="Parameters to pass to the component constructor",
        )
    )

    variables: SchemaField = field(
        default_factory=lambda: SchemaField(
            dict,
            required=False,
            default={},
            description="Initial values for NNX variables in the component",
        )
    )

    rngs: SchemaField = field(
        default_factory=lambda: SchemaField(
            dict,
            required=False,
            default={},
            description="RNG stream configurations specific to this component",
        )
    )

    load_state_from: SchemaField = field(
        default_factory=lambda: SchemaField(
            str,
            required=False,
            default=None,
            description="Path to load initial component state from",
        )
    )
