"""Configuration schemas: the fields a configuration must and may carry.

A schema is a :class:`ConfigSchema` subclass whose class attributes are :class:`SchemaField`
instances. Its fields include every base class's (a subclass redefining a name wins), and
``validate`` returns a new dictionary with the defaults applied, each default its own copy.
Types follow what TOML produces: an integer where a float is expected becomes a float, and a
boolean is neither an integer nor a float.
"""

import copy
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """A configuration does not satisfy its schema."""


@dataclass(frozen=True)
class SchemaField:
    """Definition of a field in a configuration schema.

    Attributes:
        type: The expected type: ``str``, ``int``, ``float``, ``bool``, ``list``, ``dict``, or a
            :class:`ConfigSchema` subclass for a nested table.
        required: Whether the field must be present.
        default: Value used when an optional field is absent; each validation receives a copy.
        validator: Optional predicate the value must satisfy.
        description: Optional description of the field.
    """

    type: "SchemaType"
    required: bool = True
    default: Any = None
    validator: Callable[[Any], bool] | None = None
    description: str | None = None


class ConfigSchema:
    """Base class for configuration schemas.

    Subclasses define fields as class attributes holding :class:`SchemaField` instances.

    Examples:
        ```python
        class Training(ConfigSchema):
            batch_size: SchemaField = SchemaField(int)
            learning_rate: SchemaField = SchemaField(float, required=False, default=1e-3)


        class FineTuning(Training):
            freeze_backbone: SchemaField = SchemaField(bool, required=False, default=True)
        ```
    """

    @classmethod
    def get_schema_fields(cls) -> dict[str, SchemaField]:
        """Return every field of the schema, its bases' included.

        Returns:
            Field name to :class:`SchemaField`; a subclass's definition replaces its base's.
        """
        fields: dict[str, SchemaField] = {}
        for klass in reversed(cls.__mro__):
            fields.update(
                {name: attr for name, attr in vars(klass).items() if isinstance(attr, SchemaField)}
            )
        return fields

    @classmethod
    def validate(cls, config: dict[str, Any]) -> dict[str, Any]:
        """Validate a configuration dictionary against the schema.

        Args:
            config: The configuration dictionary to validate.

        Returns:
            A new dictionary with defaults applied, floats widened from integers and nested
            schemas validated in turn.

        Raises:
            ValidationError: If a required field is missing, a field is unknown, has the wrong
                type or fails its validator; a nested failure names the field path.
        """
        schema_fields = cls.get_schema_fields()
        unknown = sorted(set(config) - set(schema_fields))
        if unknown:
            raise ValidationError(f"Unknown fields in configuration: {', '.join(unknown)}")

        validated: dict[str, Any] = {}
        for name, schema_field in schema_fields.items():
            if name not in config:
                if schema_field.required:
                    raise ValidationError(f"Required field '{name}' is missing")
                default = copy.deepcopy(schema_field.default)
                if _is_nested_schema(schema_field.type):
                    default = _validated_value(name, default, schema_field.type)
                validated[name] = default
                continue
            value = _validated_value(name, config[name], schema_field.type)
            if schema_field.validator is not None and not schema_field.validator(value):
                raise ValidationError(f"Field '{name}' failed custom validation")
            validated[name] = value
        return validated

    @classmethod
    def create(cls, config: dict[str, Any]) -> dict[str, Any]:  # noqa: DOC502
        """Create a validated configuration dictionary from the schema.

        Args:
            config: The configuration dictionary to validate

        Returns:
            A validated configuration dictionary with defaults applied

        Raises:
            ValidationError: If the configuration fails validation
        """
        return cls.validate(config)


def _is_nested_schema(expected_type: "SchemaType") -> bool:
    """Return whether a field's type is a :class:`ConfigSchema` subclass.

    Args:
        expected_type: The field's type.

    Returns:
        ``True`` for a nested schema.
    """
    return isinstance(expected_type, type) and issubclass(expected_type, ConfigSchema)


def _validated_value(name: str, value: Any, expected_type: "SchemaType") -> Any:
    """Return ``value`` checked against ``expected_type``, a nested schema validated.

    Args:
        name: The field's name, for the error message.
        value: The configured value.
        expected_type: The field's type.

    Returns:
        ``value``, as a float for a float field given an integer, or the nested schema's
        validated dictionary.

    Raises:
        ValidationError: If the value does not have the type, or a nested schema refuses it.
    """
    if _is_nested_schema(expected_type):
        if not isinstance(value, dict):
            raise ValidationError(f"Field '{name}' should be a table for {expected_type.__name__}")
        try:
            return expected_type.validate(value)
        except ValidationError as error:
            raise ValidationError(f"Field '{name}': {error}") from error
    if not is_schema_type_valid(value, expected_type):
        raise ValidationError(f"Field '{name}' should be of type {expected_type}")
    if expected_type is float:
        return float(value)
    return value


SchemaType = type[Any]


def is_schema_type_valid(value: Any, expected_type: SchemaType) -> bool:
    """Return whether ``value`` has ``expected_type`` as TOML values are typed.

    Args:
        value: The configured value.
        expected_type: ``str``, ``int``, ``float``, ``bool``, ``list``, ``dict`` or a
            :class:`ConfigSchema` subclass.

    Returns:
        ``True`` when the value fits: a boolean fits only ``bool``, an integer also fits
        ``float``, and a nested schema is fitted by a dictionary it validates.
    """
    if _is_nested_schema(expected_type):
        return isinstance(value, dict) and _is_valid_nested(expected_type, value)
    if expected_type is bool:
        return isinstance(value, bool)
    accepted = _ACCEPTED_TYPES.get(expected_type)
    return accepted is not None and not isinstance(value, bool) and isinstance(value, accepted)


_ACCEPTED_TYPES: dict[type, tuple[type, ...]] = {
    str: (str,),
    int: (int,),
    float: (int, float),
    list: (list,),
    dict: (dict,),
}
"""Python types a TOML value of each non-boolean field type may have."""


def _is_valid_nested(schema: type[ConfigSchema], value: dict[str, Any]) -> bool:
    """Return whether ``schema`` validates ``value``.

    Args:
        schema: The nested schema.
        value: The configured table.

    Returns:
        ``True`` if ``schema.validate`` accepts ``value``.
    """
    try:
        schema.validate(value)
    except ValidationError:
        return False
    return True
