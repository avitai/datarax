"""Shared validation helpers for image operators."""

import logging

from datarax.core.config import OperatorConfig


logger = logging.getLogger(__name__)


def validate_field_key_shape(
    data_shapes: dict[str, tuple[int, ...]],
    field_key: str,
) -> tuple[int, ...]:
    """Validate field_key exists in data_shapes and return the shape.

    Args:
        data_shapes: Dictionary mapping field keys to their shapes.
        field_key: The field key to validate.

    Returns:
        The shape tuple for the given field_key.

    Raises:
        KeyError: If field_key not in data_shapes.
    """
    if field_key not in data_shapes:
        raise KeyError(
            f"Field key '{field_key}' not found in data_shapes. "
            f"Available keys: {list(data_shapes.keys())}"
        )
    return data_shapes[field_key]


def resolve_mode_parameters(
    config: OperatorConfig,
    *,
    range_field: str,
    fixed_field: str,
    default_range: tuple[float, float],
    default_fixed: float,
) -> None:
    """Refuse the parameter a config's mode ignores and fill in the one it uses.

    A stochastic operator draws each record's value from ``range_field``; a deterministic
    operator applies ``fixed_field``. Both fields default to ``None``, so a value set for the
    other mode is refused rather than silently ignored.

    Args:
        config: A frozen operator config; the resolved field is set with ``object.__setattr__``.
        range_field: Name of the ``(min, max)`` field stochastic mode draws from.
        fixed_field: Name of the field deterministic mode applies.
        default_range: The range a stochastic config uses when ``range_field`` is unset.
        default_fixed: The value a deterministic config uses when ``fixed_field`` is unset.

    Raises:
        ValueError: If the other mode's field is set, or the range is not ``(min, max)`` with
            ``min <= max``.
    """
    if config.stochastic:
        if getattr(config, fixed_field) is not None:
            raise ValueError(
                f"{fixed_field} applies only when stochastic=False; "
                f"a stochastic operator draws from {range_field}"
            )
        value_range = getattr(config, range_field)
        if value_range is None:
            object.__setattr__(config, range_field, default_range)
        else:
            _check_range(range_field, value_range)
        return

    if getattr(config, range_field) is not None:
        raise ValueError(
            f"{range_field} applies only when stochastic=True; "
            f"a deterministic operator applies {fixed_field}"
        )
    if getattr(config, fixed_field) is None:
        object.__setattr__(config, fixed_field, default_fixed)


def _check_range(range_field: str, value_range: object) -> None:
    """Raise ``ValueError`` unless ``value_range`` is a ``(min, max)`` tuple with ``min <= max``.

    Args:
        range_field: Name of the field being checked, for the message.
        value_range: The value given for that field.

    Raises:
        ValueError: If it is not a two-tuple, or its minimum exceeds its maximum.
    """
    if not isinstance(value_range, tuple) or len(value_range) != 2:
        raise ValueError(f"{range_field} must be a tuple of length 2, got {value_range}")
    if value_range[0] > value_range[1]:
        raise ValueError(f"{range_field} must be (min, max) with min <= max, got {value_range}")
