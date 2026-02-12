"""Shared validation helpers for image operators."""


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
