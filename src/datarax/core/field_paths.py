"""Dotted paths to record fields: ``"image"``, or ``"audio.signal"`` for a nested field.

Operators name the fields they read and write with these paths, and composites that combine
operator outputs address the same fields through them.
"""

from typing import Any


def get_field(data: dict, path: str) -> Any:
    """Return the field of ``data`` at ``path``.

    Args:
        data: The record.
        path: A key, or dot-separated keys into nested dicts.

    Returns:
        The field value.

    Raises:
        KeyError: If a key on the path is missing or holds something other than a dict.
    """
    current: Any = data
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Field '{path}' not found in data")
        current = current[key]
    return current


def set_field(data: dict, path: str, value: Any) -> dict:  # noqa: DOC502
    """Return a copy of ``data`` with ``value`` stored at ``path``.

    Missing nested dicts on the path are created; ``data`` itself is not modified.

    Args:
        data: The record.
        path: A key, or dot-separated keys into nested dicts.
        value: The value to store.

    Returns:
        The updated record.

    Raises:
        ValueError: If a key on the path holds something other than a dict.
    """
    return _set_at(data, path.split("."), value, path)


def _set_at(data: dict, keys: list[str], value: Any, path: str) -> dict:
    key, rest = keys[0], keys[1:]
    if not rest:
        return {**data, key: value}
    nested = data.get(key, {})
    if not isinstance(nested, dict):
        raise ValueError(f"Cannot create nested path '{path}': '{key}' is not a dict")
    return {**data, key: _set_at(nested, rest, value, path)}
