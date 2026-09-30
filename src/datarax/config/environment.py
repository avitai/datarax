"""Environment variables that override values of a loaded configuration.

An override is ``DATARAX_CONFIG__<KEY>[__<KEY>...]=<value>``: the prefix keeps configuration
apart from the operational ``DATARAX_*`` variables (test, benchmark and activation settings),
and the names after it, matched case-insensitively, select a value the configuration already
has. The string is read as the type of the value it replaces, so the configuration file is the
schema of what can be overridden:

- ``bool``: ``true`` or ``false`` (any case), nothing else;
- ``int`` and ``float``: Python's number syntax (an ``int`` string also reads as ``float``);
- ``str``: the string as given;
- ``list``: a TOML array, e.g. ``[32, 32]``.

A string that does not read as the type, a name the configuration does not have, or a name that
selects a table is refused.
"""

import copy
import logging
import os
import tomllib
from collections.abc import Mapping
from typing import Any


logger = logging.getLogger(__name__)

CONFIG_ENV_PREFIX = "DATARAX_CONFIG__"
"""Prefix of the environment variables that override configuration values."""


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
    config: dict[str, Any],
    *,
    prefix: str = CONFIG_ENV_PREFIX,
    separator: str = "__",
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return a copy of ``config`` with the overrides in ``environ`` applied.

    ``DATARAX_CONFIG__TRAINING__BATCH_SIZE=64`` replaces ``config["training"]["batch_size"]``,
    read as that value's type (module docstring).

    Args:
        config: The configuration; it is not modified.
        prefix: Prefix of the override variables.
        separator: Separator between nested names.
        environ: The variables to read; the process environment when ``None``.

    Returns:
        A deep copy of ``config`` with every override applied.
    """
    result = copy.deepcopy(config)
    variables = os.environ if environ is None else environ
    for name in sorted(variables):
        if name.startswith(prefix) and len(name) > len(prefix):
            keys = [key.lower() for key in name[len(prefix) :].split(separator)]
            _override(result, keys, variables[name], name)
    return result


def _override(config: dict[str, Any], keys: list[str], text: str, name: str) -> None:
    """Replace the value at ``keys`` in ``config`` with ``text`` read as that value's type.

    Reading ``text`` raises ``ValueError`` when the value there is a table or ``text`` does not
    read as its type (:func:`_read_as`).

    Args:
        config: The configuration copy being overridden.
        keys: Lower-cased names from the variable.
        text: The variable's value.
        name: The variable's name, for error messages.

    Raises:
        KeyError: If the configuration has no value at ``keys``.
    """
    table = config
    for depth, key in enumerate(keys[:-1]):
        table = table.get(key) if isinstance(table, dict) else None
        if not isinstance(table, dict):
            path = ".".join(keys[: depth + 1])
            raise KeyError(f"{name}: the configuration has no table '{path}'")
    last = keys[-1]
    if last not in table:
        path = ".".join(keys)
        raise KeyError(f"{name}: the configuration has no value '{path}'")
    table[last] = _read_as(table[last], text, name)


def _read_as(current: Any, text: str, name: str) -> Any:
    """Read ``text`` as the type of ``current``.

    Args:
        current: The value being replaced.
        text: The variable's value.
        name: The variable's name, for error messages.

    Returns:
        ``text`` converted to ``current``'s type.

    Raises:
        ValueError: If ``current`` is a table or of a type an override cannot express, or
            ``text`` does not read as its type.
    """
    kind = type(current)
    if kind is bool:
        if text.lower() not in ("true", "false"):
            raise ValueError(f"{name}={text!r}: a boolean is 'true' or 'false'")
        return text.lower() == "true"
    if kind is str:
        return text
    if kind is list:
        try:
            value = tomllib.loads(f"value = {text}")["value"]
        except tomllib.TOMLDecodeError as error:
            raise ValueError(f"{name}={text!r}: not a TOML array") from error
        if not isinstance(value, list):
            raise ValueError(f"{name}={text!r}: not a TOML array")
        return value
    if kind in (int, float):
        try:
            return kind(text)
        except ValueError as error:
            raise ValueError(f"{name}={text!r}: not {kind.__name__}") from error
    raise ValueError(f"{name}: an override cannot replace a value of type {kind.__name__}")
