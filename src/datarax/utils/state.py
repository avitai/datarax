"""State restoration helpers for NNX variables."""

from typing import Any


def restore_optional_variable_fields(
    state: dict[str, Any],
    fields: dict[str, Any],
) -> None:
    """Restore variable-backed fields only when keys are present in state."""
    for key, variable in fields.items():
        if key in state:
            variable.set_value(state[key])


def restore_iteration_variables(
    state: dict[str, Any],
    *,
    current_index: Any,
    current_epoch: Any,
    extra_fields: dict[str, Any] | None = None,
) -> None:
    """Restore common iteration fields plus any optional extras."""
    fields = {
        "current_index": current_index,
        "current_epoch": current_epoch,
    }
    if extra_fields:
        fields.update(extra_fields)
    restore_optional_variable_fields(state, fields)


def build_state_with_iteration_fields(
    state: dict[str, Any],
    *,
    current_index: Any,
    current_epoch: Any,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return state extended with common iteration counters."""
    merged = dict(state)
    merged["current_index"] = current_index
    merged["current_epoch"] = current_epoch
    if extra_fields:
        merged.update(extra_fields)
    return merged


def restore_iteration_and_fields(
    state: dict[str, Any],
    *,
    current_index: Any,
    current_epoch: Any,
    **extra_fields: Any,
) -> None:
    """Restore iteration counters plus optional named variable fields."""
    restore_iteration_variables(
        state,
        current_index=current_index,
        current_epoch=current_epoch,
        extra_fields=extra_fields or None,
    )
