"""Shared random-parameter helpers for operators."""

from jaxtyping import PyTree

from datarax.core.operator import extract_batch_size


def get_optional_batch_size(data_shapes: PyTree) -> int | None:
    """Return batch size from shape pytrees, or None for empty pytrees."""
    try:
        return extract_batch_size(data_shapes)
    except ValueError:
        return None
