"""A record's fields named by their path, the one naming every source and its refusals share.

A field is a leaf of a record's pytree, named by its path's steps joined by ``/``: a dictionary
key by the key itself, a list position as JAX prints it (``[0]``). Provenance keys, the field
names a refusal prints and the columns a source holds on disk are all named this way.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
from jaxtyping import PyTree


LEVEL = "/"
"""What joins a field's path steps, one per level of its record's nesting."""


def field_name(path: Sequence[Any]) -> str:
    """A field's name: its path's steps joined by ``/``, a dictionary key by the key itself.

    A step without a ``key`` (a list position, ``SequenceKey``) prints as JAX prints it, ``[0]``;
    ``jax.tree_util.keystr(path, simple=True, separator="/")`` would print ``0`` there instead,
    which a dictionary key named ``0`` also prints.

    Args:
        path: A leaf's path, as ``jax.tree_util.tree_flatten_with_path`` gives it.

    Returns:
        The field's name.
    """
    return LEVEL.join(str(getattr(step, "key", step)) for step in path)


def field_names(tree: PyTree) -> list[str]:
    """The name of every leaf of ``tree``, in JAX's flattening order.

    Args:
        tree: A record's pytree.

    Returns:
        Each leaf's field name.
    """
    return [field_name(path) for path, _ in jax.tree_util.tree_flatten_with_path(tree)[0]]


def by_field(tree: PyTree, *, is_leaf: Callable[[Any], bool] | None = None) -> dict[str, Any]:
    """Every leaf of ``tree`` keyed by its field name; :func:`nested` inverts it.

    Args:
        tree: A record's pytree, or a JSON document's mappings.
        is_leaf: What counts as one leaf, as ``jax.tree_util`` takes it; for a JSON document,
            whatever is not a mapping, so a list is one value and an empty mapping no field.

    Returns:
        Each leaf, keyed by its field name.
    """
    flat = jax.tree_util.tree_flatten_with_path(tree, is_leaf=is_leaf)[0]
    return {field_name(path): leaf for path, leaf in flat}


def nested(leaves: Mapping[str, Any]) -> dict[str, Any]:
    """Leaves keyed by field name as nested mappings, one level per path step: naming inverted.

    Args:
        leaves: Values keyed by field name, :data:`LEVEL` between levels.

    Returns:
        The nested mappings.
    """
    tree: dict[str, Any] = {}
    for name, leaf in leaves.items():
        *levels, last = name.split(LEVEL)
        node = tree
        for level in levels:
            node = node.setdefault(level, {})
        node[last] = leaf
    return tree


__all__ = ["LEVEL", "by_field", "field_name", "field_names", "nested"]
