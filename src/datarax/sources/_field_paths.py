"""A record's fields named by their path, the one naming every source and its refusals share.

A field is a leaf of a record's pytree, named by its path's steps joined by ``/``: a dictionary
key by the key itself, a list position as JAX prints it (``[0]``). Provenance keys, the field
names a refusal prints and the columns a source holds on disk are all named this way.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
from jaxtyping import PyTree


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
    return "/".join(str(getattr(step, "key", step)) for step in path)


def field_names(tree: PyTree) -> list[str]:
    """The name of every leaf of ``tree``, in JAX's flattening order.

    Args:
        tree: A record's pytree.

    Returns:
        Each leaf's field name.
    """
    return [field_name(path) for path, _ in jax.tree_util.tree_flatten_with_path(tree)[0]]


__all__ = ["field_name", "field_names"]
