"""Field nodes: one record field held as several arrays.

A field node (``Maybe``; packed fields later) is one field for naming: a spelling resolves to the
node, never to its members, and a create cannot nest a field under it. A consumer that reads a
field as an array refuses a field node, naming it with the node's own fix
(``core.field_paths.refuse_field_nodes``).

This module imports nothing from datarax, so the naming module can recognise field nodes without
importing any of them.
"""

from __future__ import annotations

from typing import Any, ClassVar


class FieldNode:
    """The base of every field node.

    Attributes:
        kind: How a refusal describes the node, after "is": ``"a Maybe (a value that can be
            missing)"``.
        fix: What a refusal tells the user to do with it, ending the sentence.
    """

    __slots__ = ()

    kind: ClassVar[str] = "a field node"
    fix: ClassVar[str] = "read its members explicitly"


def is_field_node(value: Any) -> bool:
    """Whether ``value`` is a field node.

    Args:
        value: Any node of a record's tree.

    Returns:
        Whether it is one field held as several arrays.
    """
    return isinstance(value, FieldNode)


__all__ = ["FieldNode", "is_field_node"]
