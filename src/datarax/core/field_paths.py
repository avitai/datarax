"""One naming of record fields: resolution, creation, display, name order and conversion.

A record is nested dicts of arrays and field nodes (``core.field_nodes``). A field is the tuple of
dict keys from the record's root, its key tuple. A key is a non-empty string and may hold ``/``:
``State/Province`` is one field. A name is a spelling (the keys joined by ``/``) or a key tuple.

- Reading resolves a spelling against the record's structure, by a ``/``-prefix search that
  descends only into groups (dicts), never by splitting the spelling.
- Creating a field from a spelling (C3) writes onto a node the spelling resolves to; otherwise it
  descends into the existing group with the longest spelling that is a ``/``-prefix, and the rest
  of the spelling becomes one key. A spelling never extends a field: ``meta/size`` beside the
  field ``meta`` is refused, offering the key tuple ``("meta/size",)``. A key tuple creates
  exactly its path, and is the only way to create a group.
- A refusal names a node by its spelling in quotes, adding its key tuple where a key on its path
  holds ``/`` or is empty, the only nodes whose spelling hides their structure.
"""

from __future__ import annotations

import functools
import types
import typing
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields
from typing import Any, NoReturn, overload

import jax
from jaxtyping import PyTree

from datarax.core.field_nodes import FieldNode, is_field_node


LEVEL = "/"
"""What joins the keys of a field's path in its spelling."""

type KeyTuple = tuple[str, ...]
"""A field's dict keys from the record's root; a name given as keys is held in this form."""

type FieldName = str | Sequence[str]
"""A name of one field, as a parameter takes it: a spelling, or the field's keys (a key tuple, or
a list of keys as TOML writes it). Held as ``str | KeyTuple``."""

type FieldNames = Sequence[FieldName]
"""Names of several fields, as a parameter takes them: a list or tuple of names, held as a tuple."""


class FieldNotFoundError(KeyError):
    """A name that resolves to no field of the record."""

    def __str__(self) -> str:
        """The message itself, not ``KeyError``'s quoted form.

        Returns:
            The refusal's text.
        """
        return str(self.args[0]) if self.args else ""


class FieldCreateError(ValueError):
    """A create that would break the naming rules."""


# ---------------------------------------------------------------------------------- spellings


def spelling(keys: KeyTuple) -> str:
    """A field's spelling: its keys joined by ``/``.

    Args:
        keys: The field's key tuple.

    Returns:
        The spelling.
    """
    return LEVEL.join(keys)


def is_exotic(key: str) -> bool:
    """Whether a key starts or ends with ``/`` or holds ``//``, which no name may hold.

    Args:
        key: One dict key.

    Returns:
        Whether the key is refused.
    """
    return key.startswith(LEVEL) or key.endswith(LEVEL) or LEVEL * 2 in key


def _is_refused_key(key: object) -> bool:
    return not isinstance(key, str) or key == "" or is_exotic(key)


# ------------------------------------------------------------------------------------ display


def label(keys: KeyTuple, member: str = "") -> str:
    """How a refusal names a record node: its spelling in quotes.

    The key tuple is added where a key on the path holds ``/`` or is empty, the only nodes whose
    spelling hides their structure.

    Args:
        keys: The node's key tuple; ``()`` for the record itself.
        member: What follows the node, as jax prints it: a field node's member (``.present``)
            or a list position (``[0]``).

    Returns:
        The label.
    """
    if not keys:
        return f"<record>{member}"
    text = f"{spelling(keys)!r}{member}"
    if any(LEVEL in key or key == "" for key in keys):
        text += f" (key tuple {keys!r})"
    return text


def _split_path(path: Sequence[Any]) -> tuple[KeyTuple, str]:
    keys: list[str] = []
    for index, step in enumerate(path):
        if isinstance(step, jax.tree_util.DictKey) and isinstance(step.key, str):
            keys.append(step.key)
            continue
        return tuple(keys), jax.tree_util.keystr(tuple(path[index:]))
    return tuple(keys), ""


def path_label(path: Sequence[Any]) -> str:
    """The label of a node reached by a jax key path from the record, under :func:`label`'s rule.

    Args:
        path: The path, as ``jax.tree_util.tree_flatten_with_path`` gives it.

    Returns:
        The label.
    """
    keys, member = _split_path(path)
    return label(keys, member)


def _held(name: FieldName) -> str | KeyTuple:
    """The form a name is held in: a spelling as given, a field's keys as a key tuple."""
    return name if isinstance(name, str) else tuple(name)


def name_label(name: FieldName) -> str:
    """A name as the user wrote it: a spelling in quotes, a field's keys as a key tuple.

    Args:
        name: The name.

    Returns:
        Its text.
    """
    return repr(_held(name))


def describe(node: Any) -> str:
    """What a node is, for a refusal: ``int32[2]`` for an array, the type for anything else.

    Args:
        node: A record node.

    Returns:
        Its description.
    """
    if hasattr(node, "dtype") and hasattr(node, "shape"):
        return f"{node.dtype}{list(node.shape)}"
    return type(node).__name__


# --------------------------------------------------------------------------------- resolution


def field_paths(data: Any) -> list[KeyTuple]:
    """Every field of a record (a leaf or a field node), in insertion order.

    Args:
        data: The record's data.

    Returns:
        Each field's key tuple.
    """
    found: list[KeyTuple] = []

    def walk(node: Any, keys: KeyTuple) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                walk(child, (*keys, key))
        else:
            found.append(keys)

    walk(data, ())
    return found


def _not_found(data: Any, name: FieldName) -> str:
    names = ", ".join(label(keys) for keys in field_paths(data))
    return f"{name_label(name)} names no field of the record; its fields: {names}"


def _search(data: dict, wanted: str) -> list[KeyTuple]:
    found: list[KeyTuple] = []

    def go(node: dict, keys: KeyTuple, rest: str) -> None:
        if rest in node:
            found.append((*keys, rest))
        cut = rest.find(LEVEL)
        while cut != -1:
            child = node.get(rest[:cut])
            if isinstance(child, dict):
                go(child, (*keys, rest[:cut]), rest[cut + 1 :])
            cut = rest.find(LEVEL, cut + 1)

    go(data, (), wanted)
    return sorted(found)


def _walk(data: Any, keys: KeyTuple) -> Any:
    node = data
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            raise KeyError(keys)
        node = node[key]
    return node


def resolve(data: Any, name: FieldName) -> KeyTuple:
    """The key tuple a name refers to in ``data``.

    Args:
        data: The record's data.
        name: A spelling, resolved by the ``/``-prefix search, or the field's keys, walked as
            given.

    Returns:
        The node's key tuple.

    Raises:
        FieldNotFoundError: If no node has the name, listing the record's fields.
        ValueError: If two nodes share the spelling, naming both by key tuple.
    """
    held = _held(name)
    if isinstance(held, tuple):
        try:
            _walk(data, held)
        except KeyError as error:
            raise FieldNotFoundError(_not_found(data, held)) from error
        return held
    found = _search(data, held) if isinstance(data, dict) else []
    if not found:
        raise FieldNotFoundError(_not_found(data, held))
    if len(found) > 1:
        raise ValueError(
            f"{held!r} names {len(found)} nodes of the record: "
            + ", ".join(repr(keys) for keys in found)
        )
    return found[0]


def get_field(data: Any, name: FieldName) -> Any:
    """The field a name refers to.

    Args:
        data: The record's data.
        name: A spelling or a key tuple.

    Returns:
        The field's value.
    """
    return _walk(data, resolve(data, name))


# ----------------------------------------------------------------------------------- creation


def _toml(keys: KeyTuple) -> str:
    return "[" + ", ".join(f'"{key}"' for key in keys) + "]"


def _longest_group(data: dict, wanted: str) -> tuple[KeyTuple, str]:
    """C3's descent: the group with the longest spelling that is a ``/``-prefix, and the rest."""
    best: tuple[KeyTuple, str] = ((), wanted)

    def go(node: dict, keys: KeyTuple, rest: str) -> None:
        nonlocal best
        cut = rest.find(LEVEL)
        while cut != -1:
            child = node.get(rest[:cut])
            if isinstance(child, dict):
                inner, tail = (*keys, rest[:cut]), rest[cut + 1 :]
                if len(tail) < len(best[1]):
                    best = (inner, tail)
                go(child, inner, tail)
            cut = rest.find(LEVEL, cut + 1)

    go(data, (), wanted)
    return best


def _extended_field(group: dict, key: str) -> str | None:
    """The longest key of ``group`` that names a field and is a ``/``-prefix of ``key``."""
    found = None
    cut = key.find(LEVEL)
    while cut != -1:
        child = group.get(key[:cut])
        if key[:cut] in group and not isinstance(child, dict):
            found = key[:cut]
        cut = key.find(LEVEL, cut + 1)
    return found


def _refuse_key(owner: str, name: FieldName, keys: KeyTuple) -> None:
    for key in keys:
        if _is_refused_key(key):
            raise FieldCreateError(
                f"{owner} creates {name_label(name)}, whose key {key!r} is empty or starts or "
                "ends with '/' or holds '//'; a key is a non-empty string without those"
            )


def _create_by_spelling(data: dict, name: str, owner: str, param: str) -> KeyTuple:
    group_keys, rest = _longest_group(data, name)
    keys = (*group_keys, rest)
    _refuse_key(owner, name, keys)
    group = _walk(data, group_keys)
    field = _extended_field(group, rest)
    if field is not None:
        field_keys = (*group_keys, field)
        field_name = spelling(field_keys)
        raise FieldCreateError(
            f"{owner} creates {name!r}, but {field_name!r} is a field "
            f"({describe(group[field])}), not a group, so {name!r} cannot be nested under it. "
            f"For a field whose name holds '/', beside {field_name!r}, pass the key tuple "
            f"{param}={keys!r} (TOML: {param} = {_toml(keys)}); otherwise name another output"
        )
    return keys


def _node_spellings(data: Any) -> dict[str, KeyTuple]:
    spelled: dict[str, KeyTuple] = {}

    def walk(node: Any, keys: KeyTuple) -> None:
        if keys:
            spelled[spelling(keys)] = keys
        if isinstance(node, dict):
            for key, child in node.items():
                walk(child, (*keys, key))

    walk(data, ())
    return spelled


def _create_by_tuple(data: dict, keys: KeyTuple, owner: str) -> KeyTuple:
    _refuse_key(owner, keys, keys)
    node: Any = data
    for depth, key in enumerate(keys[:-1]):
        node = node.get(key, {}) if isinstance(node, dict) else node
        if not isinstance(node, dict):
            through = keys[: depth + 1]
            raise FieldCreateError(
                f"{owner} creates {keys!r}, which passes through {label(through)}, a field "
                f"({describe(node)}) and not a group; name another output"
            )
    try:
        _walk(data, keys)
    except KeyError:
        pass
    else:
        return keys
    spelled = _node_spellings(data)
    for depth in range(1, len(keys) + 1):
        new = keys[:depth]
        existing = spelled.get(spelling(new))
        if existing is not None and existing != new:
            raise FieldCreateError(
                f"{owner} creates {keys!r}, and {label(existing)} already has the spelling "
                f"{spelling(new)!r}; two nodes of a record never share one"
            )
    return keys


def create_path(  # noqa: DOC502 - _create_by_tuple and _create_by_spelling raise it
    data: Any, name: FieldName, *, owner: str, param: str
) -> KeyTuple:
    """The key tuple a create of ``name`` writes (C3).

    Args:
        data: The record the create receives.
        name: A spelling or a key tuple.
        owner: Who creates, for a refusal: the operator's class name.
        param: The parameter that holds the name, for the key tuple a refusal offers.

    Returns:
        The key tuple: an existing node's (a write) or a new field's.

    Raises:
        FieldCreateError: If the create would extend a field, pass through one, give a node a
            spelling another node has, or hold an empty or exotic key.
    """
    record = data if isinstance(data, dict) else {}
    held = _held(name)
    if isinstance(held, tuple):
        return _create_by_tuple(record, held, owner)
    try:
        return resolve(record, held)
    except FieldNotFoundError:
        return _create_by_spelling(record, held, owner, param)


def _set_at(data: dict, keys: KeyTuple, value: Any) -> dict:
    head, rest = keys[0], keys[1:]
    if not rest:
        return {**data, head: value}
    return {**data, head: _set_at(data.get(head, {}), rest, value)}


def set_field(
    data: dict, name: FieldName, value: Any, *, owner: str = "set_field", param: str = "name"
) -> dict:
    """A copy of ``data`` with ``value`` at the field ``name`` writes or creates (C3).

    Args:
        data: The record's data; not modified.
        name: A spelling or a key tuple.
        value: The value to store.
        owner: Who writes, for a refusal.
        param: The parameter that holds the name, for a refusal.

    Returns:
        The updated record.
    """
    return _set_at(data, create_path(data, name, owner=owner, param=param), value)


# ---------------------------------------------------------------------------- field nodes


@overload
def refuse_field_nodes(tree: FieldNode, consumer: str, where: Sequence[Any] = ()) -> NoReturn: ...


@overload
def refuse_field_nodes(tree: PyTree, consumer: str, where: Sequence[Any] = ()) -> None: ...


def refuse_field_nodes(tree: PyTree, consumer: str, where: Sequence[Any] = ()) -> None:
    """Refuse a field node anywhere in ``tree``: ``consumer`` would read it as an array.

    The check reads only the tree's structure, so it costs nothing inside a compiled step. Given
    a field node itself, it never returns.

    Args:
        tree: What ``consumer`` is about to treat as arrays.
        consumer: Who refuses and why, for the message: ``"MapOperator maps fn over every leaf"``.
        where: Where ``tree`` sits in the record, so a label names the field from the root: its
            key tuple, or the jax key path a flattening of the record gave it.

    Raises:
        TypeError: Naming every field node found, with each node type's own fix.
    """
    prefix = tuple(jax.tree_util.DictKey(step) if isinstance(step, str) else step for step in where)
    flat = jax.tree_util.tree_flatten_with_path(tree, is_leaf=is_field_node)[0]
    by_type: dict[type, list[str]] = {}
    for path, leaf in flat:
        if is_field_node(leaf):
            by_type.setdefault(type(leaf), []).append(path_label((*prefix, *path)))
    if by_type:
        reasons = " ".join(
            f"{', '.join(labels)} is {node_type.kind}. {node_type.fix}."
            for node_type, labels in by_type.items()
        )
        raise TypeError(f"{consumer}, and {reasons}")


# ---------------------------------------------------------------------------- name order


def name_order(name: FieldName) -> tuple[int, KeyTuple]:
    """The order names are stored in: spellings first, then key tuples, each by code point.

    Python's own sort raises on a mix of strings and tuples.

    Args:
        name: A spelling or the field's keys.

    Returns:
        The sort key.
    """
    held = _held(name)
    return (0, (held,)) if isinstance(held, str) else (1, held)


def sorted_names(names: FieldNames) -> tuple[str | KeyTuple, ...]:
    """``names`` in name order, so two orders of one selection are one value.

    Args:
        names: Spellings and fields' keys.

    Returns:
        The names as held, in :func:`name_order`.
    """
    return tuple(sorted((_held(name) for name in names), key=name_order))


# ---------------------------------------------------------------------------- conversion


def _is_keys(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str)


def as_field_name(value: Any, parameter: str) -> str | KeyTuple:
    """A value for a parameter that names one field, as it is held: a spelling or a key tuple.

    A list or tuple of strings is one key tuple, as TOML writes it (``["meta", "score"]``).

    Args:
        value: The value given.
        parameter: The parameter's name, for a refusal.

    Returns:
        The name.

    Raises:
        TypeError: If the value is neither a string nor a non-empty list of strings.
    """
    if isinstance(value, str):
        return value
    if _is_keys(value) and value and all(isinstance(k, str) for k in value):
        return tuple(value)
    raise TypeError(
        f"{parameter} names one field: a string (its spelling) or a list of strings (its key "
        f"tuple, one key per level); got {value!r}"
    )


def as_field_names(value: Any, parameter: str) -> tuple[str | KeyTuple, ...]:
    """A value for a parameter that names several fields, held as a tuple of names.

    Args:
        value: The value given; each element is a spelling or a list of strings (a key tuple).
        parameter: The parameter's name, for a refusal.

    Returns:
        The names, in the order given.

    Raises:
        TypeError: If the value is a bare string or not a list or tuple.
    """
    if not _is_keys(value):
        raise TypeError(
            f"{parameter} names several fields: a list whose elements are spellings (strings) "
            f"or key tuples (lists of strings); got {value!r}"
        )
    return tuple(as_field_name(element, f"each element of {parameter}") for element in value)


_CONVERTERS: Mapping[object, Callable[[Any, str], Any]] = {
    FieldName: as_field_name,
    FieldNames: as_field_names,
}


@functools.cache
def field_parameters(config_type: type) -> tuple[tuple[str, Callable[[Any, str], Any]], ...]:
    """The fields of a dataclass annotated ``FieldName`` or ``FieldNames`` (or either ``| None``).

    Args:
        config_type: A dataclass type.

    Returns:
        ``(field name, converter)`` per annotated field, read once per type.
    """
    hints = typing.get_type_hints(config_type)
    found = []
    for item in fields(config_type):
        hint = hints.get(item.name)
        is_union = typing.get_origin(hint) in (typing.Union, types.UnionType)
        options = typing.get_args(hint) if is_union else (hint,)
        for option in options:
            converter = _CONVERTERS.get(option)
            if converter is not None:
                found.append((item.name, converter))
    return tuple(found)


# ------------------------------------------------------------------- layout of stored columns


def field_name(path: Sequence[Any]) -> str:
    """A leaf's name: its path's steps joined by ``/``, a dictionary key by the key itself.

    A step without a ``key`` (a list position, ``SequenceKey``) prints as JAX prints it, ``[0]``;
    ``jax.tree_util.keystr(path, simple=True, separator="/")`` would print ``0`` there instead,
    which a dictionary key named ``0`` also prints.

    Args:
        path: A leaf's path, as ``jax.tree_util.tree_flatten_with_path`` gives it.

    Returns:
        The leaf's name.
    """
    return LEVEL.join(str(getattr(step, "key", step)) for step in path)


def field_names(tree: PyTree) -> list[str]:
    """The name of every leaf of ``tree``, in JAX's flattening order.

    Args:
        tree: A record's pytree.

    Returns:
        Each leaf's name.
    """
    return [field_name(path) for path, _ in jax.tree_util.tree_flatten_with_path(tree)[0]]


def by_field(tree: PyTree, *, is_leaf: Callable[[Any], bool] | None = None) -> dict[str, Any]:
    """Every leaf of ``tree`` keyed by its name; :func:`nested` inverts it.

    Args:
        tree: A record's pytree, or a JSON document's mappings.
        is_leaf: What counts as one leaf, as ``jax.tree_util`` takes it; for a JSON document,
            whatever is not a mapping, so a list is one value and an empty mapping no field.

    Returns:
        Each leaf, keyed by its name.
    """
    flat = jax.tree_util.tree_flatten_with_path(tree, is_leaf=is_leaf)[0]
    return {field_name(path): leaf for path, leaf in flat}


def nested(leaves: Mapping[str, Any]) -> dict[str, Any]:
    """Leaves keyed by name as nested mappings, one level per path step.

    Args:
        leaves: Values keyed by name, :data:`LEVEL` between levels.

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


__all__ = [
    "LEVEL",
    "FieldCreateError",
    "FieldName",
    "FieldNames",
    "FieldNotFoundError",
    "KeyTuple",
    "as_field_name",
    "as_field_names",
    "by_field",
    "create_path",
    "describe",
    "field_name",
    "field_names",
    "field_parameters",
    "field_paths",
    "get_field",
    "is_exotic",
    "label",
    "name_label",
    "name_order",
    "nested",
    "path_label",
    "refuse_field_nodes",
    "resolve",
    "set_field",
    "sorted_names",
    "spelling",
]
