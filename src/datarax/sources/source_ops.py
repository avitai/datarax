"""Operations a data source is built from (public helpers for source authors).

The bundled sources (HFEagerSource, TFDSEagerSource, HFStreamingSource,
TFDSStreamingSource, MemorySource) and sources in other packages delegate to
these helpers for:
- Wrapped index resolution, in the order a key selects (``resolve_wrapped_indices``)
- A worker's share of the records (``partition_length``)
- A named-dataset source's repr (``format_source_repr``)
- Key filters resolved at construction (``KeyFilter``)

In-memory sources read through ``datarax.sources.EagerSource``; streams build on
``datarax.sources.StreamingSourceBase``.
"""

from __future__ import annotations

from collections.abc import Mapping, Set as AbstractSet
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from datarax.core import index_words
from datarax.core.block_order import BlockOrder, order_positions
from datarax.core.field_paths import (
    field_listing,
    field_paths,
    FieldNotFoundError,
    KeyTuple,
    resolve,
)


# What a record keeps, by key: the value kept whole (None) or cut by the plan below it.
type KeepPlan = dict[str, KeepPlan | None]


def partition_length(length: int, num_workers: int = 1, shard_id: int = 0) -> int:
    """Return how many records worker ``shard_id`` of ``num_workers`` serves.

    The worker serves positions ``[shard_id::num_workers]`` of the dataset order.

    Args:
        length: Total number of records in the dataset.
        num_workers: Number of workers the dataset is split across.
        shard_id: This worker's index.

    Returns:
        ``len(range(shard_id, length, num_workers))``, computed by integer division, which
        ``len`` cannot do past ``2**63 - 1`` positions.
    """
    return max(0, -(-(length - shard_id) // num_workers))


def resolve_wrapped_indices(
    start: int | ArrayLike,
    size: int,
    length: int,
    key: jax.Array | None,
    *,
    block: BlockOrder | None = None,
    num_workers: int = 1,
    shard_id: int = 0,
) -> jax.Array:
    """Return the global record indices for a wrapped slice of a worker's record order.

    The dataset order is ``arange(length)``, the keyed bijection of it when a ``key`` is given,
    or ``block``'s block order of it with a block order too
    (:func:`~datarax.core.block_order.order_positions`; the pipeline passes its epoch key
    exactly when it shuffles); each index costs O(1), so a slice costs O(size) at every
    dataset size and no order is stored. Worker ``shard_id`` of
    ``num_workers`` serves positions ``[shard_id::num_workers]`` of that order, and its
    logical positions ``start + arange(size)`` wrap at its own length. The same arguments
    always yield the same indices. Positions and indices are 64-bit, exact for every length up
    to ``2**63`` (:func:`~datarax.core.index_words.wrapped_positions`).

    Args:
        start: Starting logical position: a Python int of any size, its two uint32 words
            ``(hi, lo)`` (a position of the worker's order, below its length), or a traced
            int32 ``jax.Array``.
        size: Number of records to return (static Python int).
        length: Total number of records in the dataset.
        key: The key selecting the order, or ``None`` for the sequential order.
        block: The block order, or ``None`` for the global order a key selects.
        num_workers: Number of workers the dataset is split across.
        shard_id: This worker's index.

    Returns:
        uint32 ``(size, 2)`` global record indices, each as its words ``(hi, lo)``.
    """
    worker_length = partition_length(length, num_workers, shard_id)
    positions = index_words.wrapped_positions(start, size, worker_length)
    if num_workers > 1 or shard_id:
        scaled = index_words.multiply_word(
            (positions[:, 0], positions[:, 1]), np.uint32(num_workers)
        )
        high, low = index_words.add(scaled, index_words.split_constant(shard_id))
        positions = jnp.stack([high, low], axis=-1)
    if key is None and block is None:
        return positions
    return order_positions(positions, length, key, block)


def format_source_repr(
    class_name: str,
    dataset_name: str | None,
    split_name: str | None,
    length: int | None,
    extra_fields: dict[str, Any] | None = None,
) -> str:
    """Format a stable source repr string with optional extra fields."""
    fields: list[tuple[str, Any]] = [
        ("dataset", f"{dataset_name}:{split_name}"),
        ("length", length),
    ]
    if extra_fields:
        fields.extend(extra_fields.items())
    serialized = ", ".join(f"{key}={value}" for key, value in fields)
    return f"{class_name}({serialized})"


@dataclass(frozen=True, slots=True)
class KeyFilter:
    """``include_keys`` or ``exclude_keys`` resolved once against a dataset's fields.

    A filter holds field names (N3), so ``meta/score`` names the nested field ``score`` of the
    group ``meta``. The names resolve at construction, an unknown one refused listing the fields,
    and resolution also decides what a record keeps: each record is then only cut to that decision,
    and a record no filter names is returned as it is. A TFDS stream and an ArrayRecord read filter
    every record they decode, so nothing is decided per record.

    Attributes:
        include: The key tuples kept, with everything below them, or ``None`` for every field.
        exclude: The key tuples dropped, or ``None`` for none.
        plan: What a record keeps, decided at resolution: each kept key's value is kept whole
            (``None``) or cut by the plan below it; ``None`` when the filter names nothing. Not
            part of the filter's identity, which is ``include`` and ``exclude``.
    """

    include: frozenset[KeyTuple] | None = None
    exclude: frozenset[KeyTuple] | None = None
    plan: KeepPlan | None = field(default=None, compare=False, hash=False, repr=False)

    def __post_init__(self) -> None:
        """Refuse a filter that names fields without the plan resolution decides.

        Raises:
            TypeError: If ``include`` or ``exclude`` is given without a plan.
        """
        if self.plan is None and (self.include is not None or self.exclude is not None):
            raise TypeError(
                "a KeyFilter that names fields decides what a record keeps against the "
                "dataset's fields: build it with "
                "KeyFilter.resolved(owner, fields, include_keys, exclude_keys)"
            )

    @classmethod
    def resolved(  # noqa: DOC502 - _resolved_names raises the ValueError
        cls,
        owner: str,
        fields: dict[str, Any],
        include_keys: AbstractSet[str] | None,
        exclude_keys: AbstractSet[str] | None,
    ) -> KeyFilter:
        """The filter a configuration names, resolved against the dataset's fields.

        Args:
            owner: Who names the filter, for the refusal: the configuration's class name.
            fields: The dataset's data and provenance fields as the record holds them: dicts
                nested by group, whose leaves are fields of any value.
            include_keys: The names kept, or ``None``.
            exclude_keys: The names dropped, or ``None``.

        Returns:
            The resolved filter.

        Raises:
            ValueError: If a name is not a field of the dataset, listing its fields.
        """
        include = _resolved_names(owner, fields, "include_keys", include_keys)
        exclude = _resolved_names(owner, fields, "exclude_keys", exclude_keys)
        if include is None and exclude is None:
            return cls()
        return cls(include=include, exclude=exclude, plan=_plan(fields, (), include, exclude))

    def keep(self, record: Any) -> dict[str, Any]:
        """The fields of ``record`` the filter keeps; a group it empties is dropped.

        Args:
            record: A decoded record, or a dataset's features as nested mappings, holding the
                fields the filter was resolved against or a subset of them; a group is a mapping
                and anything else one field, kept or dropped whole.

        Returns:
            ``record`` itself when the filter names nothing, else its kept fields, nested as
            ``record``.
        """
        return record if self.plan is None else _applied(record, self.plan)


def _resolved_names(
    owner: str, tree: Any, parameter: str, names: AbstractSet[str] | None
) -> frozenset[KeyTuple] | None:
    if names is None:
        return None
    resolved = set()
    for name in sorted(names):
        try:
            resolved.add(resolve(tree, name))
        except FieldNotFoundError as error:
            raise ValueError(
                f"{owner} names {name!r} in {parameter}, which the dataset does not have; "
                f"its fields are {field_listing(field_paths(tree))}"
            ) from error
    return frozenset(resolved)


def _plan(
    node: Any,
    path: KeyTuple,
    include: frozenset[KeyTuple] | None,
    exclude: frozenset[KeyTuple] | None,
) -> KeepPlan:
    """What a record keeps of ``node``, the fields at ``path``, decided once at resolution.

    A group is opened only where a name lies below it; any other kept node is kept whole.
    """
    plan: KeepPlan = {}
    for key, child in node.items():
        here = (*path, key)
        is_kept, below = _include_below(here, include)
        if not is_kept or (exclude is not None and here in exclude):
            continue
        if isinstance(child, Mapping) and (below is not None or _excludes_below(here, exclude)):
            inner = _plan(child, here, below, exclude)
            if inner or not child:
                plan[key] = inner
        else:
            plan[key] = None
    return plan


def _excludes_below(here: KeyTuple, exclude: frozenset[KeyTuple] | None) -> bool:
    """Whether ``exclude`` drops a node strictly below the one at ``here``."""
    return exclude is not None and any(
        len(keys) > len(here) and keys[: len(here)] == here for keys in exclude
    )


def _applied(record: Mapping[str, Any], plan: KeepPlan) -> dict[str, Any]:
    """``record`` cut to ``plan``: a value the plan opens but which is no mapping is kept whole."""
    kept: dict[str, Any] = {}
    for key, value in record.items():
        if key not in plan:
            continue
        below = plan[key]
        kept[key] = (
            value if below is None or not isinstance(value, Mapping) else _applied(value, below)
        )
    return kept


def _include_below(
    here: KeyTuple, include: frozenset[KeyTuple] | None
) -> tuple[bool, frozenset[KeyTuple] | None]:
    """Whether ``include`` keeps the node at ``here``, and what it still selects below it.

    A node named, or under a named one, is kept whole (``None`` below); a node above a named one
    is kept for the names below it; any other node is dropped.
    """
    if include is None or here in include:
        return True, None
    return any(keys[: len(here)] == here for keys in include), include
