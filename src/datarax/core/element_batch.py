"""``Element`` and ``Batch``: a record and a batch of records, as frozen pytrees of arrays.

Both are dataclasses registered with ``jax.tree_util.register_dataclass`` whose every field is a
pytree child. Nothing that is not an array is part of them, so they pass through every JAX and
Flax NNX transform, a batch's values never enter a treedef (two batches differing only in their
values share one compiled program), and a ``Batch`` of axis values or shardings is a prefix for
``vmap``, ``shard_map`` and placement. Provenance such as file names stays on the host with the
source; a record is named by its index.

A record's identity is a 64-bit index held as two uint32 words ``(hi, lo)``, the epoch it is
served in and its draw within that epoch. Together with an operator's base key they decide the
record's randomness (``datarax.core.prng.per_record_keys``). A record's index has the top bit
clear; one with it set names no record: the all-ones ``PADDING_INDEX`` marks padding, and
``(2**31, row)`` names a row of a batch built without identities.

A field whose value a record may lack is a ``datarax.core.Maybe`` in ``data``, at any depth: a
pytree node of the values and a ``present`` flag per record, which every batch operation and
placement moves with its rows.

What datarax records about a record is typed, beside the user's ``state``: its weight in losses
and metrics (``Element.weight``, ``Batch.weights``; 1 for a record, 0 for a padding row), the
values an operator hid but kept as targets (``masked``) or filled in (``imputed``), as bool marks
in the data's own nesting, and on a batch the partners and coefficients a mixing operator used
(``Batch.mix``). ``state`` and ``batch_state`` hold only the user's names.

Operations over batches are pure functions in ``datarax.core.batch_ops``.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, KW_ONLY, replace as dataclass_replace
from typing import Self

import jax
import numpy as np
from jaxtyping import PyTree


type ArrayValue = jax.Array | np.ndarray
"""An array on the device or the host: what every identity field of a record or batch holds."""

PADDING_INDEX: np.ndarray = np.full(2, np.iinfo(np.uint32).max, dtype=np.uint32)
"""The index of a row that is not a record: both uint32 words all ones."""
PADDING_INDEX.setflags(write=False)

MARK_ATTRIBUTES = ("masked", "imputed")
"""The attributes of a record and a batch that hold bool marks in the data's nesting."""


def _zero_ordinal() -> np.ndarray:
    return np.zeros((), dtype=np.int32)


def _empty() -> dict[str, PyTree]:
    return {}


def _unit_weight() -> np.ndarray:
    return np.ones((), dtype=np.float32)


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class Mix:
    """What a whole-batch mixing operator (MixUp, CutMix) did to each row of a batch.

    A loss for mixed records reads both, one value per row:
    ``lams * loss(y) + (1 - lams) * loss(y[partners])``. Partners are row numbers of the batch the
    operator returned, so they name rows of that batch only: ``take``, ``slice_rows``, ``split``,
    ``concatenate`` and ``compact`` move rows without renumbering them.

    Attributes:
        partners: int32 ``(B,)``, the row each record was mixed with.
        lams: ``(B,)``, the fraction of each record kept; a mixing operator draws one per batch
            and writes it to every row.
    """

    partners: ArrayValue
    lams: ArrayValue


class MixedTwiceError(ValueError):
    """A mixing operator was given a batch another one already mixed.

    A batch carries one ``Mix``, which a loss reads; a second mix would replace the first's
    partners and coefficients. Inside a DAG the refusal names both stages.

    Attributes:
        mixer: The refused operator, or its stage in a DAG.
        first: The stage that mixed the batch first, when known.
    """

    def __init__(self, mixer: str, first: str | None = None) -> None:
        """Name the refused mixer and, when known, the stage that mixed first.

        Args:
            mixer: The refused operator, or its stage in a DAG.
            first: The stage that mixed the batch first, or ``None``.
        """
        self.mixer = mixer
        self.first = first
        already = f"already mixed by {first}" if first is not None else "already mixed"
        super().__init__(
            f"{mixer} mixes a batch {already}. A loss reads one partner and one coefficient per "
            "row. Apply one mixing operator per batch."
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class Element:
    """One record: its values, its processing state and its identity.

    Attributes:
        data: The record's values and their intrinsic structure (masks, lengths, segment ids);
            a field the record may lack is a ``Maybe``.
        state: What the user's processing records about the record.
        index: The record's identity in its source, uint32 ``(2,)`` as ``(hi, lo)``, or
            ``None`` for a record built without one: stacked into a batch, such records are
            named by their rows (``batch_ops.from_stacked``), so no two share a key.
        epoch: int32 scalar, the pass over the data the record is served in.
        draw: int32 scalar, which draw of the record within its epoch (0 unless served again).
        weight: float32 scalar, the record's weight in losses and metrics; 1 by default.
        masked: bool marks in the data's nesting, at a field or a prefix of its shape: the
            value is kept as the target and hidden from the model. Empty by default.
        imputed: bool marks in the data's nesting: the value was missing and an operator filled
            it. Empty by default.
    """

    data: PyTree
    _: KW_ONLY
    state: PyTree = field(default_factory=_empty)
    index: ArrayValue | None = None
    epoch: ArrayValue = field(default_factory=_zero_ordinal)
    draw: ArrayValue = field(default_factory=_zero_ordinal)
    weight: ArrayValue = field(default_factory=_unit_weight)
    masked: PyTree = field(default_factory=_empty)
    imputed: PyTree = field(default_factory=_empty)

    def replace(self, **fields: PyTree) -> Self:
        """Return a copy with ``fields`` replaced.

        Args:
            **fields: New values, by field name.

        Returns:
            The new element; this one is unchanged.
        """
        return dataclass_replace(self, **fields)

    def update_data(self, updates: Mapping[str, PyTree]) -> Self:
        """Return a copy whose data mapping has ``updates`` merged in.

        ``update_data`` merges keys, as ``Batch`` reads look keys up: a key holding "/" is one
        key, and the post-stage check refuses a record where it shares a spelling with another
        node.

        Args:
            updates: Fields to add or replace in ``data``.

        Returns:
            The new element; this one is unchanged.

        Raises:
            TypeError: If ``data`` is not a mapping.
        """
        if not isinstance(self.data, Mapping):
            raise TypeError(
                f"update_data merges into a mapping; this element's data is "
                f"{type(self.data).__name__}. Use replace(data=...) instead."
            )
        return self.replace(data={**self.data, **updates})

    def update_state(self, updates: Mapping[str, PyTree]) -> Self:
        """Return a copy whose state mapping has ``updates`` merged in.

        Args:
            updates: Entries to add or replace in ``state``.

        Returns:
            The new element; this one is unchanged.
        """
        return self.replace(state={**self.state, **updates})


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class Batch:
    """A batch of records: the fields of ``Element`` with a leading record axis ``B``.

    ``batch["image"]``, ``batch.get("mask")`` and ``"mask" in batch`` read ``data``; a key that
    is not a string raises. A batch is not a mapping: ``dict``, ``**``, iteration and ``len``
    raise rather than return the data keys alone, which would silently drop the other fields.
    A computation over the batch's values maps over ``batch.data``; ``jax.tree.map`` over the
    whole batch also reaches its identities, weights, marks, mix and state.

    The constructor validates nothing: jax builds a ``Batch`` from tracers and from axis or
    sharding prefixes. ``batch_ops.from_arrays`` and ``batch_ops.from_stacked`` build one from
    host or device arrays and check the record axis.

    Attributes:
        data: The records' values, leading axis ``B``, static shapes.
        states: The user's per-record processing state, leading axis ``B``.
        indices: uint32 ``(B, 2)``, each record's 64-bit index as ``(hi, lo)``.
        epochs: int32 ``(B,)``, each record's epoch.
        draws: int32 ``(B,)``, each record's draw within its epoch.
        weights: float32 ``(B,)``, each record's weight: 1 for a record and 0 for a row whose
            index is ``PADDING_INDEX``, set wherever an index is written.
        masked: bool marks ``(B, *p)`` in the data's nesting, ``p`` a prefix of the field's
            per-record shape: values kept as targets and hidden from the model.
        imputed: bool marks ``(B, *p)`` in the data's nesting: values an operator filled in.
        mix: The partners and coefficients of the mixing operator that mixed the batch, or
            ``None``.
        batch_state: The user's batch-level arrays, without a record axis.
    """

    data: PyTree
    _: KW_ONLY
    states: PyTree
    indices: ArrayValue
    epochs: ArrayValue
    draws: ArrayValue
    weights: ArrayValue
    masked: PyTree
    imputed: PyTree
    mix: Mix | None
    batch_state: PyTree

    __iter__ = None  # a Batch is not iterable; iterate over batch.data

    def __getitem__(self, name: str) -> PyTree:
        """Return the data field ``name``.

        Args:
            name: A key of ``data``.

        Returns:
            ``data[name]``.

        Raises:
            TypeError: If ``name`` is not a string (``batch[0]`` has no meaning: see
                ``batch_ops.element``).
        """
        if not isinstance(name, str):
            raise TypeError(
                f"a Batch is read by field name, got {type(name).__name__}; "
                "use batch_ops.element(batch, i) for a record"
            )
        return self.data[name]

    def get(self, name: str, default: PyTree | None = None) -> PyTree | None:
        """Return the data field ``name``, or ``default`` when there is none.

        Args:
            name: A key of ``data``.
            default: What to return when ``data`` has no ``name``.

        Returns:
            ``data[name]`` or ``default``.
        """
        return self.data[name] if name in self else default

    def __contains__(self, name: object) -> bool:
        """Whether ``data`` has the field ``name``."""
        return isinstance(name, str) and name in self.data

    @property
    def batch_size(self) -> int:
        """The number of rows, ``B``: static, also under tracing."""
        return self.indices.shape[0]

    def replace(self, **fields: PyTree) -> Self:
        """Return a copy with ``fields`` replaced.

        Args:
            **fields: New values, by field name.

        Returns:
            The new batch; this one is unchanged.
        """
        return dataclass_replace(self, **fields)


__all__ = [
    "MARK_ATTRIBUTES",
    "PADDING_INDEX",
    "ArrayValue",
    "Batch",
    "Element",
    "Mix",
    "MixedTwiceError",
]
