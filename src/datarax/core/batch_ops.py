"""Operations over ``Batch``: pure functions over the pytree, one module for all of them.

Inside a compiled step they trace into the step and fuse with what consumes them, so a slice or a
concatenation costs no copy of its own. On a host batch of NumPy arrays the construction and
regrouping operations (``from_arrays``, ``from_stacked``, ``element``, ``slice_rows``, ``take``,
``split``, ``as_chunk``, ``concatenate``, ``stack``) stay on the host, slices and reshapes as NumPy
views, until the batch is placed. The padding operations (``mask``, ``compact``,
``record_count``) are step operations and use ``jax.numpy``. Called eagerly on device arrays every
operation works, one dispatch per leaf.

A padding row carries ``PADDING_INDEX`` and ``state_keys.WEIGHT`` 0; filtering keeps a batch's
static shape by turning rows into padding rather than removing them.
"""

from collections.abc import Callable, Mapping, Sequence
from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Sharding
from jax.typing import ArrayLike
from jaxtyping import PyTree

from datarax.core.element_batch import ArrayValue, Batch, Element, PADDING_INDEX
from datarax.core.state_keys import WEIGHT


def _namespace(*leaves: object) -> ModuleType:
    """NumPy when every leaf is a host array, so host work stays on the host; else jax.numpy."""
    if all(isinstance(leaf, np.ndarray | np.generic) for leaf in leaves):
        return np
    return jnp


def _record_axis(tree: PyTree, what: str) -> int:
    """Return the leading-axis length every leaf of ``tree`` shares.

    Args:
        tree: The arrays to check.
        what: How the error names ``tree``.

    Returns:
        The shared length.

    Raises:
        ValueError: If ``tree`` has no leaves, a leaf has no leading axis, or the lengths differ.
    """
    leaves = jax.tree.leaves(tree)
    if not leaves:
        raise ValueError(f"{what} has no arrays, so no record axis")
    shapes: list[tuple[int, ...]] = [tuple(getattr(leaf, "shape", ())) for leaf in leaves]
    lengths = {shape[0] for shape in shapes if shape}
    if len(lengths) != 1 or not all(shapes):
        found = sorted({str(shape[0]) if shape else "none" for shape in shapes})
        raise ValueError(
            f"{what} does not have one record axis: its leaves' leading axes are {found}"
        )
    return lengths.pop()


def _row_indices(size: int, xp: ModuleType) -> ArrayValue:
    """Row ``i`` named as record ``(0, i)``: the identity of a row built without one."""
    return xp.stack([xp.zeros(size, xp.uint32), xp.arange(size, dtype=xp.uint32)], axis=1)


def _map_rows(batch: Batch, fn: Callable[[ArrayValue], ArrayValue]) -> Batch:
    """Apply ``fn`` to every per-record leaf; the batch-level state is kept as it is."""
    rows: Batch = jax.tree.map(fn, batch.replace(batch_state=None))
    return rows.replace(batch_state=batch.batch_state)


def _is_padding(indices: ArrayValue) -> jax.Array:
    return jnp.all(jnp.asarray(indices) == PADDING_INDEX, axis=-1)


def _weights(batch: Batch) -> jax.Array:
    """Each row's weight: the batch's own, or 1."""
    states: Mapping[str, PyTree] = batch.states
    if WEIGHT in states:
        return jnp.asarray(states[WEIGHT])
    return jnp.ones(batch.batch_size, jnp.float32)


def from_arrays(  # noqa: DOC502 - _record_axis raises the ValueError
    data: PyTree,
    *,
    states: PyTree | None = None,
    batch_state: PyTree | None = None,
    indices: ArrayValue | None = None,
    epochs: ArrayValue | None = None,
) -> Batch:
    """Build a batch from arrays with a leading record axis.

    Without ``indices``, row ``i`` is record ``(0, i)``, so each row keys its randomness on its
    position: the same arrays draw the same values on every call, and no two rows share a key.
    Without ``epochs``, every row is of epoch 0. Every row is draw 0.

    Args:
        data: The records' values, every leaf with the record axis first.
        states: Per-record state with the same record axis; none by default.
        batch_state: Batch-level arrays; none by default.
        indices: uint32 ``(B, 2)`` record names ``(hi, lo)``; row positions by default.
        epochs: int32 ``(B,)`` epochs the rows were served in; 0 by default.

    Returns:
        The batch, its identities NumPy arrays when ``data`` is.

    Raises:
        ValueError: If ``data`` and ``states`` do not share one record axis.
    """
    states = {} if states is None else states
    size = _record_axis((data, states), "a batch's data and states")
    xp = _namespace(*jax.tree.leaves(data))
    return Batch(
        data,
        states=states,
        indices=_row_indices(size, xp) if indices is None else indices,
        epochs=xp.zeros(size, xp.int32) if epochs is None else epochs,
        draws=xp.zeros(size, xp.int32),
        batch_state={} if batch_state is None else batch_state,
    )


def from_stacked(records: Element) -> Batch:
    """Build a batch from an ``Element`` whose every leaf carries a leading record axis.

    That is what batching ``Element`` records produces: Grain's ``batch``, a stack of the
    records' leaves, or a ``vmap`` returning ``Element``. The identities are the records' own;
    records built without an index are named by their rows, as ``from_arrays`` names them.

    Args:
        records: The stacked records.

    Returns:
        The batch, with no batch-level state.

    Raises:
        ValueError: If the records' leaves do not share one record axis, or the index is not
            ``(B, 2)``.
    """
    size = _record_axis(records, "the stacked records")
    index = records.index
    if index is None:
        index = _row_indices(size, _namespace(*jax.tree.leaves(records)))
    elif index.shape != (size, 2):
        raise ValueError(
            f"stacked record indices must be ({size}, 2) uint32 words; got {index.shape}"
        )
    return Batch(
        records.data,
        states=records.state,
        indices=index,
        epochs=records.epoch,
        draws=records.draw,
        batch_state={},
    )


def element(batch: Batch, i: int | jax.Array) -> Element:
    """Return row ``i`` as an ``Element``: its values, state and identity.

    Args:
        batch: The batch.
        i: The row, a Python int or a traced integer.

    Returns:
        The record; the batch-level state is not part of it.
    """
    row = _map_rows(batch, lambda x: x[i])
    return Element(row.data, state=row.states, index=row.indices, epoch=row.epochs, draw=row.draws)


def slice_rows(batch: Batch, start: int, stop: int) -> Batch:
    """Return rows ``start:stop``, with the batch-level state; a NumPy view on the host.

    Args:
        batch: The batch.
        start: The first row.
        stop: One past the last row.

    Returns:
        The rows as a batch.
    """
    return _map_rows(batch, lambda x: x[start:stop])


def take(batch: Batch, rows: ArrayValue) -> Batch:
    """Return the given rows, in the given order, with the batch-level state.

    Args:
        batch: The batch.
        rows: Row numbers; a row may repeat.

    Returns:
        The rows as a batch.
    """
    return _map_rows(batch, lambda x: x[rows])


def split(batch: Batch, parts: int) -> list[Batch]:  # noqa: DOC502 - _part_size raises
    """Split the rows into ``parts`` equal batches, each with the batch-level state.

    Args:
        batch: The batch.
        parts: How many batches to make.

    Returns:
        The parts, in row order.

    Raises:
        ValueError: If ``parts`` does not divide the batch size.
    """
    size = _part_size(batch, parts)
    return [slice_rows(batch, k * size, (k + 1) * size) for k in range(parts)]


def as_chunk(batch: Batch, parts: int) -> Batch:  # noqa: DOC502 - _part_size raises
    """The rows as a ``(parts, B / parts, ...)`` chunk, equal to ``stack(split(batch, parts))``.

    Every per-record leaf is reshaped, not copied, so a host batch's chunk holds views of its
    rows; the batch-level state is repeated per part, as each part of a split carries it.

    Args:
        batch: The batch.
        parts: Batches in the chunk.

    Returns:
        The chunk.

    Raises:
        ValueError: If ``parts`` does not divide the batch size.
    """
    size = _part_size(batch, parts)
    rows = _map_rows(batch, lambda x: x.reshape(parts, size, *x.shape[1:]))
    repeated = jax.tree.map(lambda leaf: _namespace(leaf).stack([leaf] * parts), batch.batch_state)
    return rows.replace(batch_state=repeated)


def _part_size(batch: Batch, parts: int) -> int:
    """Rows per part when the batch is cut into ``parts`` equal batches.

    Args:
        batch: The batch.
        parts: How many batches to make.

    Returns:
        The rows of each part.

    Raises:
        ValueError: If ``parts`` does not divide the batch size.
    """
    if parts <= 0 or batch.batch_size % parts:
        raise ValueError(f"{parts} parts do not divide a batch of {batch.batch_size} rows")
    return batch.batch_size // parts


def concatenate(batches: Sequence[Batch]) -> Batch:
    """Join the rows of ``batches`` in order; the batch-level state is the first batch's.

    Args:
        batches: Batches with one structure.

    Returns:
        One batch holding every row.

    Raises:
        ValueError: If ``batches`` is empty.
    """
    if not batches:
        raise ValueError("concatenate needs at least one batch")
    rows = [batch.replace(batch_state=None) for batch in batches]
    joined = jax.tree.map(lambda *xs: _namespace(*xs).concatenate(xs, axis=0), *rows)
    return joined.replace(batch_state=batches[0].batch_state)


def stack[T: (Batch, Element)](items: Sequence[T]) -> T:
    """Stack ``items`` along a new leading axis.

    Batches become a ``(K, B, ...)`` chunk: every leaf is stacked, the batch-level state
    included, so scanning the chunk hands each step one whole batch. Records become one
    ``Element`` whose every leaf has a record axis, which ``from_stacked`` turns into a batch.

    Args:
        items: Batches, or records, with one structure.

    Returns:
        The stacked items.

    Raises:
        ValueError: If ``items`` is empty.
    """
    if not items:
        raise ValueError("stack needs at least one item")
    return jax.tree.map(lambda *xs: _namespace(*xs).stack(xs, axis=0), *items)


def record_count(batch: Batch) -> jax.Array:
    """The number of rows that are records, not padding: a traced int32 scalar.

    Args:
        batch: The batch.

    Returns:
        The count.
    """
    return jnp.sum(~_is_padding(batch.indices), dtype=jnp.int32)


def mask(batch: Batch, keep: ArrayLike) -> Batch:
    """Turn the rows not kept into padding, at the batch's shape.

    A row not kept gets ``PADDING_INDEX`` and weight 0; a kept row keeps its weight (1 when the
    batch carries none). A padding row stays padding. Values are untouched.

    Args:
        batch: The batch; ``states`` is a mapping.
        keep: bool ``(B,)``, the rows to keep.

    Returns:
        The masked batch.
    """
    keep = jnp.asarray(keep, bool) & ~_is_padding(batch.indices)
    weights = _weights(batch)
    return batch.replace(
        states={**batch.states, WEIGHT: jnp.where(keep, weights, jnp.zeros_like(weights))},
        indices=jnp.where(keep[:, None], batch.indices, PADDING_INDEX),
    )


def compact(batch: Batch, keep: ArrayLike) -> tuple[Batch, jax.Array]:
    """Move the kept rows first, in order, and make the rest padding, at the batch's shape.

    Args:
        batch: The batch; ``states`` is a mapping.
        keep: bool ``(B,)``, the rows to keep; a padding row is never kept.

    Returns:
        The compacted batch and the number of kept rows, a traced int32 scalar.
    """
    keep = jnp.asarray(keep, bool) & ~_is_padding(batch.indices)
    count = jnp.sum(keep, dtype=jnp.int32)
    order = jnp.nonzero(keep, size=batch.batch_size, fill_value=0)[0]
    taken = _map_rows(batch, lambda x: jnp.take(x, order, axis=0))
    tail = jnp.arange(batch.batch_size) >= count
    weights = _weights(taken)
    return (
        taken.replace(
            states={**taken.states, WEIGHT: jnp.where(tail, jnp.zeros_like(weights), weights)},
            indices=jnp.where(tail[:, None], PADDING_INDEX, taken.indices),
        ),
        count,
    )


def shardings(batch: Batch, row: Sharding, replicated: Sharding) -> Batch:
    """The sharding of every leaf: per-record fields on ``row``, batch-level on ``replicated``.

    Pass the result to ``substrax.spmd.place_batch_on_shards``. One sharding over the batch axis
    for the whole batch cannot place a batch-level leaf, which has no batch axis.

    Args:
        batch: The batch to place.
        row: The sharding of every leaf with a record axis, which it splits.
        replicated: The sharding of the batch-level state, identical on every host.

    Returns:
        A batch of shardings with ``batch``'s structure.
    """
    per_record: Batch = jax.tree.map(lambda _: row, batch.replace(batch_state=None))
    return per_record.replace(batch_state=jax.tree.map(lambda _: replicated, batch.batch_state))


__all__ = [
    "compact",
    "concatenate",
    "element",
    "from_arrays",
    "from_stacked",
    "mask",
    "record_count",
    "shardings",
    "slice_rows",
    "split",
    "stack",
    "take",
]
