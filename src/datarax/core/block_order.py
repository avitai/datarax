"""The block order: records read a few contiguous blocks at a time, every record once per epoch.

The global order (:func:`~datarax.core.index_shuffle.shuffle_positions`) sends consecutive
positions to unrelated records, so a source read from storage larger than the page cache reads a
device page per record. A block order keeps each batch's records inside a few runs of consecutive
records, which a memory map or a file reader serves with sequential reads. This one is CorgiPile's
buffer form (Xu, Zhang, Luo et al., "Stochastic Gradient Descent without Full Data Shuffle",
arXiv 2206.05830, Algorithm 1): the records split into blocks of :attr:`BlockOrder.block_size`
consecutive records; the full blocks are permuted; the permuted blocks are grouped
:attr:`BlockOrder.blocks_per_buffer` to a buffer, and each buffer's positions are permuted. One
block per buffer is the plain block order.

Like the global order it is computed per position from the keyed cipher, so no order is stored, a
batch costs O(batch), and the same key and position always name the same record. A position ``p``
of an order over ``L`` records with blocks of ``b`` and ``n`` blocks to a buffer:

1. its buffer ``q = p // (n b)`` and its offset ``o = p mod (n b)`` there;
2. ``o' = cipher(o)`` over the buffer's length, keyed by ``fold_in(fold_in(key, 1), q)`` (every
   buffer holds ``n b`` positions but the last, which holds the rest);
3. its slot ``s = q n + o' // b`` among the permuted blocks and its record ``r = o' mod b`` there;
4. the block in slot ``s``: ``cipher(s)`` over the ``L // b`` full blocks, keyed by
   ``fold_in(key, 0)``, or the short final block itself when ``s`` is past them;
5. the record ``block b + r``.

Each step is a bijection, so every record is named once. The short final block, when ``b`` does
not divide ``L``, is never permuted with the full blocks: it stays in the last slot and so in the
last buffer, shuffled with that buffer's other records. Its at most ``b - 1`` records therefore
always fall in an epoch's last buffer, as the last block of MosaicML Streaming's block shuffle
stays last (``streaming/base/shuffle/py1br.py``). Positions and records are 64-bit, as two uint32
words (:mod:`datarax.core.index_words`), exact for every length up to ``2**63``.
"""

from __future__ import annotations

import dataclasses
import operator

import jax
import jax.numpy as jnp
import numpy as np

from datarax.core import index_words
from datarax.core.index_shuffle import shuffle_positions


_BUFFER_STREAM = 1
"""``fold_in`` datum of the buffers' keys: each buffer's positions are permuted under its own."""

_BLOCK_STREAM = 0
"""``fold_in`` datum of the key permuting the full blocks."""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class BlockOrder:
    """An order reading blocks of consecutive records, a few blocks to a buffer.

    Static configuration: hashable, so a compiled program is keyed by its value, and never state
    of a module. A buffer's records are read while its blocks are resident, so a read from
    storage holds about ``block_size * blocks_per_buffer`` records' bytes in the page cache.

    Attributes:
        block_size: Consecutive records in a block, at least 1.
        blocks_per_buffer: Blocks whose records are shuffled together, at least 1; 1 is the
            plain block order.
    """

    block_size: int
    blocks_per_buffer: int

    def __post_init__(self) -> None:
        """Refuse an order no record can be read under.

        Raises:
            TypeError: If a field is not an integer (``bool`` included).
            ValueError: If a field is below 1, or a buffer holds ``2**32`` records or more.
        """
        for name in ("block_size", "blocks_per_buffer"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | np.integer):
                raise TypeError(f"{name} must be an integer, got {value!r}")
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}")
            object.__setattr__(self, name, operator.index(value))
        if self.buffer_size >> index_words.WORD_BITS:
            raise ValueError(
                "block_size * blocks_per_buffer, the records of a buffer, must be below 2**32; "
                f"got {self.block_size} * {self.blocks_per_buffer} = {self.buffer_size}"
            )

    @property
    def buffer_size(self) -> int:
        """Records in a full buffer."""
        return self.block_size * self.blocks_per_buffer


def buffer_bytes(order: BlockOrder, length: int, record_bytes: int) -> int:
    """The page cache a run read from storage in ``order`` keeps resident: one buffer's records.

    A buffer's records are read while its blocks are resident, so a run holds about
    ``min(block_size * blocks_per_buffer, length)`` records' stored bytes: a buffer, or every
    record of a source smaller than one.

    Args:
        order: The block order.
        length: Records the source holds, at least 1.
        record_bytes: Stored bytes per record, at least 0.

    Returns:
        The bytes.

    Raises:
        ValueError: If ``length`` is below 1 or ``record_bytes`` is negative.
    """
    if length < 1:
        raise ValueError(f"length must be at least 1, got {length}")
    if record_bytes < 0:
        raise ValueError(f"record_bytes must be at least 0, got {record_bytes}")
    return min(order.buffer_size, length) * record_bytes


def order_positions(
    positions: jax.Array, length: int, key: jax.Array | None, block: BlockOrder | None
) -> jax.Array:
    """The record at each position of an epoch's order: sequential, global or by blocks.

    The one place an order is applied. Without a key the order is sequential; with a key and no
    block order it is the global order, :func:`~datarax.core.index_shuffle.shuffle_positions`;
    with both it is ``block``'s. JAX-traceable and composable with ``jit``, ``vmap`` and
    ``scan``: ``positions`` and ``key`` may be traced, ``length`` and ``block`` are static.

    Args:
        positions: uint32 ``(..., 2)`` positions in ``[0, length)``, each a 64-bit integer as
            its words ``(hi, lo)``.
        length: Records the order covers, in ``[1, 2**63]``.
        key: The epoch's key selecting the order, or ``None`` for the sequential order.
        block: The block order, or ``None`` for the global order.

    Returns:
        uint32 ``(..., 2)`` record indices ``(hi, lo)``, shaped like ``positions``.

    Raises:
        ValueError: If a block order is given without a key, ``length`` is out of range, or
            ``positions`` is not uint32 words.
    """
    index_words.check_length(length)
    positions = index_words.checked_words(positions)
    if key is None:
        if block is not None:
            raise ValueError(
                "a block order permutes blocks under a key, and none was given: pass the "
                "epoch's key, or no block order for the sequential order"
            )
        return positions
    if block is None:
        return shuffle_positions(positions, length, key)
    return _block_positions(positions, length, key, block)


def _block_positions(
    positions: jax.Array, length: int, key: jax.Array, block: BlockOrder
) -> jax.Array:
    """The records of ``block``'s order at ``positions`` (module docstring, steps 1-5)."""
    size, buffer = block.block_size, block.buffer_size
    flat = positions.reshape(-1, 2)
    (high, low), offset = index_words.divmod_word((flat[:, 0], flat[:, 1]), buffer)
    shuffled = _shuffled_offsets((high, low), offset, length, key, buffer)
    slot = index_words.add(
        index_words.multiply_word((high, low), np.uint32(block.blocks_per_buffer)),
        (jnp.zeros_like(shuffled), shuffled // np.uint32(size)),
    )
    owner = _block_in_slot(slot, length, key, size)
    record = index_words.add(
        index_words.multiply_word(owner, np.uint32(size)),
        (jnp.zeros_like(shuffled), shuffled % np.uint32(size)),
    )
    return jnp.stack(record, axis=-1).reshape(positions.shape)


def _shuffled_offsets(
    buffer_index: tuple[jax.Array, jax.Array],
    offset: jax.Array,
    length: int,
    key: jax.Array,
    buffer: int,
) -> jax.Array:
    """Each offset permuted within its buffer, under the buffer's own key (step 2).

    Every buffer but the last holds ``buffer`` positions and the last the remaining ``tail``;
    both permutations are computed and each row takes its buffer's. An offset outside the
    last buffer is replaced by 0 before the last buffer's permutation, so the cycle walk only
    ever starts inside its range.
    """
    buffers = -(-length // buffer)
    tail = length - (buffers - 1) * buffer
    keys = _row_keys(jax.random.fold_in(key, _BUFFER_STREAM), buffer_index)
    full = _permute_words(offset, buffer, keys)
    if tail == buffer:
        return full
    last_high, last_low = index_words.split_constant(buffers - 1)
    is_last = (buffer_index[0] == last_high) & (buffer_index[1] == last_low)
    partial = _permute_words(jnp.where(is_last, offset, np.uint32(0)), tail, keys)
    return jnp.where(is_last, partial, full)


def _block_in_slot(
    slot: tuple[jax.Array, jax.Array], length: int, key: jax.Array, size: int
) -> tuple[jax.Array, jax.Array]:
    """The block in each slot: a permutation of the full blocks, the short block last (step 4)."""
    full_blocks = length // size
    if full_blocks == 0:
        return slot
    if length % size == 0:
        inside = jnp.stack(slot, axis=-1)
        permuted = shuffle_positions(inside, full_blocks, jax.random.fold_in(key, _BLOCK_STREAM))
        return permuted[:, 0], permuted[:, 1]
    tail_high, tail_low = index_words.split_constant(full_blocks)
    is_tail = (slot[0] == tail_high) & (slot[1] == tail_low)
    inside = jnp.stack(
        [jnp.where(is_tail, np.uint32(0), slot[0]), jnp.where(is_tail, np.uint32(0), slot[1])],
        axis=-1,
    )
    permuted = shuffle_positions(inside, full_blocks, jax.random.fold_in(key, _BLOCK_STREAM))
    return (
        jnp.where(is_tail, tail_high, permuted[:, 0]),
        jnp.where(is_tail, tail_low, permuted[:, 1]),
    )


def _row_keys(key: jax.Array, index: tuple[jax.Array, jax.Array]) -> jax.Array:
    """``fold_in(fold_in(key, hi), lo)`` for each row's 64-bit ``index``, the low word last."""
    return jax.vmap(lambda high, low: jax.random.fold_in(jax.random.fold_in(key, high), low))(
        index[0], index[1]
    )


def _permute_words(offset: jax.Array, length: int, keys: jax.Array) -> jax.Array:
    """Each row's ``offset`` (below ``2**32``) through its own key's order of ``[0, length)``."""
    words = jnp.stack([jnp.zeros_like(offset), offset], axis=-1)
    permuted = jax.vmap(lambda word, row_key: shuffle_positions(word, length, row_key))(words, keys)
    return permuted[:, 1]


__all__ = ["BlockOrder", "buffer_bytes", "order_positions"]
