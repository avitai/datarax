"""The block order: records read a few contiguous blocks at a time, every record once per epoch.

A buffer-(b, n) order (CorgiPile, Xu et al., arXiv 2206.05830, Algorithm 1) splits the records
into blocks of ``b`` consecutive records, permutes the full blocks, groups them ``n`` to a buffer
and permutes each buffer's positions, so a batch reads from the few blocks its buffer holds. It is
computed per position from the keyed cipher, like the global order: no order is stored, a batch
costs O(batch), and the same key, epoch and position always name the same record. The short final
block, when ``b`` does not divide the length, stays last and is shuffled inside the last buffer.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from substrax.testing.compiles import expect_compiles

from datarax.core.block_order import BlockOrder, buffer_bytes, order_positions
from datarax.core.index_shuffle import shuffle_positions
from datarax.core.index_words import from_words, HostIntegers, MAX_RECORDS, to_words


# (block size, blocks per buffer): one-record blocks, plain blocks (n = 1), a buffer that does not
# divide the lengths below, and the research's buffer 256 x 64.
_ORDERS = ((1, 1), (1, 4), (7, 1), (7, 3), (256, 64))


def _lengths(block_size: int, blocks: int) -> list[int]:
    """Lengths at the order's boundaries: a block, a buffer, one past each, and a prime."""
    buffer = block_size * blocks
    edges = {1, block_size - 1, block_size, block_size + 1, buffer, buffer + 1, 1009}
    return sorted(length for length in edges if length >= 1)


_CASES = [
    (block_size, blocks, length)
    for block_size, blocks in _ORDERS
    for length in _lengths(block_size, blocks)
]


def _words(values: HostIntegers) -> jax.Array:
    return jnp.asarray(to_words(values))


_NAMED = jax.jit(order_positions, static_argnames=("length", "block"))
"""The order as one compiled program per length and order; eager dispatch walks op by op."""


def _named(positions: jax.Array, length: int, key: jax.Array, order: BlockOrder) -> np.ndarray:
    """The records at ``positions`` of the order, as host integers."""
    return from_words(_NAMED(positions, length=length, key=key, block=order))


def _names(length: int, order: BlockOrder, key: jax.Array) -> np.ndarray:
    """The record at every position of the order, as host integers."""
    return _named(_words(np.arange(length)), length, key, order)


def _order(block_size: int, blocks: int) -> BlockOrder:
    return BlockOrder(block_size=block_size, blocks_per_buffer=blocks)


class TestEveryRecordOnce:
    @pytest.mark.parametrize(("block_size", "blocks", "length"), _CASES)
    def test_every_position_names_a_distinct_record(
        self, block_size: int, blocks: int, length: int
    ) -> None:
        names = _names(length, _order(block_size, blocks), jax.random.key(0))
        np.testing.assert_array_equal(np.sort(names), np.arange(length))

    @pytest.mark.parametrize("seed", range(8))
    def test_every_record_once_whatever_the_key(self, seed: int) -> None:
        names = _names(1009, _order(7, 3), jax.random.key(seed))
        np.testing.assert_array_equal(np.sort(names), np.arange(1009))


class TestBlocks:
    @pytest.mark.parametrize(("block_size", "blocks", "length"), _CASES)
    def test_each_buffer_holds_whole_blocks(
        self, block_size: int, blocks: int, length: int
    ) -> None:
        """The positions of a buffer name exactly the records of at most n whole blocks."""
        names = _names(length, _order(block_size, blocks), jax.random.key(1))
        buffer = block_size * blocks
        for start in range(0, length, buffer):
            held = names[start : start + buffer]
            owners = np.unique(held // block_size)
            assert len(owners) <= blocks
            whole = np.concatenate(
                [np.arange(o * block_size, min((o + 1) * block_size, length)) for o in owners]
            )
            np.testing.assert_array_equal(np.sort(held), whole)

    @pytest.mark.parametrize(("block_size", "length"), [(7, 1009), (256, 1009), (7, 21)])
    def test_plain_blocks_keep_each_block_at_consecutive_positions(
        self, block_size: int, length: int
    ) -> None:
        names = _names(length, _order(block_size, 1), jax.random.key(2))
        for block in np.unique(names // block_size):
            positions = np.flatnonzero(names // block_size == block)
            np.testing.assert_array_equal(positions, np.arange(positions[0], positions[-1] + 1))

    @pytest.mark.parametrize(("block_size", "blocks", "length"), [(7, 3, 1009), (256, 64, 70_001)])
    def test_the_short_block_falls_in_the_last_buffer(
        self, block_size: int, blocks: int, length: int
    ) -> None:
        assert length % block_size, "the case must have a short final block"
        names = _names(length, _order(block_size, blocks), jax.random.key(3))
        buffer = block_size * blocks
        last_buffer_start = (length - 1) // buffer * buffer
        tail = np.arange(length // block_size * block_size, length)
        positions = np.flatnonzero(np.isin(names, tail))
        assert positions.min() >= last_buffer_start

    def test_fewer_records_than_a_block_are_one_shuffled_block(self) -> None:
        names = _names(5, _order(7, 3), jax.random.key(4))
        np.testing.assert_array_equal(np.sort(names), np.arange(5))

    @pytest.mark.parametrize("seed", range(8))
    def test_a_buffer_is_not_read_in_stored_order(self, seed: int) -> None:
        """Within a buffer consecutive positions rarely name consecutive records."""
        names = _names(1 << 15, _order(256, 64), jax.random.key(seed))
        adjacent = np.mean(np.diff(names) == 1)
        assert adjacent < 0.01


class TestLengthsPastInt32:
    @pytest.mark.parametrize("length", [MAX_RECORDS, MAX_RECORDS - 12_345, (1 << 32) + 17])
    def test_the_last_buffer_holds_whole_blocks_and_the_tail(self, length: int) -> None:
        order = _order(256, 64)
        buffer = 256 * 64
        start = (length - 1) // buffer * buffer
        positions = _words(np.arange(start, length, dtype=np.uint64))
        names = _named(positions, length, jax.random.key(5), order)
        owners = np.unique(names // np.uint64(256))
        assert len(owners) <= 64
        assert names.max() < length
        assert len(np.unique(names)) == length - start

    @pytest.mark.parametrize("length", [MAX_RECORDS, (1 << 40) + 3])
    def test_sampled_positions_name_distinct_records_in_range(self, length: int) -> None:
        rng = np.random.default_rng(6)
        positions = np.unique(
            np.concatenate(
                [
                    rng.integers(0, length, 4096, dtype=np.uint64),
                    np.array([0, length - 1], dtype=np.uint64),
                ]
            )
        )
        names = _named(_words(positions), length, jax.random.key(7), _order(256, 64))
        assert names.max() < length
        assert len(np.unique(names)) == len(positions)


class TestPureFunction:
    def test_the_same_key_and_position_name_the_same_record_however_batched(self) -> None:
        order, key, length = _order(7, 3), jax.random.key(8), 1009
        whole = _names(length, order, key)
        parts = [
            _named(_words(np.arange(start, min(start + 37, length))), length, key, order)
            for start in range(0, length, 37)
        ]
        np.testing.assert_array_equal(np.concatenate(parts), whole)

    def test_eager_and_compiled_name_the_same_records(self) -> None:
        order, key, positions = _order(7, 3), jax.random.key(8), _words(np.arange(200, 264))
        np.testing.assert_array_equal(
            from_words(order_positions(positions, 1009, key, order)),
            _named(positions, 1009, key, order),
        )

    @pytest.mark.parametrize("seed", range(8))
    def test_another_key_puts_other_blocks_in_the_buffers(self, seed: int) -> None:
        """The blocks each buffer holds change with the key, not only their records' places."""
        order, length = _order(7, 3), 1009

        def held(key: jax.Array) -> list[list[int]]:
            owners = _names(length, order, key) // 7
            return [sorted(set(owners[start : start + 21])) for start in range(0, length, 21)]

        assert held(jax.random.key(seed)) != held(jax.random.key(seed + 100))

    @pytest.mark.parametrize(
        ("length", "first", "second"),
        [(1 << 15, 0, 1), (MAX_RECORDS, 5, 5 + (1 << 32))],
        ids=["neighbours", "buffers-2**32-apart"],
    )
    def test_each_buffer_is_shuffled_under_its_own_key(
        self, length: int, first: int, second: int
    ) -> None:
        """Two buffers' offsets within their blocks follow different permutations.

        The record's place in its block is the shuffled offset modulo the block size, so equal
        sequences would mean one key served both buffers; the second case has buffer indices
        whose low words are equal, which a key folding only the low word would merge.
        """
        order, key = _order(256, 64), jax.random.key(17)
        buffer = order.buffer_size

        def places(index: int) -> np.ndarray:
            start = index * buffer
            positions = _words(np.arange(start, start + buffer, dtype=np.uint64))
            return _named(positions, length, key, order) % np.uint64(256)

        assert not np.array_equal(places(first), places(second))

    def test_it_is_not_the_global_order(self) -> None:
        key = jax.random.key(9)
        global_order = from_words(shuffle_positions(_words(np.arange(1009)), 1009, key))
        assert not np.array_equal(_names(1009, _order(7, 3), key), global_order)


class TestTheGlobalOrderIsUnchanged:
    @pytest.mark.parametrize("length", [1, 1009, (1 << 32) + 1])
    def test_without_a_block_order_the_names_are_the_global_orders(self, length: int) -> None:
        key = jax.random.key(10)
        positions = _words(np.arange(min(length, 512), dtype=np.uint64))
        np.testing.assert_array_equal(
            np.asarray(order_positions(positions, length, key, None)),
            np.asarray(shuffle_positions(positions, length, key)),
        )

    def test_without_a_key_the_order_is_sequential(self) -> None:
        positions = _words(np.arange(100))
        np.testing.assert_array_equal(
            np.asarray(order_positions(positions, 100, None, None)), np.asarray(positions)
        )


class TestRefusals:
    @pytest.mark.parametrize(
        ("fields", "named"),
        [
            ({"block_size": 0, "blocks_per_buffer": 1}, "block_size"),
            ({"block_size": 4, "blocks_per_buffer": 0}, "blocks_per_buffer"),
            ({"block_size": 1 << 16, "blocks_per_buffer": 1 << 16}, "blocks_per_buffer"),
            ({"block_size": True, "blocks_per_buffer": 1}, "block_size"),
            ({"block_size": 4, "blocks_per_buffer": 2.0}, "blocks_per_buffer"),
        ],
    )
    def test_an_order_no_record_can_be_read_under_is_refused(
        self, fields: dict[str, object], named: str
    ) -> None:
        with pytest.raises((ValueError, TypeError), match=named):
            BlockOrder(**fields)  # type: ignore[arg-type]

    def test_a_block_order_without_a_key_is_refused(self) -> None:
        with pytest.raises(ValueError, match="key"):
            order_positions(_words(np.arange(4)), 10, None, _order(2, 2))

    @pytest.mark.parametrize("length", [0, MAX_RECORDS + 1])
    def test_a_length_past_the_records_a_source_holds_is_refused(self, length: int) -> None:
        with pytest.raises(ValueError, match="length"):
            order_positions(_words([0]), length, jax.random.key(0), _order(2, 2))

    @pytest.mark.parametrize(
        "positions", [jnp.arange(4, dtype=jnp.int32), jnp.zeros((4, 3), jnp.uint32)]
    )
    def test_positions_other_than_uint32_words_are_refused(self, positions: jax.Array) -> None:
        with pytest.raises(ValueError, match="words"):
            order_positions(positions, 10, jax.random.key(0), _order(2, 2))

    def test_an_order_is_frozen(self) -> None:
        order = _order(2, 2)
        with pytest.raises(dataclasses.FrozenInstanceError):
            order.block_size = 3  # type: ignore[misc]


class TestBufferBytes:
    """The page cache a block order keeps resident: one buffer's records, or every record."""

    @pytest.mark.parametrize(
        ("length", "expected"), [(1000, 6 * 24), (6, 6 * 24), (4, 4 * 24), (1, 24)]
    )
    def test_a_buffer_or_the_whole_source_if_smaller(self, length: int, expected: int) -> None:
        assert buffer_bytes(_order(3, 2), length, 24) == expected

    @pytest.mark.parametrize(("length", "record_bytes"), [(0, 24), (10, -1)])
    def test_no_records_or_negative_sizes_are_refused(self, length: int, record_bytes: int) -> None:
        with pytest.raises(ValueError, match="length|record_bytes"):
            buffer_bytes(_order(3, 2), length, record_bytes)


class TestTransforms:
    """The order runs inside the host naming's ``jit`` and ``vmap`` over keys, and nnx programs."""

    def test_under_jit_with_traced_positions_and_key(self) -> None:
        order, key = _order(7, 3), jax.random.key(11)
        jitted = jax.jit(lambda positions, key: order_positions(positions, 1009, key, order))
        np.testing.assert_array_equal(
            from_words(jitted(_words(np.arange(100, 164)), key)), _names(1009, order, key)[100:164]
        )

    def test_under_vmap_over_keys_matches_each_key(self) -> None:
        order = _order(7, 3)
        keys = jax.random.split(jax.random.key(12), 3)
        positions = _words(np.arange(1009))
        batched = jax.vmap(lambda key: order_positions(positions, 1009, key, order))(keys)
        for row, key in zip(from_words(batched), keys, strict=True):
            np.testing.assert_array_equal(row, _names(1009, order, key))

    def test_under_scan_each_batch_takes_its_positions_names(self) -> None:
        order, key = _order(7, 3), jax.random.key(13)
        batches = _words(np.arange(4 * 32)).reshape(4, 32, 2)

        @jax.jit
        def scanned(batches: jax.Array) -> jax.Array:
            def body(carry: None, batch: jax.Array) -> tuple[None, jax.Array]:
                return carry, order_positions(batch, 1009, key, order)

            return jax.lax.scan(body, None, batches)[1]

        np.testing.assert_array_equal(
            from_words(scanned(batches)).ravel(), _names(1009, order, key)[:128]
        )

    @pytest.mark.parametrize("length", [1009, (1 << 40) + 3])
    def test_one_compile_per_length_and_order_under_jit(self, length: int) -> None:
        names = jax.jit(order_positions, static_argnames=("length", "block"))
        first, second = _words(np.arange(256)), _words(np.arange(256, 512))
        first_key, second_key = jax.random.split(jax.random.key(14), 2)
        with expect_compiles(1):
            jax.block_until_ready(names(first, length=length, key=first_key, block=_order(7, 3)))
        with expect_compiles(0):
            # An equal order is the same static argument: no recompile for a new instance.
            jax.block_until_ready(names(second, length=length, key=second_key, block=_order(7, 3)))

    def test_one_compile_under_vmap_over_keys(self) -> None:
        order, positions = _order(256, 64), _words(np.arange(512))
        names = jax.jit(jax.vmap(lambda key: order_positions(positions, 70_001, key, order)))
        first, second = (
            jax.random.split(jax.random.key(15), 4),
            jax.random.split(jax.random.key(16), 4),
        )
        with expect_compiles(1):
            jax.block_until_ready(names(first))
        with expect_compiles(0):
            jax.block_until_ready(names(second))


class _Ordered(nnx.Module):
    """A module holding its key as array state and the order as a static attribute."""

    def __init__(self, order: BlockOrder, *, rngs: nnx.Rngs) -> None:
        self.order = order
        self.key = nnx.RngKey(rngs.default())

    def __call__(self, positions: jax.Array) -> jax.Array:
        return order_positions(positions, 1009, self.key[...], self.order)


class TestInFlaxModules:
    """The order is a static, hashable configuration: never a Variable, and one graph per value."""

    def test_the_order_is_no_variable_and_splits_in_tree_mode(self) -> None:
        module = _Ordered(_order(7, 3), rngs=nnx.Rngs(0))
        graphdef, state = nnx.split(module, graph=False)
        leaves = jax.tree.leaves(state)
        assert len(leaves) == 1
        assert jnp.issubdtype(leaves[0].dtype, jax.dtypes.prng_key)
        hash(graphdef)
        merged = nnx.merge(graphdef, state)
        assert merged.order == module.order

    @pytest.mark.parametrize("graph", [True, False])
    def test_under_nnx_jit_names_as_eager_and_compiles_once(self, graph: bool) -> None:
        positions = _words(np.arange(64))
        first = _Ordered(_order(7, 3), rngs=nnx.Rngs(1))
        second = _Ordered(_order(7, 3), rngs=nnx.Rngs(2))
        names = nnx.jit(lambda module, positions: module(positions), graph=graph)
        with expect_compiles(1):
            served = jax.block_until_ready(names(first, positions))
        with expect_compiles(0):
            jax.block_until_ready(names(second, positions))
        np.testing.assert_array_equal(np.asarray(served), np.asarray(first(positions)))
