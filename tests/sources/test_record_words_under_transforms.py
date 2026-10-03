"""Naming and gathering records by uint32 ``(hi, lo)`` words under the transforms they meet.

A compiled step names a batch's records with ``record_indices_at`` (a traced start and key) and
gathers them with ``get_records`` (traced words, addressed by the low word). Each source is
checked under ``jit`` with a compile count, under ``vmap`` over a batch axis of index words,
under ``lax.scan`` over batches, under ``grad`` (the gathered values carry the gradient back to
the source's rows; the integer words carry none), and with no read back to the host. The wrapped
and partitioned order, ``resolve_wrapped_indices``, is checked with a traced start for every
worker layout and for lengths past ``2**31`` and ``2**32``.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from substrax.testing.compiles import expect_compiles

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule
from datarax.core.index_words import from_words, to_words
from datarax.sources.eager_source import EagerSource
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from datarax.sources.mixed_source import MixDataSourcesConfig, MixDataSourcesNode
from datarax.sources.source_ops import resolve_wrapped_indices
from tests.test_common.step_jaxpr import host_callbacks


_SIZE = 6


class _Eager(EagerSource):
    """An eager source over given columns, stored on the host."""

    def __init__(self, values: jax.Array) -> None:
        super().__init__(StructuralConfig())
        self._store({"x": values})


def _values(length: int, offset: float = 0.0) -> jax.Array:
    """Distinct float rows ``(length, 3)``, so a wrong row is a wrong value."""
    return jnp.arange(length * 3, dtype=jnp.float32).reshape(length, 3) + offset


def _memory(length: int = 23, *, workers: int = 1, shard: int | None = None) -> MemorySource:
    return MemorySource(
        MemorySourceConfig(num_workers=workers, shard_id=shard), {"x": _values(length)}
    )


def _mixed() -> MixDataSourcesNode:
    return MixDataSourcesNode(
        MixDataSourcesConfig(weights=(0.5, 0.5)),
        [_memory(9), MemorySource(MemorySourceConfig(), {"x": _values(14, 1000.0)})],
    )


_SOURCES: dict[str, Callable[[], DataSourceModule]] = {
    "memory": _memory,
    "memory-worker": lambda: _memory(workers=3, shard=1),
    "eager": lambda: _Eager(_values(23)),
    "mixed": _mixed,
}


def test_the_callback_check_finds_a_host_callback_in_a_loop() -> None:
    def calls_back(x: jax.Array) -> jax.Array:
        def body(carry: jax.Array, _: None) -> tuple[jax.Array, None]:
            back = jax.pure_callback(np.sin, jax.ShapeDtypeStruct((), jnp.float32), carry)
            return back, None

        return jax.lax.scan(body, x, None, length=2)[0]

    assert host_callbacks(jax.make_jaxpr(calls_back)(jnp.float32(1.0))) == ["pure_callback"]


def _fetch(source: DataSourceModule) -> Callable[[jax.Array, jax.Array], dict[str, jax.Array]]:
    """The compiled step's two calls: name the records from a traced start, then gather them."""

    def fetch(start: jax.Array, key: jax.Array) -> dict[str, jax.Array]:
        return source.get_records(source.record_indices_at(start, _SIZE, key))

    return fetch


@pytest.mark.parametrize("name", sorted(_SOURCES))
class TestEverySource:
    def test_one_compile_serves_every_start_and_key_as_eager_calls_do(self, name: str) -> None:
        source = _SOURCES[name]()
        calls = [(start, jax.random.key(seed)) for start, seed in ((4, 1), (17, 2), (40, 3))]
        expected = [
            source.get_records(source.record_indices_at(start, _SIZE, key))["x"]
            for start, key in calls
        ]
        fetch = jax.jit(_fetch(source))
        first = (jnp.int32(0), jax.random.key(0))
        traced = [(jnp.int32(start), key) for start, key in calls]
        with expect_compiles(1):
            jax.block_until_ready(fetch(*first))
        with expect_compiles(0):
            served = [fetch(start, key)["x"] for start, key in traced]
        for got, want in zip(served, expected, strict=True):
            np.testing.assert_array_equal(got, want)

    def test_a_compiled_fetch_reads_nothing_back_to_the_host(self, name: str) -> None:
        """No host callback in the traced fetch, and off the CPU no device-to-host transfer.

        The CPU backend's arrays live in host memory, so the transfer guard never fires there;
        the program itself is checked on every backend.
        """
        source = _SOURCES[name]()
        start, key = jnp.int32(5), jax.random.key(1)
        assert host_callbacks(jax.make_jaxpr(_fetch(source))(start, key)) == []
        fetch = jax.jit(_fetch(source))
        jax.block_until_ready(fetch(start, key))
        if jax.default_backend() == "cpu":
            return
        with jax.transfer_guard_device_to_host("disallow"):
            served = jax.block_until_ready(fetch(start, key)["x"])
            with pytest.raises(RuntimeError, match="[Dd]isallowed"):  # the guard is live
                np.asarray(served)

    def test_under_vmap_each_row_of_words_gathers_its_records(self, name: str) -> None:
        source = _SOURCES[name]()
        keys = jax.random.split(jax.random.key(3), 4)
        words = jnp.stack([source.record_indices_at(7, _SIZE, key) for key in keys])
        assert words.shape == (4, _SIZE, 2)
        assert words.dtype == jnp.uint32
        batched = jax.jit(jax.vmap(lambda row: source.get_records(row)["x"]))(words)
        for row, expected in zip(words, batched, strict=True):
            np.testing.assert_array_equal(expected, source.get_records(row)["x"])

    def test_under_scan_each_step_serves_its_batch(self, name: str) -> None:
        source = _SOURCES[name]()
        key = jax.random.key(4)
        starts = jnp.arange(0, 5 * _SIZE, _SIZE, dtype=jnp.int32)

        @jax.jit
        def scanned(starts: jax.Array) -> jax.Array:
            def body(carry: None, start: jax.Array) -> tuple[None, jax.Array]:
                return carry, _fetch(source)(start, key)["x"]

            return jax.lax.scan(body, None, starts)[1]

        shifted = starts + 1
        with expect_compiles(1):
            served = jax.block_until_ready(scanned(starts))
        with expect_compiles(0):
            jax.block_until_ready(scanned(shifted))
        for step, start in enumerate(np.asarray(starts)):
            expected = source.get_records(source.record_indices_at(int(start), _SIZE, key))["x"]
            np.testing.assert_array_equal(served[step], expected)

    def test_the_gradient_reaches_each_gathered_row_of_the_source(self, name: str) -> None:
        """Under a jitted ``grad`` the cotangent of each served row lands on that source row.

        The words are computed inside the differentiated function from a traced start and key,
        as a train step computes them; a row served twice receives both cotangents.
        """
        source = _SOURCES[name]()
        graphdef, variables, arrays = nnx.split(source, nnx.Variable, ...)
        weights = jax.random.normal(jax.random.key(5), (_SIZE, 3))
        key = jax.random.key(6)

        def loss(arrays: nnx.State, start: jax.Array) -> jax.Array:
            module = nnx.merge(graphdef, variables, arrays)
            return jnp.sum(_fetch(module)(start, key)["x"] * weights)

        gradient = jax.jit(jax.grad(loss))(arrays, jnp.int32(3))
        named = from_words(source.record_indices_at(3, _SIZE, key)).astype(np.int64)
        leaves = [np.asarray(leaf) for leaf in jax.tree.leaves(gradient)]
        assert all(np.isfinite(leaf).all() for leaf in leaves)
        total = sum(float(leaf.sum()) for leaf in leaves)
        assert total == pytest.approx(float(weights.sum()), rel=1e-5)
        if name in {"memory", "memory-worker", "eager"}:
            (rows,) = leaves
            expected = np.zeros_like(rows)
            np.add.at(expected, named, np.asarray(weights))
            np.testing.assert_allclose(rows, expected, rtol=1e-6)

    def test_the_integer_words_carry_no_gradient(self, name: str) -> None:
        source = _SOURCES[name]()
        words = source.record_indices_at(2, _SIZE, jax.random.key(7))

        def loss(words: jax.Array) -> jax.Array:
            return jnp.sum(source.get_records(words)["x"])

        gradient = jax.grad(loss, allow_int=True)(words)
        assert gradient.dtype == jax.dtypes.float0
        assert gradient.shape == words.shape


def _reference(start: int, size: int, length: int, workers: int, shard: int) -> np.ndarray:
    """The worker's wrapped positions as global positions, in Python integers."""
    worker_length = -(-(length - shard) // workers)
    return np.asarray(
        [shard + ((start + offset) % worker_length) * workers for offset in range(size)],
        dtype=np.uint64,
    )


@pytest.mark.parametrize("length", [10, 23, (1 << 31) + 1, (1 << 32) + 1, 1 << 40])
@pytest.mark.parametrize(("workers", "shard"), [(1, 0), (3, 2), (8, 5)])
@pytest.mark.parametrize("shuffle", [False, True], ids=["ordered", "shuffled"])
def test_a_traced_start_names_what_an_integer_start_names(
    length: int, workers: int, shard: int, shuffle: bool
) -> None:
    """One compile per layout serves every traced start, each equal to the host-integer form."""
    key = jax.random.key(8) if shuffle else None

    @jax.jit
    def names(start: jax.Array) -> jax.Array:
        return resolve_wrapped_indices(start, 8, length, key, num_workers=workers, shard_id=shard)

    starts = (0, 3, 9, (1 << 31) - 5)
    expected = [
        resolve_wrapped_indices(start, 8, length, key, num_workers=workers, shard_id=shard)
        for start in starts
    ]
    arguments = [jnp.int32(start) for start in starts]
    with expect_compiles(1):
        jax.block_until_ready(names(arguments[0]))
    with expect_compiles(0):
        traced = [names(argument) for argument in arguments]
    for start, got, want in zip(starts, traced, expected, strict=True):
        assert got.dtype == jnp.uint32
        np.testing.assert_array_equal(got, want)
        if not shuffle:
            np.testing.assert_array_equal(
                from_words(got), _reference(start, 8, length, workers, shard)
            )


def test_under_vmap_over_starts_each_start_names_its_positions() -> None:
    """The pipeline names a crossing batch with ``record_indices_at`` vmapped over starts."""
    length, starts = (1 << 32) + 1, jnp.asarray([0, 7, (1 << 31) - 1], jnp.int32)
    batched = jax.jit(jax.vmap(lambda start: resolve_wrapped_indices(start, 4, length, None)))(
        starts
    )
    for start, words in zip(np.asarray(starts), batched, strict=True):
        np.testing.assert_array_equal(
            from_words(words), np.arange(int(start), int(start) + 4, dtype=np.uint64)
        )
    np.testing.assert_array_equal(to_words(np.asarray(from_words(batched))), batched)
