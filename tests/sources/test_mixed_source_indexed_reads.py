"""``MixDataSourcesNode``'s indexed read: the names it gives, their order, and their transforms.

``record_indices_at(start, size, key)`` names mix positions ``start .. start + size`` (wrapped at
the epoch's length). Grain's selection says which child serves each position and at which of
its positions; the child names that position in its own order, keyed by ``fold_in(key, c)``
when the pipeline shuffles and sequential otherwise. ``get_records`` gathers each named record
from its child. The naming is a pure function of ``(key, start, size)``, so it traces once per
layout and any split of a range names what the whole range names.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from substrax.testing.compiles import expect_compiles

from datarax.core.index_words import from_words, to_words
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from datarax.sources.mixed_source import MixDataSourcesConfig, MixDataSourcesNode
from tests.test_common.mixing import offsets_of
from tests.test_common.step_jaxpr import host_callbacks


_LENGTHS = (9, 14)
_BASES = (0.0, 100.0)


def _mix(weights: tuple[float, ...] = (0.3, 0.7)) -> MixDataSourcesNode:
    children = [
        MemorySource(MemorySourceConfig(), {"x": base + np.arange(n, dtype=np.float32)})
        for n, base in zip(_LENGTHS, _BASES, strict=True)
    ]
    return MixDataSourcesNode(MixDataSourcesConfig(weights=weights), children)


def _owned(mix: MixDataSourcesNode, names: jax.Array) -> tuple[np.ndarray, np.ndarray]:
    """Each named record's child and its index within that child."""
    indices = from_words(names).astype(np.int64)
    offsets = np.asarray(offsets_of(_LENGTHS), np.int64)
    owners = np.searchsorted(offsets, indices, side="right") - 1
    return owners, indices - offsets[owners]


_KEYS = {"ordered": None, "keyed": jax.random.key(7)}


class TestTheNames:
    @pytest.mark.parametrize("order", sorted(_KEYS))
    def test_the_records_named_are_the_records_gathered(self, order: str) -> None:
        mix = _mix()
        names = mix.record_indices_at(3, 12, _KEYS[order])
        owners, local = _owned(mix, names)
        served = np.asarray(mix.get_records(names)["x"])
        expected = np.asarray(_BASES, np.float32)[owners] + local.astype(np.float32)
        np.testing.assert_array_equal(served, expected)

    def test_without_a_key_each_child_serves_its_records_in_order(self) -> None:
        mix = _mix()
        owners, local = _owned(mix, mix.record_indices_at(0, len(mix), None))
        for child in range(len(_LENGTHS)):
            mine = local[owners == child]
            np.testing.assert_array_equal(mine, np.arange(len(mine)))

    def test_with_a_key_each_child_serves_its_own_keyed_order(self) -> None:
        """The mix routes the key and never applies it: child c is keyed fold_in(key, c)."""
        mix = _mix()
        key = jax.random.key(11)
        owners, local = _owned(mix, mix.record_indices_at(0, len(mix), key))
        for child, source in enumerate(mix.sources):
            mine = local[owners == child]
            own = from_words(source.record_indices_at(0, len(mine), jax.random.fold_in(key, child)))
            np.testing.assert_array_equal(mine, own.astype(np.int64))

    @pytest.mark.parametrize("order", sorted(_KEYS))
    def test_an_epoch_names_each_record_at_most_once(self, order: str) -> None:
        names = from_words(_mix().record_indices_at(0, len(_mix()), _KEYS[order]))
        assert len(np.unique(names)) == len(names)

    def test_two_keys_name_different_orders(self) -> None:
        mix = _mix()
        first = mix.record_indices_at(0, len(mix), jax.random.key(0))
        second = mix.record_indices_at(0, len(mix), jax.random.key(1))
        assert not np.array_equal(first, second)

    @pytest.mark.parametrize("order", sorted(_KEYS))
    def test_rows_past_the_epoch_name_records_of_the_mix(self, order: str) -> None:
        """A crossing batch's rows past the end are discarded by the pipeline, yet stay valid."""
        mix = _mix()
        names = from_words(mix.record_indices_at(len(mix) - 3, 9, _KEYS[order]))
        assert (names < sum(_LENGTHS)).all()
        np.asarray(mix.get_records(jnp.asarray(to_words(names)))["x"])

    @pytest.mark.parametrize("order", sorted(_KEYS))
    @pytest.mark.parametrize("cut", [1, 5, 11])
    def test_a_split_range_names_what_the_whole_range_names(self, order: str, cut: int) -> None:
        """A pure function of (key, start, size): how a range is split never changes its names."""
        mix, key = _mix(), _KEYS[order]
        whole = mix.record_indices_at(2, 16, key)
        parts = np.concatenate(
            [mix.record_indices_at(2, cut, key), mix.record_indices_at(2 + cut, 16 - cut, key)]
        )
        np.testing.assert_array_equal(parts, whole)


def _names(order: str) -> Callable[..., jax.Array]:
    mix = _mix()
    if order == "keyed":
        return jax.jit(lambda start, key: mix.record_indices_at(start, 6, key))
    return jax.jit(lambda start: mix.record_indices_at(start, 6, None))


class TestTransforms:
    @pytest.mark.parametrize("order", sorted(_KEYS))
    def test_one_compile_per_layout_serves_every_start(self, order: str) -> None:
        mix, names = _mix(), _names(order)
        starts = (0, 5, 17, 40)
        keys = [jax.random.key(seed) for seed in range(len(starts))]

        def arguments(start: int, key: jax.Array) -> tuple:
            return (jnp.int32(start), key) if order == "keyed" else (jnp.int32(start),)

        calls = [arguments(s, k) for s, k in zip(starts, keys, strict=True)]
        with expect_compiles(1):
            jax.block_until_ready(names(*calls[0]))
        with expect_compiles(0):
            served = [names(*call) for call in calls]
        for got, start, key in zip(served, starts, keys, strict=True):
            want = mix.record_indices_at(start, 6, key if order == "keyed" else None)
            assert got.dtype == jnp.uint32
            np.testing.assert_array_equal(got, want)

    def test_vmap_over_keys_equals_the_loop(self) -> None:
        mix = _mix()
        keys = jax.random.split(jax.random.key(4), 3)
        starts = jnp.asarray([0, 0, 7], jnp.int32)
        batched = jax.jit(jax.vmap(lambda s, k: mix.record_indices_at(s, 6, k)))(starts, keys)
        for row, start, key in zip(batched, starts, keys, strict=True):
            np.testing.assert_array_equal(row, mix.record_indices_at(int(start), 6, key))

    @pytest.mark.parametrize("order", sorted(_KEYS))
    def test_scan_over_starts_equals_the_loop(self, order: str) -> None:
        mix, key = _mix(), _KEYS[order]
        starts = jnp.arange(0, 30, 6, dtype=jnp.int32)

        @jax.jit
        def scanned(starts: jax.Array) -> jax.Array:
            def body(carry: None, start: jax.Array) -> tuple[None, jax.Array]:
                return carry, mix.record_indices_at(start, 6, key)

            return jax.lax.scan(body, None, starts)[1]

        served = scanned(starts)
        for step, start in enumerate(np.asarray(starts)):
            np.testing.assert_array_equal(served[step], mix.record_indices_at(int(start), 6, key))

    def test_the_naming_and_gather_hold_no_host_callback(self) -> None:
        mix = _mix()

        def fetch(start: jax.Array, key: jax.Array) -> dict[str, jax.Array]:
            return mix.get_records(mix.record_indices_at(start, 6, key))

        jaxpr = jax.make_jaxpr(fetch)(jnp.int32(0), jax.random.key(0))
        assert host_callbacks(jaxpr) == []

    def test_a_module_argument_traces_once_under_nnx_jit(self) -> None:
        fetch = nnx.jit(lambda mix, start: mix.get_records(mix.record_indices_at(start, 6))["x"])
        mix, zero, four = _mix(), jnp.int32(0), jnp.int32(4)
        with expect_compiles(1):
            jax.block_until_ready(fetch(mix, zero))
        with expect_compiles(0):
            served = fetch(mix, four)
        np.testing.assert_array_equal(
            served, np.asarray(mix.get_records(mix.record_indices_at(4, 6))["x"])
        )
