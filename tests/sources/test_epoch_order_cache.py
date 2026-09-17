"""One permutation per epoch: a shuffled source's order is computed once per key and reused.

``resolve_wrapped_indices`` built ``jax.random.permutation(key, length)`` on every batch, an
O(length) cost paid ceil(length / batch_size) times per epoch. The array-backed sources now
keep the epoch's order in an ``EpochOrderCache`` that recomputes it only when the key
changes, and the served order for a given key is exactly what the full permutation gave.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from datarax.sources.source_ops import EpochOrderCache, resolve_wrapped_indices


_N = 37


def _key(seed: int) -> jax.Array:
    return jax.random.key(seed)


class TestEpochOrderCache:
    def test_first_call_is_the_full_permutation(self) -> None:
        cache = EpochOrderCache(_N)
        order = cache.order_for(_key(3))
        np.testing.assert_array_equal(
            np.asarray(order), np.asarray(jax.random.permutation(_key(3), _N))
        )

    def test_the_same_key_reuses_the_stored_order(self) -> None:
        cache = EpochOrderCache(_N)
        first = np.asarray(cache.order_for(_key(3)))
        # Overwrite the stored order: a second call with the same key must serve the stored
        # array, not recompute it, which is the whole point of the cache.
        cache._order[...] = jnp.arange(_N, dtype=jnp.int32)[::-1]
        second = np.asarray(cache.order_for(_key(3)))
        np.testing.assert_array_equal(second, np.arange(_N)[::-1])
        assert not np.array_equal(first, second)

    def test_a_new_key_recomputes(self) -> None:
        cache = EpochOrderCache(_N)
        cache.order_for(_key(3))
        order = cache.order_for(_key(4))
        np.testing.assert_array_equal(
            np.asarray(order), np.asarray(jax.random.permutation(_key(4), _N))
        )

    def test_works_under_jit_with_state_writes(self) -> None:
        cache = EpochOrderCache(_N)

        @nnx.jit
        def first_index(module: EpochOrderCache, key: jax.Array) -> jax.Array:
            return module.order_for(key)[0]

        expected = int(jax.random.permutation(_key(5), _N)[0])
        assert int(first_index(cache, _key(5))) == expected
        assert int(first_index(cache, _key(5))) == expected
        assert bool(cache._filled[...])


class TestServedOrderIsUnchanged:
    """A shuffled memory source serves the indices the per-batch permutation served."""

    def test_record_indices_match_the_full_permutation(self) -> None:
        source = MemorySource(
            MemorySourceConfig(shuffle=True),
            data={"x": np.arange(_N, dtype=np.float32)},
            rngs=nnx.Rngs(0, shuffle=0),
        )
        key = _key(9)
        for start in (0, 5, 30):
            served = np.asarray(source.record_indices_at(start, 8, key))
            reference = np.asarray(resolve_wrapped_indices(start, 8, _N, True, key))
            np.testing.assert_array_equal(served, reference)

    def test_resolve_wrapped_indices_takes_a_precomputed_order(self) -> None:
        key = _key(9)
        order = jax.random.permutation(key, _N)
        with_order = resolve_wrapped_indices(5, 8, _N, True, key, order=order)
        without = resolve_wrapped_indices(5, 8, _N, True, key)
        np.testing.assert_array_equal(np.asarray(with_order), np.asarray(without))
