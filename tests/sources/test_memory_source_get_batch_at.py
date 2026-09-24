"""Contracts for ``MemorySource.get_batch_at(start, size, key)``.

Two modes:

- **Sequential**: when the source is constructed without shuffle
  (``MemorySourceConfig(shuffle=False)``), ``get_batch_at`` returns the
  contiguous slice ``[start, start + size)``.
- **Shuffled**: when the source has ``shuffle=True``, ``get_batch_at``
  returns a ``size``-element slice of a deterministic permutation
  derived from ``key``. Same ``(start, size, key)`` always returns the
  same output; different ``key`` yields a different permutation.

The shuffled mode contract is what ``Pipeline.scan`` consumes when
training requires per-epoch shuffling.

Test contract index:

A. Sequential (already covered by the spike, included here for
   completeness):

   1. ``test_sequential_returns_contiguous_slice``
   2. ``test_sequential_wraps_at_end_of_source``

B. Shuffled mode:

   3. ``test_shuffled_is_deterministic_for_fixed_key`` — same ``key``,
      same ``(start, size)`` → identical output.
   4. ``test_shuffled_differs_across_keys`` — different ``key`` →
      different permutation (statistically; we accept any-leaf-differs).
   5. ``test_shuffled_covers_all_records_over_one_full_epoch`` — calling
      ``get_batch_at`` for every index in ``[0, length)`` (in chunks of
      size ``size``) yields a permutation of the source records — every
      record appears exactly once.
   6. ``test_shuffled_handles_partial_final_batch`` — when
      ``start + size > length``, the function still returns ``size``
      records (wrap-around).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from datarax.sources.memory_source import MemorySource, MemorySourceConfig


# ---------- A. Sequential mode ----------


def test_sequential_returns_contiguous_slice() -> None:
    src = MemorySource(
        MemorySourceConfig(shuffle=False),
        {"x": jnp.arange(8, dtype=jnp.float32)},
    )

    batch = src.get_batch_at(start=2, size=4, key=jax.random.key(0))
    np.testing.assert_array_equal(np.asarray(batch["x"]), np.array([2.0, 3.0, 4.0, 5.0]))


def test_sequential_wraps_at_end_of_source() -> None:
    src = MemorySource(
        MemorySourceConfig(shuffle=False),
        {"x": jnp.arange(8, dtype=jnp.float32)},
    )

    # start=6, size=4 → indices 6, 7, 0, 1 (wrap)
    batch = src.get_batch_at(start=6, size=4, key=jax.random.key(0))
    np.testing.assert_array_equal(np.asarray(batch["x"]), np.array([6.0, 7.0, 0.0, 1.0]))


# ---------- B. Shuffled mode ----------


def test_shuffled_is_deterministic_for_fixed_key() -> None:
    src = MemorySource(
        MemorySourceConfig(shuffle=True),
        {"x": jnp.arange(16, dtype=jnp.float32)},
        rngs=nnx.Rngs(shuffle=0),
    )
    key = jax.random.key(42)

    a = src.get_batch_at(start=0, size=4, key=key)
    b = src.get_batch_at(start=0, size=4, key=key)

    np.testing.assert_array_equal(np.asarray(a["x"]), np.asarray(b["x"]))


def test_shuffled_differs_across_keys() -> None:
    src = MemorySource(
        MemorySourceConfig(shuffle=True),
        {"x": jnp.arange(16, dtype=jnp.float32)},
        rngs=nnx.Rngs(shuffle=0),
    )

    a = src.get_batch_at(start=0, size=4, key=jax.random.key(0))
    b = src.get_batch_at(start=0, size=4, key=jax.random.key(1))

    assert not np.array_equal(np.asarray(a["x"]), np.asarray(b["x"])), (
        "different keys must produce different shuffled batches"
    )


def test_shuffled_covers_all_records_over_one_full_epoch() -> None:
    """Walking through start=0, batch_size, 2*batch_size, ... yields a permutation."""
    length = 16
    batch_size = 4
    src = MemorySource(
        MemorySourceConfig(shuffle=True),
        {"x": jnp.arange(length, dtype=jnp.float32)},
        rngs=nnx.Rngs(shuffle=0),
    )
    key = jax.random.key(7)

    seen: list[float] = []
    for start in range(0, length, batch_size):
        batch = src.get_batch_at(start=start, size=batch_size, key=key)
        seen.extend(np.asarray(batch["x"]).tolist())

    # Each value 0..length-1 should appear exactly once
    assert sorted(seen) == [float(i) for i in range(length)], (
        f"epoch did not cover every record exactly once: {sorted(seen)}"
    )


def test_shuffled_handles_partial_final_batch() -> None:
    """When start+size exceeds length, output still has `size` records (wrap)."""
    src = MemorySource(
        MemorySourceConfig(shuffle=True),
        {"x": jnp.arange(8, dtype=jnp.float32)},
        rngs=nnx.Rngs(shuffle=0),
    )

    batch = src.get_batch_at(start=6, size=4, key=jax.random.key(3))
    # 4 elements out of a length-8 source even though start=6
    assert batch["x"].shape == (4,)


# ---------- C. Record identity ----------


def test_record_indices_name_the_records_get_batch_at_serves() -> None:
    """``record_indices_at`` returns the stable index of every record in the same batch."""
    src = MemorySource(
        MemorySourceConfig(shuffle=True),
        {"x": jnp.arange(16, dtype=jnp.float32) * 10},
        rngs=nnx.Rngs(shuffle=0),
    )
    key = jax.random.key(5)

    ids = src.record_indices_at(start=3, size=4, key=key)
    batch = src.get_batch_at(start=3, size=4, key=key)

    np.testing.assert_array_equal(np.asarray(batch["x"]), np.asarray(ids) * 10.0)


def test_workers_serve_disjoint_global_records_covering_the_source() -> None:
    """Worker k serves global positions ``[k::num_workers]`` of the (shuffled) order."""
    length = 10
    data = {"x": jnp.arange(length, dtype=jnp.float32)}
    key = jax.random.key(1)
    whole = MemorySource(MemorySourceConfig(shuffle=True), data, rngs=nnx.Rngs(shuffle=0))
    order = np.asarray(whole.record_indices_at(start=0, size=length, key=key))

    served: list[np.ndarray] = []
    for shard_id in range(3):
        worker = MemorySource(
            MemorySourceConfig(shuffle=True, num_workers=3, shard_id=shard_id),
            data,
            rngs=nnx.Rngs(shuffle=0),
        )
        worker_ids = np.asarray(worker.record_indices_at(start=0, size=len(worker), key=key))
        np.testing.assert_array_equal(worker_ids, order[shard_id::3])
        np.testing.assert_array_equal(
            np.asarray(worker.get_batch_at(start=0, size=len(worker), key=key)["x"]), worker_ids
        )
        served.append(worker_ids)

    assert [len(ids) for ids in served] == [4, 3, 3]
    assert sorted(np.concatenate(served).tolist()) == list(range(length))


# ---------- D. Indexed-access capability ----------


def test_supports_indexed_access_is_true() -> None:
    """MemorySource is random-access, so Pipeline must take the indexed path.

    A False here silently reroutes Pipeline.__iter__ through the streaming
    path, which never terminates because MemorySource.get_batch wraps
    around instead of returning an empty batch.
    """
    src = MemorySource(
        MemorySourceConfig(shuffle=False),
        {"x": jnp.arange(8, dtype=jnp.float32)},
    )
    assert src.supports_indexed_access() is True


# ---------- E. List data is a record store, not a batch source ----------


def _record_list() -> list[dict]:
    return [{"x": np.full((2,), i, np.float32), "name": f"r{i}"} for i in range(6)]


def test_list_data_has_no_indexed_access() -> None:
    """A batch is a gather over columns; a list of records has none, so it is not indexed."""
    source = MemorySource(MemorySourceConfig(), _record_list())

    assert source.supports_indexed_access() is False
    with pytest.raises(TypeError, match="dict of arrays"):
        source.get_records(jnp.arange(2, dtype=jnp.int32))


def test_list_data_still_serves_its_records_on_the_host() -> None:
    """Indexing, iteration and the host ``get_batch`` keep the records as they were given."""
    records = _record_list()
    source = MemorySource(MemorySourceConfig(), records)

    assert source[3]["name"] == "r3"
    assert [record["name"] for record in source][:2] == ["r0", "r1"]
    assert [record["name"] for record in source.get_batch(2)] == ["r0", "r1"]
