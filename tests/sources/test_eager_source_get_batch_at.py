"""Contracts for ``EagerSourceBase.get_batch_at(start, size, key)``.

The shared base provides indexed batch access for every eager-loaded
source (``HFEagerSource``, ``TFDSEagerSource``, etc.) via inheritance.
Tests use a minimal ``_FakeEagerSource`` subclass so the contract is
exercised without HuggingFace / TFDS network dependencies.

Test contract index:

A. Sequential mode:

   1. ``test_eager_get_batch_at_returns_contiguous_slice``
   2. ``test_eager_get_batch_at_wraps_at_end_of_source``
   3. ``test_eager_get_batch_at_does_not_advance_internal_index`` — the
      load-bearing stateless contract: ``self.index`` is unchanged
      after the call.

B. Shuffled mode:

   4. ``test_eager_shuffled_get_batch_at_is_deterministic_for_fixed_key``
   5. ``test_eager_shuffled_get_batch_at_differs_across_keys``
   6. ``test_eager_shuffled_get_batch_at_covers_full_epoch`` — every
      record exactly once across one full pass.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax.core.config import StructuralConfig
from datarax.sources._source_base import EagerSourceBase


class _FakeEagerSource(EagerSourceBase):
    """Minimal in-memory source for exercising the EagerSourceBase contract."""

    def __init__(self, data: dict, *, is_random_order: bool = False, seed: int = 0) -> None:
        super().__init__(StructuralConfig())
        self.data = nnx.data(data)
        leaves = jax.tree.leaves(data)
        self.length = int(leaves[0].shape[0]) if leaves else 0
        self.index = nnx.Variable(jnp.int32(0))
        self.epoch = nnx.Variable(jnp.int32(0))
        self._seed = seed
        self._is_random_order = is_random_order
        self.dataset_name = "fake"
        self.split_name = "all"
        self._dataset_info = None
        self._cache = None  # not used by tests but required by reset()


# ---------- A. Sequential mode ----------


def test_eager_get_batch_at_returns_contiguous_slice() -> None:
    src = _FakeEagerSource({"x": jnp.arange(8, dtype=jnp.float32)})
    batch = src.get_batch_at(start=2, size=4, key=jax.random.key(0))

    np.testing.assert_array_equal(np.asarray(batch["x"]), np.array([2.0, 3.0, 4.0, 5.0]))


def test_eager_get_batch_at_wraps_at_end_of_source() -> None:
    src = _FakeEagerSource({"x": jnp.arange(8, dtype=jnp.float32)})
    batch = src.get_batch_at(start=6, size=4, key=jax.random.key(0))

    np.testing.assert_array_equal(np.asarray(batch["x"]), np.array([6.0, 7.0, 0.0, 1.0]))


def test_eager_get_batch_at_does_not_advance_internal_index() -> None:
    """The contract Pipeline.scan relies on: source has no hidden iteration state."""
    src = _FakeEagerSource({"x": jnp.arange(8, dtype=jnp.float32)})
    src.index[...] = jnp.int32(5)  # baseline non-zero state

    src.get_batch_at(start=0, size=4, key=jax.random.key(0))

    assert int(src.index[...]) == 5, "get_batch_at must not advance source.index"


# ---------- B. Shuffled mode ----------


def test_eager_shuffled_get_batch_at_is_deterministic_for_fixed_key() -> None:
    src = _FakeEagerSource({"x": jnp.arange(16, dtype=jnp.float32)}, is_random_order=True)
    key = jax.random.key(42)

    a = src.get_batch_at(start=0, size=4, key=key)
    b = src.get_batch_at(start=0, size=4, key=key)

    np.testing.assert_array_equal(np.asarray(a["x"]), np.asarray(b["x"]))


def test_eager_shuffled_get_batch_at_differs_across_keys() -> None:
    src = _FakeEagerSource({"x": jnp.arange(16, dtype=jnp.float32)}, is_random_order=True)

    a = src.get_batch_at(start=0, size=4, key=jax.random.key(0))
    b = src.get_batch_at(start=0, size=4, key=jax.random.key(1))

    assert not np.array_equal(np.asarray(a["x"]), np.asarray(b["x"]))


def test_eager_shuffled_get_batch_at_covers_full_epoch() -> None:
    length = 16
    batch_size = 4
    src = _FakeEagerSource({"x": jnp.arange(length, dtype=jnp.float32)}, is_random_order=True)
    key = jax.random.key(7)

    seen: list[float] = []
    for start in range(0, length, batch_size):
        batch = src.get_batch_at(start=start, size=batch_size, key=key)
        seen.extend(np.asarray(batch["x"]).tolist())

    assert sorted(seen) == [float(i) for i in range(length)]
