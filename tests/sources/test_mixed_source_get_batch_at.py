"""Contracts for ``MixDataSourcesNode.get_batch_at`` — weighted interleaved mix.

Each output position deterministically chooses a source via weighted
categorical sampling, picks a local index uniformly within that source,
and dispatches to the source's own ``get_batch_at``. The result is a
batch of ``size`` records sampled in proportion to the configured
weights.

Test contract index:

A. Construction validation:

   1. ``test_mixed_rejects_sources_with_incompatible_element_specs`` —
      sources must produce records with identical structure.

B. Sampling semantics:

   2. ``test_mixed_get_batch_at_is_deterministic_for_fixed_key``
   3. ``test_mixed_get_batch_at_differs_across_keys``
   4. ``test_mixed_get_batch_at_returns_size_records``
   5. ``test_mixed_get_batch_at_respects_weights_in_distribution`` —
      over many positions, source-A records appear roughly
      ``weight_A / sum(weights)`` of the time.

C. JIT compatibility:

   6. ``test_mixed_get_batch_at_traces_under_jit`` — calling under
      ``jax.jit`` does not raise; output shape is correct.
   7. ``test_mixed_get_batch_at_accepts_traced_start``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from datarax.sources.mixed_source import MixDataSourcesConfig, MixDataSourcesNode


# ---------- Helpers ----------


def _source(values: list[float], *, key_name: str = "x") -> MemorySource:
    return MemorySource(
        MemorySourceConfig(shuffle=False),
        {key_name: jnp.asarray(values, dtype=jnp.float32)},
    )


def _disjoint_pair() -> tuple[MemorySource, MemorySource]:
    """Two sources whose values do not overlap, so we can identify the source from a record."""
    src_a = _source([0.0, 1.0, 2.0, 3.0])  # values in [0, 4)
    src_b = _source([100.0, 101.0, 102.0, 103.0])  # values in [100, 104)
    return src_a, src_b


# ---------- A. Construction validation ----------


def test_mixed_rejects_sources_with_incompatible_element_specs() -> None:
    """Sources must produce records with the same element_spec to be mix-able."""
    src_a = _source([0.0, 1.0], key_name="x")
    src_b = MemorySource(
        MemorySourceConfig(shuffle=False),
        {"y": jnp.asarray([0.0, 1.0], dtype=jnp.float32)},  # different key
    )

    with pytest.raises(ValueError, match="element_spec"):
        MixDataSourcesNode(
            MixDataSourcesConfig(num_sources=2, weights=(0.5, 0.5)),
            [src_a, src_b],
        )


# ---------- B. Sampling semantics ----------


def test_mixed_get_batch_at_is_deterministic_for_fixed_key() -> None:
    src_a, src_b = _disjoint_pair()
    mix = MixDataSourcesNode(
        MixDataSourcesConfig(num_sources=2, weights=(0.5, 0.5)),
        [src_a, src_b],
    )

    key = jax.random.key(7)
    batch_1 = mix.get_batch_at(start=0, size=8, key=key)
    batch_2 = mix.get_batch_at(start=0, size=8, key=key)

    np.testing.assert_array_equal(np.asarray(batch_1["x"]), np.asarray(batch_2["x"]))


def test_mixed_get_batch_at_differs_across_keys() -> None:
    src_a, src_b = _disjoint_pair()
    mix = MixDataSourcesNode(
        MixDataSourcesConfig(num_sources=2, weights=(0.5, 0.5)),
        [src_a, src_b],
    )

    batch_a = mix.get_batch_at(start=0, size=16, key=jax.random.key(0))
    batch_b = mix.get_batch_at(start=0, size=16, key=jax.random.key(1))

    assert not np.array_equal(np.asarray(batch_a["x"]), np.asarray(batch_b["x"]))


def test_mixed_get_batch_at_returns_size_records() -> None:
    src_a, src_b = _disjoint_pair()
    mix = MixDataSourcesNode(
        MixDataSourcesConfig(num_sources=2, weights=(0.7, 0.3)),
        [src_a, src_b],
    )

    batch = mix.get_batch_at(start=0, size=12, key=jax.random.key(0))
    assert batch["x"].shape == (12,)


def test_mixed_get_batch_at_respects_weights_in_distribution() -> None:
    """Over many positions, source-A frequency ≈ weight_A / sum(weights)."""
    src_a, src_b = _disjoint_pair()  # A has values in [0, 4), B has [100, 104)
    mix = MixDataSourcesNode(
        MixDataSourcesConfig(num_sources=2, weights=(0.8, 0.2)),
        [src_a, src_b],
    )

    batch = mix.get_batch_at(start=0, size=512, key=jax.random.key(0))
    values = np.asarray(batch["x"])

    # Records < 50 came from A; records >= 50 came from B.
    from_a = int(np.sum(values < 50.0))
    from_b = int(np.sum(values >= 50.0))

    fraction_a = from_a / (from_a + from_b)
    # 80/20 weighting; allow ±5% tolerance for sampling noise at n=512.
    assert 0.75 <= fraction_a <= 0.85, (
        f"expected ~80% from source A; observed {fraction_a:.2%} ({from_a}/{from_a + from_b})"
    )


# ---------- C. JIT compatibility ----------


def test_mixed_get_batch_at_traces_under_jit() -> None:
    src_a, src_b = _disjoint_pair()
    mix = MixDataSourcesNode(
        MixDataSourcesConfig(num_sources=2, weights=(0.5, 0.5)),
        [src_a, src_b],
    )

    @nnx.jit
    def fetch(mix: MixDataSourcesNode, start: jax.Array, key: jax.Array) -> jax.Array:
        return mix.get_batch_at(start, 4, key)["x"]

    out = fetch(mix, jnp.int32(0), jax.random.key(0))
    assert out.shape == (4,)


def test_mixed_get_batch_at_accepts_traced_start() -> None:
    src_a, src_b = _disjoint_pair()
    mix = MixDataSourcesNode(
        MixDataSourcesConfig(num_sources=2, weights=(0.5, 0.5)),
        [src_a, src_b],
    )

    out = mix.get_batch_at(start=jnp.int32(2), size=4, key=jax.random.key(0))
    assert out["x"].shape == (4,)
