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
   2. ``test_mixed_rejects_incompatible_element_specs_naming_the_field`` —
      the rejection names the differing field and both of its shapes.

B. Sampling semantics:

   3. ``test_mixed_get_batch_at_is_deterministic_for_fixed_key``
   4. ``test_mixed_get_batch_at_differs_across_keys``
   5. ``test_mixed_get_batch_at_returns_size_records``
   6. ``test_mixed_get_batch_at_respects_weights_in_distribution`` —
      over many positions, source-A records appear roughly
      ``weight_A / sum(weights)`` of the time.

C. JIT compatibility:

   7. ``test_mixed_get_batch_at_traces_under_jit`` — calling under
      ``jax.jit`` does not raise; output shape is correct.
   8. ``test_mixed_get_batch_at_accepts_traced_start``.
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


def test_mixed_rejects_incompatible_element_specs_naming_the_field() -> None:
    """The rejection names the differing field and both of its shapes."""
    src_a = _source([0.0, 1.0])
    src_b = MemorySource(
        MemorySourceConfig(shuffle=False),
        {"x": jnp.zeros((2, 3), dtype=jnp.float32)},
    )

    with pytest.raises(ValueError, match=r"\['x'\].*\(3,\).*\(\)"):
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


def test_mixed_record_indices_name_the_source_and_record_served() -> None:
    """A mixed record's id is its source's offset plus its index within that source."""
    src_a, src_b = _disjoint_pair()  # A holds 0..3, B holds 100..103
    mix = MixDataSourcesNode(
        MixDataSourcesConfig(num_sources=2, weights=(0.5, 0.5)),
        [src_a, src_b],
    )
    key = jax.random.key(3)

    ids = np.asarray(mix.record_indices_at(start=0, size=32, key=key))
    values = np.asarray(mix.get_batch_at(start=0, size=32, key=key)["x"])

    assert set(ids.tolist()) <= set(range(8))
    expected = np.where(ids < 4, ids.astype(np.float32), 100.0 + (ids - 4).astype(np.float32))
    np.testing.assert_array_equal(values, expected)


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


def test_mixed_source_repr_lists_children_and_weights() -> None:
    """S2: repr enumerates child-source reprs and mixing weights for checkpoint validation."""
    src_a, src_b = _disjoint_pair()
    mix = MixDataSourcesNode(
        MixDataSourcesConfig(num_sources=2, weights=(0.25, 0.75)),
        [src_a, src_b],
    )

    r = repr(mix)
    assert "MixDataSourcesNode" in r
    assert "MemorySource" in r  # child reprs are embedded
    assert "0.25" in r and "0.75" in r
    assert "length=8" in r


# ---------- D. Record naming ----------


def test_mixed_record_indices_name_the_records_served_even_with_shuffled_children() -> None:
    """A mixed record's index is its source's offset plus the child's RECORD index.

    With a shuffled child, the child's record at a position is not the position, so the index
    must name the record, or per-record randomness is keyed on the wrong record.
    """
    values_a = np.arange(8, dtype=np.float32)
    values_b = 100.0 + np.arange(8, dtype=np.float32)
    children = [
        MemorySource(MemorySourceConfig(shuffle=True), {"x": jnp.asarray(v)}, rngs=nnx.Rngs(s))
        for s, v in ((1, values_a), (2, values_b))
    ]
    mix = MixDataSourcesNode(MixDataSourcesConfig(num_sources=2, weights=(0.5, 0.5)), children)
    key = jax.random.key(5)

    ids = np.asarray(mix.record_indices_at(start=0, size=64, key=key))
    served = np.asarray(mix.get_batch_at(start=0, size=64, key=key)["x"])

    np.testing.assert_array_equal(served, np.concatenate([values_a, values_b])[ids])
    np.testing.assert_array_equal(np.asarray(mix.get_records(jnp.asarray(ids))["x"]), served)
