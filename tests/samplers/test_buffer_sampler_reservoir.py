"""Tests for ``BufferSampler`` reservoir-write mode.

Verifies the single-draw acceptance form of reservoir sampling:
``rand_idx = randint(0, max(seen, 1))``, accept if ``rand_idx < capacity``.
Each item that is ever written has equal expected retention probability
``capacity / N`` after ``N`` total writes (unbiased reservoir).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from datarax.samplers.buffer_sampler import BufferSampler, BufferSamplerConfig


def _scalar_spec() -> dict:
    return {"x": jax.ShapeDtypeStruct(shape=(), dtype=jnp.int32)}


def _scalar(value: int) -> dict:
    return {"x": jnp.asarray(value, dtype=jnp.int32)}


def test_reservoir_initial_writes_fill_buffer_in_order() -> None:
    """First ``capacity`` writes go into slots 0..capacity-1 sequentially.

    During warmup (``count < capacity``), reservoir mode writes at
    ``write_index`` deterministically (which advances 0, 1, 2, ..., capacity-1).
    """
    config = BufferSamplerConfig(
        capacity=4, prefill=4, sample_size=1, write_mode="reservoir", seed=0
    )
    sampler = BufferSampler(config, element_spec=_scalar_spec(), rngs=nnx.Rngs(0))

    for v in [10, 20, 30, 40]:
        sampler.next(_scalar(v))

    contents = np.asarray(sampler.buffer["x"]).tolist()
    assert contents == [10, 20, 30, 40]


def test_reservoir_buffer_size_never_exceeds_capacity() -> None:
    """After ``count == capacity``, every additional write either replaces or skips."""
    config = BufferSamplerConfig(
        capacity=4, prefill=4, sample_size=1, write_mode="reservoir", seed=7
    )
    sampler = BufferSampler(config, element_spec=_scalar_spec(), rngs=nnx.Rngs(0))

    for v in range(1, 21):  # 20 writes into capacity=4
        sampler.next(_scalar(v))

    contents = np.asarray(sampler.buffer["x"])
    assert contents.shape == (4,)
    assert int(sampler.count[...]) == 4
    # Every retained value must have been seen during the 20 writes.
    assert set(int(x) for x in contents).issubset(set(range(1, 21)))


def test_reservoir_mean_retention_rate_is_capacity_over_n() -> None:
    """Mean retention rate across trials is ``capacity / N`` (unbiased reservoir).

    With ``capacity=5`` and ``N=50``, expected mean is exactly 0.1. Trials run
    under ``nnx.jit`` so the reservoir's ``lax.cond`` body compiles once
    instead of re-tracing per write.
    """
    capacity = 5
    n_writes = 50
    n_trials = 60

    @nnx.jit
    def jitted_step(sampler: BufferSampler, value: dict) -> tuple:
        return sampler.next(value, mask=jnp.array(True))

    retained_counts = np.zeros(n_writes, dtype=np.int32)
    for trial_seed in range(n_trials):
        config = BufferSamplerConfig(
            capacity=capacity,
            prefill=capacity,
            sample_size=1,
            write_mode="reservoir",
            seed=trial_seed,
        )
        sampler = BufferSampler(config, element_spec=_scalar_spec(), rngs=nnx.Rngs(0))
        for v in range(n_writes):
            jitted_step(sampler, _scalar(v))
        contents = np.asarray(sampler.buffer["x"])
        for v in contents:
            retained_counts[int(v)] += 1

    rates = retained_counts / n_trials
    expected = capacity / n_writes  # 0.1
    mean_rate = float(rates.mean())
    assert abs(mean_rate - expected) < 0.01, (
        f"Mean retention rate {mean_rate:.4f} differs from expected {expected:.4f} — "
        "reservoir sampling is biased."
    )
