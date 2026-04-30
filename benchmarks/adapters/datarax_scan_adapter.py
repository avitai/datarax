"""Datarax scan-epoch adapter for the benchmark framework.

A second measurement dimension for the same Datarax pipeline:

- ``DataraxAdapter`` (in ``datarax_adapter.py``) measures iterator-style
  execution via ``__iter__`` / ``step()``. Each call pays NNX module
  marshalling cost on the host for ``nnx.split`` / ``nnx.merge`` around
  the JIT'd step body.

- ``DataraxScanAdapter`` (this module) measures whole-epoch execution
  via ``Pipeline.scan(...)``, which compiles an ``@nnx.scan`` body
  once and caches it on the Pipeline instance. Subsequent calls with
  the same ``(step_fn, length)`` signature reuse the cached graph; the
  per-batch NNX marshalling cost is paid at compile time, not per
  call.

The reducing step function below collapses the per-batch output to a
single scalar so the scan does not allocate the full ``(N, B, *)``
stacked output buffer in HBM. This isolates iteration cost from the
cost of materializing N copies of per-batch outputs.
"""

from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp

from benchmarks.adapters import register
from benchmarks.adapters.base import IterationResult
from benchmarks.adapters.datarax_adapter import DataraxAdapter
from datarax.performance.synchronization import block_until_ready_tree


def _reducing_step(batch: Any) -> jax.Array:
    """Step function that touches every leaf and returns one scalar.

    The ``.sum()`` ensures XLA cannot elide the upstream work, and
    the cast to float32 keeps the output dtype consistent regardless
    of input (some scenarios use int32 tokens). Module-scope identity
    keeps ``Pipeline.scan``'s cache key stable across warmup and
    timed calls.
    """
    leaves = jax.tree.leaves(batch)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    total = jnp.asarray(0.0, dtype=jnp.float32)
    for leaf in leaves:
        total = total + jnp.asarray(leaf, dtype=jnp.float32).sum()
    return total


@register
class DataraxScanAdapter(DataraxAdapter):
    """Datarax in whole-epoch ``nnx.scan`` mode.

    Reuses ``DataraxAdapter.setup`` to build the same Pipeline; only
    the iteration / timing path differs.
    """

    _pipeline: Any  # populated by inherited DataraxAdapter.setup
    _batch_byte_estimate: int | None = None

    @property
    def name(self) -> str:
        """Return the adapter display name, distinct from the iterator-mode adapter."""
        return "Datarax-scan"

    def _per_batch_byte_estimate(self) -> int:
        """Estimate bytes per batch by inspecting one fresh batch.

        Used because the reducing step_fn collapses output to a scalar,
        so we cannot read total_bytes from the scan result the way iter
        mode does. The estimate is taken once (during ``warmup``) and
        cached.
        """
        if self._batch_byte_estimate is not None:
            return self._batch_byte_estimate
        # Run one step explicitly to measure batch size.
        batch = self._pipeline.step()
        block_until_ready_tree(batch)
        leaves = jax.tree.leaves(batch)
        bytes_per_batch = sum(int(arr.nbytes) for arr in leaves)
        # Reset position so warmup proceeds from a clean state.
        self._pipeline._position[...] = jnp.int32(0)
        self._batch_byte_estimate = bytes_per_batch
        return bytes_per_batch

    def warmup(self, num_batches: int = 3) -> None:
        """Trigger compilation of the scan body and prime its cache.

        ``Pipeline.scan`` caches the compiled body keyed on
        ``(step_fn, length)``. Calling it once during warmup with the
        same arguments the timed call will use ensures the timed call
        hits the cache.
        """
        self._per_batch_byte_estimate()
        outputs = self._pipeline.scan(_reducing_step, length=num_batches)
        block_until_ready_tree(outputs)

    def iterate(self, num_batches: int) -> IterationResult:
        """Run the entire epoch via ``Pipeline.scan`` and time the call.

        Per-batch latencies are not directly observable in scan mode
        (the loop runs inside one XLA graph). ``per_batch_times`` is
        therefore reported as the wall clock divided across batches —
        a uniform attribution rather than a measured distribution.
        """
        start = time.perf_counter()
        outputs = self._pipeline.scan(_reducing_step, length=num_batches)
        block_until_ready_tree(outputs)
        wall_clock = time.perf_counter() - start

        config = self._config
        if config is None:
            raise RuntimeError("setup() must be called before iterate()")
        num_elements = num_batches * config.batch_size
        bytes_per_batch = self._per_batch_byte_estimate()
        total_bytes = bytes_per_batch * num_batches

        per_batch = wall_clock / max(num_batches, 1)
        return IterationResult(
            num_batches=num_batches,
            num_elements=num_elements,
            total_bytes=total_bytes,
            wall_clock_sec=wall_clock,
            per_batch_times=[per_batch] * num_batches,
            first_batch_time=per_batch,
            extra_metrics={"scan_mode": 1.0},
        )
