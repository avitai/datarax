"""Internal microbenchmark decomposition for benchmark scenarios.

Decomposes each scenario's iteration into component times:
- Source time: data generation / loading from source
- Transform time: operator execution (CPU or GPU)
- Transfer time: host-to-device data transfer
- Overhead: framework control flow, Python, batching

This provides deeper performance insight than aggregate throughput alone.

Protocol ref: Section 8.1 of the performance audit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from benchmarks.adapters.base import PipelineAdapter, ScenarioConfig


@dataclass(frozen=True, slots=True)
class MicrobenchmarkResult:
    """Decomposed timing breakdown for a single scenario run.

    All times are in seconds. Overhead is computed as wall_clock minus the
    sum of the three measured components.

    Attributes:
        scenario_id: Scenario that was profiled.
        variant_name: Variant name.
        num_batches: Number of batches measured.
        wall_clock_sec: Total elapsed time.
        source_sec: Total time spent in data source iteration.
        transform_sec: Total time in transform / materialization.
        transfer_sec: Total time in data transfer (device_put, block_until_ready).
        overhead_sec: Framework overhead (wall_clock - source - transform - transfer).
        per_batch_source: Per-batch source times.
        per_batch_transform: Per-batch transform times.
        per_batch_transfer: Per-batch transfer times.
    """

    scenario_id: str
    variant_name: str
    num_batches: int
    wall_clock_sec: float
    source_sec: float
    transform_sec: float
    transfer_sec: float
    overhead_sec: float
    per_batch_source: list[float] = field(default_factory=list)
    per_batch_transform: list[float] = field(default_factory=list)
    per_batch_transfer: list[float] = field(default_factory=list)

    @property
    def goodput_ratio(self) -> float:
        """Productive time / wall clock time."""
        productive = self.source_sec + self.transform_sec + self.transfer_sec
        if self.wall_clock_sec <= 0:
            return 0.0
        return productive / self.wall_clock_sec

    def summary_dict(self) -> dict[str, Any]:
        """Return a summary dictionary for reporting."""
        return {
            "scenario_id": self.scenario_id,
            "variant": self.variant_name,
            "num_batches": self.num_batches,
            "wall_clock_sec": round(self.wall_clock_sec, 6),
            "source_sec": round(self.source_sec, 6),
            "transform_sec": round(self.transform_sec, 6),
            "transfer_sec": round(self.transfer_sec, 6),
            "overhead_sec": round(self.overhead_sec, 6),
            "goodput_ratio": round(self.goodput_ratio, 4),
            "source_pct": round(self.source_sec / self.wall_clock_sec * 100, 1)
            if self.wall_clock_sec > 0
            else 0.0,
            "transform_pct": round(self.transform_sec / self.wall_clock_sec * 100, 1)
            if self.wall_clock_sec > 0
            else 0.0,
            "transfer_pct": round(self.transfer_sec / self.wall_clock_sec * 100, 1)
            if self.wall_clock_sec > 0
            else 0.0,
            "overhead_pct": round(self.overhead_sec / self.wall_clock_sec * 100, 1)
            if self.wall_clock_sec > 0
            else 0.0,
        }


def run_microbenchmark(
    adapter: PipelineAdapter,
    config: ScenarioConfig,
    data: Any,
    num_batches: int = 50,
    warmup_batches: int = 5,
) -> MicrobenchmarkResult:
    """Profile a scenario with decomposed timing.

    Runs the adapter lifecycle with fine-grained timing instrumentation
    to separate source, transform, and transfer costs.

    Args:
        adapter: The pipeline adapter to profile.
        config: Scenario configuration.
        data: Pre-generated synthetic data dict.
        num_batches: Number of batches to time.
        warmup_batches: Warmup batches (not timed).

    Returns:
        MicrobenchmarkResult with detailed timing breakdown.
    """
    adapter.setup(config, data)
    adapter.warmup(warmup_batches)

    per_batch_source: list[float] = []
    per_batch_transform: list[float] = []
    per_batch_transfer: list[float] = []

    overall_start = time.perf_counter()

    iterator = iter(adapter._iterate_batches())  # noqa: SLF001
    batch_count = 0

    try:
        for _ in range(num_batches):
            # Source phase: fetch next batch from iterator
            t0 = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            t1 = time.perf_counter()
            per_batch_source.append(t1 - t0)

            # Transform + materialization phase
            arrays = adapter._materialize_batch(batch)  # noqa: SLF001
            t2 = time.perf_counter()
            per_batch_transform.append(t2 - t1)

            # Transfer phase: ensure data is on device
            _block_until_ready(arrays)
            t3 = time.perf_counter()
            per_batch_transfer.append(t3 - t2)

            batch_count += 1
    finally:
        close_fn = getattr(iterator, "close", None)
        if callable(close_fn):
            close_fn()

    overall_end = time.perf_counter()
    wall = overall_end - overall_start

    source_total = sum(per_batch_source)
    transform_total = sum(per_batch_transform)
    transfer_total = sum(per_batch_transfer)
    overhead = max(0.0, wall - source_total - transform_total - transfer_total)

    adapter.teardown()

    return MicrobenchmarkResult(
        scenario_id=config.scenario_id,
        variant_name=config.extra.get("variant_name", "default"),
        num_batches=batch_count,
        wall_clock_sec=wall,
        source_sec=source_total,
        transform_sec=transform_total,
        transfer_sec=transfer_total,
        overhead_sec=overhead,
        per_batch_source=per_batch_source,
        per_batch_transform=per_batch_transform,
        per_batch_transfer=per_batch_transfer,
    )


def _block_until_ready(arrays: list[Any]) -> None:
    """Block until all JAX arrays are materialized on device."""
    for arr in arrays:
        if hasattr(arr, "block_until_ready"):
            arr.block_until_ready()
