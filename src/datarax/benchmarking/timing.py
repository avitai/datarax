"""Framework-agnostic timing with GPU sync support.

Replaces Timer from pipeline_throughput.py (fixes P5: time.time() → time.perf_counter())
and the timing loop from AdvancedProfiler.profile_pipeline() (fixes P1 decomposition).

Design ref: Section 6.2.3 of the benchmark report.
"""

import time
from dataclasses import dataclass
from typing import Any
from collections.abc import Callable, Iterator


@dataclass
class TimingSample:
    """Result of timing an iteration through a data pipeline.

    Attributes:
        wall_clock_sec: Total wall-clock time for the iteration.
        per_batch_times: List of per-batch durations in seconds.
        first_batch_time: Time from iteration start to first batch completion,
            capturing pipeline startup/JIT compilation overhead.
        num_batches: Number of batches consumed.
        num_elements: Total elements processed (via count_fn).
    """

    wall_clock_sec: float
    per_batch_times: list[float]
    first_batch_time: float
    num_batches: int
    num_elements: int


class TimingCollector:
    """Framework-agnostic timing with GPU sync support.

    Uses time.perf_counter() exclusively for accurate benchmarking.
    Supports configurable GPU synchronization via sync_fn.

    Args:
        sync_fn: GPU synchronization function called after each batch.
            For JAX: lambda: jnp.array(0.).block_until_ready()
            For PyTorch: torch.cuda.synchronize
            For CPU-only: None (default)
    """

    def __init__(self, sync_fn: Callable[[], None] | None = None):
        """Initialize TimingProfiler.

        Args:
            sync_fn: GPU synchronization function called after each batch.
        """
        self.sync_fn = sync_fn or (lambda: None)

    def measure_iteration(
        self,
        iterator: Iterator,
        num_batches: int | None = None,
        count_fn: Callable[[Any], int] | None = None,
    ) -> TimingSample:
        """Measure exactly num_batches from an iterator.

        Args:
            iterator: Data iterator to measure.
            num_batches: Max batches to consume. None = exhaust iterator.
            count_fn: Function to count elements in a batch.
                Receives the batch, returns int. Default: 1 per batch.

        Returns:
            TimingSample with timing data.
        """
        per_batch_times: list[float] = []
        first_batch_time = 0.0
        total_elements = 0
        count = count_fn or (lambda _: 1)

        overall_start = time.perf_counter()

        for i, batch in enumerate(iterator):
            if num_batches is not None and i >= num_batches:
                break

            batch_start = time.perf_counter()
            self.sync_fn()
            batch_end = time.perf_counter()

            if i == 0:
                first_batch_time = batch_end - overall_start

            per_batch_times.append(batch_end - batch_start)
            total_elements += count(batch)

        wall_clock = time.perf_counter() - overall_start

        return TimingSample(
            wall_clock_sec=wall_clock,
            per_batch_times=per_batch_times,
            first_batch_time=first_batch_time,
            num_batches=len(per_batch_times),
            num_elements=total_elements,
        )
