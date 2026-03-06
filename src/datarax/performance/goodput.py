"""Goodput/badput telemetry for pipeline performance visibility.

Tracks time breakdown per batch:
- Source time: data loading from source
- Transform time: operator execution
- Transfer time: host-to-device transfers
- Overhead: control flow, Python, framework overhead

Goodput ratio = productive_time / wall_time, where productive_time =
source + transform + transfer. Overhead is the remainder.

Usage::

    tracker = GoodputTracker()
    for batch in pipeline:
        tracker.start_batch()
        with tracker.time_source():
            data = source.get_batch()
        with tracker.time_transform():
            result = transform(data)
        with tracker.time_transfer():
            device_data = jax.device_put(result)
        tracker.end_batch()

    print(tracker.summary())
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Generator
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GoodputMetrics:
    """Summary metrics from a GoodputTracker."""

    total_batches: int
    wall_clock_sec: float
    source_sec: float
    transform_sec: float
    transfer_sec: float
    productive_sec: float
    overhead_sec: float
    goodput_ratio: float


class GoodputTracker:
    """Lightweight pipeline telemetry tracker.

    Records per-batch time breakdowns and computes goodput/badput ratios.
    Thread-safe for single-producer usage (one batch at a time).
    """

    def __init__(self) -> None:
        """Initialize the tracker with zeroed counters."""
        self._batch_start: float = 0.0
        self._wall_start: float = 0.0
        self._wall_end: float = 0.0
        self._total_batches: int = 0

        self.per_batch_source: list[float] = []
        self.per_batch_transform: list[float] = []
        self.per_batch_transfer: list[float] = []
        self.per_batch_wall: list[float] = []

        # Accumulators for current batch
        self._current_source: float = 0.0
        self._current_transform: float = 0.0
        self._current_transfer: float = 0.0

    def start_batch(self) -> None:
        """Mark the beginning of a new batch."""
        now = time.monotonic()
        if self._total_batches == 0:
            self._wall_start = now
        self._batch_start = now
        self._current_source = 0.0
        self._current_transform = 0.0
        self._current_transfer = 0.0

    def record_source(self, duration_sec: float) -> None:
        """Record source loading time for current batch."""
        self._current_source += duration_sec

    def record_transform(self, duration_sec: float) -> None:
        """Record transform execution time for current batch."""
        self._current_transform += duration_sec

    def record_transfer(self, duration_sec: float) -> None:
        """Record host-to-device transfer time for current batch."""
        self._current_transfer += duration_sec

    @contextlib.contextmanager
    def time_source(self) -> Generator[None, None, None]:
        """Context manager to time source loading."""
        start = time.monotonic()
        yield
        self._current_source += time.monotonic() - start

    @contextlib.contextmanager
    def time_transform(self) -> Generator[None, None, None]:
        """Context manager to time transform execution."""
        start = time.monotonic()
        yield
        self._current_transform += time.monotonic() - start

    @contextlib.contextmanager
    def time_transfer(self) -> Generator[None, None, None]:
        """Context manager to time host-to-device transfer."""
        start = time.monotonic()
        yield
        self._current_transfer += time.monotonic() - start

    def end_batch(self) -> None:
        """Mark the end of the current batch and record times."""
        now = time.monotonic()
        self._wall_end = now
        self._total_batches += 1

        batch_wall = now - self._batch_start
        self.per_batch_source.append(self._current_source)
        self.per_batch_transform.append(self._current_transform)
        self.per_batch_transfer.append(self._current_transfer)
        self.per_batch_wall.append(batch_wall)

    def reset(self) -> None:
        """Clear all accumulated metrics."""
        self._batch_start = 0.0
        self._wall_start = 0.0
        self._wall_end = 0.0
        self._total_batches = 0
        self.per_batch_source.clear()
        self.per_batch_transform.clear()
        self.per_batch_transfer.clear()
        self.per_batch_wall.clear()

    def summary(self) -> GoodputMetrics:
        """Compute summary metrics from accumulated data."""
        if self._total_batches == 0:
            return GoodputMetrics(
                total_batches=0,
                wall_clock_sec=0.0,
                source_sec=0.0,
                transform_sec=0.0,
                transfer_sec=0.0,
                productive_sec=0.0,
                overhead_sec=0.0,
                goodput_ratio=0.0,
            )

        wall = self._wall_end - self._wall_start
        source = sum(self.per_batch_source)
        transform = sum(self.per_batch_transform)
        transfer = sum(self.per_batch_transfer)
        productive = source + transform + transfer
        overhead = max(0.0, wall - productive)
        goodput = productive / wall if wall > 0 else 0.0

        return GoodputMetrics(
            total_batches=self._total_batches,
            wall_clock_sec=wall,
            source_sec=source,
            transform_sec=transform,
            transfer_sec=transfer,
            productive_sec=productive,
            overhead_sec=overhead,
            goodput_ratio=goodput,
        )
