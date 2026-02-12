"""Background 10Hz resource sampling.

Extracted from AdvancedMonitor's background sampling logic.
Fixes P1 (god-class decomposition) and P8 (tangled dependencies).

Design ref: Section 6.2.3 of the benchmark report.
"""

import threading
import time
from dataclasses import dataclass

import psutil


@dataclass
class ResourceSample:
    """Single resource measurement at a point in time.

    Attributes:
        timestamp: Time of measurement (perf_counter).
        cpu_percent: CPU utilization percentage.
        rss_mb: Resident set size in MB.
        gpu_util: GPU utilization percentage (None if no GPU).
        gpu_mem_mb: GPU memory used in MB (None if no GPU).
    """

    timestamp: float
    cpu_percent: float
    rss_mb: float
    gpu_util: float | None
    gpu_mem_mb: float | None


@dataclass
class ResourceSummary:
    """Aggregated resource usage over a monitoring period.

    Attributes:
        peak_rss_mb: Maximum RSS observed.
        mean_rss_mb: Average RSS across all samples.
        peak_gpu_mem_mb: Maximum GPU memory (None if no GPU).
        mean_gpu_util: Average GPU utilization (None if no GPU).
        memory_growth_mb: Last RSS minus first RSS (positive = growth).
        num_samples: Total samples collected.
        duration_sec: Time span of monitoring.
    """

    peak_rss_mb: float
    mean_rss_mb: float
    peak_gpu_mem_mb: float | None
    mean_gpu_util: float | None
    memory_growth_mb: float
    num_samples: int
    duration_sec: float


class ResourceMonitor:
    """Background 10Hz resource sampling.

    Usage::

        with ResourceMonitor() as mon:
            # ... run benchmark ...
        summary = mon.summary

    Args:
        sample_interval_sec: Seconds between samples (default 0.1 = 10Hz).
        gpu_profiler: Optional GPUMemoryProfiler for GPU metrics.
    """

    def __init__(
        self,
        sample_interval_sec: float = 0.1,
        gpu_profiler: "object | None" = None,
    ):
        """Initialize ResourceMonitor.

        Args:
            sample_interval_sec: Seconds between resource samples.
            gpu_profiler: Optional GPUMemoryProfiler for GPU metrics.
        """
        self._interval = sample_interval_sec
        self._gpu_profiler = gpu_profiler
        self._samples: list[ResourceSample] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._process = psutil.Process()

    def __enter__(self) -> "ResourceMonitor":
        """Start background sampling thread."""
        self._samples.clear()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        """Stop background sampling thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _sample_loop(self) -> None:
        """Collect samples at the configured interval until stopped."""
        while not self._stop_event.is_set():
            sample = ResourceSample(
                timestamp=time.perf_counter(),
                cpu_percent=self._process.cpu_percent(),
                rss_mb=self._process.memory_info().rss / (1024 * 1024),
                gpu_util=self._get_gpu_util(),
                gpu_mem_mb=self._get_gpu_mem(),
            )
            self._samples.append(sample)
            self._stop_event.wait(timeout=self._interval)

    def _get_gpu_util(self) -> float | None:
        if self._gpu_profiler is None:
            return None
        try:
            return self._gpu_profiler.get_utilization()  # type: ignore[attr-defined]
        except Exception:
            return None

    def _get_gpu_mem(self) -> float | None:
        if self._gpu_profiler is None:
            return None
        try:
            mem = self._gpu_profiler.get_memory_usage()  # type: ignore[attr-defined]
            return mem.get("gpu_memory_used_mb", None)
        except Exception:
            return None

    @property
    def samples(self) -> list[ResourceSample]:
        """Copy of all collected samples."""
        return list(self._samples)

    @property
    def summary(self) -> ResourceSummary:
        """Compute aggregated summary from collected samples."""
        if not self._samples:
            return ResourceSummary(
                peak_rss_mb=0,
                mean_rss_mb=0,
                peak_gpu_mem_mb=None,
                mean_gpu_util=None,
                memory_growth_mb=0,
                num_samples=0,
                duration_sec=0,
            )

        rss_values = [s.rss_mb for s in self._samples]
        gpu_utils = [s.gpu_util for s in self._samples if s.gpu_util is not None]
        gpu_mems = [s.gpu_mem_mb for s in self._samples if s.gpu_mem_mb is not None]

        return ResourceSummary(
            peak_rss_mb=max(rss_values),
            mean_rss_mb=sum(rss_values) / len(rss_values),
            peak_gpu_mem_mb=max(gpu_mems) if gpu_mems else None,
            mean_gpu_util=sum(gpu_utils) / len(gpu_utils) if gpu_utils else None,
            memory_growth_mb=rss_values[-1] - rss_values[0],
            num_samples=len(self._samples),
            duration_sec=(self._samples[-1].timestamp - self._samples[0].timestamp)
            if len(self._samples) > 1
            else 0.0,
        )
