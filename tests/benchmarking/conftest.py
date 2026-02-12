"""Shared test factories for benchmarking tests.

Provides factory functions for BenchmarkResult, TimingSample, and ResourceSummary.
Eliminates duplication across test_results.py, test_regression.py, and test_comparative.py.

Two factory patterns:
- make_timing / make_resources / make_result: Full-featured with override pattern.
  Used by test_results.py for testing the BenchmarkResult class itself.
- make_result_for_throughput: Quick factory for throughput-specific tests.
  Used by test_regression.py and test_comparative.py.
"""

from datarax.benchmarking.results import BenchmarkResult
from datarax.benchmarking.resource_monitor import ResourceSummary
from datarax.benchmarking.timing import TimingSample


def make_timing(**overrides) -> TimingSample:
    """Create a TimingSample with sensible defaults."""
    defaults = {
        "wall_clock_sec": 2.0,
        "per_batch_times": [0.1, 0.2, 0.15, 0.12, 0.18],
        "first_batch_time": 0.15,
        "num_batches": 5,
        "num_elements": 160,
    }
    defaults.update(overrides)
    return TimingSample(**defaults)


def make_resources(**overrides) -> ResourceSummary:
    """Create a ResourceSummary with sensible defaults."""
    defaults = {
        "peak_rss_mb": 512.0,
        "mean_rss_mb": 400.0,
        "peak_gpu_mem_mb": None,
        "mean_gpu_util": None,
        "memory_growth_mb": 10.0,
        "num_samples": 50,
        "duration_sec": 2.0,
    }
    defaults.update(overrides)
    return ResourceSummary(**defaults)


def make_result(**overrides) -> BenchmarkResult:
    """Create a BenchmarkResult with full defaults (all fields populated).

    Used for testing BenchmarkResult itself (serialization, methods, etc.).
    """
    defaults = {
        "framework": "Datarax",
        "scenario_id": "CV-1",
        "variant": "small",
        "timing": make_timing(),
        "resources": make_resources(),
        "environment": {"platform": "linux", "python_version": "3.12.0"},
        "config": {"batch_size": 32, "dataset_size": 1000},
        "timestamp": 1700000000.0,
    }
    defaults.update(overrides)
    return BenchmarkResult(**defaults)


def make_result_for_throughput(
    wall_clock_sec: float = 1.0,
    num_batches: int = 10,
    num_elements: int = 1000,
) -> BenchmarkResult:
    """Create a minimal BenchmarkResult for throughput-based testing.

    Computes uniform per-batch times from wall_clock_sec / num_batches.
    Used by regression and comparative tests that care about throughput ratios.
    """
    per_batch = wall_clock_sec / num_batches if num_batches > 0 else 0.0
    return make_result(
        framework="test",
        scenario_id="bench",
        timing=make_timing(
            wall_clock_sec=wall_clock_sec,
            per_batch_times=[per_batch] * num_batches,
            first_batch_time=per_batch,
            num_batches=num_batches,
            num_elements=num_elements,
        ),
        resources=None,
        environment={},
        config={},
    )
