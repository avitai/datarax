"""calibrax-native benchmark result helpers for the Datarax benchmark suite."""

from __future__ import annotations

from typing import Any

import numpy as np
from calibrax.core import BenchmarkResult, Metric
from calibrax.profiling import ResourceSummary, TimingSample


_DEFAULT_DOMAIN = "datarax_pipeline_benchmarks"
_FRAMEWORK_TAG = "framework"
_SCENARIO_TAG = "scenario_id"
_VARIANT_TAG = "variant"


def _latency_percentiles_from_timing(timing: TimingSample | None) -> dict[str, float]:
    """Compute latency percentiles in milliseconds from timing samples."""
    if timing is None or not timing.per_batch_times:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    arr = np.array(timing.per_batch_times, dtype=np.float64) * 1000.0
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def throughput_elements_per_sec(result: BenchmarkResult) -> float:
    """Primary comparison metric: elements processed per second."""
    metric = result.metrics.get("throughput")
    if metric is not None:
        return metric.value

    timing = result.timing
    if timing is None or timing.wall_clock_sec <= 0:
        return 0.0
    return timing.num_elements / timing.wall_clock_sec


def latency_percentiles_ms(result: BenchmarkResult) -> dict[str, float]:
    """Return p50/p95/p99 latency in milliseconds."""
    p50 = result.metrics.get("latency_p50")
    p95 = result.metrics.get("latency_p95")
    p99 = result.metrics.get("latency_p99")
    if p50 is not None and p95 is not None and p99 is not None:
        return {"p50": p50.value, "p95": p95.value, "p99": p99.value}
    return _latency_percentiles_from_timing(result.timing)


def result_framework(result: BenchmarkResult) -> str:
    """Get framework name from standardized tags."""
    return result.tags.get(_FRAMEWORK_TAG, "unknown")


def result_scenario_id(result: BenchmarkResult) -> str:
    """Get scenario ID from standardized tags."""
    scenario_id = result.tags.get(_SCENARIO_TAG)
    if scenario_id:
        return scenario_id
    return result.name.split("/", 1)[0]


def result_variant(result: BenchmarkResult) -> str:
    """Get variant name from standardized tags."""
    variant = result.tags.get(_VARIANT_TAG)
    if variant:
        return variant
    if "/" in result.name:
        return result.name.split("/", 1)[1]
    return "default"


def result_environment(result: BenchmarkResult) -> dict[str, Any]:
    """Get captured environment from metadata."""
    environment = result.metadata.get("environment")
    if isinstance(environment, dict):
        return environment
    return {}


def build_benchmark_result(
    *,
    framework: str,
    scenario_id: str,
    variant: str,
    timing: TimingSample,
    resources: ResourceSummary | None,
    environment: dict[str, Any],
    config: dict[str, Any],
    extra_metrics: dict[str, float] | None = None,
    domain: str = _DEFAULT_DOMAIN,
    timestamp: float | None = None,
) -> BenchmarkResult:
    """Construct a calibrax BenchmarkResult with standardized datarax tags/metrics."""
    latencies = _latency_percentiles_from_timing(timing)
    throughput = 0.0
    if timing.wall_clock_sec > 0:
        throughput = timing.num_elements / timing.wall_clock_sec

    metrics: dict[str, Metric] = {
        "throughput": Metric(value=throughput),
        "latency_p50": Metric(value=latencies["p50"]),
        "latency_p95": Metric(value=latencies["p95"]),
        "latency_p99": Metric(value=latencies["p99"]),
    }

    if resources is not None:
        metrics["peak_memory"] = Metric(value=resources.peak_rss_mb)
        if resources.peak_gpu_mem_mb is not None:
            metrics["gpu_memory"] = Metric(value=resources.peak_gpu_mem_mb)

    if extra_metrics:
        for name, value in extra_metrics.items():
            metrics[name] = Metric(value=value)

    kwargs: dict[str, Any] = {
        "name": f"{scenario_id}/{variant}",
        "domain": domain,
        "tags": {
            _FRAMEWORK_TAG: framework,
            _SCENARIO_TAG: scenario_id,
            _VARIANT_TAG: variant,
        },
        "timing": timing,
        "resources": resources,
        "metrics": metrics,
        "metadata": {"environment": environment},
        "config": config,
    }
    if timestamp is not None:
        kwargs["timestamp"] = timestamp

    return BenchmarkResult(
        **kwargs,
    )
