"""Tests for CalibraX-native benchmark result helpers."""

from __future__ import annotations

import pytest
from calibrax.core import BenchmarkResult
from calibrax.profiling import ResourceSummary, TimingSample

from benchmarks.core.result_model import (
    build_benchmark_result,
    latency_percentiles_ms,
    result_framework,
    result_scenario_id,
    result_variant,
    throughput_elements_per_sec,
)


def _make_timing(
    *,
    wall_clock_sec: float = 2.0,
    per_batch_times: tuple[float, ...] = (0.05, 0.06, 0.07, 0.08),
    num_elements: int = 400,
) -> TimingSample:
    return TimingSample(
        wall_clock_sec=wall_clock_sec,
        per_batch_times=per_batch_times,
        first_batch_time=0.1,
        num_batches=len(per_batch_times),
        num_elements=num_elements,
    )


def _make_resources() -> ResourceSummary:
    return ResourceSummary(
        peak_rss_mb=512.0,
        mean_rss_mb=500.0,
        peak_gpu_mem_mb=1024.0,
        mean_gpu_util=65.0,
        memory_growth_mb=20.0,
        num_samples=12,
        duration_sec=1.2,
    )


def test_build_benchmark_result_sets_tags_and_primary_metrics():
    result = build_benchmark_result(
        framework="Datarax",
        scenario_id="CV-1",
        variant="small",
        timing=_make_timing(),
        resources=_make_resources(),
        environment={"backend": "cpu"},
        config={"batch_size": 32},
    )

    assert isinstance(result, BenchmarkResult)
    assert result_framework(result) == "Datarax"
    assert result_scenario_id(result) == "CV-1"
    assert result_variant(result) == "small"
    assert throughput_elements_per_sec(result) == pytest.approx(200.0)
    assert "throughput" in result.metrics
    assert "latency_p50" in result.metrics
    assert "latency_p95" in result.metrics
    assert "latency_p99" in result.metrics


def test_throughput_elements_per_sec_handles_zero_wall_clock():
    result = build_benchmark_result(
        framework="Datarax",
        scenario_id="NLP-1",
        variant="tiny",
        timing=_make_timing(wall_clock_sec=0.0, num_elements=100),
        resources=None,
        environment={},
        config={},
    )

    assert throughput_elements_per_sec(result) == 0.0


def test_latency_percentiles_ms_calculates_expected_percentiles():
    timing = _make_timing(per_batch_times=(0.01, 0.02, 0.03, 0.04))
    result = build_benchmark_result(
        framework="Datarax",
        scenario_id="TAB-1",
        variant="small",
        timing=timing,
        resources=None,
        environment={},
        config={},
    )

    percentiles = latency_percentiles_ms(result)
    assert percentiles["p50"] == pytest.approx(25.0)
    assert percentiles["p95"] == pytest.approx(38.5)
    assert percentiles["p99"] == pytest.approx(39.7)
