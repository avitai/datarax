"""Shared fixtures for analysis tests.

Provides factory functions to generate synthetic BenchmarkResult objects
with controllable timing distributions for stability/report/gap testing.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from datarax.benchmarking.results import BenchmarkResult
from datarax.benchmarking.timing import TimingSample


def make_result(
    framework: str = "Datarax",
    scenario_id: str = "CV-1",
    variant: str = "small",
    throughput: float = 1000.0,
    cv: float = 0.05,
    num_batches: int = 20,
    batch_size: int = 32,
) -> BenchmarkResult:
    """Create a synthetic BenchmarkResult with controllable properties.

    Args:
        framework: Framework name.
        scenario_id: Scenario ID.
        variant: Size variant name.
        throughput: Target throughput in elements/sec.
        cv: Coefficient of variation for per-batch times.
        num_batches: Number of batches in the timing sample.
        batch_size: Elements per batch.
    """
    num_elements = num_batches * batch_size
    wall_clock = num_elements / throughput if throughput > 0 else 1.0
    mean_batch_time = wall_clock / num_batches

    rng = np.random.default_rng(42)
    std = mean_batch_time * cv
    per_batch_times = rng.normal(mean_batch_time, std, size=num_batches).tolist()
    # Ensure no negative times
    per_batch_times = [max(t, mean_batch_time * 0.1) for t in per_batch_times]

    timing = TimingSample(
        wall_clock_sec=wall_clock,
        per_batch_times=per_batch_times,
        first_batch_time=per_batch_times[0] * 1.5,
        num_batches=num_batches,
        num_elements=num_elements,
    )

    return BenchmarkResult(
        framework=framework,
        scenario_id=scenario_id,
        variant=variant,
        timing=timing,
        resources=None,
        environment={"platform": {"backend": "cpu", "device_count": 1}},
        config={
            "batch_size": batch_size,
            "dataset_size": num_elements,
            "element_shape": [32, 32, 3],
            "transforms": [],
            "seed": 42,
        },
        timestamp=time.time(),
    )


def make_comparative_results(
    frameworks: dict[str, dict[str, float]] | None = None,
    scenarios: list[str] | None = None,
) -> dict[str, list[BenchmarkResult]]:
    """Build a results dict for ComparativeResults.

    Args:
        frameworks: Mapping of framework_name -> {scenario_id: throughput}.
            If None, uses defaults with Datarax + 3 alternatives.
        scenarios: List of scenario IDs. Defaults to CV-1, NLP-1, PC-1.

    Returns:
        Dict of adapter_name -> list[BenchmarkResult].
    """
    if scenarios is None:
        scenarios = ["CV-1", "NLP-1", "PC-1"]

    if frameworks is None:
        frameworks = {
            "Datarax": {"CV-1": 1200.0, "NLP-1": 800.0, "PC-1": 500.0},
            "Grain": {"CV-1": 1000.0, "NLP-1": 750.0, "PC-1": 300.0},
            "PyTorch DataLoader": {"CV-1": 1100.0, "NLP-1": 900.0, "PC-1": 200.0},
            "tf.data": {"CV-1": 950.0, "NLP-1": 850.0, "PC-1": 250.0},
        }

    results: dict[str, list[BenchmarkResult]] = {}
    for fw_name, scenario_throughputs in frameworks.items():
        fw_results = []
        for sid in scenarios:
            if sid in scenario_throughputs:
                fw_results.append(
                    make_result(
                        framework=fw_name,
                        scenario_id=sid,
                        throughput=scenario_throughputs[sid],
                    )
                )
        results[fw_name] = fw_results

    return results


@pytest.fixture
def mock_results() -> dict[str, list[BenchmarkResult]]:
    """Default comparative results with Datarax + 3 alternatives."""
    return make_comparative_results()


@pytest.fixture
def stable_result() -> BenchmarkResult:
    """A stable result with CV=0.05."""
    return make_result(cv=0.05)


@pytest.fixture
def unstable_result() -> BenchmarkResult:
    """An unstable result with CV=0.15."""
    return make_result(cv=0.15)
