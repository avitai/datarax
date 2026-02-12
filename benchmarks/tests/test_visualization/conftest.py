"""Shared fixtures for visualization tests.

Provides mock ComparativeResults for chart generation testing.
"""

from __future__ import annotations

import time

import pytest

from benchmarks.tests.test_analysis.conftest import make_comparative_results, make_result
from datarax.benchmarking.results import BenchmarkResult


@pytest.fixture
def mock_results():
    """Create ComparativeResults with 4 frameworks x 3 scenarios."""
    from benchmarks.runners.full_runner import ComparativeResults

    return ComparativeResults(
        results=make_comparative_results(),
        environment={"platform": {"backend": "cpu", "device_count": 1}},
        platform="cpu",
        timestamp=time.time(),
    )


@pytest.fixture
def mock_results_with_scaling():
    """Create ComparativeResults with scaling dimension variants."""
    from benchmarks.runners.full_runner import ComparativeResults

    # Create results at different batch sizes for scaling curves
    results: dict[str, list[BenchmarkResult]] = {"Datarax": [], "Grain": []}
    for batch_size in [16, 32, 64, 128]:
        for fw, base_tp in [("Datarax", 1000.0), ("Grain", 800.0)]:
            results[fw].append(
                make_result(
                    framework=fw,
                    scenario_id="CV-1",
                    throughput=base_tp * (batch_size / 32),
                    batch_size=batch_size,
                )
            )

    return ComparativeResults(
        results=results,
        environment={"platform": {"backend": "cpu"}},
        platform="cpu",
        timestamp=time.time(),
    )
