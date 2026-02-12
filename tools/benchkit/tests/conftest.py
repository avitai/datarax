"""Shared test fixtures and helpers for benchkit tests."""

import pytest

from benchkit.models import Metric, MetricDef, MetricPriority, Point, Run


def make_run(framework_metrics: dict[str, dict[str, float]], **kwargs) -> Run:
    """Create a Run from {framework: {metric: value}} mapping.

    Convenience factory for tests. Each framework gets a Point named
    "CV-1/{framework}" with scenario "CV-1".
    """
    points = []
    for fw, metrics in framework_metrics.items():
        points.append(
            Point(
                name=f"CV-1/{fw}",
                scenario="CV-1",
                tags={"framework": fw},
                metrics={k: Metric(v) for k, v in metrics.items()},
            )
        )
    return Run(points=points, **kwargs)


THROUGHPUT_DEF = MetricDef(
    "throughput",
    "elem/s",
    "higher",
    group="Throughput",
    priority=MetricPriority.PRIMARY,
)
LATENCY_DEF = MetricDef(
    "latency_p50",
    "ms",
    "lower",
    group="Latency",
    priority=MetricPriority.PRIMARY,
)
STANDARD_METRIC_DEFS = {
    "throughput": THROUGHPUT_DEF,
    "latency_p50": LATENCY_DEF,
}


@pytest.fixture
def sample_run():
    """A standard two-framework run for testing.

    Datarax: throughput=5000, latency_p50=12 (with CI bounds)
    Grain:   throughput=4800, latency_p50=14
    """
    return Run(
        points=[
            Point(
                "CV-1/small",
                scenario="CV-1",
                tags={"framework": "Datarax", "variant": "small"},
                metrics={
                    "throughput": Metric(5000.0),
                    "latency_p50": Metric(12.0, lower=10.0, upper=14.0),
                },
            ),
            Point(
                "CV-1/small",
                scenario="CV-1",
                tags={"framework": "Grain", "variant": "small"},
                metrics={
                    "throughput": Metric(4800.0),
                    "latency_p50": Metric(14.0),
                },
            ),
        ],
        commit="abc123",
        branch="main",
        environment={"cpu": "AMD Ryzen", "gpu": "RTX 4090"},
        metric_defs=dict(STANDARD_METRIC_DEFS),
    )
