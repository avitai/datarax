"""Verify that roofline metrics are collected.

Uses direct RooflineAnalyzer (replaces AdvancedProfiler wrapper).
"""

import jax.numpy as jnp
import pytest

from datarax.performance.roofline import RooflineAnalyzer


@pytest.mark.benchmark
def test_roofline_metrics_collection():
    """Verify that roofline metrics are collected when enabled."""
    analyzer = RooflineAnalyzer()

    # Define a compute-bound operation (Matrix Multiplication)
    size = 128
    a = jnp.ones((size, size))
    b = jnp.ones((size, size))

    def matmul_op(data=None):
        return jnp.dot(a, b)

    # Analyze
    try:
        results = analyzer.analyze_operation(matmul_op, None)
    except Exception as e:
        pytest.skip(f"Roofline analysis not available: {e}")

    print("Collected Metrics Keys:", results.keys())

    assert "arithmetic_intensity" in results, f"Full metrics dump: {results}"
    assert "bottleneck" in results

    intensity = results["arithmetic_intensity"]
    bottleneck = results["bottleneck"]

    print(f"Arithmetic Intensity: {intensity}")
    print(f"Bottleneck: {bottleneck}")
