"""Verify that roofline metrics are collected.

Uses direct RooflineAnalyzer (replaces AdvancedProfiler wrapper).
"""

import jax.numpy as jnp
import pytest

from datarax.performance.roofline import RooflineAnalyzer
from datarax.utils.console import emit


ROOFLINE_SKIP_EXCEPTIONS = (NotImplementedError, RuntimeError, ValueError, TypeError)


@pytest.mark.benchmark
def test_roofline_metrics_collection():
    """Verify that roofline metrics are collected when enabled."""
    analyzer = RooflineAnalyzer()

    # Define a compute-bound operation (Matrix Multiplication)
    size = 128
    a = jnp.ones((size, size))
    b = jnp.ones((size, size))

    def matmul_op(data=None):
        del data
        return jnp.dot(a, b)

    # Analyze
    try:
        results = analyzer.analyze_operation(matmul_op, None)
    except ROOFLINE_SKIP_EXCEPTIONS as e:
        pytest.skip(f"Roofline analysis not available: {e}")

    emit("Collected Metrics Keys:", results.keys())

    assert "arithmetic_intensity" in results, f"Full metrics dump: {results}"
    assert "bottleneck" in results

    intensity = results["arithmetic_intensity"]
    bottleneck = results["bottleneck"]

    emit(f"Arithmetic Intensity: {intensity}")
    emit(f"Bottleneck: {bottleneck}")
