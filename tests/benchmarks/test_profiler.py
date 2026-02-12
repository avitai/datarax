"""Tests for GPU memory profiler and memory optimizer.

Tests for AdaptiveOperation are in test_mega_profiler.py.
Tests for BenchmarkResult (replacing ProfileResult) are in tests/benchmarking/test_results.py.
"""

import jax.numpy as jnp

from datarax.benchmarking.profiler import (
    GPUMemoryProfiler,
    MemoryOptimizer,
)


class TestGPUMemoryProfiler:
    """Tests for GPU memory profiling."""

    def test_memory_profiler_initialization(self):
        """Test GPU memory profiler initialization."""
        profiler = GPUMemoryProfiler()
        assert profiler is not None
        assert isinstance(profiler.has_gpu, bool)

    def test_get_memory_usage(self):
        """Test memory usage measurement."""
        profiler = GPUMemoryProfiler()
        memory_info = profiler.get_memory_usage()

        assert isinstance(memory_info, dict)
        assert "gpu_memory_used_mb" in memory_info
        assert "gpu_memory_total_mb" in memory_info
        assert isinstance(memory_info["gpu_memory_used_mb"], int | float)
        assert isinstance(memory_info["gpu_memory_total_mb"], int | float)

    def test_get_utilization(self):
        """Test GPU utilization method (used by ResourceMonitor)."""
        profiler = GPUMemoryProfiler()
        util = profiler.get_utilization()
        assert isinstance(util, float)
        assert util >= 0

    def test_analyze_memory_pattern(self):
        """Test memory pattern analysis."""
        profiler = GPUMemoryProfiler()

        # Test with empty measurements
        suggestions = profiler.analyze_memory_pattern([])
        assert suggestions == []

        # Test with stable memory pattern
        stable_measurements = [
            {"gpu_memory_used_mb": 100, "gpu_memory_utilization": 0.1},
            {"gpu_memory_used_mb": 102, "gpu_memory_utilization": 0.1},
            {"gpu_memory_used_mb": 101, "gpu_memory_utilization": 0.1},
        ]
        suggestions = profiler.analyze_memory_pattern(stable_measurements)
        assert isinstance(suggestions, list)

        # Test with high utilization pattern
        high_util_measurements = [
            {"gpu_memory_used_mb": 1000, "gpu_memory_utilization": 0.95},
            {"gpu_memory_used_mb": 1000, "gpu_memory_utilization": 0.95},
            {"gpu_memory_used_mb": 1000, "gpu_memory_utilization": 0.95},
        ]
        suggestions = profiler.analyze_memory_pattern(high_util_measurements)
        assert len(suggestions) > 0
        assert any("90%" in s or "high" in s.lower() for s in suggestions)


class TestMemoryOptimizer:
    """Tests for memory optimization."""

    def test_memory_optimizer_initialization(self):
        """Test memory optimizer initialization."""
        optimizer = MemoryOptimizer()
        assert optimizer is not None

    def test_analyze_simple_function(self):
        """Test memory analysis of simple function."""
        optimizer = MemoryOptimizer()

        def simple_function(data):
            """Simple function for testing."""
            return jnp.ones((10, 10))

        analysis = optimizer.analyze_pipeline_memory(simple_function, {})

        assert isinstance(analysis, dict)
        assert "baseline_memory_mb" in analysis
        assert "peak_memory_mb" in analysis
        assert "suggestions" in analysis
        assert isinstance(analysis["suggestions"], list)

    def test_analyze_memory_intensive_function(self):
        """Test memory analysis of memory-intensive function."""
        optimizer = MemoryOptimizer()

        def memory_intensive_function(data):
            """Memory-intensive function for testing."""
            large_array = jnp.ones((500, 500))
            return jnp.sum(large_array)

        analysis = optimizer.analyze_pipeline_memory(memory_intensive_function, {})

        assert isinstance(analysis, dict)
        assert analysis["peak_usage_mb"] >= 0
        assert 0 <= analysis.get("memory_efficiency", 0) <= 1
