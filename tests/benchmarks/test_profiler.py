"""Tests for GPU memory profiler and memory optimizer.

Tests for AdaptiveOperation are in test_mega_profiler.py.
Tests for BenchmarkResult (replacing ProfileResult) are in tests/benchmarking/test_results.py.
"""

from typing import Any

import jax.numpy as jnp
import numpy as np
from calibrax.profiling import (
    GPUMemoryProfiler,
    MemoryAnalysis,
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
            del data
            return jnp.ones((10, 10))

        analysis = optimizer.analyze_pipeline_memory(simple_function, {})

        assert isinstance(analysis, MemoryAnalysis)
        assert analysis.baseline_memory_mb >= 0
        assert analysis.peak_memory_mb >= 0
        assert isinstance(analysis.peak_usage_mb, float)
        assert isinstance(analysis.suggestions, tuple)

    def test_analyze_memory_intensive_function(self):
        """Test memory analysis of memory-intensive function.

        ``calibrax.MemoryOptimizer.analyze_pipeline_memory`` measures
        baseline RSS, then end-of-call RSS as "peak". On macOS the
        kernel can compress or release pages between the two readings,
        producing negative or below-floor deltas even when a 200 MB
        array is held alive across the call. The retry loop accepts
        the first reading that clears the floor; if every attempt
        falls below it the test fails with the best-case value so a
        real regression in the optimizer still surfaces.
        """
        optimizer = MemoryOptimizer()

        # 5000 x 5000 float64 = 200 MB, dominates noise on typical hosts.
        side = 5000
        ref_holder: list[Any] = []

        def memory_intensive_function(data):
            """Memory-intensive function for testing."""
            del data
            large_array = np.ones((side, side), dtype=np.float64)
            ref_holder.append(large_array)  # keep alive across analyze step
            return float(large_array.sum())

        analyses: list[MemoryAnalysis] = []
        try:
            for _ in range(3):
                ref_holder.clear()
                analysis = optimizer.analyze_pipeline_memory(memory_intensive_function, {})
                assert analysis is not None
                analyses.append(analysis)
                if analysis.peak_usage_mb >= 50.0:
                    break
        finally:
            ref_holder.clear()

        best = max(analyses, key=lambda a: a.peak_usage_mb)
        assert isinstance(best, MemoryAnalysis)
        # 200 MB allocation; require at least 50 MB peak delta in the best of 3 attempts.
        assert best.peak_usage_mb >= 50.0, (
            f"Expected peak_usage_mb >= 50 MB for a 200 MB allocation in "
            f"{len(analyses)} attempts, best was {best.peak_usage_mb:.1f} MB"
        )
        assert best.memory_efficiency <= 1
