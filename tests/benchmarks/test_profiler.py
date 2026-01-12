"""Tests for the advanced profiling system."""

import tempfile
from pathlib import Path

import jax.numpy as jnp
import pytest

from datarax.benchmarking.profiler import (
    AdvancedProfiler,
    GPUMemoryProfiler,
    MemoryOptimizer,
    ProfileResult,
)


class TestGPUMemoryProfiler:
    """Tests for GPU memory profiling."""

    def test_memory_profiler_initialization(self):
        """Test GPU memory profiler initialization."""
        profiler = GPUMemoryProfiler()
        assert profiler is not None
        # Should work regardless of GPU availability
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
            # Create large arrays
            large_array = jnp.ones((500, 500))
            return jnp.sum(large_array)

        analysis = optimizer.analyze_pipeline_memory(memory_intensive_function, {})

        assert isinstance(analysis, dict)
        assert analysis["peak_usage_mb"] >= 0
        assert 0 <= analysis.get("memory_efficiency", 0) <= 1


class TestAdvancedProfiler:
    """Tests for the advanced profiler."""

    def test_profiler_initialization(self):
        """Test profiler initialization."""
        profiler = AdvancedProfiler()
        assert profiler is not None
        assert profiler.gpu_profiler is not None
        assert profiler.memory_optimizer is not None

        # Test with disabled features
        profiler_no_gpu = AdvancedProfiler(enable_gpu_profiling=False)
        assert profiler_no_gpu.gpu_profiler is None

        profiler_no_memory = AdvancedProfiler(enable_memory_profiling=False)
        assert profiler_no_memory.memory_optimizer is None

    def test_profile_simple_pipeline(self):
        """Test profiling a simple pipeline function."""
        profiler = AdvancedProfiler()

        def simple_pipeline(sample_data):
            """Simple pipeline for testing."""
            x = jnp.ones((32, 10))
            y = jnp.sum(x, axis=1)
            return y

        result = profiler.profile_pipeline(
            simple_pipeline,
            sample_data={},
            num_iterations=3,
            warmup_iterations=1,
        )

        assert isinstance(result, ProfileResult)
        assert isinstance(result.timing_metrics, dict)
        assert isinstance(result.memory_metrics, dict)
        assert isinstance(result.gpu_metrics, dict)
        assert isinstance(result.optimization_suggestions, list)

        # Check timing metrics
        assert "mean_time_s" in result.timing_metrics
        assert "iterations_per_second" in result.timing_metrics
        assert result.timing_metrics["mean_time_s"] > 0

    def test_profile_error_handling(self):
        """Test profiler error handling."""
        profiler = AdvancedProfiler()

        def failing_pipeline(sample_data):
            """Pipeline that raises an error."""
            raise ValueError("Test error")

        result = profiler.profile_pipeline(
            failing_pipeline,
            sample_data={},
            num_iterations=2,
        )

        assert isinstance(result, ProfileResult)
        assert "error" in result.timing_metrics

    def test_clear_snapshots(self):
        """Test clearing memory snapshots."""
        profiler = AdvancedProfiler()

        # Add some snapshots
        profiler._memory_snapshots = [{"test": 1}, {"test": 2}]
        assert len(profiler._memory_snapshots) == 2

        # Clear snapshots
        profiler.clear_snapshots()
        assert len(profiler._memory_snapshots) == 0


class TestProfileResult:
    """Tests for ProfileResult."""

    def test_profile_result_creation(self):
        """Test ProfileResult creation."""
        result = ProfileResult(
            timing_metrics={"test_time": 1.0},
            memory_metrics={"test_memory": 100.0},
            gpu_metrics={"test_gpu": 50.0},
            optimization_suggestions=["Test suggestion"],
        )

        assert result.timing_metrics == {"test_time": 1.0}
        assert result.memory_metrics == {"test_memory": 100.0}
        assert result.gpu_metrics == {"test_gpu": 50.0}
        assert result.optimization_suggestions == ["Test suggestion"]
        assert result.timestamp > 0

    def test_profile_result_serialization(self):
        """Test ProfileResult serialization."""
        result = ProfileResult(
            timing_metrics={"test_time": 1.0},
            memory_metrics={"test_memory": 100.0},
            gpu_metrics={"test_gpu": 50.0},
        )

        # Test to_dict
        data = result.to_dict()
        assert isinstance(data, dict)
        assert data["timing_metrics"] == {"test_time": 1.0}
        assert data["memory_metrics"] == {"test_memory": 100.0}
        assert data["gpu_metrics"] == {"test_gpu": 50.0}

    def test_profile_result_save_load(self):
        """Test ProfileResult save and load."""
        result = ProfileResult(
            timing_metrics={"test_time": 1.0},
            memory_metrics={"test_memory": 100.0},
            gpu_metrics={"test_gpu": 50.0},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_profile.json"

            # Save result
            result.save(filepath)
            assert filepath.exists()

            # Load result
            loaded_result = ProfileResult.load(filepath)
            assert loaded_result.timing_metrics == result.timing_metrics
            assert loaded_result.memory_metrics == result.memory_metrics
            assert loaded_result.gpu_metrics == result.gpu_metrics


# Integration test with actual Datarax components
@pytest.mark.integration
class TestProfilerIntegration:
    """Integration tests with Datarax components."""

    def test_profile_datarax_pipeline(self):
        """Test profiling an actual Datarax pipeline."""
        profiler = AdvancedProfiler()

        # Create simple Datarax pipeline
        def datarax_pipeline(sample_data):
            """Simple Datarax-style pipeline."""
            # Simulate data processing
            data = jnp.array([[1, 2, 3], [4, 5, 6]])

            # Transform data
            normalized = (data - jnp.mean(data)) / jnp.std(data)

            # Batch data
            batched = jnp.reshape(normalized, (1, -1))

            return batched

        result = profiler.profile_pipeline(
            datarax_pipeline,
            sample_data={},
            num_iterations=5,
            warmup_iterations=1,
        )

        # Verify results
        assert isinstance(result, ProfileResult)
        assert result.timing_metrics["mean_time_s"] > 0
        assert result.timing_metrics["iterations_per_second"] > 0

        # Should have some metadata
        assert "backend" in result.metadata
        # device may not be present if no devices, but hardware_config should be
        assert "hardware_config" in result.metadata
