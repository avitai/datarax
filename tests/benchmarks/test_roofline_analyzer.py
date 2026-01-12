"""Tests for RooflineAnalyzer performance optimization class.

This module tests the hardware-aware optimization infrastructure that analyzes
operations based on the roofline model to identify performance bottlenecks.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from datarax.performance.roofline import RooflineAnalyzer


class TestRooflineAnalyzer:
    """Test suite for RooflineAnalyzer."""

    def test_hardware_detection(self):
        """Test automatic hardware detection and configuration."""
        analyzer = RooflineAnalyzer()

        # Should detect current hardware
        assert analyzer.hardware_name in ["tpu_v5e", "h100", "a100", "cpu"]
        assert analyzer.hw_specs is not None
        assert analyzer.hw_specs.peak_flops_bf16 > 0
        assert analyzer.hw_specs.hbm_bandwidth > 0
        assert analyzer.hw_specs.critical_intensity > 0

    def test_tpu_specs(self):
        """Test TPU v5e hardware specifications."""
        analyzer = RooflineAnalyzer(hardware="tpu_v5e")

        assert analyzer.hardware_name == "tpu_v5e"
        assert analyzer.hw_specs.peak_flops_bf16 == 1.97e14
        assert analyzer.hw_specs.hbm_bandwidth == 8.2e11
        assert analyzer.hw_specs.vmem_bandwidth == 18e12
        assert analyzer.hw_specs.critical_intensity == 240
        assert analyzer.hw_specs.optimal_batch_size == 240
        assert analyzer.hw_specs.matrix_unit_size == (128, 128)

    def test_gpu_specs(self):
        """Test GPU hardware specifications."""
        analyzer = RooflineAnalyzer(hardware="h100")

        assert analyzer.hardware_name == "h100"
        assert analyzer.hw_specs.peak_flops_bf16 == 9.89e14
        assert analyzer.hw_specs.hbm_bandwidth == 3.35e12
        assert analyzer.hw_specs.critical_intensity == 298
        assert analyzer.hw_specs.optimal_batch_size == 298

    def test_arithmetic_intensity_calculation(self):
        """Test arithmetic intensity calculation for operations."""
        analyzer = RooflineAnalyzer(hardware="tpu_v5e")

        # Matrix multiplication
        a = jnp.ones((512, 1024), dtype=jnp.bfloat16)
        b = jnp.ones((1024, 2048), dtype=jnp.bfloat16)

        analysis = analyzer.analyze_operation(jnp.matmul, a, b)

        # Check analysis results
        assert "arithmetic_intensity" in analysis
        assert "bottleneck" in analysis
        assert "flops_utilization" in analysis
        assert "recommendations" in analysis

        # Matrix multiplication should have high arithmetic intensity
        assert analysis["arithmetic_intensity"] > 100
        assert analysis["bottleneck"] in ["compute", "memory"]

    def test_memory_bound_detection(self):
        """Test detection of memory-bound operations."""
        analyzer = RooflineAnalyzer(hardware="tpu_v5e")

        # Element-wise operation (memory-bound)
        x = jnp.ones((1000, 1000), dtype=jnp.bfloat16)

        def elementwise_op(x):
            return jnp.sin(x) + jnp.cos(x)

        analysis = analyzer.analyze_operation(elementwise_op, x)

        # Should be memory-bound
        assert analysis["bottleneck"] == "memory"
        assert analysis["arithmetic_intensity"] < analyzer.hw_specs.critical_intensity

        # Should recommend optimization
        recommendations = analysis["recommendations"]
        assert len(recommendations) > 0
        assert any("memory-bound" in r.lower() for r in recommendations)

    def test_compute_bound_detection(self):
        """Test detection of compute-bound operations."""
        analyzer = RooflineAnalyzer(hardware="tpu_v5e")

        # Large matrix multiplication (compute-bound)
        a = jnp.ones((2048, 2048), dtype=jnp.bfloat16)
        b = jnp.ones((2048, 2048), dtype=jnp.bfloat16)

        analysis = analyzer.analyze_operation(jnp.matmul, a, b)

        # Should be compute-bound
        assert analysis["bottleneck"] == "compute"
        assert analysis["arithmetic_intensity"] > analyzer.hw_specs.critical_intensity

    def test_batch_size_recommendations(self):
        """Test batch size optimization recommendations."""
        analyzer = RooflineAnalyzer(hardware="tpu_v5e")

        # Small batch size
        small_batch = jnp.ones((32, 512), dtype=jnp.bfloat16)
        weights = jnp.ones((512, 256), dtype=jnp.bfloat16)

        analysis = analyzer.analyze_operation(jnp.matmul, small_batch, weights)

        # Should recommend larger batch size
        recommendations = analysis["recommendations"]
        assert any("batch size" in r.lower() for r in recommendations)
        assert any("240" in r for r in recommendations)  # TPU optimal batch size

    def test_shape_alignment_recommendations(self):
        """Test shape alignment recommendations for hardware."""
        analyzer = RooflineAnalyzer(hardware="tpu_v5e")

        # Unaligned shapes for TPU
        unaligned = jnp.ones((37, 789), dtype=jnp.bfloat16)

        analysis = analyzer.analyze_operation(lambda x: x * 2, unaligned)

        # Should recommend shape alignment
        recommendations = analysis["recommendations"]
        assert any("shape" in r.lower() or "align" in r.lower() for r in recommendations)

    def test_efficiency_calculation(self):
        """Test efficiency and utilization calculations."""
        analyzer = RooflineAnalyzer(hardware="tpu_v5e")

        # Well-optimized operation
        a = jnp.ones((1024, 1024), dtype=jnp.bfloat16)
        b = jnp.ones((1024, 1024), dtype=jnp.bfloat16)

        analysis = analyzer.analyze_operation(jnp.matmul, a, b)

        # Check efficiency metrics
        assert "efficiency" in analysis
        assert "flops_utilization" in analysis
        assert 0 <= analysis["efficiency"] <= 1.0
        assert 0 <= analysis["flops_utilization"] <= 1.0

    def test_performance_profiling(self):
        """Test performance timing and profiling."""
        analyzer = RooflineAnalyzer(hardware="cpu")  # Use CPU for consistent testing

        x = jnp.ones((100, 100), dtype=jnp.float32)

        analysis = analyzer.analyze_operation(lambda x: x @ x.T, x)

        # Should have timing information
        assert "theoretical_time_ms" in analysis
        assert "actual_time_ms" in analysis
        assert analysis["theoretical_time_ms"] > 0
        assert analysis["actual_time_ms"] > 0

    def test_optimal_batch_size_finder(self):
        """Test automatic optimal batch size discovery."""
        analyzer = RooflineAnalyzer(hardware="tpu_v5e")

        sample = jnp.ones((512,), dtype=jnp.bfloat16)
        optimal_batch = analyzer.find_optimal_batch_size(sample)

        # Should find batch size >= critical batch size
        assert optimal_batch >= analyzer.hw_specs.optimal_batch_size

    def test_operation_optimization(self):
        """Test automatic operation optimization."""
        analyzer = RooflineAnalyzer(hardware="tpu_v5e")

        # Memory-bound operation
        def inefficient_op(x):
            y = jnp.sqrt(x)
            y = jnp.sin(y)
            y = jnp.exp(y)
            return jnp.tanh(y)

        # Optimize for better arithmetic intensity
        optimized_op = analyzer.optimize_for_arithmetic_intensity(
            inefficient_op, target_intensity=240
        )

        # Test with small input
        x = jnp.ones((32, 512), dtype=jnp.bfloat16)

        # Both should produce valid results
        original_result = inefficient_op(x)
        optimized_result = optimized_op(x)

        assert original_result.shape == optimized_result.shape
        assert jnp.isfinite(optimized_result).all()


class TestHardwareAdaptation:
    """Test suite for hardware adaptation utilities."""

    def test_shape_optimization_tpu(self):
        """Test shape optimization for TPU."""
        analyzer = RooflineAnalyzer(hardware="tpu_v5e")

        # Unaligned tensors
        tensors = [jnp.ones((37, 100)), jnp.ones((64, 129)), jnp.ones((200, 200))]

        optimized = analyzer.optimize_shapes(tensors)

        # Should pad to multiples of 128 for TPU
        assert optimized[0].shape == (128, 128)  # Padded from (37, 100)
        assert optimized[1].shape == (128, 256)  # Padded from (64, 129) - 64->128, 129->256
        assert optimized[2].shape == (256, 256)  # Padded from (200, 200)

    def test_shape_optimization_gpu(self):
        """Test shape optimization for GPU."""
        analyzer = RooflineAnalyzer(hardware="h100")

        # Unaligned tensors
        tensors = [jnp.ones((15, 33)), jnp.ones((64, 65))]

        optimized = analyzer.optimize_shapes(tensors)

        # Should pad to multiples of 16 for GPU tensor cores
        assert optimized[0].shape[0] % 16 == 0
        assert optimized[0].shape[1] % 16 == 0
        assert optimized[1].shape[0] % 16 == 0
        assert optimized[1].shape[1] % 16 == 0

    def test_precision_casting(self):
        """Test automatic precision casting for hardware."""
        import warnings

        analyzer_tpu = RooflineAnalyzer(hardware="tpu_v5e")
        analyzer_gpu = RooflineAnalyzer(hardware="h100")

        # Mixed precision inputs - suppress float64 warning since we're testing casting
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*float64.*")
            tensors = [jnp.ones((10, 10), dtype=jnp.float32), jnp.ones((10, 10), dtype=jnp.float64)]

        # TPU should cast to bfloat16
        tpu_casted = analyzer_tpu.cast_to_optimal_precision(tensors)
        assert all(t.dtype == jnp.bfloat16 for t in tpu_casted)

        # GPU should support bfloat16 on H100
        gpu_casted = analyzer_gpu.cast_to_optimal_precision(tensors)
        assert all(t.dtype == jnp.bfloat16 for t in gpu_casted)

    def test_memory_layout_optimization(self):
        """Test memory layout optimization for different hardware."""
        analyzer = RooflineAnalyzer(hardware="tpu_v5e")

        # Check optimal memory layout
        assert analyzer.hw_specs.memory_layout == "row_major"
        assert analyzer.hw_specs.use_vmem_optimization


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return {
        "small_batch": jnp.ones((32, 512), dtype=jnp.bfloat16),
        "large_batch": jnp.ones((512, 512), dtype=jnp.bfloat16),
        "unaligned": jnp.ones((37, 789), dtype=jnp.bfloat16),
        "aligned_tpu": jnp.ones((256, 512), dtype=jnp.bfloat16),
    }


def test_integration_with_nnx_module(sample_data):
    """Test integration with NNX modules."""

    class OptimizedModule(nnx.Module):
        def __init__(self, input_features: int, output_features: int, *, rngs: nnx.Rngs):
            super().__init__()
            self.analyzer = RooflineAnalyzer()
            self.input_features = input_features
            self.dense = nnx.Linear(input_features, output_features, rngs=rngs)

        def __call__(self, x: jax.Array) -> jax.Array:
            # For this test, we use the input directly
            # In real usage, we'd pad the linear layer weights instead
            result = self.dense(x)
            return result

    # Test with properly sized input that matches Linear layer
    rngs = nnx.Rngs(42)
    # Create input with correct dimensions
    x = jnp.ones((37, 789), dtype=jnp.bfloat16)  # unaligned shape
    module = OptimizedModule(789, 256, rngs=rngs)  # Match input features

    output = module(x)
    assert output.shape == (37, 256)  # Output shape preserved
