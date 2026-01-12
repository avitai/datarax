"""Roofline analysis for hardware-aware performance optimization.

This module provides tools for analyzing operations based on the roofline model
to identify performance bottlenecks and suggest optimizations.
"""

import time
from dataclasses import dataclass
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp


@dataclass
class HardwareSpecs:
    """Hardware specifications for roofline analysis."""

    peak_flops_bf16: float
    hbm_bandwidth: float
    critical_intensity: float
    optimal_batch_size: int
    matrix_unit_size: tuple[int, int] | None = None
    vmem_bandwidth: float | None = None
    memory_layout: str = "row_major"
    use_vmem_optimization: bool = False
    tensor_core_shapes: list[tuple[int, int, int]] | None = None
    preferred_tile_size: int = 128


# Hardware specifications from JAX scaling book
HARDWARE_SPECS = {
    "tpu_v5e": HardwareSpecs(
        peak_flops_bf16=1.97e14,
        hbm_bandwidth=8.2e11,
        vmem_bandwidth=18e12,  # 22x faster than HBM
        critical_intensity=240,  # FLOPs/byte
        matrix_unit_size=(128, 128),
        optimal_batch_size=240,
        memory_layout="row_major",
        use_vmem_optimization=True,
        preferred_tile_size=128,
    ),
    "h100": HardwareSpecs(
        peak_flops_bf16=9.89e14,
        hbm_bandwidth=3.35e12,
        critical_intensity=298,
        tensor_core_shapes=[(16, 16, 8), (32, 8, 16)],
        optimal_batch_size=298,
        memory_layout="NHWC",
        use_vmem_optimization=False,
        preferred_tile_size=16,
    ),
    "a100": HardwareSpecs(
        peak_flops_bf16=3.12e14,
        hbm_bandwidth=1.55e12,
        critical_intensity=201,
        tensor_core_shapes=[(16, 16, 8)],
        optimal_batch_size=128,
        memory_layout="NHWC",
        use_vmem_optimization=False,
        preferred_tile_size=16,
    ),
    "cpu": HardwareSpecs(
        peak_flops_bf16=1e12,  # Approximate for modern CPU
        hbm_bandwidth=1e11,
        critical_intensity=10,
        optimal_batch_size=32,
        memory_layout="row_major",
        use_vmem_optimization=False,
        preferred_tile_size=64,
    ),
}


class RooflineAnalyzer:
    """Analyze operations based on roofline model for performance optimization."""

    def __init__(self, hardware: str = "auto"):
        """Initialize analyzer with hardware configuration.

        Args:
            hardware: Target hardware ('tpu_v5e', 'h100', 'a100', 'cpu', 'auto')
        """
        if hardware == "auto":
            hardware = self._detect_hardware()

        self.hardware_name = hardware
        self.hw_specs = HARDWARE_SPECS.get(hardware, HARDWARE_SPECS["cpu"])

    def _detect_hardware(self) -> str:
        """Automatically detect current hardware."""
        backend = jax.default_backend()

        if backend == "tpu":
            # Try to detect TPU version
            device = jax.devices()[0]
            device_kind = str(device).lower()
            if "v5" in device_kind:
                return "tpu_v5e"
            return "tpu_v5e"  # Default TPU

        elif backend == "gpu":
            # Try to detect GPU type
            device = jax.devices()[0]
            device_kind = str(device).lower()
            if "h100" in device_kind:
                return "h100"
            elif "a100" in device_kind:
                return "a100"
            return "h100"  # Default to newer GPU

        else:
            return "cpu"

    def analyze_operation(
        self, func: Callable, *args: Any, output_shape: tuple | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Analyze a JAX operation using roofline model.

        Args:
            func: Function to analyze
            args: Arguments to the function
            output_shape: Optional output shape for memory estimation
            kwargs: Keyword arguments to the function

        Returns:
            Analysis dict with performance metrics and recommendations
        """
        # Estimate FLOPs and memory access
        flops = self._estimate_flops(func, args)
        bytes_accessed = self._estimate_memory_access(args, output_shape)

        # Calculate arithmetic intensity
        arithmetic_intensity = flops / bytes_accessed if bytes_accessed > 0 else float("inf")

        # Determine bottleneck
        is_compute_bound = arithmetic_intensity > self.hw_specs.critical_intensity

        # Theoretical performance bounds
        compute_time = flops / self.hw_specs.peak_flops_bf16
        memory_time = bytes_accessed / self.hw_specs.hbm_bandwidth
        theoretical_time = max(compute_time, memory_time)

        # Measure actual performance
        compiled_func = jax.jit(func)

        # Warmup
        for _ in range(3):
            result = compiled_func(*args, **kwargs)
            if hasattr(result, "block_until_ready"):
                result.block_until_ready()

        # Benchmark
        start_time = time.time()
        for _ in range(10):
            result = compiled_func(*args, **kwargs)
            if hasattr(result, "block_until_ready"):
                result.block_until_ready()
        actual_time = (time.time() - start_time) / 10

        # Calculate efficiency
        efficiency = min(theoretical_time / actual_time, 1.0) if actual_time > 0 else 0.0
        utilization = min((flops / actual_time) / self.hw_specs.peak_flops_bf16, 1.0)

        analysis = {
            "arithmetic_intensity": arithmetic_intensity,
            "critical_intensity": self.hw_specs.critical_intensity,
            "bottleneck": "compute" if is_compute_bound else "memory",
            "theoretical_time_ms": theoretical_time * 1000,
            "actual_time_ms": actual_time * 1000,
            "efficiency": efficiency,
            "flops_utilization": utilization,
            "recommendations": self._generate_recommendations(
                arithmetic_intensity, efficiency, args
            ),
        }

        return analysis

    def _estimate_flops(self, func: Callable, args: Tuple) -> float:
        """Estimate FLOPs for common operations."""
        # Check for matrix multiplication
        if hasattr(func, "__name__"):
            if "matmul" in func.__name__ or "@" in str(func):
                if len(args) >= 2:
                    a, b = args[0], args[1]
                    if hasattr(a, "shape") and hasattr(b, "shape"):
                        # Standard matrix multiplication FLOPs
                        if len(a.shape) >= 2 and len(b.shape) >= 2:
                            m = a.shape[-2]
                            k = a.shape[-1]
                            n = b.shape[-1] if len(b.shape) >= 2 else b.shape[0]
                            batch_size = 1
                            for dim in a.shape[:-2]:
                                batch_size *= dim
                            return 2 * batch_size * m * k * n

        # Default estimation based on tensor operations
        total_elements = sum(x.size for x in args if hasattr(x, "size"))
        return total_elements * 2  # Rough estimate

    def _estimate_memory_access(self, args, output_shape: tuple | None = None) -> float:
        """Estimate memory access in bytes."""
        input_bytes = sum(
            x.size * x.dtype.itemsize for x in args if hasattr(x, "size") and hasattr(x, "dtype")
        )

        if output_shape:
            # Assume bfloat16 output
            output_bytes = int(jnp.prod(jnp.array(output_shape))) * 2
        else:
            # Conservative estimate
            output_bytes = input_bytes

        return float(input_bytes + output_bytes)

    def _generate_recommendations(
        self, arithmetic_intensity: float, efficiency: float, args: Tuple
    ) -> list[str]:
        """Generate optimization recommendations."""
        recommendations = []

        # Check arithmetic intensity
        if arithmetic_intensity < self.hw_specs.critical_intensity:
            recommendations.append(
                f"Memory-bound operation (intensity: {arithmetic_intensity:.1f} < "
                f"{self.hw_specs.critical_intensity}). Increase batch size or fuse operations."
            )

        # Check efficiency
        if efficiency < 0.7:
            recommendations.append(
                f"Low efficiency ({efficiency:.2f}). Check for shape misalignments "
                f"or suboptimal compilation."
            )

        # Check batch sizes
        for arg in args:
            if hasattr(arg, "shape") and len(arg.shape) > 0:
                batch_size = arg.shape[0]
                if batch_size < self.hw_specs.optimal_batch_size:
                    recommendations.append(
                        f"Small batch size ({batch_size} < "
                        f"{self.hw_specs.optimal_batch_size}). Increase for better utilization."
                    )
                    break

        # Check shape alignment
        for arg in args:
            if hasattr(arg, "shape"):
                for dim in arg.shape:
                    if (
                        self.hw_specs.preferred_tile_size
                        and dim % self.hw_specs.preferred_tile_size != 0
                    ):
                        recommendations.append(
                            f"Shape dimension {dim} not aligned to "
                            f"{self.hw_specs.preferred_tile_size}. "
                            f"Consider padding for better performance."
                        )
                        break

        if not recommendations:
            recommendations.append("Operation is well-optimized for this hardware.")

        return recommendations

    def find_optimal_batch_size(
        self, sample_input: jax.Array, target_hardware: str | None = None
    ) -> int:
        """Find optimal batch size for compute-bound operation.

        Args:
            sample_input: Sample input tensor
            target_hardware: Optional target hardware override

        Returns:
            Optimal batch size
        """
        hw_specs = (
            self.hw_specs
            if target_hardware is None
            else HARDWARE_SPECS.get(target_hardware, self.hw_specs)
        )
        return hw_specs.optimal_batch_size

    def optimize_for_arithmetic_intensity(
        self, operation: Callable, target_intensity: float = 240
    ) -> Callable:
        """Optimize operation for target arithmetic intensity.

        Args:
            operation: Operation to optimize
            target_intensity: Target arithmetic intensity

        Returns:
            Optimized operation
        """

        def optimized_operation(*args, **kwargs):
            # Estimate current arithmetic intensity
            total_flops = sum(x.size * 2 for x in args if hasattr(x, "size"))
            total_bytes = sum(x.size * x.dtype.itemsize for x in args if hasattr(x, "size"))

            current_intensity = total_flops / total_bytes if total_bytes > 0 else 0

            if current_intensity < target_intensity and args:
                # Try to increase batch size
                if hasattr(args[0], "shape") and len(args[0].shape) > 0:
                    scaling_factor = max(1, int(target_intensity / max(current_intensity, 1)))
                    scaling_factor = min(scaling_factor, 8)  # Limit scaling

                    # Create larger batch by repeating
                    scaled_args = list(args)
                    original_shape = args[0].shape
                    batch_size = original_shape[0]

                    # Repeat along batch dimension
                    scaled_args[0] = jnp.repeat(args[0], scaling_factor, axis=0)

                    # Apply operation
                    result = operation(*scaled_args, **kwargs)

                    # Take first part of result
                    return result[:batch_size]

            return operation(*args, **kwargs)

        return optimized_operation

    def optimize_shapes(self, tensors: list[jax.Array]) -> list[jax.Array]:
        """Optimize tensor shapes for hardware.

        Args:
            tensors: List of tensors to optimize

        Returns:
            List of optimized tensors
        """
        optimized = []
        target_multiple = self.hw_specs.preferred_tile_size

        for tensor in tensors:
            if not hasattr(tensor, "shape"):
                optimized.append(tensor)
                continue

            shape = list(tensor.shape)
            new_shape = shape.copy()
            padding_needed = False

            # Optimize dimensions based on hardware preferences
            # For TPU: pad all dimensions to 128
            # For GPU: pad to 16 multiples
            for i in range(len(shape)):
                dim = shape[i]
                # Don't pad dimensions that are already multiples of smaller units
                # e.g., 64 is already a good size for many operations
                if self.hardware_name == "tpu_v5e":
                    # For TPU, be aggressive with padding for matrix ops
                    if dim % target_multiple != 0 and dim < target_multiple:
                        new_dim = target_multiple
                        new_shape[i] = new_dim
                        padding_needed = True
                    elif dim % target_multiple != 0 and dim > target_multiple:
                        new_dim = ((dim + target_multiple - 1) // target_multiple) * target_multiple
                        new_shape[i] = new_dim
                        padding_needed = True
                elif dim % target_multiple != 0:
                    # For GPU/other, pad to nearest multiple
                    new_dim = ((dim + target_multiple - 1) // target_multiple) * target_multiple
                    new_shape[i] = new_dim
                    padding_needed = True

            if padding_needed:
                # Apply padding
                pad_widths = []
                for old_dim, new_dim in zip(shape, new_shape):
                    pad_widths.append((0, new_dim - old_dim))

                padded_tensor = jnp.pad(tensor, pad_widths, mode="constant", constant_values=0)
                optimized.append(padded_tensor)
            else:
                optimized.append(tensor)

        return optimized

    def cast_to_optimal_precision(self, tensors: list[jax.Array]) -> list[jax.Array]:
        """Cast tensors to optimal precision for hardware.

        Args:
            tensors: List of tensors to cast

        Returns:
            List of casted tensors
        """
        # Use bfloat16 for TPU and modern GPUs
        target_dtype = jnp.bfloat16

        casted = []
        for tensor in tensors:
            if hasattr(tensor, "astype"):
                casted.append(tensor.astype(target_dtype))
            else:
                casted.append(tensor)

        return casted
