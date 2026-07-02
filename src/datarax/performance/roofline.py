"""Roofline analysis for hardware-aware performance optimization.

This module provides tools for analyzing operations based on the roofline model
to identify performance bottlenecks and suggest optimizations.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from datarax.performance.synchronization import block_until_ready_tree


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
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

    def __init__(self, hardware: str = "auto") -> None:
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

        if backend == "gpu":
            # Try to detect GPU type
            device = jax.devices()[0]
            device_kind = str(device).lower()
            if "h100" in device_kind:
                return "h100"
            if "a100" in device_kind:
                return "a100"
            return "h100"  # Default to newer GPU

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
            block_until_ready_tree(result)

        # Benchmark
        start_time = time.time()
        for _ in range(10):
            result = compiled_func(*args, **kwargs)
            block_until_ready_tree(result)
        actual_time = (time.time() - start_time) / 10

        # Calculate efficiency
        efficiency = min(theoretical_time / actual_time, 1.0) if actual_time > 0 else 0.0
        utilization = min((flops / actual_time) / self.hw_specs.peak_flops_bf16, 1.0)

        return {
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

    def _estimate_flops(self, func: Callable, args: tuple) -> float:
        """Estimate FLOPs for common operations."""
        matmul_flops = self._matmul_flops(func, args)
        if matmul_flops is not None:
            return matmul_flops

        # Default estimation based on tensor operations
        total_elements = sum(x.size for x in args if hasattr(x, "size"))
        return total_elements * 2  # Rough estimate

    @staticmethod
    def _matmul_flops(func: Callable, args: tuple) -> float | None:
        """Return matrix-multiplication FLOPs, or ``None`` if ``func`` is not a matmul.

        Args:
            func: Callable being profiled.
            args: Positional arguments the callable was invoked with.

        Returns:
            Standard batched matmul FLOP count, or ``None`` when the operation is
            not a recognizable 2-D+ matrix multiplication.
        """
        is_matmul = hasattr(func, "__name__") and ("matmul" in func.__name__ or "@" in str(func))
        if not is_matmul or len(args) < 2:
            return None

        a, b = args[0], args[1]
        if not (hasattr(a, "shape") and hasattr(b, "shape")):
            return None
        if len(a.shape) < 2 or len(b.shape) < 2:
            return None

        m, k, n = a.shape[-2], a.shape[-1], b.shape[-1]
        batch_size = 1
        for dim in a.shape[:-2]:
            batch_size *= dim
        return 2 * batch_size * m * k * n

    def _estimate_memory_access(self, args: Any, output_shape: tuple | None = None) -> float:
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
        self, arithmetic_intensity: float, efficiency: float, args: tuple
    ) -> list[str]:
        """Generate optimization recommendations."""
        recommendations: list[str] = []
        self._append_intensity_recommendation(recommendations, arithmetic_intensity)
        self._append_efficiency_recommendation(recommendations, efficiency)
        self._append_batch_size_recommendation(recommendations, args)
        self._append_tile_alignment_recommendation(recommendations, args)
        if recommendations:
            return recommendations
        return ["Operation is well-optimized for this hardware."]

    def _append_intensity_recommendation(
        self, recommendations: list[str], arithmetic_intensity: float
    ) -> None:
        """Append recommendation for memory-bound operations."""
        if arithmetic_intensity >= self.hw_specs.critical_intensity:
            return
        recommendations.append(
            f"Memory-bound operation (intensity: {arithmetic_intensity:.1f} < "
            f"{self.hw_specs.critical_intensity}). Increase batch size or fuse operations."
        )

    def _append_efficiency_recommendation(
        self, recommendations: list[str], efficiency: float
    ) -> None:
        """Append recommendation for low utilization."""
        if efficiency >= 0.7:
            return
        recommendations.append(
            f"Low efficiency ({efficiency:.2f}). Check for shape misalignments "
            f"or suboptimal compilation."
        )

    def _append_batch_size_recommendation(self, recommendations: list[str], args: tuple) -> None:
        """Append recommendation when the effective batch size is too small."""
        for arg in args:
            if not hasattr(arg, "shape") or len(arg.shape) == 0:
                continue
            batch_size = arg.shape[0]
            if batch_size < self.hw_specs.optimal_batch_size:
                recommendations.append(
                    f"Small batch size ({batch_size} < "
                    f"{self.hw_specs.optimal_batch_size}). Increase for better utilization."
                )
                return

    def _append_tile_alignment_recommendation(
        self, recommendations: list[str], args: tuple
    ) -> None:
        """Append recommendation for misaligned tensor dimensions."""
        tile_size = self.hw_specs.preferred_tile_size
        if tile_size is None:
            return
        for arg in args:
            if not hasattr(arg, "shape"):
                continue
            for dim in arg.shape:
                if dim % tile_size != 0:
                    recommendations.append(
                        f"Shape dimension {dim} not aligned to {tile_size}. "
                        "Consider padding for better performance."
                    )
                    return

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
        del sample_input
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

        def optimized_operation(*args: Any, **kwargs: Any) -> Any:
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
        target_multiple = self.hw_specs.preferred_tile_size
        return [self._optimize_tensor_shape(tensor, target_multiple) for tensor in tensors]

    @staticmethod
    def _round_up_to_multiple(value: int, multiple: int) -> int:
        """Round ``value`` up to the nearest multiple of ``multiple``."""
        return ((value + multiple - 1) // multiple) * multiple

    def _optimize_tensor_shape(self, tensor: jax.Array, target_multiple: int) -> jax.Array:
        """Pad a single tensor so every dimension is a multiple of ``target_multiple``.

        Any dimension already aligned is left untouched; a tensor without a ``shape``
        attribute is returned unchanged.

        Args:
            tensor: Tensor to align.
            target_multiple: Hardware-preferred tile size to align dimensions to.

        Returns:
            The padded tensor, or the original when no padding is required.
        """
        if not hasattr(tensor, "shape"):
            return tensor

        shape = list(tensor.shape)
        # Rounding up already covers the "pad small dims up to one tile" case, so TPU
        # and GPU share identical logic once the tile size is chosen.
        new_shape = [self._round_up_to_multiple(dim, target_multiple) for dim in shape]
        if new_shape == shape:
            return tensor

        pad_widths = [(0, new - old) for old, new in zip(shape, new_shape, strict=False)]
        return jnp.pad(tensor, pad_widths, mode="constant", constant_values=0)

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
