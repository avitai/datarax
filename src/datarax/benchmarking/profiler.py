"""Hardware-adaptive profiling components for Datarax pipelines.

Provides GPU memory profiling, hardware-adaptive operations, and memory
optimization, adhering to JAX, Flax NNX, and Grain best practices.
"""

import gc
import warnings
from typing import Any
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np


class AdaptiveOperation:
    """Helper for hardware-adaptive operations (from JAX Guide)."""

    def __init__(self):
        """Initialize AdaptiveOperation with auto-detected hardware config."""
        self.hw_config = self.detect_hardware_and_optimize()
        self.backend = jax.default_backend()

    def detect_hardware_and_optimize(self) -> dict[str, Any]:
        """Automatically detect hardware and apply optimal settings."""
        backend = jax.default_backend()

        # Default config
        config = {
            "precision": jnp.float32,
            "tile_size": 64,
            "critical_batch_size": 32,
            "memory_layout": "row_major",
            "use_vmem_optimization": False,
            "platform": "cpu",
        }

        if backend == "tpu":
            config.update(
                {
                    "precision": jnp.bfloat16,
                    "tile_size": 128,
                    "critical_batch_size": 240,
                    "memory_layout": "row_major",  # TPU prefers padding to 128
                    "use_vmem_optimization": True,
                    "platform": "tpu",
                }
            )
        elif backend == "gpu":
            try:
                devices = jax.devices()
                device_kind = getattr(devices[0], "device_kind", "unknown").lower()
                if "h100" in device_kind or "a100" in device_kind:
                    config.update(
                        {
                            "precision": jnp.bfloat16,
                            "tile_size": 16,  # TensorCore
                            "critical_batch_size": 298,  # H100 specific
                            "platform": "gpu_modern",
                        }
                    )
                else:
                    config.update(
                        {
                            "precision": jnp.float32,
                            "tile_size": 32,
                            "critical_batch_size": 128,
                            "platform": "gpu_legacy",
                        }
                    )
            except Exception:
                pass

        return config

    def optimize_shapes(self, *shapes):
        """Optimize tensor shapes for current hardware."""
        tile_size = self.hw_config["tile_size"]
        optimized_shapes = []

        for shape in shapes:
            opt_shape = list(shape)
            # Optimize last two dimensions for matrix operations
            for i in [-2, -1]:
                if len(opt_shape) >= abs(i):
                    dim = opt_shape[i]
                    if dim % tile_size != 0:
                        # Pad to next multiple of tile_size
                        opt_shape[i] = ((dim + tile_size - 1) // tile_size) * tile_size

            optimized_shapes.append(tuple(opt_shape))

        return optimized_shapes

    def optimize_grain_dataset(self, ds: Any, ram_budget_mb: int = 4096) -> Any:
        """Apply Grain auto-optimization based on RAM budget.

        Uses grain.experimental.pick_performance_config if available.
        """
        try:
            import grain.python as grain

            if not hasattr(grain, "experimental") or not hasattr(
                grain.experimental, "pick_performance_config"
            ):
                warnings.warn(
                    "grain.experimental.pick_performance_config not found. "
                    "Skipping Grain optimization."
                )
                return ds

            # Get auto-tuned configuration
            try:
                performance_config = grain.experimental.pick_performance_config(
                    ds=ds,
                    ram_budget_mb=ram_budget_mb,
                    max_workers=None,  # Use all available
                )

                # Apply optimizations
                if hasattr(ds, "to_iter_dataset") and performance_config.read_options:
                    ds = ds.to_iter_dataset(read_options=performance_config.read_options)

                if hasattr(ds, "mp_prefetch") and performance_config.multiprocessing_options:
                    ds = ds.mp_prefetch(performance_config.multiprocessing_options)

                return ds
            except Exception as e:
                warnings.warn(f"Failed to apply Grain optimization: {e}")
                return ds

        except ImportError:
            return ds


class GPUMemoryProfiler:
    """GPU memory profiling and optimization suggestions."""

    def __init__(self):
        """Initialize GPU memory profiler."""
        try:
            self.has_gpu = len(jax.devices("gpu")) > 0
        except (RuntimeError, ValueError):
            self.has_gpu = False

    def get_memory_usage(self) -> dict[str, float]:
        """Get current GPU memory usage statistics."""
        if not self.has_gpu:
            return {"gpu_memory_used_mb": 0.0, "gpu_memory_total_mb": 0.0}

        try:
            try:
                device = jax.devices("gpu")[0]
                if hasattr(device, "memory_stats"):
                    stats = device.memory_stats()
                    if stats:
                        used_mb = stats.get("bytes_in_use", 0) / 1024 / 1024
                        limit_mb = (
                            stats.get("bytes_limit", 0) / 1024 / 1024
                            if stats.get("bytes_limit")
                            else 0
                        )
                        return {
                            "gpu_memory_used_mb": used_mb,
                            "gpu_memory_total_mb": limit_mb,
                            "gpu_memory_utilization": used_mb / limit_mb if limit_mb > 0 else 0,
                            "num_gpu_devices": len(jax.devices("gpu")),
                            "pool_bytes_mb": stats.get("pool_bytes", 0) / 1024 / 1024,
                        }
            except Exception:
                pass

            devices = jax.devices("gpu")
            total_used = 0.0
            total_available = 0.0

            for device in devices:
                if hasattr(jax.lib.xla_bridge, "get_memory_info"):
                    memory_info = jax.lib.xla_bridge.get_memory_info(device)
                    total_used += memory_info.bytes_in_use / 1024 / 1024
                    total_available += memory_info.bytes_limit / 1024 / 1024
                else:
                    total_available += 8192  # Assume 8GB per GPU as fallback

            return {
                "gpu_memory_used_mb": total_used,
                "gpu_memory_total_mb": total_available,
                "gpu_memory_utilization": total_used / total_available
                if total_available > 0
                else 0.0,
                "num_gpu_devices": len(devices),
            }
        except Exception:
            return {"gpu_memory_used_mb": 0.0, "gpu_memory_total_mb": 0.0}

    def get_utilization(self) -> float:
        """Get current GPU utilization percentage.

        Used by ResourceMonitor for background sampling.
        """
        mem = self.get_memory_usage()
        return mem.get("gpu_memory_utilization", 0.0) * 100

    def analyze_memory_pattern(self, measurements: list[dict[str, float]]) -> list[str]:
        """Analyze memory usage patterns and provide optimization suggestions."""
        if not measurements:
            return []

        suggestions: list[str] = []

        usage_values = [m.get("gpu_memory_used_mb", 0) for m in measurements]
        utilization_values = [m.get("gpu_memory_utilization", 0) for m in measurements]

        if not usage_values:
            return suggestions

        if len(usage_values) >= 3:
            trend = np.polyfit(range(len(usage_values)), usage_values, 1)[0]
            if trend > 10:
                suggestions.append(
                    "Potential memory leak detected. Consider using JAX's garbage collection "
                    "or clearing unused variables in long-running pipelines."
                )

        max_utilization = max(utilization_values) if utilization_values else 0
        avg_utilization = np.mean(utilization_values) if utilization_values else 0

        if max_utilization > 0.9:
            suggestions.append(
                "High GPU memory utilization detected (>90%). Consider reducing batch size "
                "or using gradient checkpointing to reduce memory usage."
            )
        elif avg_utilization > 0.8:
            suggestions.append(
                "Consistently high GPU memory usage (>80%). Monitor for potential "
                "out-of-memory errors."
            )

        return suggestions


class MemoryOptimizer:
    """Memory optimization recommendations and automatic optimizations."""

    def __init__(self):
        """Initialize memory optimizer."""
        pass

    def analyze_pipeline_memory(self, pipeline_fn: Callable, sample_data: Any) -> dict[str, Any]:
        """Analyze memory usage of a pipeline function."""
        baseline_memory = self._get_system_memory()

        gc.collect()
        self._get_system_memory()

        try:
            result = pipeline_fn(sample_data)
            if hasattr(result, "block_until_ready"):
                result.block_until_ready()
            elif isinstance(result, list | tuple):
                for item in jax.tree_util.tree_leaves(result):
                    if hasattr(item, "block_until_ready"):
                        item.block_until_ready()
        except Exception as e:
            return {"error": str(e), "memory_analysis": None}

        end_memory = self._get_system_memory()
        gc.collect()
        post_gc_memory = self._get_system_memory()

        peak_usage = end_memory - baseline_memory
        retained_memory = post_gc_memory - baseline_memory

        analysis = {
            "baseline_memory_mb": baseline_memory,
            "peak_memory_mb": end_memory,
            "peak_usage_mb": peak_usage,
            "retained_memory_mb": retained_memory,
            "memory_efficiency": (peak_usage - retained_memory) / peak_usage
            if peak_usage > 0
            else 1.0,
            "suggestions": self._generate_memory_suggestions(peak_usage, retained_memory),
        }

        return analysis

    def _get_system_memory(self) -> float:
        """Get current system memory usage in MB."""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return sum(len(obj) for obj in gc.get_objects() if hasattr(obj, "__len__")) / 1024

    def _generate_memory_suggestions(self, peak_usage: float, retained_memory: float) -> list[str]:
        """Generate memory optimization suggestions."""
        suggestions: list[str] = []

        if peak_usage > 1000:
            suggestions.append(
                "High memory usage detected. Consider processing data in smaller batches "
                "or using JAX's device_put with sharding for large arrays."
            )

        efficiency = (peak_usage - retained_memory) / peak_usage if peak_usage > 0 else 1.0
        if efficiency < 0.7:
            suggestions.append(
                "Low memory efficiency detected. Consider using explicit del statements "
                "for large temporary arrays or using JAX's compilation cache clearing."
            )

        return suggestions
