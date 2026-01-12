"""Advanced profiling system for Datarax pipelines.

This module provides detailed profiling capabilities including GPU memory
tracking, optimization suggestions, and detailed performance analysis, adhering
to JAX, Flax NNX, and Grain best practices.
"""

import gc
import json
import os
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

import jax
import jax.numpy as jnp
import jax.profiler
import numpy as np

from datarax.monitoring.metrics import MetricsCollector
from datarax.performance.roofline import RooflineAnalyzer


@dataclass
class ProfileResult:
    """Result of an advanced profiling session.

    Attributes:
        timing_metrics: Dictionary of timing measurements
        memory_metrics: Dictionary of memory usage measurements
        gpu_metrics: Dictionary of GPU-specific metrics
        optimization_suggestions: List of optimization recommendations
        metadata: Additional profiling metadata
        timestamp: When the profiling was performed
    """

    timing_metrics: dict[str, float]
    memory_metrics: dict[str, float]
    gpu_metrics: dict[str, float]
    optimization_suggestions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert ProfileResult to dictionary for serialization."""
        return {
            "timing_metrics": self.timing_metrics,
            "memory_metrics": self.memory_metrics,
            "gpu_metrics": self.gpu_metrics,
            "optimization_suggestions": self.optimization_suggestions,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    def save(self, filepath: str | Path) -> None:
        """Save profile result to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, filepath: str | Path) -> "ProfileResult":
        """Load profile result from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(**data)


class AdaptiveOperation:
    """Helper for hardware-adaptive operations (from JAX Guide)."""

    def __init__(self):
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
                if len(opt_shape) > abs(i):
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
            # Note: We assume ds is a MapDataset or IterDataset that Grain supports
            # Since pick_performance_config takes the dataset as input to sample from it
            try:
                performance_config = grain.experimental.pick_performance_config(
                    ds=ds,
                    ram_budget_mb=ram_budget_mb,
                    max_workers=None,  # Use all available
                )

                # Apply optimizations
                # Check if we can apply them (depends on dataset type)
                if hasattr(ds, "to_iter_dataset") and performance_config.read_options:
                    ds = ds.to_iter_dataset(read_options=performance_config.read_options)

                if hasattr(ds, "mp_prefetch") and performance_config.multiprocessing_options:
                    ds = ds.mp_prefetch(performance_config.multiprocessing_options)

                return ds
            except Exception as e:
                warnings.warn(f"Failed to apply Grain optimization: {e}")
                return ds

        except ImportError:
            # warnings.warn("Grain not installed. Skipping Grain optimization.")
            return ds


class GPUMemoryProfiler:
    """GPU memory profiling and optimization suggestions."""

    def __init__(self):
        """Initialize GPU memory profiler."""
        try:
            self.has_gpu = len(jax.devices("gpu")) > 0
        except (RuntimeError, ValueError):
            # Happens if 'gpu' backend is not available/initialized
            self.has_gpu = False
        if not self.has_gpu:
            # warnings.warn("No GPU devices detected. GPU profiling will be limited.")
            pass

    def get_memory_usage(self) -> dict[str, float]:
        """Get current GPU memory usage statistics."""
        if not self.has_gpu:
            return {"gpu_memory_used_mb": 0.0, "gpu_memory_total_mb": 0.0}

        try:
            # try to access detailed memory stats first which is available
            # on newer JAX versions (0.4.14+) for some backends
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

            # Fallback to xla_bridge or nvidia-smi specific logic
            devices = jax.devices("gpu")
            total_used = 0.0
            total_available = 0.0

            for device in devices:
                # Use JAX's memory info if available
                if hasattr(jax.lib.xla_bridge, "get_memory_info"):
                    memory_info = jax.lib.xla_bridge.get_memory_info(device)
                    total_used += memory_info.bytes_in_use / 1024 / 1024  # Convert to MB
                    total_available += memory_info.bytes_limit / 1024 / 1024
                else:
                    # Fallback: estimate based on device memory
                    # This is approximate as JAX doesn't always expose detailed memory info
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
            # warnings.warn(f"Could not get GPU memory info: {e}")
            return {"gpu_memory_used_mb": 0.0, "gpu_memory_total_mb": 0.0}

    def analyze_memory_pattern(self, measurements: list[dict[str, float]]) -> list[str]:
        """Analyze memory usage patterns and provide optimization suggestions."""
        if not measurements:
            return []

        suggestions: list[str] = []

        # Extract memory usage values
        usage_values = [m.get("gpu_memory_used_mb", 0) for m in measurements]
        utilization_values = [m.get("gpu_memory_utilization", 0) for m in measurements]

        if not usage_values:
            return suggestions

        # Check for memory leaks (consistent upward trend)
        if len(usage_values) >= 3:
            trend = np.polyfit(range(len(usage_values)), usage_values, 1)[0]
            if trend > 10:  # More than 10MB increase per measurement
                suggestions.append(
                    "Potential memory leak detected. Consider using JAX's garbage collection "
                    "or clearing unused variables in long-running pipelines."
                )

        # Check for high memory utilization
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
        # Get baseline memory
        baseline_memory = self._get_system_memory()

        # Run pipeline and track memory
        gc.collect()  # Clear any existing garbage
        self._get_system_memory()

        try:
            result = pipeline_fn(sample_data)
            # Force computation if result is lazy
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

        # Calculate memory metrics
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
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            # Fallback: use gc stats if psutil not available
            return sum(len(obj) for obj in gc.get_objects() if hasattr(obj, "__len__")) / 1024

    def _generate_memory_suggestions(self, peak_usage: float, retained_memory: float) -> list[str]:
        """Generate memory optimization suggestions."""
        suggestions: list[str] = []

        if peak_usage > 1000:  # More than 1GB peak usage
            suggestions.append(
                "High memory usage detected. Consider processing data in smaller batches "
                "or using JAX's device_put with sharding for large arrays."
            )

        efficiency = (peak_usage - retained_memory) / peak_usage if peak_usage > 0 else 1.0
        if efficiency < 0.7:  # Less than 70% memory is freed
            suggestions.append(
                "Low memory efficiency detected. Consider using explicit del statements "
                "for large temporary arrays or using JAX's compilation cache clearing."
            )

        return suggestions


@dataclass
class ProfilerConfig:
    """Configuration for AdvancedProfiler."""

    warmup_steps: int = 5
    measure_steps: int = 20
    enable_gpu_profiling: bool = True
    enable_memory_profiling: bool = True
    enable_roofline_analysis: bool = False
    enable_trace: bool = False
    trace_dir: str = os.path.join(tempfile.gettempdir(), "jaxrouter_traces")
    enable_pgle: bool = False  # Profile Guided Latency Estimator


class AdvancedProfiler:
    """Advanced profiling system with detailed performance analysis.

    Adheres to JAX, Flax NNX, and Grain best practices for profiling.
    """

    def __init__(
        self,
        enable_gpu_profiling: bool = True,
        enable_memory_profiling: bool = True,
        enable_roofline_analysis: bool = False,
        config: ProfilerConfig | None = None,
    ):
        """Initialize advanced profiler."""
        if config is not None:
            self.config = config
        else:
            self.config = ProfilerConfig(
                enable_gpu_profiling=enable_gpu_profiling,
                enable_memory_profiling=enable_memory_profiling,
                enable_roofline_analysis=enable_roofline_analysis,
            )

        # Initialize sub-profilers
        self.gpu_profiler = GPUMemoryProfiler() if self.config.enable_gpu_profiling else None
        self.memory_optimizer = MemoryOptimizer() if self.config.enable_memory_profiling else None
        self.roofline_analyzer = (
            RooflineAnalyzer() if self.config.enable_roofline_analysis else None
        )
        self.metrics_collector = MetricsCollector()
        self.adaptive_op = AdaptiveOperation()

        self._memory_snapshots: list[dict[str, float]] = []

        if self.config.enable_pgle or self.config.enable_gpu_profiling:
            self._configure_xla_flags()

    def _configure_xla_flags(self):
        """Configure Optimized XLA Flags for GPU Performance."""
        # See jax_guide.md 5.1.1 Essential XLA Flags
        flags = [
            "--xla_gpu_enable_latency_hiding_scheduler=true",
            "--xla_gpu_triton_gemm_any=True",
            "--xla_gpu_all_gather_combine_threshold_bytes=134217728",  # 128MB
            "--xla_gpu_reduce_scatter_combine_threshold_bytes=134217728",  # 128MB
        ]

        current_flags = os.environ.get("XLA_FLAGS", "")
        # Only add flags not already present to avoid duplication/conflicts
        for flag in flags:
            if flag.split("=")[0] not in current_flags:
                current_flags += " " + flag

        os.environ["XLA_FLAGS"] = current_flags.strip()
        # Note: Auto-PGLE might require JAX config flags in newer versions

    def profile(self, func: Callable[[], Any], name: str | None = None) -> ProfileResult:
        """Profile a zero-argument function using the configured settings."""
        wrapper = lambda _: func()
        return self.profile_pipeline(pipeline_fn=wrapper, sample_data=None, name=name)

    def profile_pipeline(
        self,
        pipeline_fn: Callable,
        sample_data: Any,
        num_iterations: int | None = None,
        warmup_iterations: int | None = None,
        collect_memory_snapshots: bool | None = None,
        name: str | None = None,
    ) -> ProfileResult:
        """Profile a pipeline function suited for JAX/Flax/Grain pipelines."""

        # Defaults from config if not provided
        iterations = num_iterations if num_iterations is not None else self.config.measure_steps
        warmup = warmup_iterations if warmup_iterations is not None else self.config.warmup_steps
        do_memory = (
            collect_memory_snapshots
            if collect_memory_snapshots is not None
            else self.config.enable_memory_profiling
        )
        name = name or "pipeline_profile"

        timing_metrics: dict[str, float] = {}
        memory_metrics: dict[str, float] = {}
        gpu_metrics: dict[str, float] = {}
        suggestions: list[str] = []

        # 1. Warmup (Critical for JAX compilation)
        # JAX Guide: "Warmup before measuring - First run includes compilation time"
        for _ in range(warmup):
            try:
                result = pipeline_fn(sample_data)
                self._block_until_ready(result)
            except Exception:
                pass  # Ignore warmup errors (e.g. init issues)

        # Clear compilation cache/garbage for clean measurement
        jax.clear_caches()
        gc.collect()

        # 2. Trace execution if enabled (JAX Profiler)
        if self.config.enable_trace:
            jax.profiler.start_trace(self.config.trace_dir)

        # 3. Main Benchmark Loop
        iteration_times = []
        for i in range(iterations):
            if do_memory:
                if self.gpu_profiler:
                    self._memory_snapshots.append(self.gpu_profiler.get_memory_usage())
                if i == iterations - 1:  # Capture detailed snapshot on last iteration
                    try:
                        jax.profiler.save_device_memory_profile(
                            os.path.join(self.config.trace_dir, "memory.prof")
                        )
                    except Exception:
                        pass

            start_time = time.perf_counter()
            try:
                result = pipeline_fn(sample_data)
                self._block_until_ready(result)  # JAX Guide: "Always use .block_until_ready()"
            except Exception as e:
                timing_metrics["error"] = str(e)
                break
            end_time = time.perf_counter()
            iteration_times.append(end_time - start_time)

        if self.config.enable_trace:
            jax.profiler.stop_trace()
            suggestions.append(f"Trace saved to {self.config.trace_dir}. View with TensorBoard.")
        if do_memory:
            prof_path = os.path.join(self.config.trace_dir, "memory.prof")
            suggestions.append(f"Device memory profile saved to {prof_path}. View with pprof.")

        # 4. Analysis
        if iteration_times:
            self._analyze_timing(iteration_times, timing_metrics)

        if self.memory_optimizer:
            mem_analysis = self.memory_optimizer.analyze_pipeline_memory(pipeline_fn, sample_data)
            memory_metrics.update(mem_analysis)
            suggestions.extend(mem_analysis.get("suggestions", []))

        if self.gpu_profiler:
            gpu_metrics.update(self.gpu_profiler.get_memory_usage())
            suggestions.extend(self.gpu_profiler.analyze_memory_pattern(self._memory_snapshots))

        if self.roofline_analyzer:
            self._run_roofline_analysis(pipeline_fn, sample_data, gpu_metrics, suggestions)

        # Add hardware optimization suggestions
        hw_config = self.adaptive_op.hw_config
        suggestions.append(
            f"Running on {hw_config['platform']}. Recommended Precision: {hw_config['precision']}"
        )
        suggestions.append(
            f"Target Batch Size > {hw_config['critical_batch_size']} for optimal efficiency."
        )

        return ProfileResult(
            timing_metrics=timing_metrics,
            memory_metrics=memory_metrics,
            gpu_metrics=gpu_metrics,
            optimization_suggestions=suggestions,
            metadata={
                "iterations": iterations,
                "warmup": warmup,
                "backend": jax.default_backend(),
                "device": str(jax.devices()[0]) if jax.devices() else "unknown",
                "hardware_config": str(hw_config),
            },
        )

    def _block_until_ready(self, result: Any):
        """Recursively block until JAX arrays are ready."""
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        elif isinstance(result, list | tuple):
            for item in jax.tree_util.tree_leaves(result):
                if hasattr(item, "block_until_ready"):
                    item.block_until_ready()

    def _analyze_timing(self, times: List[float], metrics: Dict[str, float]):
        """Compute detailed timing statistics."""
        times_np = np.array(times)
        metrics.update(
            {
                "mean_time_s": float(np.mean(times_np)),
                "std_time_s": float(np.std(times_np)),
                "min_time_s": float(np.min(times_np)),
                "max_time_s": float(np.max(times_np)),
                "p50_time_s": float(np.percentile(times_np, 50)),
                "p95_time_s": float(np.percentile(times_np, 95)),
                "p99_time_s": float(np.percentile(times_np, 99)),
                "iterations_per_second": 1.0 / float(np.mean(times_np))
                if np.mean(times_np) > 0
                else 0.0,
                "total_time_s": float(np.sum(times_np)),
            }
        )

    def _run_roofline_analysis(self, func, sample_data, metrics, suggestions):
        """Run roofline analysis if enabled."""
        try:
            args = (sample_data,)
            results = self.roofline_analyzer.analyze_operation(func, *args)
            for k, v in results.items():
                if k == "recommendations":
                    suggestions.extend(v)
                else:
                    metrics[f"roofline_{k}"] = v
        except Exception as e:
            metrics["roofline_error"] = str(e)

    def clear_snapshots(self) -> None:
        """Clear collected memory snapshots."""
        self._memory_snapshots.clear()
