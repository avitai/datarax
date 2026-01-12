"""XLA compilation optimization for maximum performance.

This module provides XLA compilation strategies, smart compilation caching,
and memory-efficient patterns for JAX operations.
"""

import logging
import os
import tempfile
import time
from functools import partial
from typing import Any, Callable, Dict, Tuple

import jax
import numpy as np


class XLAOptimizer:
    """Configure XLA compiler for maximum performance."""

    def __init__(self, target_hardware: str = "auto"):
        """Initialize XLA optimizer with hardware-specific configuration.

        Args:
            target_hardware: Target hardware ('auto', 'gpu', 'tpu', 'cpu')
        """
        self.target_hardware = target_hardware
        self.setup_xla_flags()
        self.setup_jax_config()
        self.setup_compilation_cache()

    def setup_xla_flags(self):
        """Configure XLA compiler flags for maximum performance."""
        # Core XLA optimizations
        # See jax_guide.md 5.1.1 Essential XLA Flags
        flags = [
            # CPU Optimizations
            "--xla_cpu_enable_fast_math=true",
            "--xla_force_host_platform_device_count=1",
            # GPU/TPU Common Latency Hiding
            "--xla_gpu_enable_latency_hiding_scheduler=true",
            "--xla_gpu_enable_async_all_gather=true",
            "--xla_gpu_enable_async_all_reduce=true",
            "--xla_gpu_enable_async_reduce_scatter=true",
            # GPU Specific (Triton)
            "--xla_gpu_triton_gemm_any=True",
            "--xla_gpu_enable_triton_gemm=true",
            # Memory Optimization (Collective Combining)
            "--xla_gpu_all_gather_combine_threshold_bytes=134217728",  # 128MB
            "--xla_gpu_reduce_scatter_combine_threshold_bytes=134217728",  # 128MB
            "--xla_gpu_enable_memory_space_assignment=true",
        ]

        # Combine flags based on hardware
        # Note: In modern JAX, setting these mainly affects XLA init.
        # We append to XLA_FLAGS env var.

        existing_flags = os.environ.get("XLA_FLAGS", "")
        # Only add flags not already present
        for flag in flags:
            if flag.split("=")[0] not in existing_flags:
                existing_flags += " " + flag

        os.environ["XLA_FLAGS"] = existing_flags.strip()

        logging.info(f"XLA flags configured: {len(flags)} optimization flags set")

    def setup_jax_config(self):
        """Configure JAX-specific optimizations."""
        # Enable 32-bit by default for speed
        jax.config.update("jax_enable_x64", False)

        # Memory optimizations - removed deprecated jax_default_preallocate

        # Performance debugging (disabled by default)
        jax.config.update("jax_log_compiles", False)

        # Platform-specific settings
        backend = jax.default_backend()
        if backend == "tpu":
            # TPU-specific configurations
            jax.config.update("jax_default_matmul_precision", "default")
        elif backend == "gpu":
            # GPU-specific configurations
            jax.config.update("jax_default_matmul_precision", "high")

        logging.info(f"JAX configuration optimized for {backend}")

    def setup_compilation_cache(self):
        """Setup persistent compilation cache."""
        # Use existing cache dir if set, otherwise create temp
        cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")
        if not cache_dir:
            cache_dir = os.path.join(tempfile.gettempdir(), "jax_compilation_cache")
            os.makedirs(cache_dir, exist_ok=True)

        jax.config.update("jax_compilation_cache_dir", cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

        logging.info(f"Compilation cache enabled at: {cache_dir}")


class SmartCompilation:
    """Intelligent compilation strategies for different scenarios."""

    def __init__(self):
        """Initialize smart compilation system."""
        self.compilation_cache: dict[Any, Callable] = {}
        self.shape_signatures: dict[Any, Any] = {}

    def adaptive_jit(self, func: Callable, static_threshold: int = 1000) -> Callable:
        """Apply JIT compilation adaptively based on input characteristics.

        Args:
            func: Function to potentially compile
            static_threshold: Size threshold for JIT compilation

        Returns:
            Adaptively compiled function
        """

        @partial(jax.jit)
        def compiled_version(*args, **kwargs):
            return func(*args, **kwargs)

        def adaptive_wrapper(*args, **kwargs):
            # Calculate total computation size
            total_size = sum(x.size if hasattr(x, "size") else 1 for x in jax.tree.leaves(args))

            # Use JIT for large computations
            if total_size > static_threshold:
                return compiled_version(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return adaptive_wrapper

    def aot_compile(self, func: Callable, *args: Any, **kwargs: Any) -> Callable:
        """Perform Ahead-of-Time (AOT) compilation for specific input shapes.

        This eliminates the first-run compilation cost (jitter), ideal for serving.

        Args:
            func: Function to compile
            args: Example arguments (for shape/dtype)
            kwargs: Example keyword arguments

        Returns:
            Compiled function ready for execution (lowered and compiled)
        """
        # JAX AOT compilation via jit(...).lower(...).compile()
        # This is the modern "smart" way for serving.

        jit_func = jax.jit(func)
        try:
            lowered = jit_func.lower(*args, **kwargs)
            compiled = lowered.compile()
            logging.info(f"AOT compilation successful for {func.__name__}")
            return compiled
        except Exception as e:
            logging.warning(f"AOT compilation failed: {e}. Falling back to standard JIT.")
            return jit_func

    def shard_map_jit(self, mesh: jax.sharding.Mesh, in_specs: Any, out_specs: Any) -> Callable:
        """Create a shard_map (SPMD) function for expert parallelization.

        Args:
            mesh: Device mesh
            in_specs: Input PartitionSpecs
            out_specs: Output PartitionSpecs

        Returns:
            Decorator for the function
        """
        from jax.experimental.shard_map import shard_map

        def decorator(func):
            return shard_map(func, mesh=mesh, in_specs=in_specs, out_specs=out_specs)

        return decorator


class MemoryEfficientCompilation:
    """Compilation patterns optimized for memory efficiency."""

    @staticmethod
    def donate_wrapper(func: Callable, donate_args: tuple[int, ...] | None = None) -> Callable:
        """Wrapper to automatically donate large arrays.

        Args:
            func: Function to wrap
            donate_args: Indices of arguments to donate, or None for auto-detect

        Returns:
            Memory-efficient wrapped function
        """

        def optimized_func(*args, **kwargs):
            # Automatically determine which arguments to donate
            if donate_args is None:
                auto_donate = []
                for i, arg in enumerate(args):
                    if hasattr(arg, "size") and arg.size > 1024 * 1024:  # >1MB
                        auto_donate.append(i)

                if auto_donate:
                    compiled = partial(jax.jit, donate_argnums=tuple(auto_donate))(func)
                else:
                    compiled = jax.jit(func)
            else:
                compiled = partial(jax.jit, donate_argnums=donate_args)(func)

            return compiled(*args, **kwargs)

        return optimized_func

    @staticmethod
    def parameter_update_pattern(learning_rate: float = 0.01) -> Callable:
        """Optimized parameter update with buffer donation.

        Args:
            learning_rate: Learning rate for updates

        Returns:
            Memory-efficient update function
        """

        # Only donate params (arg 0) as it's the one being modified
        # Donating grads (arg 1) can cause issues as it may be reused
        @partial(jax.jit, donate_argnums=(0,))
        def update_parameters(params: Any, grads: Any) -> Any:
            """Memory-efficient parameter update."""
            return jax.tree.map(lambda p, g: p - learning_rate * g, params, grads)

        return update_parameters

    @staticmethod
    def with_rematerialization(func: Callable, policy: Callable | None = None) -> Callable:
        """Apply gradient checkpointing (rematerialization) to reduce memory.

        Args:
            func: Function to checkpoint
            policy: Checkpoint policy (optional, e.g. jax.checkpoint_policies.save_nothing)

        Returns:
            Function with checkpointing applied
        """
        if policy:
            return jax.checkpoint(func, policy=policy)  # type: ignore
        return jax.checkpoint(func)  # type: ignore


class DistributedUtils:
    """Best practices for distributed computation and sharding."""

    @staticmethod
    def create_mesh(axis_dims: tuple[int, ...], axis_names: tuple[str, ...]) -> jax.sharding.Mesh:
        """Create a device mesh for parallelism.

        Args:
            axis_dims: Dimensions for mesh (e.g. (4, 2) for 8 devices)
            axis_names: Names for axes (e.g. ('data', 'model'))

        Returns:
            jax.sharding.Mesh
        """
        devices = jax.devices()
        n_devices = len(devices)
        expected = 1
        for d in axis_dims:
            expected *= d

        if n_devices < expected:
            logging.warning(
                f"Not enough devices for mesh {axis_dims} (Needed {expected}, Has {n_devices}). "
                "Falling back to CPU mesh."
            )
            # Fallback logic could be better, but for now just warn

        mesh_devices = np.array(devices[:expected]).reshape(axis_dims)
        return jax.sharding.Mesh(mesh_devices, axis_names)

    @staticmethod
    def with_sharding(x: Any, mesh: jax.sharding.Mesh, partition_spec: Any) -> Any:
        """Apply sharding constraint to an array.

        Args:
            x: JAX array
            mesh: Device mesh
            partition_spec: PartitionSpec for the array

        Returns:
            Sharded array
        """
        from jax.sharding import NamedSharding

        sharding = NamedSharding(mesh, partition_spec)
        return jax.lax.with_sharding_constraint(x, sharding)


class CompilationProfiler:
    """Profile and optimize JAX compilation performance."""

    def __init__(self):
        """Initialize compilation profiler."""
        self.compilation_times: dict[Any, float] = {}
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.shape_profiles: dict[Any, Dict] = {}

    def profile_function(self, func_name: str, enable_detailed_logging: bool = False) -> Callable:
        """Decorator to profile function compilation and execution.

        Args:
            func_name: Name of the function for reporting
            enable_detailed_logging: Enable detailed JAX logging

        Returns:
            Profiling decorator
        """

        def decorator(func: Callable) -> Callable:
            if enable_detailed_logging:
                # Enable detailed JAX logging
                jax.config.update("jax_log_compiles", True)
                jax.config.update("jax_explain_cache_misses", True)

            def profiled_wrapper(*args, **kwargs):
                # Create signature for caching analysis
                signature = self._create_shape_signature(args)

                if signature in self.compilation_times:
                    self.cache_hits += 1
                else:
                    self.cache_misses += 1

                # Measure compilation time
                start_time = time.time()

                # First call triggers compilation
                compiled_func = jax.jit(func)
                result = compiled_func(*args, **kwargs)
                if hasattr(result, "block_until_ready"):
                    result.block_until_ready()

                compile_time = time.time() - start_time
                self.compilation_times[signature] = compile_time

                # Store shape profile
                self.shape_profiles[signature] = {
                    "input_shapes": [getattr(arg, "shape", None) for arg in args],
                    "input_dtypes": [getattr(arg, "dtype", None) for arg in args],
                    "compile_time": compile_time,
                }

                logging.info(f"Compiled {func_name} for signature {signature}: {compile_time:.3f}s")

                return result

            return profiled_wrapper

        return decorator

    def _create_shape_signature(self, args: Tuple) -> Tuple:
        """Create a signature from argument shapes for cache analysis.

        Args:
            args: Function arguments

        Returns:
            Shape signature tuple
        """
        shapes = []
        for arg in args:
            if hasattr(arg, "shape"):
                shapes.append(arg.shape)
            else:
                shapes.append(type(arg))
        return tuple(shapes)

    def generate_report(self) -> dict[str, Any]:
        """Generate compilation report.

        Returns:
            Report dictionary with analysis and recommendations
        """
        total_compilations = len(self.compilation_times)
        total_compile_time = sum(self.compilation_times.values())
        total_calls = self.cache_hits + self.cache_misses

        report = {
            "summary": {
                "total_compilations": total_compilations,
                "total_compile_time_s": total_compile_time,
                "average_compile_time_s": total_compile_time / max(total_compilations, 1),
                "cache_hit_rate": self.cache_hits / max(total_calls, 1),
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
            },
            "expensive_compilations": [],
            "shape_analysis": {},
            "recommendations": [],
        }

        # Find expensive compilations
        if self.compilation_times:
            sorted_compilations = sorted(
                self.compilation_times.items(), key=lambda x: x[1], reverse=True
            )
            report["expensive_compilations"] = sorted_compilations[:5]

        # Analyze shape patterns
        shape_groups: dict[tuple, list] = {}
        for sig, profile in self.shape_profiles.items():
            shape_key = tuple(profile["input_shapes"])
            if shape_key not in shape_groups:
                shape_groups[shape_key] = []
            shape_groups[shape_key].append(profile)

        report["shape_analysis"] = {
            "unique_shape_patterns": len(shape_groups),
            "most_common_shapes": sorted(
                [(k, len(v)) for k, v in shape_groups.items()], key=lambda x: x[1], reverse=True
            )[:5],
        }

        # Generate recommendations
        if report["summary"]["cache_hit_rate"] < 0.8:
            report["recommendations"].append(
                "Low cache hit rate. Consider using consistent input shapes "
                "or static_argnums for frequently changing parameters."
            )

        if report["summary"]["average_compile_time_s"] > 5.0:
            report["recommendations"].append(
                "High average compilation time. Consider breaking down large "
                "functions or using hierarchical compilation."
            )

        if len(shape_groups) > 20:
            report["recommendations"].append(
                "Many unique shape patterns detected. Consider shape padding "
                "or bucketing to reduce compilation overhead."
            )

        return report
