"""XLA compilation optimization for maximum performance.

This module provides XLA compilation strategies, smart compilation caching,
and memory-efficient patterns for JAX operations.
"""

import logging
import os
import tempfile
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import jax
import numpy as np


logger = logging.getLogger(__name__)


def get_xla_flags(backend: str) -> list[str]:
    """Return hardware-specific XLA compiler flags for the given backend.

    Selects flags based on the backend to avoid setting backend-specific flags
    (e.g. GPU Triton flags) on incompatible backends which would cause a fatal
    XLA parse error.

    Args:
        backend: Target backend ('gpu', 'tpu', 'cpu'). Pass ``jax.default_backend()``
                 for auto-detection.

    Returns:
        List of ``--xla_*`` flag strings appropriate for the backend.
    """
    if backend == "gpu":
        # Async collective flags (async_all_gather, async_all_reduce,
        # async_reduce_scatter) and memory_space_assignment were removed in
        # XLA 2024.10+ — these optimizations are now always-on.  Setting the
        # old flag names causes a FATAL parse error in parse_flags_from_env.cc,
        # which also crashes co-resident TF/PyTorch-XLA runtimes that share
        # the XLA_FLAGS env var.
        return [
            "--xla_gpu_enable_latency_hiding_scheduler=true",
            "--xla_gpu_strict_conv_algorithm_picker=false",
            "--xla_gpu_triton_gemm_any=True",
            "--xla_gpu_enable_triton_gemm=true",
            "--xla_gpu_all_gather_combine_threshold_bytes=134217728",
            "--xla_gpu_reduce_scatter_combine_threshold_bytes=134217728",
        ]
    if backend == "tpu":
        return [
            "--xla_tpu_enable_async_collective_fusion=true",
            "--xla_enable_async_all_gather=true",
        ]
    if backend == "cpu":
        return [
            "--xla_cpu_enable_fast_math=true",
        ]
    return []


def apply_xla_flags(backend: str) -> None:
    """Apply hardware-specific XLA flags to the ``XLA_FLAGS`` environment variable.

    Merges flags returned by :func:`get_xla_flags` into the existing
    ``XLA_FLAGS`` env var, skipping any flag whose name is already present.

    Args:
        backend: Target backend ('gpu', 'tpu', 'cpu').
    """
    flags = get_xla_flags(backend)
    if not flags:
        return

    existing = os.environ.get("XLA_FLAGS", "")
    for flag in flags:
        if flag.split("=")[0] not in existing:
            existing += " " + flag

    os.environ["XLA_FLAGS"] = existing.strip()
    logger.info("XLA flags configured for %s: %d flags set", backend, len(flags))


class XLAOptimizer:
    """Configure XLA compiler for maximum performance."""

    def __init__(self, target_hardware: str = "auto") -> None:
        """Initialize XLA optimizer with hardware-specific configuration.

        Args:
            target_hardware: Target hardware ('auto', 'gpu', 'tpu', 'cpu')
        """
        self.target_hardware = target_hardware
        self.setup_xla_flags()
        self.setup_jax_config()
        self.setup_compilation_cache()

    def setup_xla_flags(self) -> None:
        """Configure XLA compiler flags for maximum performance.

        Delegates to :func:`apply_xla_flags` after resolving ``"auto"``
        to the actual backend.
        """
        backend = self.target_hardware
        if backend == "auto":
            backend = jax.default_backend()

        apply_xla_flags(backend)

    def setup_jax_config(self) -> None:
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

        logger.info(f"JAX configuration optimized for {backend}")

    def setup_compilation_cache(self) -> None:
        """Setup persistent compilation cache."""
        # Use existing cache dir if set, otherwise create temp
        cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")
        if not cache_dir:
            cache_path = Path(tempfile.gettempdir()) / "jax_compilation_cache"
            cache_path.mkdir(parents=True, exist_ok=True)
            cache_dir = str(cache_path)

        jax.config.update("jax_compilation_cache_dir", cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

        logger.info(f"Compilation cache enabled at: {cache_dir}")


class SmartCompilation:
    """Intelligent compilation strategies for different scenarios."""

    def __init__(self) -> None:
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
        def compiled_version(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        def adaptive_wrapper(*args: Any, **kwargs: Any) -> Any:
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
            logger.info(f"AOT compilation successful for {func.__name__}")
            return compiled
        except (AttributeError, RuntimeError, TypeError, ValueError) as e:
            logger.warning(f"AOT compilation failed: {e}. Falling back to standard JIT.")
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

        def decorator(func: Callable) -> Any:
            return shard_map(func, mesh=mesh, in_specs=in_specs, out_specs=out_specs)

        return decorator


class MemoryEfficientCompilation:
    """Compilation patterns optimized for memory efficiency."""

    @staticmethod
    def donate_wrapper(func: Callable, donate_args: tuple[int, ...] | None = None) -> Callable:
        """Wrapper to automatically donate large arrays.

        Caches the compiled function keyed by donate_argnums to avoid
        recompilation on every call.

        Args:
            func: Function to wrap
            donate_args: Indices of arguments to donate, or None for auto-detect

        Returns:
            Memory-efficient wrapped function
        """
        # Cache compiled functions keyed by donate_argnums
        _compiled_cache: dict[tuple[int, ...], Callable] = {}

        if donate_args is not None:
            # Static donate args: compile once upfront
            compiled = partial(jax.jit, donate_argnums=donate_args)(func)
            return compiled

        def optimized_func(*args: Any, **kwargs: Any) -> Any:
            auto_donate: list[int] = []
            for i, arg in enumerate(args):
                if hasattr(arg, "size") and arg.size > 1024 * 1024:  # >1MB
                    auto_donate.append(i)

            cache_key = tuple(auto_donate)
            compiled = _compiled_cache.get(cache_key)
            if compiled is None:
                if auto_donate:
                    compiled = partial(jax.jit, donate_argnums=cache_key)(func)
                else:
                    compiled = jax.jit(func)
                _compiled_cache[cache_key] = compiled

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
            raise ValueError(
                f"Not enough devices for mesh {axis_dims}: required {expected}, found {n_devices}"
            )

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

    def __init__(self) -> None:
        """Initialize compilation profiler."""
        self.compilation_times: dict[Any, float] = {}
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.shape_profiles: dict[Any, dict] = {}

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

            def profiled_wrapper(*args: Any, **kwargs: Any) -> Any:
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

                logger.info(f"Compiled {func_name} for signature {signature}: {compile_time:.3f}s")

                return result

            return profiled_wrapper

        return decorator

    def _create_shape_signature(self, args: tuple) -> tuple:
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
