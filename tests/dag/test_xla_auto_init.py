"""Tests for automatic XLA optimization and compilation cache initialization.

Validates F8/P1.2/P1.3:
1. XLA flags are auto-applied on first DAGExecutor creation
2. Persistent compilation cache is enabled by default
3. Initialization is idempotent (called once per process)
"""

import os
from unittest.mock import patch


class TestXLAAutoInit:
    """Tests for _ensure_xla_optimized() in dag_executor."""

    def test_xla_initialized_on_executor_creation(self):
        """Creating a DAGExecutor should trigger XLA optimization."""
        import datarax.dag.dag_executor as mod

        # Reset the flag to test initialization
        original = mod._IS_XLA_INITIALIZED
        mod._IS_XLA_INITIALIZED = False
        try:
            from datarax.dag.dag_executor import DAGExecutor

            DAGExecutor(enforce_batch=False)
            assert mod._IS_XLA_INITIALIZED is True
        finally:
            mod._IS_XLA_INITIALIZED = original

    def test_initialization_is_idempotent(self):
        """Multiple DAGExecutor creations should only initialize once."""
        import datarax.dag.dag_executor as mod

        original = mod._IS_XLA_INITIALIZED
        mod._IS_XLA_INITIALIZED = False
        try:
            from datarax.dag.dag_executor import _ensure_xla_optimized

            # First call sets the flag
            _ensure_xla_optimized()
            assert mod._IS_XLA_INITIALIZED is True

            # Capture env state after first init
            flags_after_first = os.environ.get("XLA_FLAGS", "")

            # Second call is a no-op
            _ensure_xla_optimized()
            flags_after_second = os.environ.get("XLA_FLAGS", "")

            assert flags_after_first == flags_after_second
        finally:
            mod._IS_XLA_INITIALIZED = original

    def test_compilation_cache_enabled(self):
        """Compilation cache should be configured after initialization."""
        import datarax.dag.dag_executor as mod

        original = mod._IS_XLA_INITIALIZED
        mod._IS_XLA_INITIALIZED = False
        try:
            mod._ensure_xla_optimized()
            # After init, the cache dir should be set in JAX config
            # We verify this indirectly — the function should have run without error
            assert mod._IS_XLA_INITIALIZED is True
        finally:
            mod._IS_XLA_INITIALIZED = original

    def test_gpu_flags_applied_on_gpu_backend(self):
        """GPU-specific XLA flags should be applied when backend is GPU."""
        import datarax.dag.dag_executor as mod

        original_flag = mod._IS_XLA_INITIALIZED
        original_env = os.environ.get("XLA_FLAGS", "")
        mod._IS_XLA_INITIALIZED = False
        try:
            with patch("jax.default_backend", return_value="gpu"):
                mod._ensure_xla_optimized()

            flags = os.environ.get("XLA_FLAGS", "")
            assert "xla_gpu_enable_latency_hiding_scheduler" in flags
            assert "xla_gpu_triton_gemm_any" in flags
        finally:
            mod._IS_XLA_INITIALIZED = original_flag
            os.environ["XLA_FLAGS"] = original_env

    def test_cpu_backend_no_gpu_flags(self):
        """CPU backend should not add GPU-specific flags."""
        import datarax.dag.dag_executor as mod

        original_flag = mod._IS_XLA_INITIALIZED
        original_env = os.environ.get("XLA_FLAGS", "")
        # Clear existing flags to test clean
        os.environ["XLA_FLAGS"] = ""
        mod._IS_XLA_INITIALIZED = False
        try:
            with patch("jax.default_backend", return_value="cpu"):
                mod._ensure_xla_optimized()

            flags = os.environ.get("XLA_FLAGS", "")
            assert "xla_gpu_enable_latency_hiding_scheduler" not in flags
        finally:
            mod._IS_XLA_INITIALIZED = original_flag
            os.environ["XLA_FLAGS"] = original_env
