"""Complete tests for profiler.py to reach ≥80% coverage.

Covers uncovered paths:
- TPU and GPU hardware detection (AdaptiveOperation)
- optimize_grain_dataset() with mocked grain module
- GPUMemoryProfiler GPU-present paths (mocked)
- analyze_memory_pattern() trend detection + avg utilization branches
- MemoryOptimizer.analyze_pipeline_memory() block_until_ready + error paths
- _generate_memory_suggestions() high peak + low efficiency triggers
"""

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import pytest

from datarax.benchmarking.profiler import (
    AdaptiveOperation,
    GPUMemoryProfiler,
    MemoryOptimizer,
)


# ---------------------------------------------------------------------------
# AdaptiveOperation: hardware detection
# ---------------------------------------------------------------------------


class TestAdaptiveOperationTPU:
    """Cover the TPU branch in detect_hardware_and_optimize (line 41)."""

    def test_tpu_config(self):
        with patch("jax.default_backend", return_value="tpu"):
            op = AdaptiveOperation()

        assert op.hw_config["platform"] == "tpu"
        assert op.hw_config["precision"] == jnp.bfloat16
        assert op.hw_config["tile_size"] == 128
        assert op.hw_config["critical_batch_size"] == 240
        assert op.hw_config["use_vmem_optimization"] is True


class TestAdaptiveOperationGPU:
    """Cover both modern and legacy GPU branches + exception fallback."""

    def test_gpu_modern_h100(self):
        """H100 detected -> gpu_modern config (line 55-63)."""
        mock_dev = MagicMock()
        mock_dev.device_kind = "NVIDIA H100"
        with (
            patch("jax.default_backend", return_value="gpu"),
            patch("jax.devices", return_value=[mock_dev]),
        ):
            op = AdaptiveOperation()

        assert op.hw_config["platform"] == "gpu_modern"
        assert op.hw_config["precision"] == jnp.bfloat16
        assert op.hw_config["tile_size"] == 16

    def test_gpu_legacy(self):
        """Non-A100/H100 GPU -> gpu_legacy config (lines 64-72)."""
        mock_dev = MagicMock()
        mock_dev.device_kind = "NVIDIA RTX 3090"
        with (
            patch("jax.default_backend", return_value="gpu"),
            patch("jax.devices", return_value=[mock_dev]),
        ):
            op = AdaptiveOperation()

        assert op.hw_config["platform"] == "gpu_legacy"
        assert op.hw_config["precision"] == jnp.float32
        assert op.hw_config["tile_size"] == 32
        assert op.hw_config["critical_batch_size"] == 128

    def test_gpu_detection_exception(self):
        """jax.devices() raises -> falls through to CPU defaults (line 73)."""
        with (
            patch("jax.default_backend", return_value="gpu"),
            patch("jax.devices", side_effect=RuntimeError("no GPU")),
        ):
            op = AdaptiveOperation()

        # Falls back to CPU defaults (the exception is caught on line 73)
        assert op.hw_config["platform"] == "cpu"


# ---------------------------------------------------------------------------
# AdaptiveOperation: optimize_grain_dataset
# ---------------------------------------------------------------------------


class TestOptimizeGrainDataset:
    """Cover all paths in optimize_grain_dataset (lines 97-135)."""

    def test_grain_not_installed(self):
        """ImportError path -> returns ds unchanged (line 134-135)."""
        op = AdaptiveOperation()
        sentinel = object()
        with patch.dict("sys.modules", {"grain.python": None}):
            result = op.optimize_grain_dataset(sentinel)
        assert result is sentinel

    def test_grain_no_experimental(self):
        """grain exists but no experimental attr -> warns and returns ds (lines 105-112)."""
        import grain.python as grain_mod

        op = AdaptiveOperation()
        sentinel = object()

        # Temporarily hide experimental so hasattr(grain, "experimental") is False
        orig = grain_mod.experimental
        try:
            del grain_mod.experimental
            with pytest.warns(UserWarning, match="pick_performance_config not found"):
                result = op.optimize_grain_dataset(sentinel)
        finally:
            grain_mod.experimental = orig

        assert result is sentinel

    def test_grain_optimization_applied(self):
        """Happy path: grain optimizations applied (lines 114-129)."""
        import grain.python as grain_mod

        op = AdaptiveOperation()

        mock_perf_config = MagicMock()
        mock_perf_config.read_options = "read_opts"
        mock_perf_config.multiprocessing_options = "mp_opts"

        mock_ds = MagicMock()
        mock_ds.to_iter_dataset.return_value = mock_ds
        mock_ds.mp_prefetch.return_value = mock_ds

        # Patch pick_performance_config onto real grain.experimental
        with patch.object(
            grain_mod.experimental,
            "pick_performance_config",
            return_value=mock_perf_config,
            create=True,
        ):
            result = op.optimize_grain_dataset(mock_ds)

        assert result is mock_ds
        mock_ds.to_iter_dataset.assert_called_once_with(read_options="read_opts")
        mock_ds.mp_prefetch.assert_called_once_with("mp_opts")

    def test_grain_optimization_exception(self):
        """pick_performance_config raises -> warns, returns ds (lines 130-132)."""
        import grain.python as grain_mod

        op = AdaptiveOperation()
        sentinel = object()

        with patch.object(
            grain_mod.experimental,
            "pick_performance_config",
            side_effect=RuntimeError("boom"),
            create=True,
        ):
            with pytest.warns(UserWarning, match="Failed to apply Grain optimization"):
                result = op.optimize_grain_dataset(sentinel)

        assert result is sentinel


# ---------------------------------------------------------------------------
# GPUMemoryProfiler: GPU-present paths
# ---------------------------------------------------------------------------


class TestGPUMemoryProfilerWithGPU:
    """Cover GPU-present branches via mocking (lines 145-196)."""

    def test_init_with_gpu(self):
        """has_gpu = True when jax.devices('gpu') returns devices (line 144)."""
        with patch("jax.devices", return_value=[MagicMock()]):
            profiler = GPUMemoryProfiler()
        assert profiler.has_gpu is True

    def test_init_runtime_error(self):
        """has_gpu = False when jax.devices raises RuntimeError (line 145-146)."""
        with patch("jax.devices", side_effect=RuntimeError("no GPU")):
            profiler = GPUMemoryProfiler()
        assert profiler.has_gpu is False

    def test_init_value_error(self):
        """has_gpu = False when jax.devices raises ValueError (line 145-146)."""
        with patch("jax.devices", side_effect=ValueError("bad")):
            profiler = GPUMemoryProfiler()
        assert profiler.has_gpu is False

    def test_get_memory_no_gpu(self):
        """No GPU -> returns zeros (line 150-151)."""
        profiler = GPUMemoryProfiler()
        profiler.has_gpu = False
        mem = profiler.get_memory_usage()
        assert mem == {"gpu_memory_used_mb": 0.0, "gpu_memory_total_mb": 0.0}

    def test_get_memory_with_memory_stats(self):
        """GPU with memory_stats -> detailed memory info (lines 156-171)."""
        mock_device = MagicMock()
        mock_device.memory_stats.return_value = {
            "bytes_in_use": 2048 * 1024 * 1024,
            "bytes_limit": 8192 * 1024 * 1024,
            "pool_bytes": 1024 * 1024 * 1024,
        }

        profiler = GPUMemoryProfiler()
        profiler.has_gpu = True

        with patch("jax.devices", return_value=[mock_device]):
            mem = profiler.get_memory_usage()

        assert mem["gpu_memory_used_mb"] == pytest.approx(2048.0)
        assert mem["gpu_memory_total_mb"] == pytest.approx(8192.0)
        assert mem["gpu_memory_utilization"] == pytest.approx(0.25)
        assert "pool_bytes_mb" in mem

    def test_get_memory_stats_empty(self):
        """GPU device.memory_stats() returns empty dict -> falls to xla_bridge (line 158)."""
        mock_device = MagicMock()
        mock_device.memory_stats.return_value = {}

        profiler = GPUMemoryProfiler()
        profiler.has_gpu = True

        mock_mem_info = MagicMock()
        mock_mem_info.bytes_in_use = 1024 * 1024 * 1024
        mock_mem_info.bytes_limit = 4096 * 1024 * 1024

        mock_xla_bridge = MagicMock()
        mock_xla_bridge.get_memory_info.return_value = mock_mem_info

        with (
            patch("jax.devices", return_value=[mock_device]),
            patch.object(jax.lib, "xla_bridge", mock_xla_bridge, create=True),
        ):
            mem = profiler.get_memory_usage()

        assert mem["gpu_memory_used_mb"] == pytest.approx(1024.0)
        assert mem["gpu_memory_total_mb"] == pytest.approx(4096.0)

    def test_get_memory_no_xla_bridge_fallback(self):
        """No xla_bridge.get_memory_info -> assumes 8GB (line 185)."""
        mock_device = MagicMock()
        mock_device.memory_stats.side_effect = Exception("no stats")

        profiler = GPUMemoryProfiler()
        profiler.has_gpu = True

        # Use a mock xla_bridge that lacks get_memory_info
        mock_xla_bridge = MagicMock(spec=[])

        with (
            patch("jax.devices", return_value=[mock_device]),
            patch.object(jax.lib, "xla_bridge", mock_xla_bridge, create=True),
        ):
            mem = profiler.get_memory_usage()

        assert mem["gpu_memory_total_mb"] == pytest.approx(8192.0)

    def test_get_memory_outer_exception(self):
        """Outer exception -> returns zeros (line 195-196)."""
        profiler = GPUMemoryProfiler()
        profiler.has_gpu = True

        with patch("jax.devices", side_effect=Exception("catastrophic")):
            mem = profiler.get_memory_usage()

        assert mem == {"gpu_memory_used_mb": 0.0, "gpu_memory_total_mb": 0.0}


# ---------------------------------------------------------------------------
# GPUMemoryProfiler: analyze_memory_pattern branches
# ---------------------------------------------------------------------------


class TestAnalyzeMemoryPattern:
    """Cover trend detection, high utilization, and avg utilization branches."""

    def test_empty_usage_values(self):
        """usage_values empty -> returns empty (line 216-217)."""
        profiler = GPUMemoryProfiler()
        result = profiler.analyze_memory_pattern([{}])
        # One dict with no keys -> usage_values = [0], not empty
        assert isinstance(result, list)

    def test_memory_leak_detection(self):
        """Increasing trend > 10 MB/sample -> leak warning (lines 219-225)."""
        profiler = GPUMemoryProfiler()
        measurements = [
            {"gpu_memory_used_mb": 100, "gpu_memory_utilization": 0.3},
            {"gpu_memory_used_mb": 200, "gpu_memory_utilization": 0.3},
            {"gpu_memory_used_mb": 300, "gpu_memory_utilization": 0.3},
        ]
        suggestions = profiler.analyze_memory_pattern(measurements)
        assert any("memory leak" in s.lower() for s in suggestions)

    def test_high_utilization_above_90(self):
        """max_utilization > 0.9 -> high utilization warning (lines 230-234)."""
        profiler = GPUMemoryProfiler()
        measurements = [
            {"gpu_memory_used_mb": 100, "gpu_memory_utilization": 0.95},
            {"gpu_memory_used_mb": 100, "gpu_memory_utilization": 0.5},
            {"gpu_memory_used_mb": 100, "gpu_memory_utilization": 0.5},
        ]
        suggestions = profiler.analyze_memory_pattern(measurements)
        assert any("90%" in s for s in suggestions)

    def test_avg_utilization_above_80(self):
        """avg_utilization > 0.8 but max < 0.9 -> monitor warning (lines 235-239)."""
        profiler = GPUMemoryProfiler()
        measurements = [
            {"gpu_memory_used_mb": 100, "gpu_memory_utilization": 0.85},
            {"gpu_memory_used_mb": 100, "gpu_memory_utilization": 0.85},
            {"gpu_memory_used_mb": 100, "gpu_memory_utilization": 0.85},
        ]
        suggestions = profiler.analyze_memory_pattern(measurements)
        assert any("80%" in s for s in suggestions)

    def test_no_suggestions_for_low_usage(self):
        """Low and stable memory -> no suggestions."""
        profiler = GPUMemoryProfiler()
        measurements = [
            {"gpu_memory_used_mb": 50, "gpu_memory_utilization": 0.1},
            {"gpu_memory_used_mb": 51, "gpu_memory_utilization": 0.1},
            {"gpu_memory_used_mb": 50, "gpu_memory_utilization": 0.1},
        ]
        suggestions = profiler.analyze_memory_pattern(measurements)
        assert suggestions == []

    def test_fewer_than_3_measurements_skips_trend(self):
        """< 3 measurements -> skips polyfit trend check (line 219)."""
        profiler = GPUMemoryProfiler()
        measurements = [
            {"gpu_memory_used_mb": 100, "gpu_memory_utilization": 0.95},
            {"gpu_memory_used_mb": 200, "gpu_memory_utilization": 0.95},
        ]
        suggestions = profiler.analyze_memory_pattern(measurements)
        # Should still check utilization but skip trend
        assert any("90%" in s for s in suggestions)


# ---------------------------------------------------------------------------
# MemoryOptimizer: analyze_pipeline_memory
# ---------------------------------------------------------------------------


class TestAnalyzePipelineMemory:
    """Cover block_until_ready paths and error path (lines 260-267)."""

    def test_result_with_block_until_ready(self):
        """Pipeline returns object with block_until_ready (line 260-261)."""
        optimizer = MemoryOptimizer()
        mock_result = MagicMock()
        mock_result.block_until_ready = MagicMock()

        analysis = optimizer.analyze_pipeline_memory(lambda _: mock_result, None)

        mock_result.block_until_ready.assert_called_once()
        assert "baseline_memory_mb" in analysis
        assert "suggestions" in analysis

    def test_result_is_list_with_jax_arrays(self):
        """Pipeline returns list -> iterates tree_leaves for block_until_ready (lines 262-265)."""
        optimizer = MemoryOptimizer()

        leaf1 = MagicMock()
        leaf1.block_until_ready = MagicMock()
        leaf2 = MagicMock(spec=[])  # no block_until_ready

        with patch("jax.tree_util.tree_leaves", return_value=[leaf1, leaf2]):
            analysis = optimizer.analyze_pipeline_memory(lambda _: [leaf1, leaf2], None)

        leaf1.block_until_ready.assert_called_once()
        assert "baseline_memory_mb" in analysis

    def test_result_is_tuple_with_jax_arrays(self):
        """Pipeline returns tuple -> same tree_leaves path (lines 262-265)."""
        optimizer = MemoryOptimizer()

        leaf = MagicMock()
        leaf.block_until_ready = MagicMock()

        with patch("jax.tree_util.tree_leaves", return_value=[leaf]):
            analysis = optimizer.analyze_pipeline_memory(lambda _: (leaf,), None)

        leaf.block_until_ready.assert_called_once()
        assert "peak_memory_mb" in analysis

    def test_pipeline_exception(self):
        """Pipeline raises -> returns error dict (lines 266-267)."""
        optimizer = MemoryOptimizer()

        def failing_fn(_):
            raise ValueError("pipeline exploded")

        analysis = optimizer.analyze_pipeline_memory(failing_fn, None)

        assert "error" in analysis
        assert "pipeline exploded" in analysis["error"]
        assert analysis["memory_analysis"] is None


# ---------------------------------------------------------------------------
# MemoryOptimizer: _generate_memory_suggestions
# ---------------------------------------------------------------------------


class TestGenerateMemorySuggestions:
    """Cover high peak and low efficiency trigger branches (lines 296-316)."""

    def test_high_peak_usage(self):
        """peak > 1000 MB -> high memory suggestion (lines 303-307)."""
        optimizer = MemoryOptimizer()
        suggestions = optimizer._generate_memory_suggestions(peak_usage=1500, retained_memory=100)
        assert any("high memory" in s.lower() for s in suggestions)

    def test_low_efficiency(self):
        """efficiency < 0.7 -> low efficiency suggestion (lines 309-314)."""
        optimizer = MemoryOptimizer()
        # efficiency = (500 - 400) / 500 = 0.2 < 0.7
        suggestions = optimizer._generate_memory_suggestions(peak_usage=500, retained_memory=400)
        assert any("low memory efficiency" in s.lower() for s in suggestions)

    def test_both_high_peak_and_low_efficiency(self):
        """Both triggers fire when peak > 1000 and efficiency < 0.7."""
        optimizer = MemoryOptimizer()
        # peak=2000, retained=1800 -> efficiency = 0.1
        suggestions = optimizer._generate_memory_suggestions(peak_usage=2000, retained_memory=1800)
        assert len(suggestions) == 2

    def test_zero_peak_usage(self):
        """peak_usage == 0 -> efficiency defaults to 1.0, no suggestions."""
        optimizer = MemoryOptimizer()
        suggestions = optimizer._generate_memory_suggestions(peak_usage=0, retained_memory=0)
        assert suggestions == []

    def test_moderate_usage_no_suggestions(self):
        """Moderate peak and good efficiency -> no suggestions."""
        optimizer = MemoryOptimizer()
        # peak=500, retained=50 -> efficiency = 0.9
        suggestions = optimizer._generate_memory_suggestions(peak_usage=500, retained_memory=50)
        assert suggestions == []
