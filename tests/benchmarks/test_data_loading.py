"""Benchmark tests for data loading performance with different data sources.

This module contains comprehensive benchmark tests for data loading,
including performance tests, error handling, and edge cases.
"""

import platform
import time
from typing import Any
from unittest.mock import Mock, MagicMock, patch

import jax
import jax.numpy as jnp
import pytest
import numpy as np

# Detect macOS - TensorFlow crashes on macOS ARM64 due to Metal/GPU detection issues
# https://github.com/tensorflow/tensorflow/issues/52138
IS_MACOS = platform.system() == "Darwin"

# Import TFDS and HF modules
# These dependencies are now included in the test-cpu dependencies
# in pyproject.toml, so we don't need to skip these tests
from tests.test_common.data_generators import generate_image_data

from datarax.dag import DAGExecutor
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from datarax.sources.array_record_source import ArrayRecordSourceModule, ArrayRecordSourceConfig


# Set skip flags to False since we now have these dependencies in test-cpu
SKIP_TFDS = False
SKIP_HF = False


# Mark all tests in this file as benchmarks
pytestmark = pytest.mark.benchmark


@pytest.fixture
def benchmark_image_data() -> list[dict[str, Any]]:
    """Generate image data for benchmarking."""
    data = generate_image_data(num_samples=5000, image_height=32, image_width=32)
    # Convert to list of dictionaries for compatibility with MemorySource
    images = data["image"]
    labels = data["label"]
    return [{"image": images[i], "label": labels[i]} for i in range(len(images))]


def benchmark_stream_iteration(data_stream, num_batches=100):
    """Benchmark the given data stream's iteration performance."""
    # data_stream is already a configured DAGExecutor
    executor = data_stream

    # Warmup
    warmup_count = 0
    for batch in executor:
        warmup_count += 1
        if warmup_count >= 5:
            break

    # For DAGExecutor, we can't easily reset the iterator state, so we'll skip the reset
    # This means warmup and actual benchmark use the same stream, which is acceptable

    # Measure performance
    start_time = time.time()
    batches_processed = 0
    examples_processed = 0

    for batch in executor:
        batches_processed += 1

        # Count examples
        if isinstance(batch, dict) and "image" in batch:
            examples_processed += batch["image"].shape[0]
        else:
            # Fallback - assume first dimension is batch size
            first_value = jax.tree_util.tree_leaves(batch)[0]
            examples_processed += first_value.shape[0]

        # Stop after processing specified number of batches
        if batches_processed >= num_batches:
            break

    end_time = time.time()
    duration = end_time - start_time

    # Calculate metrics
    examples_per_second = examples_processed / duration
    batches_per_second = batches_processed / duration

    result = {
        "duration_seconds": duration,
        "batches_processed": batches_processed,
        "examples_processed": examples_processed,
        "examples_per_second": examples_per_second,
        "batches_per_second": batches_per_second,
    }

    return result


def test_inmemory_source_benchmark(benchmark, benchmark_image_data):
    """Benchmark the MemorySource performance."""

    def setup_and_run():
        config = MemorySourceConfig()
        source = MemorySource(config, benchmark_image_data)
        data_stream = DAGExecutor().add(source).batch(32)
        return benchmark_stream_iteration(data_stream)

    # Run benchmark
    result = benchmark(setup_and_run)

    # Print results for debugging
    print("\nMemorySource Performance:")
    print(f"  Duration: {result['duration_seconds']:.4f}s")
    print(f"  Examples/second: {result['examples_per_second']:.2f}")
    print(f"  Batches/second: {result['batches_per_second']:.2f}")

    # Verify performance is reasonable
    assert result["examples_per_second"] > 1000, "InMemory source performance below threshold"


@pytest.mark.skipif(
    IS_MACOS,
    reason="TensorFlow crashes on macOS ARM64 (tensorflow/tensorflow#52138)"
)
def test_tfds_source_benchmark(benchmark):
    """Benchmark the TFDSSource performance."""
    # Skip this test if tensorflow_datasets is not available
    pytest.importorskip("tensorflow_datasets")

    from datarax.sources import TFDSSource
    from datarax.sources.tfds_source import TfdsDataSourceConfig

    def setup_and_run():
        # Use a small dataset for benchmarking, let Pipeline handle batching
        config = TfdsDataSourceConfig(name="mnist", split="train[:1000]")
        source = TFDSSource(config)
        data_stream = DAGExecutor().add(source).batch(32)
        return benchmark_stream_iteration(data_stream, num_batches=30)

    # Run benchmark
    result = benchmark(setup_and_run)

    # Print results for debugging
    print("\nTFDSSource Performance:")
    print(f"  Duration: {result['duration_seconds']:.4f}s")
    print(f"  Examples/second: {result['examples_per_second']:.2f}")
    print(f"  Batches/second: {result['batches_per_second']:.2f}")

    # Verify performance is reasonable
    assert result["examples_per_second"] > 100, "TFDS source performance below threshold"


@pytest.mark.skipif(
    IS_MACOS,
    reason="HuggingFace datasets may have TensorFlow backend issues on macOS ARM64"
)
def test_hf_source_benchmark(benchmark):
    """Benchmark the HFSource performance."""
    # Skip this test if datasets is not available
    pytest.importorskip("datasets")

    from datarax.sources import HFSource
    from datarax.sources.hf_source import HfDataSourceConfig

    def setup_and_run():
        # Use a small dataset for benchmarking
        # HFSource now automatically converts PIL images to JAX arrays
        hf_config = HfDataSourceConfig(name="mnist", split="train[:1000]")
        source = HFSource(hf_config)

        data_stream = DAGExecutor().add(source).batch(32)

        return benchmark_stream_iteration(data_stream, num_batches=30)

    # Run benchmark
    result = benchmark(setup_and_run)

    # Print results for debugging
    print("\nHFSource Performance:")
    print(f"  Duration: {result['duration_seconds']:.4f}s")
    print(f"  Examples/second: {result['examples_per_second']:.2f}")
    print(f"  Batches/second: {result['batches_per_second']:.2f}")

    # Verify performance is reasonable
    assert result["examples_per_second"] > 100, "HF source performance below threshold"


def test_comparison_benchmark(benchmark, benchmark_image_data):
    """Benchmark and compare different data sources using the same data."""
    # Create temporary directory for saving/loading data if needed
    # import tempfile
    # temp_dir = tempfile.mkdtemp()

    # Prepare benchmark function
    def run_benchmark():
        # In-memory source (baseline)
        config = MemorySourceConfig()
        source = MemorySource(config, benchmark_image_data)
        data_stream = DAGExecutor().add(source).batch(32)
        inmemory_result = benchmark_stream_iteration(data_stream, num_batches=30)

        results = {"inmemory": inmemory_result}

        # Optional: Add other data sources if available and comparable
        # This is just a placeholder for future extension

        return results

    # Run benchmark
    results = benchmark(run_benchmark)

    # Print comparison
    print("\nData Source Performance Comparison:")
    for source_name, metrics in results.items():
        print(f"  {source_name.upper()}:")
        print(f"    Examples/second: {metrics['examples_per_second']:.2f}")
        print(f"    Batches/second: {metrics['batches_per_second']:.2f}")

    # Verify results exist
    assert "inmemory" in results
    assert results["inmemory"]["examples_per_second"] > 0


# Additional comprehensive tests for better coverage


class TestDataLoadingPerformance:
    """Additional tests for data loading performance and edge cases."""

    def test_empty_data_benchmark(self, benchmark):
        """Test performance with empty data source."""

        def setup_and_run():
            config = MemorySourceConfig()
            source = MemorySource(config, [])  # Empty data
            data_stream = DAGExecutor().add(source).batch(32)

            # Should handle empty data gracefully
            count = 0
            for _ in data_stream:
                count += 1
                if count > 10:  # Safety limit
                    break
            return {"count": count}

        result = benchmark(setup_and_run)
        assert result["count"] == 0, "Empty data should produce no batches"

    def test_single_item_data(self, benchmark):
        """Test with single data item."""

        def setup_and_run():
            data = [{"value": jnp.array([1.0])}]
            config = MemorySourceConfig()
            source = MemorySource(config, data)
            data_stream = DAGExecutor().add(source).batch(1)

            count = 0
            for batch in data_stream:
                count += 1
                assert "value" in batch
                if count > 10:  # Safety limit
                    break
            return {"count": count}

        result = benchmark(setup_and_run)
        assert result["count"] > 0, "Single item should produce at least one batch"

    def test_large_batch_size(self, benchmark, benchmark_image_data):
        """Test with very large batch size."""

        def setup_and_run():
            config = MemorySourceConfig()
            source = MemorySource(config, benchmark_image_data[:100])  # Use subset
            # Large batch size (larger than dataset)
            data_stream = DAGExecutor().add(source).batch(1000)

            batch_count = 0
            total_items = 0
            for batch in data_stream:
                batch_count += 1
                # Count items in batch
                if isinstance(batch, dict) and "image" in batch:
                    total_items += batch["image"].shape[0]
                if batch_count > 10:  # Safety limit
                    break

            return {"batch_count": batch_count, "total_items": total_items}

        result = benchmark(setup_and_run)
        assert result["batch_count"] > 0, "Should produce at least one batch"

    def test_benchmark_stream_iteration_edge_cases(self):
        """Test the benchmark_stream_iteration function with edge cases."""
        # Test with empty iterator (no batches)
        mock_executor = Mock()
        mock_executor.__iter__ = Mock(return_value=iter([]))

        result = benchmark_stream_iteration(mock_executor, num_batches=1)
        assert result["batches_processed"] == 0
        assert result["examples_processed"] == 0  # No batches means no examples

        # Test with dict batch without 'image' key
        # Note: The warmup will consume this batch, so we need more batches
        mock_executor = Mock()
        test_batch = {"data": jnp.ones((5, 10))}
        # Create enough batches for warmup (5) + actual test (1)
        batches = [test_batch] * 6
        mock_executor.__iter__ = Mock(return_value=iter(batches))

        result = benchmark_stream_iteration(mock_executor, num_batches=1)
        assert result["examples_processed"] == 5  # Should use first dimension

        # Test with zero batches requested
        mock_executor = Mock()
        mock_executor.__iter__ = Mock(return_value=iter([]))

        result = benchmark_stream_iteration(mock_executor, num_batches=0)
        assert result["batches_processed"] == 0

    def test_warmup_functionality(self):
        """Test that warmup phase works correctly."""
        # Create mock executor with controlled iteration
        batches = [{"image": jnp.ones((32, 32, 32, 3))} for _ in range(10)]
        mock_executor = Mock()
        mock_executor.__iter__ = Mock(return_value=iter(batches))

        result = benchmark_stream_iteration(mock_executor, num_batches=3)

        # Warmup consumes 5 batches, then we process 3 more
        # But since we can't reset the iterator, both warmup and actual use same stream
        assert result["batches_processed"] == 3
        assert result["examples_processed"] == 3 * 32

    def test_performance_metrics_calculation(self):
        """Test that performance metrics are calculated correctly."""
        # Create predictable data
        batch = {"image": jnp.ones((16, 32, 32, 3))}
        batches = [batch for _ in range(10)]
        mock_executor = Mock()
        mock_executor.__iter__ = Mock(return_value=iter(batches))

        result = benchmark_stream_iteration(mock_executor, num_batches=5)

        assert result["batches_processed"] == 5
        assert result["examples_processed"] == 5 * 16
        assert result["examples_per_second"] > 0
        assert result["batches_per_second"] > 0
        assert result["duration_seconds"] > 0

        # Check consistency
        assert (
            abs(
                result["examples_per_second"]
                - (result["examples_processed"] / result["duration_seconds"])
            )
            < 0.01
        )
        assert (
            abs(
                result["batches_per_second"]
                - (result["batches_processed"] / result["duration_seconds"])
            )
            < 0.01
        )


class TestDataSourceIntegration:
    """Test integration with different data sources."""

    def test_memory_source_with_transforms(self, benchmark_image_data):
        """Test MemorySource with additional transforms in pipeline."""
        from datarax.operators import ElementOperator, ElementOperatorConfig

        # Simple transform function - adapted to Element-based API
        def normalize(element, key):
            """Normalize image values to [0, 1] range."""
            new_data = dict(element.data)
            if "image" in new_data:
                new_data["image"] = new_data["image"] / 255.0
            return element.replace(data=new_data)

        config = MemorySourceConfig()
        source = MemorySource(config, benchmark_image_data[:100])
        transform_config = ElementOperatorConfig(stochastic=False)
        transform = ElementOperator(transform_config, fn=normalize)

        # Build pipeline with transform
        executor = DAGExecutor().add(source).batch(32).add(transform)

        # Verify it works
        batch_count = 0
        for batch in executor:
            batch_count += 1
            if isinstance(batch, dict) and "image" in batch:
                # Check normalization was applied
                assert jnp.max(batch["image"]) <= 1.0
            if batch_count >= 3:
                break

        assert batch_count > 0, "Pipeline with transforms should produce batches"

    def test_memory_source_error_handling(self):
        """Test MemorySource error handling."""
        # Test with invalid data type
        config = MemorySourceConfig()
        with pytest.raises((TypeError, ValueError)):
            source = MemorySource(config, "not a list")  # Invalid data type
            executor = DAGExecutor().add(source).batch(32)
            # Try to iterate
            next(iter(executor))

    def test_executor_configuration(self):
        """Test DAGExecutor configuration options."""
        data = [{"value": jnp.array([i])} for i in range(10)]
        config = MemorySourceConfig()
        source = MemorySource(config, data)

        # Test with caching disabled
        executor = DAGExecutor(enable_caching=False).add(source).batch(2)
        assert executor.enable_caching is False

        # Test with JIT compilation (even if not fully implemented)
        executor = DAGExecutor(jit_compile=True).add(source).batch(2)
        assert executor.jit_compile is True

        # Test with batch enforcement disabled (should fail)
        with pytest.raises(ValueError, match="Batch-first enforcement"):
            executor = DAGExecutor(enforce_batch=True).add(source)
            # Missing batch node should raise error when iterating
            next(iter(executor))

    def test_fixture_generation(self, benchmark_image_data):
        """Test that the benchmark_image_data fixture generates correct data."""
        assert isinstance(benchmark_image_data, list)
        assert len(benchmark_image_data) == 5000

        # Check first item structure
        first_item = benchmark_image_data[0]
        assert isinstance(first_item, dict)
        assert "image" in first_item
        assert "label" in first_item

        # Check image shape
        image = first_item["image"]
        assert image.shape == (32, 32, 3)

        # Check data types - label should be an integer type
        label = first_item["label"]
        assert isinstance(label, int | np.integer) or hasattr(label, "dtype")


class TestBenchmarkUtilities:
    """Test the benchmark utility functions."""

    def test_benchmark_decorator(self):
        """Test that benchmark decorator is properly applied."""
        # The pytestmark should apply to all tests
        assert pytestmark == pytest.mark.benchmark

    def test_skip_flags(self):
        """Test that skip flags are correctly set."""
        assert SKIP_TFDS is False
        assert SKIP_HF is False

    def test_importorskip_usage(self):
        """Test that pytest.importorskip works as expected."""
        # Should skip if module not available
        with pytest.raises(pytest.skip.Exception):
            pytest.importorskip("nonexistent_module_xyz")

        # Should not skip for available modules
        jax_module = pytest.importorskip("jax")
        assert jax_module is not None


class TestArrayRecordSourcePerformance:
    """Performance tests for ArrayRecordSourceModule."""

    @pytest.fixture
    def mock_grain_source(self):
        """Create mock Grain ArrayRecordDataSource for performance tests."""
        mock = MagicMock()
        mock.__len__.return_value = 1000
        mock.__getitem__.side_effect = lambda idx: {"data": np.array([idx])}
        return mock

    @pytest.mark.performance
    def test_iteration_speed(self, mock_grain_source):
        """Test iteration performance."""
        import time

        with patch("grain.python.ArrayRecordDataSource", return_value=mock_grain_source):
            config = ArrayRecordSourceConfig()
            source = ArrayRecordSourceModule(config, paths="dummy.array_record")

            start = time.time()
            count = 0
            for element in source:
                count += 1
                if count >= 1000:
                    break
            elapsed = time.time() - start

            # Should iterate fast (< 1 second for 1000 elements)
            assert elapsed < 1.0
            assert count == 1000

    @pytest.mark.performance
    def test_memory_usage(self, mock_grain_source):
        """Test memory usage remains bounded."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        with patch("grain.python.ArrayRecordDataSource", return_value=mock_grain_source):
            config = ArrayRecordSourceConfig()
            source = ArrayRecordSourceModule(config, paths="dummy.array_record")

            # Iterate through many elements
            for i, element in enumerate(source):
                if i >= 10000:
                    break

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            # Memory increase should be reasonable (< 100MB)
            assert memory_increase < 100


if __name__ == "__main__":
    # Allow running directly for manual testing
    pytest.main(["-xvs", __file__])
