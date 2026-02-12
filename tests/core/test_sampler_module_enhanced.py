"""Tests for enhanced SamplerModule functionality.

This module contains thorough tests for the enhanced SamplerModule that
leverages advanced features from DataraxModule, including caching, statistics,
RNG handling, and state management.
"""

from dataclasses import dataclass
from collections.abc import Iterator

import pytest
import flax.nnx as nnx

from datarax.core.sampler import SamplerModule
from datarax.core.config import StructuralConfig
from datarax.utils.prng import create_rngs


@dataclass
class SimpleTestSamplerConfig(StructuralConfig):
    """Configuration for SimpleTestSampler."""

    dataset_size: int = 10


class SimpleTestSampler(SamplerModule):
    """Simple sampler for testing enhanced functionality."""

    def __init__(
        self,
        config: SimpleTestSamplerConfig | None = None,
        *,
        dataset_size: int = 10,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        # Handle optional config: if no config provided, create one from kwargs
        if config is None:
            config = SimpleTestSamplerConfig(dataset_size=dataset_size)

        super().__init__(config, rngs=rngs, name=name or "SimpleTestSampler")
        self.dataset_size = config.dataset_size
        self.current_index = nnx.Variable(0)

    def _sample_impl(self, n: int) -> list[int]:
        """Simple sampling implementation for testing."""
        # Validate dataset size
        if self.dataset_size < 0:
            raise ValueError(f"Dataset size must be non-negative, got {self.dataset_size}")

        # Reset index for each sampling call
        self.current_index.set_value(0)
        indices = []
        for i in range(n):
            # Cycle through dataset if n > dataset_size
            idx = i % self.dataset_size if self.dataset_size > 0 else 0
            indices.append(idx)
        return indices

    def __iter__(self) -> Iterator[int]:
        """Iterator implementation."""
        self.current_index.set_value(0)
        return self

    def __next__(self) -> int:
        """Get next index."""
        if self.current_index.get_value() >= self.dataset_size:
            raise StopIteration
        idx = self.current_index.get_value()
        self.current_index.set_value(self.current_index.get_value() + 1)
        return idx

    def __len__(self) -> int:
        """Return dataset size."""
        return self.dataset_size

    def sample(self, n: int) -> list[int]:
        """Override sample to use our _sample_impl method."""
        # Set dataset_size if needed
        if hasattr(self, "dataset_size") and getattr(self, "dataset_size", None) is None:
            setattr(self, "dataset_size", n)

        # Use our custom implementation
        return self._sample_impl(n)


class TestSamplerModuleEnhanced:
    """Test enhanced SamplerModule functionality."""

    def test_initialization_with_enhanced_features(self):
        """Test SamplerModule initialization with enhanced features."""
        config = SimpleTestSamplerConfig(
            dataset_size=5,
            cacheable=True,
            batch_stats_fn=lambda x: {"count": len(x)},
        )
        sampler = SimpleTestSampler(config, rngs=create_rngs(seed=42))

        # Check enhanced features are available
        assert hasattr(sampler, "_cache")
        assert sampler.config.batch_stats_fn is not None  # Accessed through config
        assert hasattr(sampler, "rngs")
        assert sampler.config.cacheable is True

    def test_default_stochastic_for_samplers(self):
        """Test that samplers default to non-stochastic mode."""
        sampler = SimpleTestSampler()

        # Samplers should default to non-stochastic
        assert sampler.stochastic is False

    def test_caching_functionality(self):
        """Test caching works for sample operations."""
        config = SimpleTestSamplerConfig(dataset_size=5, cacheable=True)
        sampler = SimpleTestSampler(config)

        # First call using enhanced interface
        result1 = sampler(5)
        assert len(sampler._cache) == 1

        # Second call with same data should use cache
        result2 = sampler(5)

        # Results should be identical (from cache)
        assert result1 == result2

    def test_statistics_computation(self):
        """Test statistics computation functionality."""

        def compute_stats(indices):
            return {"count": len(indices), "max": max(indices) if indices else 0}

        config = SimpleTestSamplerConfig(dataset_size=5, batch_stats_fn=compute_stats)
        sampler = SimpleTestSampler(config)

        # Call enhanced interface
        sampler(3)

        # Should have computed statistics
        assert hasattr(sampler, "_last_computed_stats")
        stats = sampler._last_computed_stats
        assert stats is not None
        assert stats["count"] == 3
        assert stats["max"] == 2

    def test_precomputed_statistics(self):
        """Test precomputed statistics functionality."""
        precomputed = {"dataset_size": 100, "type": "test"}
        config = SimpleTestSamplerConfig(dataset_size=5, precomputed_stats=precomputed)
        sampler = SimpleTestSampler(config)

        # Statistics should be available
        stats = sampler._compute_statistics([])
        assert stats["dataset_size"] == 100
        assert stats["type"] == "test"

    def test_rng_integration(self):
        """Test RNG integration with enhanced features."""
        rngs = create_rngs(seed=42)
        # Use stochastic config to test RNG stream requirements
        config = SimpleTestSamplerConfig(dataset_size=5, stochastic=True, stream_name="default")
        sampler = SimpleTestSampler(config, rngs=rngs)

        # Should have RNG streams
        assert sampler.rngs is not None
        assert hasattr(sampler, "stream_name")

        # Should be able to get required streams for stochastic sampler
        required_streams = sampler.requires_rng_streams()
        assert required_streams is not None
        assert len(required_streams) > 0

    def test_cache_key_computation(self):
        """Test cache key computation for sampling operations."""
        config = SimpleTestSamplerConfig(dataset_size=5, cacheable=True)
        sampler = SimpleTestSampler(config)

        # Different parameters should produce different cache keys
        key1 = sampler._compute_cache_key(5)
        key2 = sampler._compute_cache_key(3)

        assert key1 != key2

    def test_state_management(self):
        """Test enhanced state management."""
        sampler = SimpleTestSampler(dataset_size=5)

        # Perform some operations to change state
        sampler(3)

        # Get state
        state = sampler.get_state()

        # State should contain all necessary information
        assert isinstance(state, dict)
        # Just check that we have some state
        assert len(state) > 0

        # Should be able to restore state
        new_sampler = SimpleTestSampler(dataset_size=5)
        new_sampler.set_state(state)

    def test_cache_reset(self):
        """Test cache reset functionality."""
        config = SimpleTestSamplerConfig(dataset_size=5, cacheable=True)
        sampler = SimpleTestSampler(config)

        # Add something to cache
        sampler(3)
        assert len(sampler._cache) > 0

        # Reset cache
        sampler.reset_cache()
        assert len(sampler._cache) == 0

    def test_enhanced_call_interface(self):
        """Test the enhanced __call__ interface."""
        config = SimpleTestSamplerConfig(dataset_size=5, cacheable=True)
        sampler = SimpleTestSampler(config)

        # Enhanced interface should work
        result = sampler(3)
        assert isinstance(result, list)
        assert len(result) == 3
        assert result == [0, 1, 2]


class TestSamplerModuleIntegration:
    """Test SamplerModule integration with other components."""

    def test_dag_composition_support(self):
        """Test that enhanced SamplerModule works with DAG composition."""
        from datarax.dag import OperatorNode

        sampler = SimpleTestSampler(dataset_size=5)

        # Should be able to create OperatorNode with sampler
        # This tests basic compatibility with DAG infrastructure
        node = OperatorNode(sampler)
        assert node.operator == sampler

    def test_pipeline_integration(self):
        """Test integration with pipeline infrastructure."""
        config = SimpleTestSamplerConfig(dataset_size=5, cacheable=True)
        sampler = SimpleTestSampler(config)

        # Should work with pipeline-style calls
        result = sampler(3)
        assert len(result) == 3

    def test_consistency_with_other_modules(self):
        """Test consistency with other enhanced modules."""
        config = SimpleTestSamplerConfig(dataset_size=5, cacheable=True)
        sampler = SimpleTestSampler(config)

        # Should have same enhanced interface as other modules
        assert hasattr(sampler, "_cache")
        assert hasattr(sampler, "get_state")
        assert hasattr(sampler, "set_state")
        assert hasattr(sampler, "reset_cache")


class TestSamplerModuleErrorHandling:
    """Test error handling in enhanced SamplerModule."""

    def test_invalid_dataset_size(self):
        """Test handling of invalid dataset size."""
        with pytest.raises(ValueError):
            sampler = SimpleTestSampler(dataset_size=-1)
            sampler(5)

    def test_empty_sampling(self):
        """Test handling of empty sampling requests."""
        sampler = SimpleTestSampler(dataset_size=5)

        result = sampler(0)
        assert result == []

    def test_caching_with_non_cacheable_data(self):
        """Test caching behavior with non-cacheable data."""
        config = SimpleTestSamplerConfig(dataset_size=5, cacheable=False)
        sampler = SimpleTestSampler(config)

        # Should not cache when cacheable=False
        sampler(3)
        assert sampler._cache is None  # Cache should be None when not cacheable

    def test_statistics_computation_errors(self):
        """Test handling of statistics computation errors."""

        def failing_stats(indices):
            raise ValueError("Statistics computation failed")

        config = SimpleTestSamplerConfig(dataset_size=5, batch_stats_fn=failing_stats)
        sampler = SimpleTestSampler(config)

        # Should handle errors gracefully
        result = sampler._compute_statistics([1, 2, 3])
        assert result is None

    def test_rng_without_streams(self):
        """Test behavior when RNG streams are not available."""
        sampler = SimpleTestSampler(dataset_size=5, rngs=None)

        # Should still work without RNG
        result = sampler(3)
        assert len(result) == 3


class TestSamplerModuleCoverage:
    """Test complete coverage of SamplerModule functionality."""

    def test_stochastic_modes(self):
        """Test SamplerModule in stochastic and non-stochastic modes."""
        # Non-stochastic (default)
        config_det = SimpleTestSamplerConfig(dataset_size=5, stochastic=False)
        sampler_det = SimpleTestSampler(config_det)
        assert sampler_det.stochastic is False

        # Stochastic requires stream_name
        config_stoch = SimpleTestSamplerConfig(
            dataset_size=5, stochastic=True, stream_name="sample"
        )
        sampler_stoch = SimpleTestSampler(config_stoch, rngs=create_rngs(seed=42))
        assert sampler_stoch.stochastic is True

    def test_complex_sampling_patterns(self):
        """Test complex sampling patterns."""
        sampler = SimpleTestSampler(dataset_size=10)

        # Test various sampling sizes
        for n in [1, 3, 5, 10]:
            result = sampler.sample(n)
            assert len(result) == n
            assert all(0 <= idx < 10 for idx in result)

    def test_performance_with_large_datasets(self):
        """Test performance doesn't degrade with large datasets."""
        import time

        sampler = SimpleTestSampler(dataset_size=1000)

        start_time = time.time()
        result = sampler(100)
        end_time = time.time()

        # Should complete quickly
        assert end_time - start_time < 1.0
        assert len(result) == 100

    def test_memory_efficiency(self):
        """Test memory efficiency of sampling operations."""
        config = SimpleTestSamplerConfig(dataset_size=1000, cacheable=True)
        sampler = SimpleTestSampler(config)

        # Multiple calls should not cause memory issues
        for _ in range(10):
            result = sampler(50)
            assert len(result) == 50

    def test_serialization_compatibility(self):
        """Test serialization compatibility with enhanced features."""
        config = SimpleTestSamplerConfig(dataset_size=5, cacheable=True)
        sampler = SimpleTestSampler(config)

        # Perform some operations
        sampler(3)

        # Should be able to get and set state
        state = sampler.get_state()
        assert isinstance(state, dict)

        new_sampler = SimpleTestSampler(dataset_size=5)
        new_sampler.set_state(state)


class TestSamplerModuleDocumentation:
    """Test that documentation examples work correctly."""

    def test_basic_usage_example(self):
        """Test basic SamplerModule usage example."""
        # Create a simple sampler
        sampler = SimpleTestSampler(dataset_size=10)

        # Sample indices
        indices = sampler.sample(5)

        assert len(indices) == 5
        assert all(isinstance(idx, int) for idx in indices)
        assert all(0 <= idx < 10 for idx in indices)

    def test_caching_example(self):
        """Test caching functionality example."""
        config = SimpleTestSamplerConfig(dataset_size=5, cacheable=True)
        sampler = SimpleTestSampler(config)

        # First call using enhanced interface
        result1 = sampler(3)
        # Second call should use cache
        result2 = sampler(3)

        assert result1 == result2
        assert len(sampler._cache) > 0

    def test_statistics_example(self):
        """Test statistics computation example."""

        def compute_stats(indices):
            return {"count": len(indices), "sum": sum(indices)}

        config = SimpleTestSamplerConfig(dataset_size=5, batch_stats_fn=compute_stats)
        sampler = SimpleTestSampler(config)

        # Use enhanced interface
        sampler(3)

        # Should have computed statistics
        assert hasattr(sampler, "_last_computed_stats")
        stats = sampler._last_computed_stats
        assert stats["count"] == 3
        assert stats["sum"] == sum([0, 1, 2])


class TestSamplerModuleAdditionalCoverage:
    """Additional tests to ensure complete coverage of SamplerModule."""

    def test_reset_method(self):
        """Test the reset method for epoch management."""
        sampler = SimpleTestSampler(dataset_size=10)

        # Call reset with seed
        sampler.reset(seed=42)
        # Method should complete without errors (it's a no-op in base class)

        # Call reset without seed
        sampler.reset()
        # Should also complete without errors

    def test_negative_sampling_error(self):
        """Test that negative sampling raises ValueError."""
        sampler = SimpleTestSampler(dataset_size=5)

        with pytest.raises(ValueError) as exc_info:
            sampler(-1)
        assert "non-negative" in str(exc_info.value)

    def test_compute_cache_key_internal(self):
        """Test _compute_cache_key method directly."""
        config = SimpleTestSamplerConfig(dataset_size=5, cacheable=True)
        sampler = SimpleTestSampler(config)

        # Test that cache keys are consistent for same input
        key1a = sampler._compute_cache_key(5)
        key1b = sampler._compute_cache_key(5)
        assert key1a == key1b

        # Test that cache keys differ for different inputs
        key2 = sampler._compute_cache_key(10)
        assert key1a != key2

    def test_compute_statistics_internal(self):
        """Test _compute_statistics method directly."""

        def stats_fn(indices):
            return {"mean": sum(indices) / len(indices) if indices else 0}

        config = SimpleTestSamplerConfig(dataset_size=5, batch_stats_fn=stats_fn)
        sampler = SimpleTestSampler(config)

        # Test with valid data
        stats = sampler._compute_statistics([1, 2, 3])
        assert stats["mean"] == 2.0

        # Test with empty data
        stats = sampler._compute_statistics([])
        assert stats["mean"] == 0

    def test_sample_impl_method(self):
        """Test the _sample_impl method which is overridden in SimpleTestSampler."""
        sampler = SimpleTestSampler(dataset_size=8)

        # Test _sample_impl directly
        result = sampler._sample_impl(5)
        assert len(result) == 5
        assert result == [0, 1, 2, 3, 4]

        # Test with n > dataset_size (should cycle)
        result = sampler._sample_impl(12)
        assert len(result) == 12
        assert result == [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]

    def test_iterator_protocol(self):
        """Test the iterator protocol implementation."""
        sampler = SimpleTestSampler(dataset_size=3)

        # Test __iter__ returns self
        assert sampler.__iter__() == sampler
        assert sampler.current_index.get_value() == 0

        # Test __next__ yields correct values
        assert sampler.__next__() == 0
        assert sampler.__next__() == 1
        assert sampler.__next__() == 2

        # Test StopIteration is raised
        with pytest.raises(StopIteration):
            sampler.__next__()

    def test_len_method(self):
        """Test the __len__ method."""
        sampler = SimpleTestSampler(dataset_size=7)
        assert len(sampler) == 7

    def test_state_with_stream_name(self):
        """Test that stream_name is preserved in state management."""
        sampler = SimpleTestSampler(dataset_size=5)
        sampler.stream_name = "custom_stream"

        # Get state
        state = sampler.get_state()
        assert "stream_name" in state
        assert state["stream_name"] == "custom_stream"

        # Create new sampler and restore state
        new_sampler = SimpleTestSampler(dataset_size=5)
        new_sampler.set_state(state)
        assert new_sampler.stream_name == "custom_stream"

    def test_sample_direct_call(self):
        """Test the sample method directly which has special behavior."""
        sampler = SimpleTestSampler(dataset_size=5)

        # Call sample directly (not through __call__)
        result = sampler.sample(5)
        assert len(result) == 5

        # Call again
        result = sampler.sample(5)
        assert len(result) == 5

    def test_zero_dataset_size(self):
        """Test behavior with zero dataset size."""
        sampler = SimpleTestSampler(dataset_size=0)

        # Should handle gracefully
        result = sampler(5)
        assert len(result) == 5
        assert all(idx == 0 for idx in result)  # All indices should be 0

    def test_requires_rng_streams(self):
        """Test the requires_rng_streams method."""
        # Default stream_name is None for non-stochastic samplers
        sampler = SimpleTestSampler(dataset_size=5)
        streams = sampler.requires_rng_streams()
        # Non-stochastic samplers return None (no RNG required)
        assert streams is None

        # Stochastic sampler with custom stream
        config = SimpleTestSamplerConfig(dataset_size=5, stochastic=True, stream_name="my_stream")
        sampler_stoch = SimpleTestSampler(config, rngs=create_rngs(seed=42))
        streams = sampler_stoch.requires_rng_streams()
        assert streams == ["my_stream"]

    def test_operation_stats(self):
        """Test operation statistics tracking from DataraxModule."""
        sampler = SimpleTestSampler(dataset_size=5)

        # Get initial stats
        stats = sampler.get_operation_stats()
        assert stats["applied_count"] == 0
        assert stats["skipped_count"] == 0

        # Increment applied count (inherited method)
        sampler._increment_applied_count()
        stats = sampler.get_operation_stats()
        assert stats["applied_count"] == 1

        # Increment skipped count
        sampler._increment_skipped_count()
        stats = sampler.get_operation_stats()
        assert stats["skipped_count"] == 1

        # Reset stats
        sampler.reset_operation_stats()
        stats = sampler.get_operation_stats()
        assert stats["applied_count"] == 0
        assert stats["skipped_count"] == 0

    def test_clone_method(self):
        """Test the clone method from DataraxModule."""
        sampler = SimpleTestSampler(dataset_size=5)

        # Clone the sampler
        cloned = sampler.clone()

        # Check it's a different instance
        assert cloned is not sampler

        # Check state is preserved
        assert cloned.dataset_size == 5

    def test_precomputed_stats_with_dict(self):
        """Test precomputed statistics stored as dict."""
        precomputed = {"mean": 5.0, "std": 2.0}
        config = SimpleTestSamplerConfig(dataset_size=5, precomputed_stats=precomputed)
        sampler = SimpleTestSampler(config)

        # Access precomputed stats through _compute_statistics
        stats = sampler._compute_statistics([])
        assert stats["mean"] == 5.0
        assert stats["std"] == 2.0

    def test_sample_with_dataset_size_none(self):
        """Test sample method when dataset_size starts as None."""
        # Create a modified sampler where dataset_size can be None
        sampler = SimpleTestSampler(dataset_size=5)
        sampler.dataset_size = None

        # Call sample with n
        sampler.sample(10)

        # dataset_size should be set
        assert sampler.dataset_size == 10

    def test_default_sample_implementation(self):
        """Test the default sample implementation that uses iterator."""

        @dataclass
        class IteratorOnlySamplerConfig(StructuralConfig):
            """Config for IteratorOnlySampler."""

            dataset_size: int = 10

        class IteratorOnlySampler(SamplerModule):
            """Sampler that only implements iterator protocol."""

            def __init__(
                self,
                config: IteratorOnlySamplerConfig | None = None,
                *,
                dataset_size: int = 10,
                rngs: nnx.Rngs | None = None,
            ):
                if config is None:
                    config = IteratorOnlySamplerConfig(dataset_size=dataset_size)
                super().__init__(config, rngs=rngs)
                self.dataset_size = config.dataset_size
                self.idx = 0

            def __iter__(self):
                self.idx = 0
                return self

            def __next__(self):
                if self.idx >= self.dataset_size:
                    raise StopIteration
                val = self.idx
                self.idx += 1
                return val

            def __len__(self):
                return self.dataset_size

        sampler = IteratorOnlySampler(dataset_size=5)
        result = sampler.sample(5)
        assert result == [0, 1, 2, 3, 4]

    def test_cache_hit_path(self):
        """Test the cache hit path in __call__."""
        config = SimpleTestSamplerConfig(dataset_size=5, cacheable=True)
        sampler = SimpleTestSampler(config)

        # First call - cache miss
        result1 = sampler(3)

        # Second call - should hit cache
        result2 = sampler(3)
        assert result1 == result2

        # Different input should miss cache
        result3 = sampler(5)
        assert len(result3) == 5
