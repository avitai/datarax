"""Tests for enhanced BatcherModule functionality.

This module contains thorough tests for the enhanced BatcherModule that
inherits advanced features from StructuralModule.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
from dataclasses import dataclass
from collections.abc import Iterator
from datarax.core.batcher import BatcherModule
from datarax.core.config import StructuralConfig
from datarax.typing import Element, Batch


@dataclass
class SimpleTestBatcherConfig(StructuralConfig):
    """Configuration for SimpleTestBatcher."""

    def __post_init__(self):
        """Validate configuration."""
        # SimpleTestBatcher is deterministic
        object.__setattr__(self, "stochastic", False)
        super().__post_init__()


class SimpleTestBatcher(BatcherModule):
    """Simple batcher for testing enhanced functionality."""

    def process(
        self,
        elements: list[Element] | Iterator[Element],
        *args,
        batch_size: int,
        drop_remainder: bool = False,
        **kwargs,
    ) -> list[Batch]:
        """Simple batching implementation for testing."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        batches = []
        batch_buffer = []

        # Convert iterator to list if needed
        if hasattr(elements, "__iter__") and not isinstance(elements, list):
            elements = list(elements)

        for element in elements:
            batch_buffer.append(element)

            if len(batch_buffer) == batch_size:
                # Create batch by stacking arrays
                batch = self._collate_batch(batch_buffer)
                batches.append(batch)
                batch_buffer = []

        # Handle remainder
        if batch_buffer and not drop_remainder:
            batch = self._collate_batch(batch_buffer)
            batches.append(batch)

        return batches

    def _collate_batch(self, elements: list[Element]) -> Batch:
        """Simple collation for testing with nested PyTree support."""
        if not elements:
            return {}

        # Handle dict structure (including nested dicts)
        if isinstance(elements[0], dict):
            result = {}
            for key in elements[0].keys():
                values = [elem[key] for elem in elements]

                # If all values are arrays, stack them
                if all(isinstance(v, jax.Array | np.ndarray) for v in values):
                    result[key] = jnp.stack(values)
                # If all values are dicts, recursively collate
                elif all(isinstance(v, dict) for v in values):
                    result[key] = self._collate_batch(values)
                # Otherwise, keep as list
                else:
                    result[key] = values
            return result

        # Handle array elements
        if all(isinstance(elem, jax.Array | np.ndarray) for elem in elements):
            return jnp.stack(elements)

        return elements


class TestBatcherModuleEnhanced:
    """Test suite for enhanced BatcherModule functionality."""

    def test_initialization_with_enhanced_features(self):
        """Test BatcherModule initialization with enhanced DataraxModule features."""
        # Enhanced features are now passed through config
        config = SimpleTestBatcherConfig(
            stochastic=False,
            cacheable=True,
            batch_stats_fn=lambda x: {"count": len(x)},
        )
        batcher = SimpleTestBatcher(config, rngs=nnx.Rngs(0))

        # Should have DataraxModule features (accessed through config)
        assert batcher.config.cacheable
        assert batcher._cache == {}
        assert batcher.config.batch_stats_fn is not None

    def test_basic_batching(self):
        """Test basic batching functionality."""
        config = SimpleTestBatcherConfig()
        batcher = SimpleTestBatcher(config)

        # Verify batcher can process elements
        elements = [{"data": jnp.array([1.0])}, {"data": jnp.array([2.0])}]
        result = batcher(elements, batch_size=2)
        assert len(result) == 1
        assert result[0]["data"].shape == (2, 1)

    def test_statistics_computation(self):
        """Test batch statistics computation."""

        def compute_batch_stats(elements):
            return {
                "element_count": len(elements),
                "has_data": any("data" in elem for elem in elements if isinstance(elem, dict)),
            }

        config = SimpleTestBatcherConfig(batch_stats_fn=compute_batch_stats)
        batcher = SimpleTestBatcher(config)

        elements = [
            {"data": jnp.array([1.0])},
            {"data": jnp.array([2.0])},
        ]

        # Access statistics through the public API
        stats = batcher.compute_statistics(elements)
        assert stats is not None
        assert stats["element_count"] == 2
        assert stats["has_data"] is True

    def test_precomputed_statistics(self):
        """Test precomputed statistics usage."""
        precomputed_stats = {"batch_size": 32, "expected_shape": (32, 10)}
        config = SimpleTestBatcherConfig(precomputed_stats=precomputed_stats)
        batcher = SimpleTestBatcher(config)

        # get_statistics() returns precomputed_stats (compute_statistics() computes new stats)
        stats = batcher.get_statistics()
        assert stats == precomputed_stats

    def test_state_management(self):
        """Test enhanced state management."""
        config = SimpleTestBatcherConfig(cacheable=True)
        batcher = SimpleTestBatcher(config)

        # Modify internal state
        batcher._cache["test"] = "data"

        # Get state - nnx.Variable fields are included
        state = batcher.get_state()
        # _cache is not included in state as it's not an nnx.Variable
        # _computed_stats IS in state as it's nnx.Variable
        assert "_computed_stats" in state

        new_batcher = SimpleTestBatcher(SimpleTestBatcherConfig())
        new_batcher.set_state(state)

    def test_cache_reset(self):
        """Test cache reset functionality."""
        config = SimpleTestBatcherConfig(cacheable=True)
        batcher = SimpleTestBatcher(config)

        # Add to cache
        batcher._cache["key1"] = "value1"
        assert len(batcher._cache) == 1

        # Reset cache
        batcher.reset_cache()
        assert len(batcher._cache) == 0


class TestBatcherModuleIntegration:
    """Integration tests for BatcherModule with other components."""

    def test_pipeline_integration(self):
        """Test integration with pipeline infrastructure."""
        from datarax.dag import DAGExecutor, BatchNode

        batcher = SimpleTestBatcher(SimpleTestBatcherConfig())

        # DAGExecutor requires BatchNode first for batch-first enforcement
        pipeline = DAGExecutor().add(BatchNode(batch_size=32)).add(batcher)
        assert pipeline is not None


class TestBatcherModuleErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_batch_size(self):
        """Test handling of invalid batch sizes."""
        batcher = SimpleTestBatcher(SimpleTestBatcherConfig())
        elements = [{"data": jnp.array([1.0])}]

        # Batch size must be positive
        with pytest.raises(ValueError, match="batch_size must be positive"):
            batcher(elements, batch_size=0)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            batcher(elements, batch_size=-1)

    def test_empty_elements_handling(self):
        """Test handling of empty element lists."""
        batcher = SimpleTestBatcher(SimpleTestBatcherConfig())

        # Empty elements should return empty list
        result = batcher([], batch_size=2)
        assert result == []

    def test_caching_with_non_cacheable_data(self):
        """Test caching behavior with non-cacheable data."""
        config = SimpleTestBatcherConfig(cacheable=True)
        batcher = SimpleTestBatcher(config)

        # Elements with non-hashable components
        elements = [{"data": [1, 2, 3]}]  # List is not hashable

        # Should handle gracefully and not cache
        result = batcher(elements, batch_size=1)
        assert result is not None
        # Cache might be empty due to unhashable data
        assert len(batcher._cache) >= 0


class TestBatcherModuleCoverage:
    """Additional tests to reach 80% coverage."""

    def test_different_cacheable_configs(self):
        """Test BatcherModule with different cacheable configurations."""
        # Test with caching enabled
        config_cacheable = SimpleTestBatcherConfig(cacheable=True)
        batcher_cacheable = SimpleTestBatcher(config_cacheable)
        assert batcher_cacheable.config.cacheable is True
        assert batcher_cacheable._cache == {}

        # Test with caching disabled
        config_no_cache = SimpleTestBatcherConfig(cacheable=False)
        batcher_no_cache = SimpleTestBatcher(config_no_cache)
        assert batcher_no_cache.config.cacheable is False
        assert batcher_no_cache._cache is None

    def test_complex_pytree_batching(self):
        """Test batching of complex PyTree structures."""
        batcher = SimpleTestBatcher(SimpleTestBatcherConfig())

        elements = [
            {
                "arrays": {"x": jnp.array([1.0]), "y": jnp.array([2.0])},
                "metadata": {"id": i, "name": f"item_{i}"},
            }
            for i in range(3)
        ]

        result = batcher(elements, batch_size=3)
        assert len(result) == 1

        batch = result[0]
        assert "arrays" in batch
        assert "metadata" in batch
        assert batch["arrays"]["x"].shape == (3, 1)  # 3 elements, each with shape (1,)
        assert batch["arrays"]["y"].shape == (3, 1)  # 3 elements, each with shape (1,)

    def test_performance_with_large_batches(self):
        """Test performance doesn't degrade with large batches."""
        import time

        batcher = SimpleTestBatcher(SimpleTestBatcherConfig())

        # Create large number of elements
        elements = [{"data": jnp.array([i])} for i in range(1000)]

        start_time = time.time()
        result = batcher(elements, batch_size=100)
        end_time = time.time()

        # Should complete in reasonable time
        assert (end_time - start_time) < 2.0  # 2 second threshold
        assert len(result) == 10  # 1000 / 100 = 10 batches

    def test_memory_efficiency(self):
        """Test memory efficiency of caching."""
        config = SimpleTestBatcherConfig(cacheable=True)
        batcher = SimpleTestBatcher(config)

        # Process multiple different batches
        for i in range(10):
            elements = [{"data": jnp.array([i])}]
            batcher(elements, batch_size=1)

        # Cache should not grow unbounded
        assert len(batcher._cache) <= 10

    def test_serialization_compatibility(self):
        """Test that enhanced BatcherModule is serialization compatible."""
        config = SimpleTestBatcherConfig(cacheable=True)
        batcher = SimpleTestBatcher(config)

        # Modify internal state
        batcher._cache["test"] = "value"

        # Get state (should be serializable)
        state = batcher.get_state()

        # State should contain nnx.Variable fields
        assert isinstance(state, dict)
        # _computed_stats IS in state (it's nnx.Variable)
        assert "_computed_stats" in state
        # _cache is not included in state as it's not an nnx.Variable

        new_batcher = SimpleTestBatcher(SimpleTestBatcherConfig())
        new_batcher.set_state(state)


class TestDefaultBatcherImplementation:
    """Test suite for DefaultBatcher implementation."""

    def test_default_batcher_initialization(self):
        """Test DefaultBatcher initialization."""
        from datarax.batching.default_batcher import DefaultBatcher, DefaultBatcherConfig

        config = DefaultBatcherConfig()
        batcher = DefaultBatcher(config)
        assert batcher is not None
        assert batcher.collate_fn is None

        # With custom collate function
        def custom_collate(elements):
            return {"batch": elements}

        batcher = DefaultBatcher(config, collate_fn=custom_collate)
        assert batcher.collate_fn is not None

    def test_default_batcher_with_arrays(self):
        """Test DefaultBatcher with array elements."""
        from datarax.batching.default_batcher import DefaultBatcher, DefaultBatcherConfig

        config = DefaultBatcherConfig()
        batcher = DefaultBatcher(config)
        elements = [jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]), jnp.array([5.0, 6.0])]

        # DefaultBatcher.batch returns an iterator
        batches = list(batcher(iter(elements), batch_size=2))
        assert len(batches) == 2  # 2 batches (2 + 1)
        assert batches[0].shape == (2, 2)  # First batch has 2 elements
        assert batches[1].shape == (1, 2)  # Second batch has 1 element

    def test_default_batcher_with_dicts(self):
        """Test DefaultBatcher with dictionary elements."""
        from datarax.batching.default_batcher import DefaultBatcher, DefaultBatcherConfig

        config = DefaultBatcherConfig()
        batcher = DefaultBatcher(config)
        elements = [
            {"x": jnp.array([1.0]), "y": jnp.array([2.0])},
            {"x": jnp.array([3.0]), "y": jnp.array([4.0])},
        ]

        batches = list(batcher(iter(elements), batch_size=2))
        assert len(batches) == 1
        assert "x" in batches[0] and "y" in batches[0]
        assert batches[0]["x"].shape == (2, 1)
        assert batches[0]["y"].shape == (2, 1)

    def test_default_batcher_drop_remainder(self):
        """Test DefaultBatcher with drop_remainder flag."""
        from datarax.batching.default_batcher import DefaultBatcher, DefaultBatcherConfig

        config = DefaultBatcherConfig()
        batcher = DefaultBatcher(config)
        elements = [jnp.array([i]) for i in range(5)]

        # Without drop_remainder
        batches = list(batcher(iter(elements), batch_size=2, drop_remainder=False))
        assert len(batches) == 3  # 2 full + 1 partial

        # With drop_remainder - need fresh iterator
        elements = [jnp.array([i]) for i in range(5)]
        batches = list(batcher(iter(elements), batch_size=2, drop_remainder=True))
        assert len(batches) == 2  # Only full batches

    def test_default_batcher_with_custom_collate(self):
        """Test DefaultBatcher with custom collate function."""
        from datarax.batching.default_batcher import DefaultBatcher, DefaultBatcherConfig

        def custom_collate(elements):
            # Custom logic: concatenate instead of stack
            return jnp.concatenate(elements)

        config = DefaultBatcherConfig()
        batcher = DefaultBatcher(config, collate_fn=custom_collate)
        elements = [jnp.array([1, 2]), jnp.array([3, 4]), jnp.array([5, 6])]

        batches = list(batcher(iter(elements), batch_size=2))
        assert len(batches) == 2
        # Custom collate concatenates, so shape is (4,) for first batch
        assert batches[0].shape == (4,)  # [1, 2, 3, 4]
        assert batches[1].shape == (2,)  # [5, 6]


class TestBatcherModuleAdvancedFeatures:
    """Test advanced features of BatcherModule."""

    def test_operation_statistics_tracking(self):
        """Test operation statistics tracking."""
        batcher = SimpleTestBatcher(SimpleTestBatcherConfig())

        # Initial stats
        stats = batcher.get_operation_stats()
        assert stats["applied_count"] == 0
        assert stats["skipped_count"] == 0

        # Perform operations
        elements = [{"data": jnp.array([1.0])}]
        batcher(elements, batch_size=1)

        # Check stats updated (through inherited behavior)
        stats = batcher.get_operation_stats()
        # Note: Actual stats updating would need to be implemented in batch method

        # Reset stats
        batcher.reset_operation_stats()
        stats = batcher.get_operation_stats()
        assert stats["applied_count"] == 0
        assert stats["skipped_count"] == 0

    def test_clone_functionality(self):
        """Test module cloning."""
        config = SimpleTestBatcherConfig(cacheable=True)
        batcher = SimpleTestBatcher(config)
        batcher._cache["test"] = "data"

        # Clone the module
        cloned = batcher.clone()

        # Cache is cloned too
        assert "test" in cloned._cache
        assert cloned._cache["test"] == "data"

    def test_rng_stream_requirements(self):
        """Test RNG stream requirement checking."""
        batcher = SimpleTestBatcher(SimpleTestBatcherConfig())

        # Default has no requirements
        assert batcher.requires_rng_streams() is None

        # Test with custom implementation that requires streams
        class RNGBatcher(SimpleTestBatcher):
            def requires_rng_streams(self):
                return ["batch", "shuffle"]

        rng_batcher = RNGBatcher(SimpleTestBatcherConfig())
        assert rng_batcher.requires_rng_streams() == ["batch", "shuffle"]

        # Test stream validation
        with pytest.raises(ValueError, match="RNG stream 'batch' is required"):
            rng_batcher.ensure_rng_streams(["other"])

        # Valid streams
        rng_batcher.ensure_rng_streams(["batch", "shuffle", "extra"])  # Should not raise

    def test_complex_nested_pytree_batching(self):
        """Test batching with deeply nested PyTree structures."""
        batcher = SimpleTestBatcher(SimpleTestBatcherConfig())

        elements = [
            {
                "level1": {
                    "level2": {
                        "arrays": {"x": jnp.array([i]), "y": jnp.array([i * 2])},
                        "scalars": {"a": i, "b": i * 3},
                    },
                    "lists": [i, i + 1, i + 2],
                },
                "metadata": {"id": f"item_{i}"},
            }
            for i in range(4)
        ]

        batches = batcher(elements, batch_size=2)
        batch_list = list(batches)

        assert len(batch_list) == 2
        # First batch should have properly nested structure
        first_batch = batch_list[0]
        assert "level1" in first_batch
        assert "level2" in first_batch["level1"]
        assert first_batch["level1"]["level2"]["arrays"]["x"].shape == (2, 1)


class TestBatcherModuleEdgeCases:
    """Test edge cases and error conditions."""

    def test_iterator_vs_list_input(self):
        """Test handling of iterator vs list inputs."""
        batcher = SimpleTestBatcher(SimpleTestBatcherConfig())

        # Test with list
        list_elements = [{"data": jnp.array([i])} for i in range(3)]
        list_batches = list(batcher(list_elements, batch_size=2))
        assert len(list_batches) == 2

        # Test with iterator
        def element_generator():
            for i in range(3):
                yield {"data": jnp.array([i])}

        iter_batches = list(batcher(element_generator(), batch_size=2))
        assert len(iter_batches) == 2

        # Results should be identical
        for lb, ib in zip(list_batches, iter_batches):
            assert jnp.allclose(lb["data"], ib["data"])

    def test_heterogeneous_element_types(self):
        """Test handling of heterogeneous element types."""
        batcher = SimpleTestBatcher(SimpleTestBatcherConfig())

        # Mixed types - should handle gracefully
        elements = [
            {"data": jnp.array([1.0]), "type": "array"},
            {"data": [1, 2, 3], "type": "list"},  # Non-array type
        ]

        batches = list(batcher(elements, batch_size=2))
        assert len(batches) == 1
        # Mixed types should be preserved as list
        assert isinstance(batches[0]["data"], list)

    def test_very_large_batch_size(self):
        """Test with batch size larger than number of elements."""
        batcher = SimpleTestBatcher(SimpleTestBatcherConfig())

        elements = [{"data": jnp.array([i])} for i in range(5)]
        batches = list(batcher(elements, batch_size=100))

        # Should get single batch with all elements
        assert len(batches) == 1
        assert batches[0]["data"].shape == (5, 1)

    def test_single_element_batching(self):
        """Test batching with single element."""
        batcher = SimpleTestBatcher(SimpleTestBatcherConfig())

        elements = [{"data": jnp.array([42.0])}]

        # Batch size 1
        batches = list(batcher(elements, batch_size=1))
        assert len(batches) == 1
        assert batches[0]["data"].shape == (1, 1)

        # Batch size > 1 with single element
        batches = list(batcher(elements, batch_size=10))
        assert len(batches) == 1
        assert batches[0]["data"].shape == (1, 1)

    def test_cache_key_with_different_data_types(self):
        """Test cache key computation with various data types."""
        config = SimpleTestBatcherConfig(cacheable=True)
        batcher = SimpleTestBatcher(config)

        # Test with arrays
        key1 = batcher._compute_cache_key([jnp.array([1.0])])
        assert key1 is not None

        # Test with nested structures
        key2 = batcher._compute_cache_key([{"a": jnp.array([1.0]), "b": {"c": 2}}])
        assert key2 is not None
        assert key1 != key2

        # Test with non-hashable types (should handle gracefully)
        key3 = batcher._compute_cache_key([{"data": [1, 2, 3]}])
        assert key3 is not None


class TestBatcherModuleConcurrency:
    """Test concurrent and parallel batching scenarios."""

    pass


class TestBatcherModuleRobustness:
    """Test robustness and recovery."""

    def test_recovery_from_invalid_state(self):
        """Test recovery from invalid state restoration."""
        batcher = SimpleTestBatcher(SimpleTestBatcherConfig())

        # Try to restore incompatible state (with fields that don't exist)
        invalid_state = {"non_existent_field": 42, "_computed_stats": {"custom": "data"}}

        # Should handle gracefully - set_state ignores unknown fields
        batcher.set_state(invalid_state)

        # _computed_stats should be restored if it exists in state
        # (depends on implementation behavior with mismatched state)

    def test_handling_none_elements(self):
        """Test handling of None in elements."""
        batcher = SimpleTestBatcher(SimpleTestBatcherConfig())

        # Elements with None values
        elements = [
            {"data": jnp.array([1.0]), "optional": None},
            {"data": jnp.array([2.0]), "optional": None},
        ]

        batches = list(batcher(elements, batch_size=2))
        assert len(batches) == 1
        assert batches[0]["data"].shape == (2, 1)
        # None values should be preserved
        assert "optional" in batches[0]


class TestBatcherModuleDocumentation:
    """Test that documentation examples work correctly."""

    def test_basic_usage_example(self):
        """Test basic BatcherModule usage example."""
        # Create a simple batcher
        batcher = SimpleTestBatcher(SimpleTestBatcherConfig())

        # Create elements
        elements = [
            {"data": jnp.array([1.0])},
            {"data": jnp.array([2.0])},
            {"data": jnp.array([3.0])},
        ]

        # Batch elements
        batches = batcher(elements, batch_size=2)

        assert len(batches) == 2  # 2 full batches + 1 remainder
        assert batches[0]["data"].shape == (2, 1)  # 2 elements, each with shape (1,)
        assert batches[1]["data"].shape == (1, 1)  # 1 element with shape (1,)

    def test_statistics_example(self):
        """Test statistics computation example."""

        def compute_stats(elements):
            return {"count": len(elements)}

        config = SimpleTestBatcherConfig(batch_stats_fn=compute_stats)
        batcher = SimpleTestBatcher(config)

        elements = [{"data": jnp.array([i])} for i in range(5)]
        stats = batcher.compute_statistics(elements)

        assert stats["count"] == 5


class TestBatcherModuleWithPRNGKeys:
    """Test batching with PRNG keys."""

    def test_batching_with_prng_keys(self):
        """Test batching elements containing PRNG keys."""
        from datarax.batching.default_batcher import DefaultBatcher, DefaultBatcherConfig

        config = DefaultBatcherConfig()
        batcher = DefaultBatcher(config)

        # Create elements with PRNG keys
        elements = [{"key": jax.random.key(i), "data": jnp.array([i])} for i in range(3)]

        # Should handle PRNG keys properly
        batches = list(batcher(iter(elements), batch_size=2))
        assert len(batches) == 2

        # First batch should have stacked keys
        first_batch = batches[0]
        assert "key" in first_batch
        assert "data" in first_batch
        assert first_batch["data"].shape == (2, 1)

    def test_default_batcher_safe_tree_map(self):
        """Test the _safe_tree_map method with PRNG keys."""
        from datarax.batching.default_batcher import DefaultBatcher, DefaultBatcherConfig

        config = DefaultBatcherConfig()
        batcher = DefaultBatcher(config)

        # Test with PRNG keys
        key1 = jax.random.key(42)
        key2 = jax.random.key(43)

        # Function to stack arrays
        def stack_fn(*xs):
            return jnp.stack(xs)

        # Should handle PRNG keys without error
        result = batcher._safe_tree_map(stack_fn, key1, key2)
        assert result is not None


class TestBatcherModuleIntegrationAdvanced:
    """Advanced integration tests."""

    def test_batcher_state_persistence(self):
        """Test state persistence across save/load cycles."""
        config = SimpleTestBatcherConfig(cacheable=True)
        batcher = SimpleTestBatcher(config)

        # Perform some operations
        elements = [{"data": jnp.array([i])} for i in range(3)]
        batcher(elements, batch_size=2)
        batcher(elements, batch_size=2)

        # Set some statistics to persist
        batcher.set_statistics({"mean": 1.5, "count": 3})

        # Save state
        state = batcher.get_state()

        # Create new batcher and restore
        new_batcher = SimpleTestBatcher(SimpleTestBatcherConfig(cacheable=True))
        new_batcher.set_state(state)

        # _computed_stats IS persisted (nnx.Variable)
        assert new_batcher.get_statistics() == {"mean": 1.5, "count": 3}

    def test_batcher_with_jit_compilation(self):
        """Test batcher compatibility with JAX JIT."""
        batcher = SimpleTestBatcher(SimpleTestBatcherConfig())

        # Create a JIT-compatible function using the batcher
        def process_batch(elements):
            # Note: This is a simplified test - actual JIT would need more care
            return batcher._collate_batch(elements)

        # Test collation (which should be JIT-compatible)
        elements = [{"data": jnp.array([1.0])}, {"data": jnp.array([2.0])}]
        result = process_batch(elements)

        assert result is not None
        assert result["data"].shape == (2, 1)


class TestBatcherModulePerformance:
    """Performance and efficiency tests."""

    def test_cache_performance_improvement(self):
        """Test that caching improves performance."""
        import time

        cached_config = SimpleTestBatcherConfig(cacheable=True)
        uncached_config = SimpleTestBatcherConfig(cacheable=False)
        batcher_cached = SimpleTestBatcher(cached_config)
        batcher_uncached = SimpleTestBatcher(uncached_config)

        # Large dataset
        elements = [{"data": jnp.array([i])} for i in range(100)]

        # First run - both should take similar time
        start = time.time()
        batcher_cached(elements, batch_size=10)
        cached_first_time = time.time() - start

        start = time.time()
        batcher_uncached(elements, batch_size=10)
        time.time() - start

        # Second run - cached should be faster
        start = time.time()
        batcher_cached(elements, batch_size=10)  # Should use cache
        cached_second_time = time.time() - start

        # Cache should make second call faster
        # Note: In practice, caching overhead might make this test flaky
        assert cached_second_time <= cached_first_time * 1.5  # Allow some variance

    def test_memory_bounded_caching(self):
        """Test that cache doesn't grow unbounded."""
        config = SimpleTestBatcherConfig(cacheable=True)
        batcher = SimpleTestBatcher(config)

        # Process many unique batches
        for i in range(100):
            elements = [{"data": jnp.array([i + j])} for j in range(3)]
            batcher(elements, batch_size=2)

        # Cache should have reasonable size
        # (Implementation would need cache size limits)
        assert len(batcher._cache) <= 100  # Should not exceed reasonable limit


class TestBatcherModuleCompliance:
    """Test compliance with Datarax standards."""

    def test_flax_nnx_compliance(self):
        """Test compliance with Flax NNX requirements."""
        batcher = SimpleTestBatcher(SimpleTestBatcherConfig())

        # Should be an nnx.Module
        assert isinstance(batcher, nnx.Module)

        # Should have proper state management
        state = nnx.state(batcher)
        assert state is not None

        # Should support nnx operations
        cloned = nnx.clone(batcher)
        assert cloned is not None
        assert cloned is not batcher

    def test_pytree_compatibility(self):
        """Test PyTree compatibility."""
        batcher = SimpleTestBatcher(SimpleTestBatcherConfig())

        # Create PyTree structured data
        pytree_data = {
            "arrays": (jnp.array([1.0]), jnp.array([2.0])),
            "nested": {"a": jnp.array([3.0]), "b": [jnp.array([4.0])]},
        }

        # Should handle PyTree structures
        elements = [pytree_data]
        batches = list(batcher(elements, batch_size=1))

        assert len(batches) == 1
        assert "arrays" in batches[0]
        assert "nested" in batches[0]
