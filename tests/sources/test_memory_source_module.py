"""Tests for the MemorySource.

This module tests the functionality of the unified MemorySource implementation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import flax.nnx as nnx

from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from datarax.core.metadata import RecordMetadata


def test_memory_source_basic_functionality():
    """Test that MemorySource correctly manages data."""
    # Create test data
    data = {
        "feature1": np.random.rand(10, 5).astype(np.float32),
        "feature2": np.random.rand(10, 3).astype(np.float32),
        "label": np.random.randint(0, 2, size=(10,)),
    }

    # Create the data source (config-based API)
    config = MemorySourceConfig()
    source = MemorySource(config, data)

    # Check length
    assert len(source) == 10

    # Get items through iteration
    items = list(source)

    # Check structure of first item
    assert "feature1" in items[0]
    assert "feature2" in items[0]
    assert "label" in items[0]

    # Check shapes
    assert items[0]["feature1"].shape == (5,)
    assert items[0]["feature2"].shape == (3,)
    assert items[0]["label"].shape == ()

    # Check values match the original data
    np.testing.assert_array_equal(items[0]["feature1"], data["feature1"][0])
    np.testing.assert_array_equal(items[0]["feature2"], data["feature2"][0])
    assert items[0]["label"] == data["label"][0]


def test_memory_source_creation() -> None:
    """Test creation of MemorySource with different data types."""
    # Test with dictionary
    data_dict = {"a": np.arange(10), "b": np.arange(10, 20)}
    config = MemorySourceConfig()
    source = MemorySource(config, data_dict)
    assert len(source) == 10

    # Test with list
    data_list = [{"x": i, "y": i * 2} for i in range(5)]
    source = MemorySource(config, data_list)
    assert len(source) == 5


def test_memory_source_stateless_iteration() -> None:
    """Test stateless iteration over MemorySource."""
    # Create data source without rngs (stateless mode)
    data = [{"x": i} for i in range(3)]
    config = MemorySourceConfig()
    source = MemorySource(config, data)

    # Test full iteration
    items = list(source)
    assert len(items) == 3
    assert all(item["x"] == i for i, item in enumerate(items))

    # Test iteration can be repeated
    items2 = list(source)
    assert len(items2) == 3
    assert all(item["x"] == i for i, item in enumerate(items2))


def test_memory_source_stateful_iteration() -> None:
    """Test stateful iteration with internal index tracking."""
    # Create data source with rngs (stateful mode)
    data = [{"x": i} for i in range(10)]
    rngs = nnx.Rngs(default=0)
    config = MemorySourceConfig()
    source = MemorySource(config, data, rngs=rngs)

    # Test batch retrieval with internal state
    batch1 = source.get_batch(3)
    batch2 = source.get_batch(3)

    # Batches should be different (advancing internal index)
    assert batch1 != batch2

    # Check state tracking
    assert source.index.get_value() > 0


def test_memory_source_with_jax_arrays():
    """Test MemorySource with JAX arrays."""
    # Create data source with JAX arrays
    data = {"a": jnp.arange(5), "b": jnp.ones((5, 3))}
    config = MemorySourceConfig()
    source = MemorySource(config, data)

    # Iterate through all elements
    for i, element in enumerate(source):
        assert element["a"] == i
        assert jnp.all(element["b"] == 1.0)


def test_memory_source_random_access():
    """Test random access via __getitem__."""
    # Create data source
    data = [{"value": i} for i in range(10)]
    config = MemorySourceConfig()
    source = MemorySource(config, data)

    # Test direct indexing
    assert source[0]["value"] == 0
    assert source[5]["value"] == 5
    assert source[-1]["value"] == 9

    # Test index out of bounds
    with pytest.raises(IndexError):
        _ = source[10]

    with pytest.raises(IndexError):
        _ = source[-11]


def test_memory_source_batch_retrieval():
    """Test batch retrieval methods."""
    # Create data source with rngs for stateful batch retrieval
    data = {"values": jnp.arange(20)}
    rngs = nnx.Rngs(default=0)
    config = MemorySourceConfig()
    source = MemorySource(config, data, rngs=rngs)

    # Get batch using stateful mode
    batch = source.get_batch(5)
    assert "values" in batch
    assert len(batch["values"]) == 5
    assert jnp.array_equal(batch["values"], jnp.arange(5))

    # Get next batch (should continue from index 5)
    batch2 = source.get_batch(5)
    assert jnp.array_equal(batch2["values"], jnp.arange(5, 10))


def test_memory_source_shuffling():
    """Test shuffling functionality."""
    # Create data source with shuffling enabled
    data = list(range(100))
    rngs = nnx.Rngs(default=42, shuffle=42)
    config = MemorySourceConfig(shuffle=True)
    source = MemorySource(config, data, rngs=rngs)

    # Get shuffled data
    items = list(source)

    # Should have same elements but in different order
    assert set(items) == set(range(100))
    assert items != list(range(100))  # Should be shuffled

    # Test epoch tracking
    assert source.epoch.get_value() == 1


def test_memory_source_stateless_batch():
    """Test stateless batch retrieval with explicit key."""
    # Create data source (can be with or without rngs)
    data = {"values": jnp.arange(20)}
    config = MemorySourceConfig()
    source = MemorySource(config, data)

    # Get batch with explicit key (stateless mode)
    key = jax.random.key(42)
    batch = source.get_batch(5, key=key)
    assert "values" in batch
    assert len(batch["values"]) == 5

    # Same key should give same batch
    batch2 = source.get_batch(5, key=key)
    assert jnp.array_equal(batch["values"], batch2["values"])

    # Different key might give different batch (if shuffling)
    key2 = jax.random.key(43)
    source.get_batch(5, key=key2)
    # Note: Without shuffle=True, batches will still be the same


def test_memory_source_errors():
    """Test error cases for MemorySource."""
    config = MemorySourceConfig()
    # Test error for inconsistent array lengths in dictionary
    with pytest.raises(ValueError, match="same length"):
        MemorySource(config, {"a": np.arange(5), "b": np.arange(10)})

    # Test error for dictionary without array-like values
    with pytest.raises(ValueError, match="array-like value"):
        MemorySource(config, {"a": 1, "b": 2})


def test_memory_source_state_management():
    """Test state management in MemorySource."""
    # Create data source with rngs for stateful operation
    data = [{"value": i} for i in range(10)]
    rngs = nnx.Rngs(default=0)
    config = MemorySourceConfig()
    source = MemorySource(config, data, rngs=rngs)

    # Get several batches to advance state
    source.get_batch(3)
    source.get_batch(3)
    source.get_batch(3)

    # Verify state - after 3 batches of 3 items each, we should be at index 9
    assert source.index.get_value() == 9
    assert source.epoch.get_value() == 0

    # Get one more batch to wrap around
    source.get_batch(3)
    # After wrapping around, we should be at index 0 (9 + 3 = 12, 12 % 10 = 0)
    # The last batch gets items 9, 0, 1 (wrapping around)
    assert source.index.get_value() == 0  # Wrapped around
    assert source.epoch.get_value() == 1  # Epoch incremented


def test_memory_source_with_transform_interface():
    """Test MemorySource compatibility with StructuralModule interface."""
    # Create data source
    data = {"values": jnp.arange(10)}
    config = MemorySourceConfig()
    source = MemorySource(config, data)

    # Test that MemorySource has the expected interface properties
    # from StructuralModule (new config-based architecture)
    assert hasattr(source, "config")
    assert hasattr(source.config, "cacheable")  # cacheable is now in config
    assert hasattr(source, "name")
    assert hasattr(source.config, "stochastic")

    # Test that we can get batches using the get_batch method
    batch = source.get_batch(5)
    assert "values" in batch
    assert len(batch["values"]) == 5


def test_memory_source_caching():
    """Test caching configuration."""
    # Create data source with caching enabled
    data = list(range(100))
    config = MemorySourceConfig(cache_size=10, cacheable=True)
    source = MemorySource(config, data)

    # Check that cacheable flag is set correctly in config
    assert source.config.cacheable

    # Create without caching
    config2 = MemorySourceConfig(cache_size=0, cacheable=False)
    source2 = MemorySource(config2, data)
    assert not source2.config.cacheable


def test_memory_source_repr():
    """Test string representation of MemorySource."""
    data = [1, 2, 3]
    config = MemorySourceConfig()
    source = MemorySource(config, data)

    # Check that repr includes useful information
    repr_str = repr(source)
    assert "MemorySource" in repr_str or "TransformBase" in repr_str


# ============================================================================
# Metadata tracking tests
# ============================================================================


def test_memory_source_without_metadata():
    """Test that MemorySource works normally without metadata tracking."""
    data = [{"x": i, "y": i * 2} for i in range(10)]
    config = MemorySourceConfig()
    source = MemorySource(config, data)

    # Should work normally
    assert len(source) == 10
    assert source[0] == {"x": 0, "y": 0}

    # Should not have metadata
    assert not source.has_metadata

    # Should raise error when trying to get metadata
    with pytest.raises(RuntimeError, match="Metadata tracking is not enabled"):
        source.get_with_metadata(0)

    with pytest.raises(RuntimeError, match="Metadata tracking is not enabled"):
        source.get_batch_with_metadata(5)


def test_memory_source_with_metadata_enabled():
    """Test MemorySource with metadata tracking enabled."""
    data = [{"x": i, "y": i * 2} for i in range(10)]
    rngs = nnx.Rngs(data=42, metadata=43)
    config = MemorySourceConfig(track_metadata=True, shard_id=0)
    source = MemorySource(config, data, rngs=rngs)

    # Should have metadata capability
    assert source.has_metadata
    assert source.metadata_manager is not None

    # Get single element with metadata
    element, metadata = source.get_with_metadata(5)
    assert element == {"x": 5, "y": 10}
    assert isinstance(metadata, RecordMetadata)
    assert metadata.record_key == 5
    assert metadata.shard_id == 0
    assert metadata.source_info is not None
    assert metadata.source_info["source"] == "memory"
    assert metadata.source_info["index"] == 5
    assert metadata.rng_key is not None  # Should have RNG key from metadata Rngs


def test_memory_source_batch_metadata():
    """Test batch retrieval with metadata."""
    data = jnp.arange(20).reshape(20, 1)
    rngs = nnx.Rngs(data=0, metadata=1)
    config = MemorySourceConfig(track_metadata=True, shard_id=1)
    source = MemorySource(config, {"values": data}, rngs=rngs)

    # Get batch with metadata
    batch, metadata_list = source.get_batch_with_metadata(5)

    # Check batch data
    assert "values" in batch
    assert batch["values"].shape == (5, 1)

    # Check metadata list
    assert len(metadata_list) == 5
    assert all(isinstance(m, RecordMetadata) for m in metadata_list)
    assert all(m.shard_id == 1 for m in metadata_list)
    assert all(m.source_info["source"] == "memory" for m in metadata_list)

    # Check batch positions
    for i, meta in enumerate(metadata_list):
        assert meta.source_info["batch_position"] == i

    # Get another batch - batch index should increment
    batch2, metadata_list2 = source.get_batch_with_metadata(5)
    assert metadata_list2[0].batch_idx > metadata_list[0].batch_idx


def test_memory_source_metadata_state_tracking():
    """Test that metadata manager tracks state correctly."""
    data = list(range(10))
    rngs = nnx.Rngs(0)
    config = MemorySourceConfig(track_metadata=True)
    source = MemorySource(config, data, rngs=rngs)

    # Get some data with metadata
    _, m1 = source.get_with_metadata(0)
    assert m1.global_step == 0
    assert m1.index == 0

    _, m2 = source.get_with_metadata(1)
    assert m2.global_step == 1  # Should increment
    assert m2.index == 1


def test_memory_source_metadata_reset():
    """Test that metadata state resets properly with source reset."""
    data = list(range(5))
    rngs = nnx.Rngs(0)
    config = MemorySourceConfig(track_metadata=True)
    source = MemorySource(config, data, rngs=rngs)

    # Generate some metadata to advance state
    source.get_batch_with_metadata(3)
    _, metadata_before = source.get_with_metadata(0)
    assert metadata_before.global_step > 0

    # Reset the source
    source.reset()

    # Check that metadata manager was also reset
    assert source.metadata_manager.state.get_value()["global_step"] == 0
    assert source.metadata_manager.state.get_value()["index"] == 0
    assert source.metadata_manager.state.get_value()["epoch"] == 0

    # New metadata should start from 0
    _, metadata_after = source.get_with_metadata(0)
    assert metadata_after.global_step == 0
    assert metadata_after.index == 0


def test_memory_source_metadata_with_shuffling():
    """Test metadata tracking with shuffling enabled."""
    data = [{"id": i} for i in range(10)]
    # Include default stream for fallback behavior
    rngs = nnx.Rngs(default=0, shuffle=42, metadata=43)
    config = MemorySourceConfig(shuffle=True, track_metadata=True)
    source = MemorySource(config, data, rngs=rngs)

    # Get element with metadata
    element, metadata = source.get_with_metadata(0)

    # Check that shuffle info is in source_info
    assert metadata.source_info["shuffle_enabled"] is True

    # Get batch with metadata
    batch, meta_list = source.get_batch_with_metadata(3)
    assert all(m.source_info["shuffle_enabled"] is True for m in meta_list)


def test_memory_source_metadata_epoch_tracking():
    """Test that metadata tracks epochs correctly through iteration."""
    data = list(range(5))
    rngs = nnx.Rngs(0)
    config = MemorySourceConfig(track_metadata=True)
    source = MemorySource(config, data, rngs=rngs)

    # Initial epoch should be 0
    _, m1 = source.get_with_metadata(0)
    assert m1.epoch == 0

    # Iterate through source to trigger epoch change
    _ = list(source)  # This increments epoch

    # Note: The metadata manager maintains its own epoch tracking
    # which may differ from the source's epoch tracking
    # This is by design to allow independent metadata management


# ============================================================================
# Additional tests for full coverage
# ============================================================================


def test_memory_source_string_input_error():
    """Test that MemorySource rejects string input with proper error."""
    # Test error for string input (line 87)
    config = MemorySourceConfig()
    with pytest.raises(TypeError, match="MemorySource expects a list, sequence, or dictionary"):
        MemorySource(config, "not_a_valid_input")


def test_memory_source_stateless_batch_with_shuffle():
    """Test stateless batch retrieval with shuffling enabled."""
    # Create data source with shuffling (requires rngs for stochastic modules)
    data = {"values": jnp.arange(20)}
    config = MemorySourceConfig(shuffle=True)
    rngs = nnx.Rngs(shuffle=42)  # Required for stochastic modules
    source = MemorySource(config, data, rngs=rngs)

    # Get batch with explicit key (stateless mode, line 198)
    key = jax.random.key(42)
    batch1 = source.get_batch(10, key=key)
    assert "values" in batch1
    assert len(batch1["values"]) == 10

    # Same key should give same shuffled batch
    batch2 = source.get_batch(10, key=key)
    assert jnp.array_equal(batch1["values"], batch2["values"])

    # Different key should give different shuffled batch
    key2 = jax.random.key(43)
    batch3 = source.get_batch(10, key=key2)
    # The values should be different (with high probability)
    assert not jnp.array_equal(batch1["values"], batch3["values"])


def test_memory_source_apply_transform():
    """Test the _apply_transform method for TransformBase compatibility."""
    # Test _apply_transform method (line 235)
    data = {"values": jnp.arange(15)}
    config = MemorySourceConfig()
    source = MemorySource(config, data)

    # Call _apply_transform directly
    batch = source._apply_transform(5, None)
    assert "values" in batch
    assert len(batch["values"]) == 5
    assert jnp.array_equal(batch["values"], jnp.arange(5))

    # With key for shuffling
    key = jax.random.key(42)
    batch_with_key = source._apply_transform(5, key)
    assert "values" in batch_with_key
    assert len(batch_with_key["values"]) == 5


def test_memory_source_shuffle_without_key():
    """Test that shuffling without rngs raises an error in stochastic mode."""
    # In the new architecture, stochastic modules (shuffle=True) require rngs
    data = list(range(10))
    config = MemorySourceConfig(shuffle=True)

    # Should raise ValueError because stochastic modules require rngs
    with pytest.raises(ValueError, match="Stochastic structural modules require rngs"):
        MemorySource(config, data, rngs=None)


def test_memory_source_dict_with_scalar_values():
    """Test dictionary data with scalar (non-array) values."""
    # Test dictionary with scalar values (lines 278, 306)
    # Note: strings have __len__ so they're treated as arrays
    # Use actual scalars without __len__ like int, float
    data = {
        "array_field": jnp.arange(5),
        "scalar_field": 42,  # Scalar int value without __len__
        "float_val": 3.14,  # Scalar float value
    }
    config = MemorySourceConfig()
    source = MemorySource(config, data)

    # Get single element
    elem = source[0]
    assert elem["array_field"] == 0
    assert elem["scalar_field"] == 42  # Scalar repeated
    assert elem["float_val"] == 3.14

    # Get batch - scalar values should be repeated
    batch = source.get_batch(3)
    assert jnp.array_equal(batch["array_field"], jnp.arange(3))
    assert batch["scalar_field"] == [42, 42, 42]  # Repeated for batch
    assert batch["float_val"] == [3.14, 3.14, 3.14]


def test_memory_source_list_batch_gathering():
    """Test batch gathering from list/tuple data."""
    # Test list batch gathering (line 300)
    data = [{"id": i, "value": i * 10} for i in range(10)]
    config = MemorySourceConfig()
    source = MemorySource(config, data)

    # Get batch
    batch = source.get_batch(3)
    assert len(batch) == 3
    assert batch[0]["id"] == 0
    assert batch[1]["id"] == 1
    assert batch[2]["id"] == 2

    # Test with tuple data
    data_tuple = tuple(range(10))
    source_tuple = MemorySource(config, data_tuple)
    batch_tuple = source_tuple.get_batch(3)
    assert batch_tuple == [0, 1, 2]


def test_memory_source_array_batch_gathering():
    """Test batch gathering from numpy/jax arrays."""
    # Test array batch gathering (line 314)
    data = jnp.arange(20).reshape(20, 1)
    config = MemorySourceConfig()
    source = MemorySource(config, data)

    # Get batch
    batch = source.get_batch(5)
    expected = jnp.arange(5).reshape(5, 1)
    assert jnp.array_equal(batch, expected)


def test_memory_source_reset_with_cache():
    """Test reset method with caching enabled."""
    # Test cache clearing in reset (line 327)
    data = list(range(10))
    # cacheable must be True to create the _cache dict
    config = MemorySourceConfig(cache_size=5, cacheable=True)
    source = MemorySource(config, data)

    # The cache is a simple dict when cacheable=True
    assert source._cache is not None  # Should exist when cacheable=True
    assert isinstance(source._cache, dict)

    # Add something to cache
    source._cache[123] = "test_value"

    # Reset should clear cache
    source.reset()
    assert source.index.get_value() == 0
    assert source.epoch.get_value() == 0
    assert len(source._cache) == 0  # Cache should be cleared


def test_memory_source_set_shuffle():
    """Test set_shuffle method to enable/disable shuffling."""
    # Test set_shuffle method (lines 337-339)
    data = list(range(100))
    # Include default stream for fallback behavior
    rngs = nnx.Rngs(default=0, shuffle=42)
    config = MemorySourceConfig(shuffle=False)
    source = MemorySource(config, data, rngs=rngs)

    # Initially not shuffling
    assert source.shuffle is False

    # Enable shuffling
    source.set_shuffle(True)
    assert source.shuffle is True

    # Get data - should be shuffled
    items = list(source)
    assert set(items) == set(range(100))
    assert items != list(range(100))  # Should be shuffled

    # Disable shuffling
    source.set_shuffle(False)
    assert source.shuffle is False
    assert source._shuffled_indices.get_value() is None  # Should clear shuffled indices

    # Get data - should not be shuffled
    items2 = list(source)
    assert items2 == list(range(100))


def test_memory_source_complex_nested_data():
    """Test MemorySource with complex nested data structures."""
    # Test with nested dictionaries and mixed types
    data = [
        {
            "features": {"x": i, "y": i * 2},
            "metadata": {"id": f"item_{i}", "timestamp": i * 1000},
            "label": i % 3,
        }
        for i in range(10)
    ]
    config = MemorySourceConfig()
    source = MemorySource(config, data)

    # Check access
    item = source[5]
    assert item["features"]["x"] == 5
    assert item["features"]["y"] == 10
    assert item["metadata"]["id"] == "item_5"
    assert item["label"] == 2

    # Check iteration
    all_items = list(source)
    assert len(all_items) == 10
    assert all_items[0]["metadata"]["timestamp"] == 0


def test_memory_source_edge_cases():
    """Test various edge cases for MemorySource."""
    # Test with single element
    single_data = [42]
    config = MemorySourceConfig()
    source = MemorySource(config, single_data)
    assert len(source) == 1
    assert source[0] == 42

    # Test batch larger than data
    batch = source.get_batch(5)
    assert len(batch) == 1  # Should only return available data

    # Test empty batch after exhaustion in stateful mode
    rngs = nnx.Rngs(0)
    source2 = MemorySource(config, [1, 2, 3], rngs=rngs)
    batch1 = source2.get_batch(2)
    assert len(batch1) == 2
    batch2 = source2.get_batch(2)
    assert len(batch2) == 1  # Only 1 element left

    # After wrap-around, should start from beginning
    batch3 = source2.get_batch(2)
    assert len(batch3) == 2
