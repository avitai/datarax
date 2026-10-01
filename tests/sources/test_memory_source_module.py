"""Tests for the MemorySource.

This module tests the functionality of the unified MemorySource implementation.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from datarax.core.index_words import to_words
from datarax.sources.memory_source import MemorySource, MemorySourceConfig


def _words(rows: list[int]) -> np.ndarray:
    """Record positions as the uint32 words the host read takes."""
    return to_words(np.asarray(rows, np.uint64))


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


def test_memory_source_errors():
    """Test error cases for MemorySource."""
    config = MemorySourceConfig()
    # Test error for inconsistent array lengths in dictionary
    with pytest.raises(ValueError, match="same length"):
        MemorySource(config, {"a": np.arange(5), "b": np.arange(10)})

    # Test error for dictionary without array-like values
    with pytest.raises(ValueError, match="array-like value"):
        MemorySource(config, {"a": 1, "b": 2})


def test_memory_source_with_transform_interface():
    """Test MemorySource compatibility with StructuralModule interface."""
    # Create data source
    data = {"values": jnp.arange(10)}
    config = MemorySourceConfig()
    source = MemorySource(config, data)

    # Test that MemorySource has the expected interface properties
    # from StructuralModule (new config-based architecture)
    assert hasattr(source, "config")
    assert hasattr(source, "name")
    assert hasattr(source.config, "stochastic")

    # Test that we can read records with the host read
    batch = source.get_batch(_words([0, 1, 2, 3, 4]))
    assert "values" in batch
    assert batch.batch_size == 5


def test_memory_source_repr():
    """Test string representation of MemorySource."""
    data = [1, 2, 3]
    config = MemorySourceConfig()
    source = MemorySource(config, data)

    # Check that repr includes useful information
    repr_str = repr(source)
    assert "MemorySource" in repr_str or "TransformBase" in repr_str


def test_memory_source_string_input_error():
    """Test that MemorySource rejects string input with proper error."""
    # Test error for string input (line 87)
    config = MemorySourceConfig()
    with pytest.raises(TypeError, match="MemorySource expects a list, sequence, or dictionary"):
        MemorySource(config, "not_a_valid_input")


def test_memory_source_dict_with_scalar_values():
    """Test dictionary data with scalar (non-array) values."""
    # A value without rows is every record's
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

    # Read a batch - scalar values are repeated
    batch = source.get_batch(_words([0, 1, 2]))
    np.testing.assert_array_equal(batch["array_field"], np.arange(3))
    np.testing.assert_array_equal(batch["scalar_field"], [42, 42, 42])
    np.testing.assert_array_equal(batch["float_val"], [3.14, 3.14, 3.14])


def test_memory_source_list_batch_gathering():
    """A list of records is stored as columns, so a read gathers rows of them."""
    data = [{"id": i, "value": i * 10} for i in range(10)]
    config = MemorySourceConfig()
    source = MemorySource(config, data)

    batch = source.get_batch(_words([0, 1, 2]))
    assert batch.batch_size == 3
    np.testing.assert_array_equal(batch["id"], [0, 1, 2])
    np.testing.assert_array_equal(batch["value"], [0, 10, 20])

    # A tuple of scalar records is one column
    source_tuple = MemorySource(config, tuple(range(10)))
    np.testing.assert_array_equal(source_tuple.get_batch(_words([0, 1, 2])).data, [0, 1, 2])


def test_memory_source_array_batch_gathering():
    """An array is one column, held on the host."""
    data = jnp.arange(20).reshape(20, 1)
    config = MemorySourceConfig()
    source = MemorySource(config, data)

    batch = source.get_batch(_words([0, 1, 2, 3, 4]))
    assert isinstance(batch.data, np.ndarray)
    np.testing.assert_array_equal(batch.data, np.arange(5).reshape(5, 1))


def test_memory_source_complex_nested_data():
    """Test MemorySource with complex nested data structures."""
    # Nested records: numbers become columns, strings the record's provenance
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
    assert "id" not in item["metadata"]
    assert source._provenance.value[5]["metadata/id"] == "item_5"
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

    # A record outside the source is refused rather than wrapped
    with pytest.raises(IndexError, match="outside"):
        source.get_batch(_words([0, 1]))
