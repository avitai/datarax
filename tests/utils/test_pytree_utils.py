"""Tests for pytree utility functions."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import flax.nnx as nnx

from datarax.utils.pytree_utils import (
    is_jax_array,
    is_batch_leaf,
    is_non_jax_leaf,
    get_batch_size,
    is_single_element,
    add_batch_dimension,
    remove_batch_dimension,
    split_batch,
    concatenate_batches,
    apply_to_batch_dimension,
    validate_batch_consistency,
    get_pytree_structure_info,
)

from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.typing import Element, Batch


class TestIsJaxArray:
    """Tests for is_jax_array function."""

    def test_jax_arrays(self):
        """Test with JAX arrays."""
        assert is_jax_array(jnp.array([1, 2, 3]))
        assert is_jax_array(jnp.ones((2, 3)))
        assert is_jax_array(jnp.zeros(5))

    def test_numeric_types(self):
        """Test with numeric types."""
        assert is_jax_array(1)
        assert is_jax_array(1.5)
        assert is_jax_array(True)
        assert is_jax_array(1 + 2j)

    def test_non_jax_types(self):
        """Test with non-JAX types."""
        assert not is_jax_array("string")
        assert not is_jax_array([1, 2, 3])
        assert not is_jax_array({"key": "value"})
        assert not is_jax_array(None)


class TestLeafPredicates:
    """Tests for leaf predicate functions."""

    def test_is_batch_leaf(self):
        """Test is_batch_leaf predicate."""
        # Arrays are leaves
        assert is_batch_leaf(jnp.array([1, 2, 3]))
        assert is_batch_leaf(np.array([1, 2, 3]))

        assert is_batch_leaf(1)
        assert is_batch_leaf("string")
        assert is_batch_leaf(True)

        # Containers are NOT leaves (traversed)
        assert not is_batch_leaf({"a": 1})
        assert not is_batch_leaf([1, 2])
        assert not is_batch_leaf((1, 2))

    def test_is_non_jax_leaf(self):
        """Test is_non_jax_leaf predicate."""
        # Arrays are leaves
        assert is_non_jax_leaf(jnp.array([1, 2, 3]))

        # Python lists/tuples are leaves (treated as atomic payloads)
        assert is_non_jax_leaf([1, 2, 3])
        assert is_non_jax_leaf((1, 2, 3))

        # Dicts are NOT leaves (traversed)
        assert not is_non_jax_leaf({"a": 1})

        # Scalars are not explicitly handled by first two checks, returns False
        # Scalars are not explicitly handled by first two checks, returns False
        # The function logic is: array -> True, list/tuple -> True, else -> False
        #
        # Implementation analysis:
        # - if is_array(x): return True
        # - if isinstance(x, list | tuple): return True
        # - return False
        #
        # So scalars return False, meaning they are NOT treated as leaves by this
        # specific predicate?
        #
        # If x is 1 (int), is_array(1) is False (is_array checks jax/np array).
        # So it returns False.
        # This seems correct for the specific use case of "rebatching" where we
        # want to traverse dicts but keep lists/arrays as data. Scalars might be
        # traversed if they are not considered leaves?
        #
        # Actually standard jax.tree_util.tree_map treats scalars as leaves by
        # default unless is_leaf returns True first?
        # No, tree_map traversal depends on registry. is_leaf acts as a stopper.
        # If is_leaf(scalar) is False, it continues.
        #
        # Actually scalars are not registered as nodes in pytree registry, so they
        # are leaves effectively. is_leaf is only queried for nodes.
        #
        # But wait, logic: is_leaf is a predicate.
        # If is_leaf returns False, JAX checks if it's a registered node type.
        # If it is, it flattens it.
        # If it's not registered (like int/str), it's treated as leaf anyway.
        # So returning False for int is fine.
        assert not is_non_jax_leaf(1)
        assert not is_non_jax_leaf("string")


class TestGetBatchSize:
    """Tests for get_batch_size function."""

    def test_simple_batch(self):
        """Test getting batch size from simple array."""
        elements = [Element(data={"x": jnp.ones((10,))}) for _ in range(32)]
        batch = Batch(elements)
        assert get_batch_size(batch) == 32

    def test_dict_batch(self):
        """Test getting batch size from dictionary."""
        elements = [Element(data={"x": jnp.ones((5,)), "y": jnp.ones((3,))}) for _ in range(16)]
        batch = Batch(elements)
        assert get_batch_size(batch) == 16

    def test_nested_batch(self):
        """Test getting batch size from nested structure."""
        elements = [
            Element(data={"features": jnp.ones((10,)), "labels": jnp.ones(())}) for _ in range(8)
        ]
        batch = Batch(elements)
        assert get_batch_size(batch) == 8

    def test_no_batch(self):
        """Test with empty batch."""
        empty_batch = Batch([])
        assert get_batch_size(empty_batch) == 0

    def test_dict_no_arrays(self):
        """Test dict with no arrays returns None."""
        d = {"scalar": 1, "string": "val"}
        assert get_batch_size(d) is None


class TestIsSingleElement:
    """Tests for is_single_element function."""

    def test_single_element(self):
        """Test identifying single elements."""
        # Single element
        element = Element(data={"x": jnp.array(1), "y": jnp.array([1, 2, 3])})
        assert is_single_element(element)

        # Single element with different shaped arrays
        element2 = Element(data={"a": jnp.ones((3, 4)), "b": jnp.ones((5, 6))})
        assert is_single_element(element2)

    def test_batch(self):
        """Test identifying batches."""
        # Batch with consistent first dimension
        elements = [Element(data={"x": jnp.ones((10,)), "y": jnp.ones((5,))}) for _ in range(32)]
        batch = Batch(elements)
        assert not is_single_element(batch)

    def test_edge_cases(self):
        """Test edge cases."""
        # Only scalars
        element = Element(data={"a": jnp.array(1), "b": jnp.array(2)})
        assert is_single_element(element)

        # Empty element
        empty_element = Element(data={})
        assert is_single_element(empty_element)


class TestBatchDimensionOps:
    """Tests for batch dimension operations."""

    def test_add_batch_dimension(self):
        """Test adding batch dimension."""
        element = Element(data={"x": jnp.array([1, 2, 3]), "y": jnp.array(5)})
        batched = add_batch_dimension(element)

        assert batched.batch_size == 1
        assert batched["x"].shape == (1, 3)
        assert batched["y"].shape == (1,)

    def test_remove_batch_dimension(self):
        """Test removing batch dimension."""
        element = Element(data={"x": jnp.array([1, 2, 3]), "y": jnp.array(5)})
        batch = Batch([element])
        extracted = remove_batch_dimension(batch)

        assert extracted.data["x"].shape == (3,)
        assert extracted.data["y"].shape == ()

    def test_add_remove_roundtrip(self):
        """Test add then remove is identity."""
        element = Element(data={"data": jnp.ones((5, 3))})
        result = remove_batch_dimension(add_batch_dimension(element))

        assert jnp.array_equal(result.data["data"], element.data["data"])


class TestSplitBatch:
    """Tests for split_batch function."""

    def test_split_even(self):
        """Test splitting batch evenly."""
        elements = [Element(data={"x": jnp.arange(3) + i * 3}) for i in range(4)]
        batch = Batch(elements)
        splits = split_batch(batch, 2)

        assert len(splits) == 2
        assert splits[0].batch_size == 2
        assert splits[1].batch_size == 2
        assert splits[0]["x"].shape == (2, 3)
        assert splits[1]["x"].shape == (2, 3)

    def test_split_error_not_divisible(self):
        """Test error when batch size not divisible."""
        elements = [Element(data={"x": jnp.ones((3,))}) for _ in range(5)]
        batch = Batch(elements)

        with pytest.raises(ValueError, match="not divisible"):
            split_batch(batch, 2)

    def test_split_non_batched(self):
        """Test error with single element batch."""
        element = Element(data={"x": jnp.array(5)})
        batch = Batch([element])

        with pytest.raises(ValueError, match="not divisible"):
            split_batch(batch, 2)

    def test_split_identity(self):
        """Test split with num_splits=1."""
        batch = Batch([Element(data={"x": jnp.array(1)})])
        splits = split_batch(batch, 1)
        assert len(splits) == 1
        # Check value equality, not identity since split creates new views/objects
        assert splits[0].batch_size == batch.batch_size
        assert jnp.array_equal(splits[0]["x"], batch["x"])

    def test_split_num_splits_eq_batch_size(self):
        """Test split where num_splits equals batch size."""
        elements = [Element(data={"x": jnp.array([i])}) for i in range(3)]
        batch = Batch(elements)
        splits = split_batch(batch, 3)

        assert len(splits) == 3
        for i, s in enumerate(splits):
            assert s.batch_size == 1
            assert jnp.array_equal(s["x"], jnp.array([[i]]))


class TestConcatenateBatches:
    """Tests for concatenate_batches function."""

    def test_concatenate_simple(self):
        """Test concatenating simple batches."""
        elements1 = [Element(data={"x": jnp.array([1, 2])}) for _ in range(2)]
        elements1[1] = Element(data={"x": jnp.array([3, 4])})
        batch1 = Batch(elements1)

        elements2 = [Element(data={"x": jnp.array([5, 6])}) for _ in range(2)]
        elements2[1] = Element(data={"x": jnp.array([7, 8])})
        batch2 = Batch(elements2)

        result = concatenate_batches([batch1, batch2])
        expected = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]])

        assert result.batch_size == 4
        assert jnp.array_equal(result["x"], expected)

    def test_concatenate_single(self):
        """Test concatenating single batch."""
        elements = [Element(data={"x": jnp.ones((2,))}) for _ in range(3)]
        batch = Batch(elements)
        result = concatenate_batches([batch])

        assert jnp.array_equal(result["x"], batch["x"])  # type: ignore

    def test_concatenate_empty_error(self):
        """Test error with empty list."""
        with pytest.raises(ValueError, match="empty list"):
            concatenate_batches([])

    def test_concatenate_nested(self):
        """Test concatenating nested batch structures."""
        # Batch 1
        e1 = [Element(data={"nest": {"x": jnp.array([1])}})]
        b1 = Batch(e1)
        # Batch 2
        e2 = [Element(data={"nest": {"x": jnp.array([2])}})]
        b2 = Batch(e2)

        result = concatenate_batches([b1, b2])
        assert result.batch_size == 2
        assert jnp.array_equal(result.data["nest"]["x"], jnp.array([[1], [2]]))

    def test_concatenate_mismatched_structures(self):
        """Test that concatenating mismatched structures raises error (from JAX)."""
        b1 = Batch([Element(data={"x": jnp.ones(1)})])
        b2 = Batch([Element(data={"y": jnp.ones(1)})])  # Different key

        # JAX tree_map will fail when structures don't match or stack will fail.
        # Actually Element/Batch stacking logic relies on stacking lists of leaves.
        # But BatchOps.concatenate_batches implementation:
        # It iterates elements of all batches and creates a new Batch(all_elements)
        # (which re-stacks).
        # Batch constructor creates elements list, then re-stacks.
        #
        # If elements have different keys, Batch constructor might fail or produce
        # mismatched leaves?
        # Actually Element is just data/state dicts.
        # Batch constructor:
        # element_data_list = [elem.data for elem in elements]
        # batched_data = jax.tree.map(lambda *a: stack(a), *element_data_list)
        # If keys differ, jax.tree.map might error depending on how strict it is,
        # or if dicts match. For dicts, keys must match.
        with pytest.raises((ValueError, TypeError)):
            concatenate_batches([b1, b2])


class TestApplyToBatchDimension:
    """Tests for apply_to_batch_dimension function."""

    def test_apply_mean(self):
        """Test applying mean along batch dimension."""
        # Create 3 elements with different values for x
        elements = [
            Element(data={"x": jnp.array([1.0, 2.0, 3.0])}),
            Element(data={"x": jnp.array([4.0, 5.0, 6.0])}),
            Element(data={"x": jnp.array([7.0, 8.0, 9.0])}),
        ]
        batch = Batch(elements)
        result = apply_to_batch_dimension(batch, jnp.mean)

        # Mean along batch: [mean(1,4,7), mean(2,5,8), mean(3,6,9)] = [4., 5., 6.]
        assert jnp.allclose(result["x"], jnp.array([4.0, 5.0, 6.0]))  # type: ignore

    def test_apply_with_keepdims(self):
        """Test applying with keepdims."""
        # Create proper Batch from Elements with 2D data
        elements = [Element(data={"x": jnp.ones((3, 2))}) for _ in range(4)]
        batch = Batch(elements)
        result = apply_to_batch_dimension(batch, jnp.sum, keepdims=True)

        # Type narrowing: we know "x" contains an array after transformation
        x_result = result["x"]
        assert isinstance(x_result, jax.Array), "Expected array type"
        assert x_result.shape == (1, 3, 2)
        assert jnp.allclose(x_result, 4 * jnp.ones((1, 3, 2)))

    def test_apply_to_mixed_structure(self):
        """Test applying sum to structure with arrays of different shapes."""
        elements = [
            Element(data={"array": jnp.ones((2,)), "scalar": jnp.array(i)}) for i in range(3)
        ]
        batch = Batch(elements)
        result = apply_to_batch_dimension(batch, jnp.sum)

        # array: sum along batch axis (3 ones per position) = [3, 3]
        array_result = result["array"]
        assert isinstance(array_result, jax.Array)
        assert jnp.allclose(array_result, 3 * jnp.ones(2))

        # scalar: becomes [0, 1, 2] after batching, sum = 3
        assert result["scalar"] == 3


class TestValidateBatchConsistency:
    """Tests for validate_batch_consistency function."""

    def test_consistent_batch(self):
        """Test with consistent batch."""
        elements = [
            Element(data={"x": jnp.ones((10,)), "y": jnp.ones((5,)), "z": jnp.ones((32,))})
            for _ in range(32)
        ]
        batch = Batch(elements)
        assert validate_batch_consistency(batch) is True

    def test_inconsistent_batch(self):
        """Test with inconsistent batch."""
        # Create batch with inconsistent data by bypassing validation
        elements = [Element(data={"x": jnp.ones((10,)), "y": jnp.ones((5,))}) for _ in range(5)]
        batch = Batch(elements, validate=False)
        # Manually create inconsistency by modifying the batch data
        # Use proper NNX Variable API: get_value(), modify dict, set_value()
        data_dict = batch.data.get_value()
        data_dict["x"] = jnp.ones((4, 10))  # Different batch size
        batch.data.set_value(data_dict)
        assert validate_batch_consistency(batch) is False

    def test_empty_batch(self):
        """Test with empty batch."""
        empty_batch = Batch([])
        # Empty batch is consistent (no data to be inconsistent)
        assert validate_batch_consistency(empty_batch) is True


class TestGetPytreeStructureInfo:
    """Tests for get_pytree_structure_info function."""

    def test_simple_structure(self):
        """Test getting info for simple structure."""
        elements = [Element(data={"x": jnp.ones((3,)), "y": jnp.array(i)}) for i in range(4)]
        batch = Batch(elements)
        info = get_pytree_structure_info(batch)

        assert info["type"] == "Batch"
        assert info["batch_size"] == 4
        assert info["is_batch_consistent"] is True
        assert ("x", (4, 3)) in info["leaf_shapes"]
        assert ("y", (4,)) in info["leaf_shapes"]

    def test_nested_structure(self):
        """Test getting info for nested structure."""
        elements = [
            Element(data={"features": jnp.ones((10,)), "labels": jnp.array(i, dtype=jnp.int32)})
            for i in range(8)
        ]
        batch = Batch(elements)
        info = get_pytree_structure_info(batch)

        assert info["type"] == "Batch"
        assert info["num_data_fields"] == 2
        assert info["batch_size"] == 8
        assert info["is_batch_consistent"] is True
        assert len(info["leaf_shapes"]) == 2
        assert len(info["leaf_dtypes"]) == 2

    def test_single_element_info(self):
        """Test info for single element."""
        element = Element(data={"scalar": jnp.array(5), "vector": jnp.array([1, 2, 3])})
        info = get_pytree_structure_info(element)

        assert info["is_single_element"] is True
        assert info["batch_size"] is None


class TestBatchDimensionHandling:
    """Test batch dimension detection and manipulation."""

    def test_single_element_detection(self):
        """Test detection of single elements vs batches."""
        # Single element (no batch dimension)
        single = Element(data={"image": jnp.ones((32, 32, 3)), "label": jnp.array(5)})
        assert is_single_element(single)

        # Batch of elements
        elements = [
            Element(data={"image": jnp.ones((32, 32, 3)), "label": jnp.array(5)}) for _ in range(10)
        ]
        batch = Batch(elements)
        assert not is_single_element(batch)

    def test_add_batch_dimension(self):
        """Test adding batch dimension to single element."""
        single = Element(data={"image": jnp.ones((32, 32, 3)), "label": jnp.array(5)})
        batched = add_batch_dimension(single)

        # Type narrowing: we know these are arrays after add_batch_dimension
        image_batched = batched["image"]
        label_batched = batched["label"]
        assert isinstance(image_batched, jax.Array | jax.Array)
        assert isinstance(label_batched, jax.Array | jax.Array)
        assert image_batched.shape == (1, 32, 32, 3)
        assert label_batched.shape == (1,)

    def test_remove_batch_dimension(self):
        """Test removing batch dimension from size-1 batch."""
        elements = [
            Element(data={"image": jnp.ones((32, 32, 3)), "label": jnp.array(5)}) for _ in range(1)
        ]
        batch = Batch(elements)
        single = remove_batch_dimension(batch)

        # Element data is accessed via .data attribute
        image_single = single.data["image"]
        label_single = single.data["label"]
        assert isinstance(image_single, jax.Array)
        assert isinstance(label_single, jax.Array)
        assert image_single.shape == (32, 32, 3)
        assert label_single.shape == ()

    def test_get_batch_size(self):
        """Test extracting batch size from batched data."""
        elements = [
            Element(data={"image": jnp.ones((32, 32, 3)), "label": jnp.array(5)}) for _ in range(10)
        ]
        batch = Batch(elements)
        assert get_batch_size(batch) == 10

        # Nested structure
        elements = [
            Element(data={"features": {"dense": jnp.ones((128,)), "sparse": jnp.ones((64,))}})
            for _ in range(5)
        ]
        nested = Batch(elements)
        assert get_batch_size(nested) == 5


class TestTransformBatchSemantics:
    """Test the core operator batch semantics using ElementOperator."""

    def test_operator_single_element_via_batch(self):
        """Test operator on single element (wrapped in batch)."""

        def double_fn(element: Element, key: jax.Array) -> Element:
            new_data = jax.tree.map(lambda x: x * 2, element.data)
            return element.replace(data=new_data)

        config = ElementOperatorConfig(stochastic=False)
        op = ElementOperator(config, fn=double_fn)

        element = Element(data={"value": jnp.array([1.0, 2.0, 3.0])})
        batch = Batch([element])
        result_batch = op(batch)

        # Extract result from batch
        value_result = result_batch["value"]
        assert isinstance(value_result, jax.Array | jax.Array)
        assert jnp.array_equal(value_result, jnp.array([[2.0, 4.0, 6.0]]))

    def test_operator_batch_via_vmap(self):
        """Test operator batch processing using vmap."""

        def add_one_fn(element: Element, key: jax.Array) -> Element:
            new_data = jax.tree.map(lambda x: x + 1, element.data)
            return element.replace(data=new_data)

        config = ElementOperatorConfig(stochastic=False)
        op = ElementOperator(config, fn=add_one_fn)

        elements = [Element(data={"value": jnp.array([float(i)])}) for i in range(1, 4)]
        batch = Batch(elements)
        result = op(batch)

        expected = jnp.array([[2.0], [3.0], [4.0]])
        value_result = result["value"]
        assert isinstance(value_result, jax.Array | jax.Array)
        assert jnp.array_equal(value_result, expected)

    def test_call_method_single_element_via_batch(self):
        """Test __call__ handles single element via batch wrapping."""

        def scale_fn(element: Element, key: jax.Array) -> Element:
            new_data = jax.tree.map(lambda x: x * 0.5, element.data)
            return element.replace(data=new_data)

        config = ElementOperatorConfig(stochastic=False)
        op = ElementOperator(config, fn=scale_fn)

        # Single element wrapped in batch
        single = Element(data={"value": jnp.array([2.0, 4.0])})
        batch = Batch([single])
        result = op(batch)

        # Result has batch dimension (batch_size=1)
        assert result["value"].shape == (1, 2)
        assert jnp.array_equal(result["value"], jnp.array([[1.0, 2.0]]))

    def test_call_method_batch(self):
        """Test __call__ handles batches correctly."""

        def scale_fn(element: Element, key: jax.Array) -> Element:
            new_data = jax.tree.map(lambda x: x * 0.5, element.data)
            return element.replace(data=new_data)

        config = ElementOperatorConfig(stochastic=False)
        op = ElementOperator(config, fn=scale_fn)

        # Batch of elements with distinct values
        elements = [
            Element(data={"value": jnp.array([2.0, 4.0])}),
            Element(data={"value": jnp.array([6.0, 8.0])}),
        ]
        batch = Batch(elements)
        result = op(batch)

        assert result["value"].shape == (2, 2)
        expected = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        assert jnp.array_equal(result["value"], expected)


class TestBatchStatistics:
    """Test transformations that depend on batch statistics."""

    def test_normalization_with_batch_stats(self):
        """Test normalization using pre-computed batch statistics."""
        # Pre-computed stats
        pre_stats = {"mean": {"value": jnp.array([5.0])}, "std": {"value": jnp.array([2.0])}}

        def normalize_fn(element: Element, key: jax.Array) -> Element:
            """Normalize using stats captured in closure."""

            def normalize_leaf(x: jax.Array, mean: jax.Array, std: jax.Array) -> jax.Array:
                return (x - mean) / (std + 1e-8)

            new_data = jax.tree.map(
                normalize_leaf, element.data, pre_stats["mean"], pre_stats["std"]
            )
            return element.replace(data=new_data)

        config = ElementOperatorConfig(stochastic=False)
        op = ElementOperator(config, fn=normalize_fn)

        element = Element(data={"value": jnp.array([7.0])})
        batch = Batch([element])
        result = op(batch)

        expected = (7.0 - 5.0) / 2.0
        value_result = result["value"]
        assert isinstance(value_result, jax.Array | jax.Array)
        assert jnp.allclose(value_result, jnp.array([[expected]]))

    def test_callable_stats_computation(self):
        """Test using callable to compute stats from batch."""

        def compute_batch_stats(batch: Batch) -> dict[str, Any]:
            """Compute mean and std across batch dimension."""
            # Apply jax.tree.map over the batch data
            data = batch.data.get_value()
            mean = jax.tree.map(lambda x: jnp.mean(x, axis=0), data)
            std = jax.tree.map(lambda x: jnp.std(x, axis=0), data)
            return {"mean": mean, "std": std}

        # Create 3 elements with different scalar values
        elements = [
            Element(data={"value": jnp.array([1.0])}),
            Element(data={"value": jnp.array([2.0])}),
            Element(data={"value": jnp.array([3.0])}),
        ]
        batch = Batch(elements)
        stats = compute_batch_stats(batch)

        # Batch stacks to shape (3, 1), mean along axis 0 gives shape (1,)
        assert stats["mean"]["value"].shape == (1,)
        # Mean of [1.0, 2.0, 3.0] = 2.0
        assert jnp.allclose(stats["mean"]["value"], jnp.array([2.0]))
        # Std of [1.0, 2.0, 3.0]
        expected_std = jnp.std(jnp.array([1.0, 2.0, 3.0]))
        assert jnp.allclose(stats["std"]["value"], jnp.array([expected_std]))

    def test_nested_pytree_stats(self):
        """Test statistics computation on nested PyTrees."""

        def compute_stats(batch: Batch) -> dict[str, Any]:
            mean = jax.tree.map(lambda x: jnp.mean(x, axis=0), batch)
            return {"mean": mean}

        # Create 3 elements with DIFFERENT data to test mean computation
        # Mean of dense: ([1,2] + [3,4] + [5,6]) / 3 = [3, 4]
        # Mean of sparse: ([0.1] + [0.2] + [0.3]) / 3 = [0.2]
        # Mean of labels: (0 + 1 + 2) / 3 = 1.0
        elements = [
            Element(
                data={
                    "features": {
                        "dense": jnp.array([1.0, 2.0]),
                        "sparse": jnp.array([0.1]),
                    },
                    "labels": jnp.array(0),
                }
            ),
            Element(
                data={
                    "features": {
                        "dense": jnp.array([3.0, 4.0]),
                        "sparse": jnp.array([0.2]),
                    },
                    "labels": jnp.array(1),
                }
            ),
            Element(
                data={
                    "features": {
                        "dense": jnp.array([5.0, 6.0]),
                        "sparse": jnp.array([0.3]),
                    },
                    "labels": jnp.array(2),
                }
            ),
        ]
        nested_batch = Batch(elements)

        stats = compute_stats(nested_batch)

        # Check nested structure is preserved
        assert "features" in stats["mean"]
        assert "dense" in stats["mean"]["features"]
        assert "sparse" in stats["mean"]["features"]

        # Check computed values
        assert jnp.allclose(stats["mean"]["features"]["dense"], jnp.array([3.0, 4.0]))
        assert jnp.allclose(stats["mean"]["features"]["sparse"], jnp.array([0.2]))
        assert jnp.allclose(
            stats["mean"]["labels"],
            1.0,  # Mean of [0, 1, 2]
        )

    def test_nnx_module_stats_computation(self):
        """Test using NNX module for stats computation."""

        class StatsComputer(nnx.Module):
            """NNX module that computes and stores statistics."""

            def __init__(self):
                super().__init__()
                self.count = nnx.Variable(0)

            def __call__(self, batch: Batch) -> dict[str, Any]:
                # Update count
                self.count.set_value(self.count.get_value() + 1)

                # Compute stats
                mean = jax.tree.map(lambda x: jnp.mean(x, axis=0), batch)
                std = jax.tree.map(lambda x: jnp.std(x, axis=0), batch)

                return {"mean": mean, "std": std, "count": self.count.get_value()}

        stats_module = StatsComputer()

        # Create 3 elements with DIFFERENT data to test mean computation
        # Mean of [1.0], [2.0], [3.0] = [2.0]
        elements = [
            Element(data={"data": jnp.array([1.0])}),
            Element(data={"data": jnp.array([2.0])}),
            Element(data={"data": jnp.array([3.0])}),
        ]
        batch = Batch(elements)
        stats1 = stats_module(batch)

        assert stats1["count"] == 1
        assert jnp.allclose(stats1["mean"]["data"], jnp.array([2.0]))

        # Call again to verify state is maintained
        stats2 = stats_module(batch)
        assert stats2["count"] == 2


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_batch(self):
        """Test handling of empty batches."""

        def identity_fn(element: Element, key: jax.Array) -> Element:
            return element

        config = ElementOperatorConfig(stochastic=False)
        op = ElementOperator(config, fn=identity_fn)

        # Empty batch (0 elements)
        empty_batch = Batch([])
        result = op(empty_batch)

        # Empty batch should return empty batch
        assert result.batch_size == 0

    def test_scalar_elements(self):
        """Test handling of scalar values."""

        def add_one_fn(element: Element, key: jax.Array) -> Element:
            new_data = jax.tree.map(lambda x: x + 1, element.data)
            return element.replace(data=new_data)

        config = ElementOperatorConfig(stochastic=False)
        op = ElementOperator(config, fn=add_one_fn)

        # Single element batch (scalar with batch dimension)
        elements = [Element(data={"value": jnp.array(5.0)})]
        scalar_batch = Batch(elements)
        result = op(scalar_batch)

        assert jnp.allclose(result["value"], jnp.array([6.0]))

        # Batch of scalars
        elements = [Element(data={"value": jnp.array(float(i))}) for i in [1.0, 2.0, 3.0]]
        batch = Batch(elements)
        batch_result = op(batch)

        assert jnp.array_equal(batch_result["value"], jnp.array([2.0, 3.0, 4.0]))

    def test_mixed_types_in_pytree(self):
        """Test PyTrees with mixed data types."""

        def type_preserving_fn(element: Element, key: jax.Array) -> Element:
            def process_leaf(x: Any):
                if jnp.issubdtype(x.dtype, jnp.integer):
                    return x + 1
                elif jnp.issubdtype(x.dtype, jnp.floating):
                    return x * 2.0
                else:
                    return x

            new_data = jax.tree.map(process_leaf, element.data)
            return element.replace(data=new_data)

        config = ElementOperatorConfig(stochastic=False)
        op = ElementOperator(config, fn=type_preserving_fn)

        elements = [
            Element(
                data={
                    "int_data": jnp.array(i, dtype=jnp.int32),
                    "float_data": jnp.array(float(i), dtype=jnp.float32),
                    "bool_data": jnp.array(i % 2 == 1),
                }
            )
            for i in [1, 2, 3]
        ]
        mixed = Batch(elements)

        result = op(mixed)

        assert jnp.array_equal(result["int_data"], jnp.array([2, 3, 4]))
        assert jnp.array_equal(result["float_data"], jnp.array([2.0, 4.0, 6.0]))
        # Bool data should remain unchanged (not int or float)
        assert jnp.array_equal(result["bool_data"], jnp.array([True, False, True]))
