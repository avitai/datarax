"""Tests for the Element and Batch modules."""

import pytest
import jax
import jax.numpy as jnp
from jax import tree_util

from datarax.core.element_batch import (
    Element,
    Batch,
    BatchOps,
    conditional_transform,
    iterative_transform,
    create_element,
    create_batch_from_arrays,
)
from datarax.core.metadata import Metadata


# ========================================================================
# Test Helper Functions
# ========================================================================


def transform_batch(batch: Batch, scale: float = 1.0) -> Batch:
    """Simple batch transformation helper for testing."""
    transformed_data = jax.tree.map(lambda x: x * scale, batch.data.get_value())
    return Batch.from_parts(
        data=transformed_data,
        states=batch.states.get_value(),
        metadata_list=batch._metadata_list,
        validate=False,
    )


class TestElement:
    """Test Element dataclass functionality."""

    def test_creation_empty(self):
        """Test creating empty element."""
        elem = Element()

        assert elem.data == {}
        assert elem.state == {}
        assert elem.metadata is None

    def test_creation_with_data(self):
        """Test creating element with data."""
        data = {"input": jnp.array([1, 2, 3]), "label": jnp.array([0, 1])}
        state = {"epoch": 0, "processed": False}
        metadata = Metadata(index=5)

        elem = Element(data=data, state=state, metadata=metadata)

        assert "input" in elem.data
        assert jnp.array_equal(elem.data["input"], jnp.array([1, 2, 3]))
        assert elem.state["epoch"] == 0
        assert elem.metadata.index == 5

    def test_immutability(self):
        """Test that Element is immutable."""
        elem = Element(data={"x": jnp.ones(3)})

        with pytest.raises(AttributeError):
            elem.data = {"y": jnp.zeros(3)}

    def test_replace(self):
        """Test replace method creates new instance."""
        elem = Element(data={"x": jnp.ones(3)}, state={"a": 1})

        new_elem = elem.replace(state={"a": 2, "b": 3})

        # Original unchanged
        assert elem.state == {"a": 1}

        # New instance has updates
        assert new_elem.state == {"a": 2, "b": 3}
        assert jnp.array_equal(new_elem.data["x"], elem.data["x"])

    def test_is_pytree(self):
        """Test Element is registered as PyTree."""
        elem = Element(
            data={"x": jnp.array([1, 2])}, state={"epoch": 1}, metadata=Metadata(index=0)
        )

        leaves, treedef = tree_util.tree_flatten(elem)
        reconstructed = tree_util.tree_unflatten(treedef, leaves)

        assert jnp.array_equal(reconstructed.data["x"], elem.data["x"])
        assert reconstructed.state == elem.state
        assert reconstructed.metadata.index == elem.metadata.index

    def test_update_state(self):
        """Test state update with merge behavior."""
        elem = Element(state={"a": 1, "b": 2})

        new_elem = elem.update_state({"b": 3, "c": 4})

        # Original unchanged
        assert elem.state == {"a": 1, "b": 2}

        # New has merged state
        assert new_elem.state == {"a": 1, "b": 3, "c": 4}

    def test_update_data(self):
        """Test data update with merge behavior."""
        elem = Element(data={"x": jnp.ones(3)})

        new_elem = elem.update_data({"x": jnp.zeros(3), "y": jnp.ones(2)})

        # Original unchanged
        assert jnp.array_equal(elem.data["x"], jnp.ones(3))

        # New has merged data
        assert jnp.array_equal(new_elem.data["x"], jnp.zeros(3))
        assert jnp.array_equal(new_elem.data["y"], jnp.ones(2))

    def test_transform(self):
        """Test transform method."""
        elem = Element(data={"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 4.0])})

        new_elem = elem.transform(lambda x: x * 2)

        assert jnp.array_equal(new_elem.data["a"], jnp.array([2.0, 4.0]))
        assert jnp.array_equal(new_elem.data["b"], jnp.array([6.0, 8.0]))

    def test_with_metadata(self):
        """Test metadata update."""
        elem = Element()
        metadata = Metadata(index=10, epoch=2)

        new_elem = elem.with_metadata(metadata)

        assert elem.metadata is None
        assert new_elem.metadata.index == 10
        assert new_elem.metadata.epoch == 2


class TestBatch:
    """Test Batch module functionality."""

    def test_creation_empty(self):
        """Test creating empty batch."""
        batch = Batch([])

        assert batch.batch_size == 0
        assert batch.get_data() == {}
        assert batch.get_batch_state() == {}
        assert batch.get_batch_metadata() is None

    def test_creation_from_elements(self):
        """Test creating batch from elements."""
        elements = [
            Element(data={"x": jnp.array([i, i + 1])}, state={"id": i}, metadata=Metadata(index=i))
            for i in range(4)
        ]

        batch = Batch(elements)

        assert batch.batch_size == 4
        assert "x" in batch.get_data()
        assert batch.get_data()["x"].shape == (4, 2)
        assert jnp.array_equal(batch.get_data()["x"][0], jnp.array([0, 1]))
        assert jnp.array_equal(batch.get_data()["x"][3], jnp.array([3, 4]))

    def test_creation_nested_pytree(self):
        """Test creating batch with nested PyTree data structure."""
        elements = [
            Element(
                data={
                    "vision": {"image": jnp.ones((224, 224, 3)), "depth": jnp.ones((224, 224))},
                    "text": jnp.ones((512,)),
                },
                state={},
                metadata=None,
            )
            for _ in range(4)
        ]

        batch = Batch(elements)

        assert batch.batch_size == 4
        assert batch.data.get_value()["vision"]["image"].shape == (4, 224, 224, 3)
        assert batch.data.get_value()["vision"]["depth"].shape == (4, 224, 224)
        assert batch.data.get_value()["text"].shape == (4, 512)

    def test_creation_deeply_nested_pytree(self):
        """Test creating batch with deeply nested (3+ level) PyTree structure."""
        elements = [
            Element(
                data={
                    "modality": {
                        "vision": {
                            "rgb": jnp.ones((224, 224, 3)),
                            "depth": jnp.ones((224, 224)),
                        },
                        "text": {
                            "tokens": jnp.ones((512,)),
                            "embeddings": jnp.ones((512, 768)),
                        },
                    }
                },
                state={},
                metadata=None,
            )
            for _ in range(2)
        ]

        batch = Batch(elements)

        assert batch.batch_size == 2
        assert batch.data.get_value()["modality"]["vision"]["rgb"].shape == (2, 224, 224, 3)
        assert batch.data.get_value()["modality"]["text"]["embeddings"].shape == (2, 512, 768)

    def test_get_element_nested_pytree(self):
        """Test extracting element from batch with nested PyTree."""
        elements = [
            Element(
                data={
                    "vision": {"image": jnp.ones((224, 224, 3)) * i},
                    "text": jnp.ones((512,)) * i,
                },
                state={"id": i},
                metadata=None,
            )
            for i in range(3)
        ]

        batch = Batch(elements)
        elem = batch.get_element(1)

        # Verify nested structure is preserved
        assert "vision" in elem.data
        assert "image" in elem.data["vision"]
        assert elem.data["vision"]["image"].shape == (224, 224, 3)
        assert elem.data["text"].shape == (512,)
        assert jnp.allclose(elem.data["vision"]["image"], jnp.ones((224, 224, 3)))
        assert elem.state["id"] == 1

    def test_get_element(self):
        """Test extracting single element."""
        elements = [
            Element(data={"x": jnp.array([i])}, state={"id": i}, metadata=Metadata(index=i))
            for i in range(3)
        ]

        batch = Batch(elements)
        elem = batch.get_element(1)

        assert jnp.array_equal(elem.data["x"], jnp.array([1]))
        assert elem.state["id"] == 1
        assert elem.metadata.index == 1

    def test_get_element_bounds(self):
        """Test get_element bounds checking."""
        batch = Batch([Element()])

        with pytest.raises(IndexError):
            batch.get_element(-1)

        with pytest.raises(IndexError):
            batch.get_element(1)

    def test_get_elements(self):
        """Test getting multiple elements."""
        elements = [Element(data={"x": jnp.array([i])}) for i in range(5)]
        batch = Batch(elements)

        # Test with slice
        elems = batch.get_elements(slice(1, 3))
        assert len(elems) == 2
        assert jnp.array_equal(elems[0].data["x"], jnp.array([1]))

        # Test with list
        elems = batch.get_elements([0, 2, 4])
        assert len(elems) == 3
        assert jnp.array_equal(elems[1].data["x"], jnp.array([2]))

    def test_slice(self):
        """Test batch slicing."""
        elements = [Element(data={"x": jnp.array([i])}) for i in range(6)]
        batch = Batch(elements)

        sliced = batch.slice(1, 4)

        assert sliced.batch_size == 3
        assert jnp.array_equal(sliced.get_data()["x"][0], jnp.array([1]))
        assert jnp.array_equal(sliced.get_data()["x"][2], jnp.array([3]))

    def test_split_for_devices(self):
        """Test splitting batch across devices."""
        elements = [Element(data={"x": jnp.array([i])}) for i in range(8)]
        batch = Batch(elements)

        splits = batch.split_for_devices(4)

        assert len(splits) == 4
        assert all(s.batch_size == 2 for s in splits)
        assert jnp.array_equal(splits[0].get_data()["x"][0], jnp.array([0]))
        assert jnp.array_equal(splits[3].get_data()["x"][1], jnp.array([7]))

    def test_split_for_devices_error(self):
        """Test error on incompatible split."""
        elements = [Element(data={"x": jnp.ones(2)}) for i in range(5)]
        batch = Batch(elements)

        with pytest.raises(ValueError):
            batch.split_for_devices(3)

    def test_compute_stats(self):
        """Test computing batch statistics."""
        elements = [Element(data={"x": jnp.array([i, i + 1, i + 2])}) for i in range(4)]

        batch = Batch(elements)
        stats = batch.compute_stats()

        assert "x_mean" in stats
        assert "x_std" in stats
        assert "x_min" in stats
        assert "x_max" in stats

        expected_mean = jnp.mean(jnp.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]), axis=0)
        assert jnp.allclose(stats["x_mean"], expected_mean)

    def test_batch_metadata_operations(self):
        """Test batch-level metadata."""
        batch = Batch([Element()])

        assert batch.get_batch_metadata() is None

        metadata = Metadata(index=100, epoch=5)
        batch.set_batch_metadata(metadata)

        assert batch.get_batch_metadata().index == 100
        assert batch.get_batch_metadata().epoch == 5

    def test_batch_state_operations(self):
        """Test batch-level state."""
        batch = Batch([Element()])

        assert batch.get_batch_state() == {}

        batch.update_batch_state({"loss": 0.5, "accuracy": 0.9})
        assert batch.get_batch_state()["loss"] == 0.5
        assert batch.get_batch_state()["accuracy"] == 0.9

        # Update merges
        batch.update_batch_state({"accuracy": 0.95, "step": 100})
        assert batch.get_batch_state()["accuracy"] == 0.95
        assert batch.get_batch_state()["step"] == 100
        assert batch.get_batch_state()["loss"] == 0.5

    def test_validation(self):
        """Test batch validation."""
        # Valid batch should not raise
        elements = [Element(data={"x": jnp.ones(3)}) for _ in range(4)]
        batch = Batch(elements, validate=True)
        assert batch.batch_size == 4

        # Invalid batch (mismatched shapes) should raise
        # This would require manual construction, skipping for now


class TestTransformations:
    """Test Batch.from_parts() and PyTree transformation capabilities."""

    def test_batch_from_parts_simple(self):
        """Test creating batch from pre-built parts with simple flat data."""
        data = {"image": jnp.ones((32, 224, 224, 3))}
        states = [{} for _ in range(32)]
        metadata_list = [None for _ in range(32)]

        batch = Batch.from_parts(data, states, metadata_list)

        assert batch.batch_size == 32
        assert batch.data.get_value()["image"].shape == (32, 224, 224, 3)
        assert len(batch.states.get_value()) == 32
        assert len(batch._metadata_list) == 32

    def test_batch_from_parts_nested_pytree(self):
        """Test from_parts with nested PyTree structure."""
        data = {
            "vision": {"image": jnp.ones((4, 224, 224, 3)), "depth": jnp.ones((4, 224, 224))},
            "text": jnp.ones((4, 512)),
        }
        states = [{"step": i} for i in range(4)]

        batch = Batch.from_parts(data, states)

        assert batch.batch_size == 4
        assert batch.data.get_value()["vision"]["image"].shape == (4, 224, 224, 3)
        assert batch.data.get_value()["vision"]["depth"].shape == (4, 224, 224)
        assert batch.data.get_value()["text"].shape == (4, 512)

    def test_batch_from_parts_deeply_nested_pytree(self):
        """Test from_parts with 3+ level nesting."""
        data = {
            "modality": {
                "vision": {"rgb": jnp.ones((2, 224, 224, 3)), "depth": jnp.ones((2, 224, 224))},
                "text": {"tokens": jnp.ones((2, 512)), "embeddings": jnp.ones((2, 512, 768))},
            }
        }
        states = [{} for _ in range(2)]

        batch = Batch.from_parts(data, states)

        assert batch.batch_size == 2
        assert batch.data.get_value()["modality"]["vision"]["rgb"].shape == (2, 224, 224, 3)
        assert batch.data.get_value()["modality"]["text"]["embeddings"].shape == (2, 512, 768)

    def test_batch_from_parts_with_batch_level_data(self):
        """Test from_parts preserves batch-level metadata and state."""
        data = {"x": jnp.ones((3, 5))}
        states = [{} for _ in range(3)]
        batch_metadata = Metadata(index=42)
        batch_state = {"dataset": "train", "epoch": 5}

        batch = Batch.from_parts(
            data, states, batch_metadata=batch_metadata, batch_state=batch_state
        )

        assert batch._batch_metadata.index == 42
        assert batch.batch_state.get_value()["dataset"] == "train"
        assert batch.batch_state.get_value()["epoch"] == 5

    def test_batch_from_parts_validation_inconsistent_batch_dims(self):
        """Test from_parts validation catches inconsistent batch dimensions."""
        data = {"a": jnp.ones((5, 10)), "b": jnp.ones((3, 10))}  # Different batch sizes!
        states = [{} for _ in range(5)]

        with pytest.raises(ValueError, match="Inconsistent batch dimensions"):
            Batch.from_parts(data, states, validate=True)

    def test_batch_from_parts_validation_states_length(self):
        """Test from_parts validation catches states batch dimension mismatch."""
        data = {"x": jnp.ones((5, 10))}
        # PyTree states with wrong batch dimension
        states = {"count": jnp.array([0, 1, 2])}  # Length 3, but batch size is 5

        with pytest.raises(ValueError, match="states batch dimension.*doesn't match"):
            Batch.from_parts(data, states, validate=True)

    def test_batch_from_parts_validation_metadata_length(self):
        """Test from_parts validation catches metadata_list length mismatch."""
        data = {"x": jnp.ones((5, 10))}
        states = {}  # Empty PyTree
        metadata_list = [None for _ in range(3)]  # Wrong length!

        with pytest.raises(ValueError, match="metadata_list length.*doesn't match"):
            Batch.from_parts(data, states, metadata_list=metadata_list, validate=True)

    def test_batch_from_parts_no_validation(self):
        """Test from_parts with validation disabled (faster hot path)."""
        data = {"x": jnp.ones((10, 5))}
        states = [{} for _ in range(10)]

        batch = Batch.from_parts(data, states, validate=False)

        assert batch.batch_size == 10
        assert batch.data.get_value()["x"].shape == (10, 5)

    def test_batch_from_parts_default_metadata(self):
        """Test from_parts creates default None metadata when not provided."""
        data = {"x": jnp.ones((3, 2))}
        states = [{} for _ in range(3)]

        batch = Batch.from_parts(data, states)  # No metadata_list

        assert len(batch._metadata_list) == 3
        assert all(m is None for m in batch._metadata_list)


class TestBatchOps:
    """Test BatchOps utility class."""

    def test_filter_batch(self):
        """Test filtering batch elements."""
        elements = [Element(data={"x": jnp.array([i])}, state={"id": i}) for i in range(5)]
        batch = Batch(elements)

        mask = jnp.array([True, False, True, False, True])
        filtered = BatchOps.filter_batch(batch, mask)

        assert filtered.batch_size == 3
        assert jnp.array_equal(filtered.get_data()["x"], jnp.array([[0], [2], [4]]))

    def test_filter_batch_empty(self):
        """Test filtering with all False mask."""
        elements = [Element(data={"x": jnp.array([i])}) for i in range(3)]
        batch = Batch(elements)

        mask = jnp.array([False, False, False])
        filtered = BatchOps.filter_batch(batch, mask)

        assert filtered.batch_size == 0
        # Empty batch preserves structure with zero-length arrays
        assert "x" in filtered.get_data()
        assert filtered.get_data()["x"].shape[0] == 0

    def test_filter_batch_mask_mismatch(self):
        """Test error on mask size mismatch."""
        batch = Batch([Element()])
        mask = jnp.array([True, False])

        with pytest.raises(ValueError):
            BatchOps.filter_batch(batch, mask)

    def test_concatenate_batches(self):
        """Test concatenating multiple batches."""
        batch1 = Batch([Element(data={"x": jnp.array([i])}) for i in range(3)])
        batch2 = Batch([Element(data={"x": jnp.array([i + 3])}) for i in range(2)])

        concatenated = BatchOps.concatenate_batches([batch1, batch2])

        assert concatenated.batch_size == 5
        assert jnp.array_equal(concatenated.get_data()["x"], jnp.array([[0], [1], [2], [3], [4]]))

    def test_concatenate_empty(self):
        """Test concatenating empty list."""
        result = BatchOps.concatenate_batches([])
        assert result.batch_size == 0

    def test_concatenate_single(self):
        """Test concatenating single batch."""
        batch = Batch([Element(data={"x": jnp.ones(2)})])
        result = BatchOps.concatenate_batches([batch])

        assert result.batch_size == batch.batch_size
        assert jnp.array_equal(result.get_data()["x"], batch.get_data()["x"])

    def test_update_batch_inplace(self):
        """Test in-place batch update."""
        elements = [Element(data={"x": jnp.ones(2)}) for _ in range(3)]
        batch = Batch(elements)

        new_data = {"x": jnp.zeros((3, 2)), "y": jnp.ones((3, 1))}
        updated = BatchOps.update_batch_inplace(batch, new_data)

        assert jnp.array_equal(updated.get_data()["x"], jnp.zeros((3, 2)))
        assert jnp.array_equal(updated.get_data()["y"], jnp.ones((3, 1)))


class TestControlFlow:
    """Test JAX control flow operations."""

    def test_conditional_transform(self):
        """Test conditional transformation with static condition."""
        batch = Batch([Element(data={"x": jnp.ones(3)})])

        def true_fn(b):
            return transform_batch(b, 2.0)

        def false_fn(b):
            return transform_batch(b, 0.5)

        # Test true condition
        result_true = conditional_transform(batch, true_fn, false_fn, True)
        assert jnp.array_equal(result_true.get_data()["x"], jnp.array([[2.0, 2.0, 2.0]]))

        # Test false condition
        result_false = conditional_transform(batch, true_fn, false_fn, False)
        assert jnp.array_equal(result_false.get_data()["x"], jnp.array([[0.5, 0.5, 0.5]]))

    def test_iterative_transform(self):
        """Test iterative transformation."""
        batch = Batch([Element(data={"x": jnp.ones(3)})])

        def iteration_fn(b, i):
            # Scale by 1 + 0.1 * iteration
            return transform_batch(b, 1.0 + i * 0.1)

        result = iterative_transform(batch, iteration_fn, 3)

        # 1 * 1.0 * 1.1 * 1.2 = 1.32
        expected = jnp.ones(3) * 1.32
        assert jnp.allclose(result.get_data()["x"][0], expected)


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_element(self):
        """Test element factory."""
        elem = create_element(data={"x": jnp.ones(3)}, state={"id": 1}, metadata=Metadata(index=5))

        assert jnp.array_equal(elem.data["x"], jnp.ones(3))
        assert elem.state["id"] == 1
        assert elem.metadata.index == 5

    def test_create_element_defaults(self):
        """Test element factory with defaults."""
        elem = create_element()

        assert elem.data == {}
        assert elem.state == {}
        assert elem.metadata is None

    def test_create_batch_from_arrays(self):
        """Test batch creation from arrays."""
        data = {"x": jnp.ones((4, 3)), "y": jnp.zeros((4, 2))}
        states = [{"id": i} for i in range(4)]
        metadata_list = [Metadata(index=i) for i in range(4)]

        batch = create_batch_from_arrays(data, states, metadata_list)

        assert batch.batch_size == 4
        assert jnp.array_equal(batch.get_data()["x"], jnp.ones((4, 3)))
        assert jnp.array_equal(batch.get_data()["y"], jnp.zeros((4, 2)))

        elem = batch.get_element(2)
        assert elem.state["id"] == 2
        assert elem.metadata.index == 2

    def test_create_batch_from_arrays_empty(self):
        """Test batch creation with empty arrays."""
        batch = create_batch_from_arrays({})

        assert batch.batch_size == 0
        assert batch.get_data() == {}


class TestJAXIntegration:
    """Test JAX integration and compilation."""

    def test_batch_in_jit(self):
        """Test batch operations in JIT compiled function."""

        @jax.jit
        def process_batch(batch: Batch) -> jax.Array:
            return jnp.sum(batch.get_data()["x"])

        batch = Batch([Element(data={"x": jnp.ones(3) * i}) for i in range(4)])

        result = process_batch(batch)
        expected = jnp.sum(jnp.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]))
        assert result == expected

    def test_no_recompilation(self):
        """Test that different data doesn't cause recompilation."""
        trace_count = 0

        @jax.jit
        def process(batch: Batch) -> jax.Array:
            nonlocal trace_count
            trace_count += 1
            return jnp.sum(batch.get_data()["x"])

        batch1 = Batch([Element(data={"x": jnp.ones(3) * i}) for i in range(4)])
        batch2 = Batch([Element(data={"x": jnp.ones(3) * (i + 10)}) for i in range(4)])

        result1 = process(batch1)
        first_trace = trace_count

        result2 = process(batch2)

        # Should not recompile for different data values
        assert trace_count == first_trace
        assert result1 != result2

    def test_element_in_tree_map(self):
        """Test element works with tree_map."""
        elem1 = Element(data={"x": jnp.ones(3)})
        elem2 = Element(data={"x": jnp.ones(3) * 2})

        result = jax.tree.map(lambda a, b: a + b, elem1, elem2)

        assert jnp.array_equal(result.data["x"], jnp.ones(3) * 3)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_element_batch(self):
        """Test single-element batch."""
        elem = Element(data={"x": jnp.array([1, 2, 3])})
        batch = Batch([elem])

        assert batch.batch_size == 1
        stats = batch.compute_stats()
        assert jnp.array_equal(stats["x_mean"], jnp.array([1, 2, 3]))
        assert jnp.array_equal(stats["x_std"], jnp.zeros(3))

    def test_mixed_data_types(self):
        """Test batch with mixed data types."""
        elements = [
            Element(
                data={
                    "float32": jnp.array([1.0], dtype=jnp.float32),
                    "int32": jnp.array([2], dtype=jnp.int32),
                    "bool": jnp.array([True], dtype=jnp.bool_),
                }
            )
            for _ in range(3)
        ]

        batch = Batch(elements)

        assert batch.get_data()["float32"].dtype == jnp.float32
        assert batch.get_data()["int32"].dtype == jnp.int32
        assert batch.get_data()["bool"].dtype == jnp.bool_

    def test_large_batch(self):
        """Test handling of large batches."""
        elements = [Element(data={"x": jnp.ones(100) * i}) for i in range(1000)]

        batch = Batch(elements)

        assert batch.batch_size == 1000
        assert batch.get_data()["x"].shape == (1000, 100)

        stats = batch.compute_stats()
        assert "x_mean" in stats


class TestPerformance:
    """Test performance characteristics."""

    def test_batch_creation_performance(self, benchmark):
        """Benchmark batch creation."""
        elements = [Element(data={"x": jnp.ones(100)}) for _ in range(100)]

        def create():
            return Batch(elements)

        batch = benchmark(create)
        assert batch.batch_size == 100

    def test_transform_performance(self, benchmark):
        """Benchmark transformation."""
        elements = [Element(data={"x": jnp.ones(100)}) for _ in range(100)]
        batch = Batch(elements)

        def transform():
            return transform_batch(batch, 2.0)

        result = benchmark(transform)
        assert result.batch_size == 100

    def test_filter_performance(self, benchmark):
        """Benchmark filtering."""
        elements = [Element(data={"x": jnp.ones(10)}) for _ in range(1000)]
        batch = Batch(elements)
        mask = jnp.array([i % 2 == 0 for i in range(1000)])

        def filter_op():
            return BatchOps.filter_batch(batch, mask)

        result = benchmark(filter_op)
        assert result.batch_size == 500


class TestElementGradientPreservation:
    """Test gradient preservation in Element operations."""

    def test_apply_to_data_basic(self):
        """Test Element.apply_to_data preserves gradients in basic case."""
        # Create element with JAX arrays
        element = Element(
            data={
                "features": jnp.array([1.0, 2.0, 3.0]),
                "weights": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            },
            state={"epoch": 1},
            metadata=Metadata(index=0),
        )

        # Define differentiable transformation
        def scale_transform(x):
            return x * 2.0

        # Test gradient computation through apply_to_data
        def loss_fn(features, weights):
            elem = Element(data={"features": features, "weights": weights})
            transformed = elem.apply_to_data(scale_transform)
            return jnp.sum(transformed.data["features"]) + jnp.sum(transformed.data["weights"])

        # Compute gradients
        grad_fn = jax.grad(loss_fn, argnums=(0, 1))
        grads = grad_fn(element.data["features"], element.data["weights"])

        # Gradients should be preserved (scaled by 2.0)
        expected_features_grad = jnp.array([2.0, 2.0, 2.0])
        expected_weights_grad = jnp.array([[2.0, 2.0], [2.0, 2.0]])

        assert jnp.allclose(grads[0], expected_features_grad)
        assert jnp.allclose(grads[1], expected_weights_grad)

    def test_apply_to_data_preserves_structure(self):
        """Test apply_to_data preserves element structure."""
        element = Element(
            data={"x": jnp.array([1.0, 2.0])}, state={"processed": True}, metadata=Metadata(index=5)
        )

        def double_transform(x):
            return x * 2.0

        transformed = element.apply_to_data(double_transform)

        # Data should be transformed
        assert jnp.allclose(transformed.data["x"], jnp.array([2.0, 4.0]))

        # State and metadata should be preserved
        assert transformed.state == {"processed": True}
        assert transformed.metadata.index == 5

    def test_apply_to_data_jit_compatibility(self):
        """Test apply_to_data works with JAX JIT compilation."""
        element = Element(data={"x": jnp.array([1.0, 2.0, 3.0])})

        def linear_transform(x):
            return 3.0 * x + 1.0

        @jax.jit
        def jitted_transform(elem_data):
            elem = Element(data=elem_data)
            transformed = elem.apply_to_data(linear_transform)
            return transformed.data

        result = jitted_transform(element.data)
        expected = {"x": jnp.array([4.0, 7.0, 10.0])}

        assert jnp.allclose(result["x"], expected["x"])


class TestBatchGradientPreservation:
    """Test gradient preservation in Batch operations."""


class TestComplexGradientFlows:
    """Test complex gradient flows through Element/Batch operations."""

    def test_nested_transformations(self):
        """Test gradients through nested Element transformations."""
        element = Element(data={"x": jnp.array([1.0, 2.0, 3.0])})

        def nested_transform(x):
            return jnp.sin(jnp.cos(x * 2.0)) + x

        def loss_fn(input_data):
            elem = Element(data={"x": input_data})
            # Apply transformation twice
            transformed1 = elem.apply_to_data(nested_transform)
            transformed2 = transformed1.apply_to_data(lambda x: x**2)
            return jnp.sum(transformed2.data["x"])

        # Should be differentiable
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(element.data["x"])

        assert grads is not None
        assert grads.shape == element.data["x"].shape
        assert not jnp.allclose(grads, 0.0)  # Non-zero gradients
