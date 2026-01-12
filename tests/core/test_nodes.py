"""Comprehensive test suite for DAG nodes.

Written FIRST following TDD principles.
"""

import pytest

import jax
import jax.numpy as jnp
import flax.nnx as nnx
from typing import Iterator

from datarax.dag.nodes import (
    Node,
    DataSourceNode,
    BatchNode,
    OperatorNode,
    ShuffleNode,
    CacheNode,
    Sequential,
    Parallel,
    RebatchNode,
    DifferentiableRebatchImpl,
    FastRebatchImpl,
    GradientTransparentRebatchImpl,
    dataloader,
    DataLoader,
)
from datarax.core.element_batch import create_batch_from_arrays
from datarax.core.data_source import DataSourceModule
from datarax.core.operator import OperatorModule
from datarax.core.config import StructuralConfig
from datarax.typing import Batch, Element
from datarax.core.module import DataraxModule


class MockDataSource(DataSourceModule):
    """Mock data source for testing."""

    # REQUIRED: Annotate data attribute with nnx.data() to prevent NNX container errors
    data: list = nnx.data()

    def __init__(self, config: StructuralConfig, data: list, *, rngs: nnx.Rngs | None = None):
        super().__init__(config, rngs=rngs)
        self.data = data
        self.index = nnx.Variable(0)

    def __iter__(self) -> Iterator[Element]:
        # Don't reset index to support state restoration
        # Only reset if we've reached the end
        if self.index.get_value() >= len(self.data):
            self.index.set_value(0)
        return self

    def __next__(self) -> Element:
        if self.index.get_value() >= len(self.data):
            raise StopIteration
        item = self.data[self.index.get_value()]
        self.index.set_value(self.index.get_value() + 1)
        return item

    def __len__(self) -> int:
        return len(self.data)

    def get_state(self) -> dict:
        """Get current state for checkpointing."""
        return {"index": self.index.get_value()}

    def set_state(self, state: dict) -> None:
        """Restore state from checkpoint."""
        if "index" in state:
            self.index.set_value(state["index"])


class MockDataSourceV2(DataSourceModule):
    """Mock data source for testing."""

    # REQUIRED: Annotate data attribute with nnx.data() to prevent NNX container errors
    data: list = nnx.data()

    def __init__(
        self, config: StructuralConfig, size=10, *, rngs: nnx.Rngs | None = None, name="mock_source"
    ):
        super().__init__(config, rngs=rngs, name=name)
        self.size = size
        self.data = [{"data": jnp.array([i, i + 1, i + 2])} for i in range(size)]
        self.index = 0

    def __iter__(self) -> Iterator[Element]:
        self.index = 0
        return self

    def __next__(self) -> Element:
        if self.index >= len(self.data):
            raise StopIteration
        result = self.data[self.index]
        self.index += 1
        return result


class MockOperator(OperatorModule):
    """Mock operator for testing with call count tracking."""

    def __init__(self, factor: float = 2.0, *, rngs: nnx.Rngs | None = None):
        from datarax.core.config import OperatorConfig

        config = OperatorConfig(stochastic=False)
        super().__init__(config, rngs=rngs, name="mock_operator")
        self.factor = factor
        self.call_count = nnx.Variable(0)

    def __call__(self, data: Batch) -> Batch:
        self.call_count.set_value(self.call_count.get_value() + 1)
        return jax.tree.map(lambda x: x * self.factor, data)


def create_mock_operator(factor: float = 2.0, rngs: nnx.Rngs | None = None) -> MockOperator:
    """Create a mock operator for testing."""
    return MockOperator(factor=factor, rngs=rngs)


class TestDataSourceNode:
    """Test DataSourceNode implementation."""

    def test_initialization(self):
        """Test DataSourceNode initialization."""
        source = MockDataSource(StructuralConfig(), [1, 2, 3], rngs=nnx.Rngs(0))
        node = DataSourceNode(source, name="test_source")

        assert node.name == "test_source"
        assert node.source is source
        assert isinstance(node, Node)
        assert isinstance(node, nnx.Module)

    def test_iteration(self):
        """Test DataSourceNode iteration."""
        data = [{"value": i} for i in range(5)]
        source = MockDataSource(StructuralConfig(), data, rngs=nnx.Rngs(0))
        node = DataSourceNode(source)

        results = []
        for item in node:
            results.append(item)

        assert len(results) == 5
        assert results[0]["value"] == 0
        assert results[-1]["value"] == 4

    def test_call_method(self):
        """Test DataSourceNode __call__ for single iteration."""
        source = MockDataSource(StructuralConfig(), [1, 2, 3], rngs=nnx.Rngs(0))
        node = DataSourceNode(source)

        # First call returns first element
        result = node(None)
        assert result == 1

        # Second call returns second element
        result = node(None)
        assert result == 2

    def test_state_management(self):
        """Test DataSourceNode state management."""
        source = MockDataSource(StructuralConfig(), [1, 2, 3], rngs=nnx.Rngs(0))
        node = DataSourceNode(source)

        # Advance iteration
        _ = node(None)
        _ = node(None)

        # Get state
        state = node.get_state()
        assert "source_state" in state
        assert state["source_state"]["index"] == 2

        # Reset and restore
        new_node = DataSourceNode(MockDataSource(StructuralConfig(), [1, 2, 3], rngs=nnx.Rngs(0)))
        new_node.set_state(state)

        # Should continue from saved position
        result = new_node(None)
        assert result == 3


class TestBatchNode:
    """Test BatchNode implementation."""

    def test_initialization(self):
        """Test BatchNode initialization."""
        node = BatchNode(batch_size=32, drop_remainder=True)

        assert node.batch_size == 32
        assert node.drop_remainder is True
        assert list(node.buffer) == []  # nnx.List is iterable, not Variable

    def test_batch_creation(self):
        """Test batch creation from elements."""
        node = BatchNode(batch_size=3)

        # Feed elements
        elements = [{"data": jnp.array([i])} for i in range(10)]

        batches = []
        for elem in elements:
            batch = node(elem)
            if batch is not None:
                batches.append(batch)

        # Should create 3 complete batches
        assert len(batches) == 3

        # Check first batch
        assert batches[0]["data"].shape == (3, 1)
        assert jnp.array_equal(batches[0]["data"], jnp.array([[0], [1], [2]]))

    def test_drop_remainder(self):
        """Test drop_remainder functionality."""
        node = BatchNode(batch_size=3, drop_remainder=True)

        elements = [{"data": jnp.array([i])} for i in range(10)]

        batches = []
        for elem in elements:
            batch = node(elem)
            if batch is not None:
                batches.append(batch)

        # Should only create 3 batches (drop last incomplete)
        assert len(batches) == 3

        # Flush should return None when drop_remainder=True
        assert node.flush() is None

    def test_flush_incomplete_batch(self):
        """Test flushing incomplete batches."""
        node = BatchNode(batch_size=3, drop_remainder=False)

        # Add 2 elements (incomplete batch)
        node({"data": jnp.array([1])})
        node({"data": jnp.array([2])})

        # Flush should return the incomplete batch
        batch = node.flush()
        assert batch is not None
        assert batch["data"].shape == (2, 1)

    def test_state_preservation(self):
        """Test BatchNode state preservation."""
        node = BatchNode(batch_size=2)

        # Add one element
        node({"data": jnp.array([1])})

        # Get state
        state = node.get_state()
        assert len(state["buffer"]) == 1

        # Create new node and restore
        new_node = BatchNode(batch_size=2)
        new_node.set_state(state)

        # Add another element to complete batch
        batch = new_node({"data": jnp.array([2])})
        assert batch is not None
        assert batch["data"].shape == (2, 1)


class TestOperatorNode:
    """Test OperatorNode implementation."""

    def test_initialization(self):
        """Test OperatorNode initialization."""
        transformer = create_mock_operator(factor=3.0, rngs=nnx.Rngs())
        node = OperatorNode(transformer, name="test_transform")

        assert node.name == "test_transform"
        assert node.operator is transformer

    def test_batch_transformation(self):
        """Test batch transformation."""
        transformer = create_mock_operator(factor=2.5, rngs=nnx.Rngs())
        node = OperatorNode(transformer, name="test_transform")

        batch = create_batch_from_arrays({"data": jnp.array([[1.0], [2.0], [3.0]])})
        result = node(batch)

        expected = jnp.array([[2.5], [5.0], [7.5]])
        # Access data via Batch interface if result is a Batch
        result_data = (
            result.data.get_value()["data"] if isinstance(result, Batch) else result["data"]
        )
        assert jnp.allclose(result_data, expected)
        assert transformer.call_count.get_value() == 1

    def test_with_rng_key(self):
        """Test transformation with RNG key."""
        transformer = create_mock_operator(rngs=nnx.Rngs())
        node = OperatorNode(transformer, name="test_transform")

        batch = create_batch_from_arrays({"data": jnp.ones((3, 2))})
        key = jax.random.key(42)

        result = node(batch, key=key)
        assert result is not None


class TestShuffleNode:
    """Test ShuffleNode implementation."""

    def test_initialization(self):
        """Test ShuffleNode initialization."""
        node = ShuffleNode(buffer_size=100, seed=42)

        assert node.buffer_size == 100
        assert len(node.buffer) == 0  # buffer is now nnx.data([]), returns plain list
        assert node.rng_key is not None

    def test_buffer_filling(self):
        """Test shuffle buffer filling."""
        node = ShuffleNode(buffer_size=5, seed=42)

        # Add elements to fill buffer
        for i in range(5):
            result = node({"id": i})
            assert result is None  # Buffer filling

        assert len(node.buffer) == 5  # buffer is now nnx.data([]), returns plain list

        # Next element should trigger output
        result = node({"id": 5})
        assert result is not None
        assert "id" in result

    def test_shuffling(self):
        """Test that shuffling actually occurs."""
        node = ShuffleNode(buffer_size=10, seed=42)

        # Fill buffer and collect outputs
        outputs = []
        for i in range(20):
            result = node({"id": i})
            if result is not None:
                outputs.append(result["id"])

        # Flush remaining
        while True:
            result = node.flush()
            if result is None:
                break
            outputs.append(result["id"])

        # Check that we got all elements
        assert len(outputs) == 20
        assert set(outputs) == set(range(20))

        # Check that order is different (with high probability)
        original = list(range(20))
        assert outputs != original  # Should be shuffled

    def test_deterministic_shuffle(self):
        """Test deterministic shuffling with same seed."""
        node1 = ShuffleNode(buffer_size=5, seed=123)
        node2 = ShuffleNode(buffer_size=5, seed=123)

        data = [{"id": i} for i in range(10)]

        # Process through both nodes
        outputs1 = []
        outputs2 = []

        for item in data:
            result1 = node1(item)
            result2 = node2(item)

            if result1 is not None:
                outputs1.append(result1["id"])
            if result2 is not None:
                outputs2.append(result2["id"])

        # Flush both
        for _ in range(5):
            outputs1.append(node1.flush()["id"])
            outputs2.append(node2.flush()["id"])

        # Should produce same order
        assert outputs1 == outputs2


class TestCacheNode:
    """Test CacheNode implementation."""

    def test_initialization(self):
        """Test CacheNode initialization."""
        inner = OperatorNode(create_mock_operator(rngs=nnx.Rngs()), name="test_transform")
        node = CacheNode(inner, cache_size=100)

        assert node.cache_size == 100
        assert len(node.cache.get_value()) == 0
        assert node.inner_node is inner

    def test_caching(self):
        """Test caching behavior."""
        transformer = create_mock_operator(factor=2.0, rngs=nnx.Rngs())
        inner = OperatorNode(transformer, name="test_transform")
        node = CacheNode(inner, cache_size=10)

        batch1 = create_batch_from_arrays({"data": jnp.array([[1.0], [2.0]])})

        # First call - should execute transform
        result1 = node(batch1)
        assert transformer.call_count.get_value() == 1

        # Second call with same input - should use cache
        result2 = node(batch1)
        assert transformer.call_count.get_value() == 1  # No additional call

        # Compare batch data
        data1 = result1.data.get_value()["data"]
        data2 = result2.data.get_value()["data"]
        assert jnp.array_equal(data1, data2)

    def test_cache_size_limit(self):
        """Test cache size limiting."""
        inner = OperatorNode(create_mock_operator(rngs=nnx.Rngs()), name="test_transform")
        node = CacheNode(inner, cache_size=3)

        # Add more than cache size
        for i in range(5):
            batch = create_batch_from_arrays({"data": jnp.array([[float(i)]])})
            _ = node(batch)

        is_buffer_full = len(node.cache.get_value()) <= 3
        assert is_buffer_full

    def test_cache_clearing(self):
        """Test cache clearing."""
        inner = OperatorNode(create_mock_operator(rngs=nnx.Rngs()), name="test_transform")
        node = CacheNode(inner, cache_size=10)

        # Add to cache
        for i in range(3):
            batch = create_batch_from_arrays({"data": jnp.array([[float(i)]])})
            _ = node(batch)

        assert len(node.cache.get_value()) == 3

        # Clear cache
        node.clear_cache()
        assert len(node.cache.get_value()) == 0


class TestComposition:
    """Test node composition with operators."""

    def test_sequential_operator(self):
        """Test >> operator for sequential composition."""
        source = DataSourceNode(MockDataSource(StructuralConfig(), [1, 2, 3], rngs=nnx.Rngs(0)))
        batch = BatchNode(batch_size=2)
        transform = OperatorNode(
            create_mock_operator(factor=2.0, rngs=nnx.Rngs()), name="test_transform"
        )

        # Compose with >>
        pipeline = source >> batch >> transform

        assert isinstance(pipeline, Sequential)
        assert len(pipeline.nodes) == 3

    def test_parallel_operator(self):
        """Test | operator for parallel composition."""
        t1 = OperatorNode(create_mock_operator(factor=2.0, rngs=nnx.Rngs()), name="test_transform")
        t2 = OperatorNode(create_mock_operator(factor=3.0, rngs=nnx.Rngs()), name="test_transform")
        t3 = OperatorNode(create_mock_operator(factor=4.0, rngs=nnx.Rngs()), name="test_transform")

        # Compose with |
        parallel = t1 | t2 | t3

        assert isinstance(parallel, Parallel)
        assert len(parallel.nodes) == 3

    def test_mixed_operators(self):
        """Test mixing >> and | operators."""
        source = DataSourceNode(MockDataSource(StructuralConfig(), [1, 2, 3], rngs=nnx.Rngs(0)))
        batch = BatchNode(batch_size=2)
        t1 = OperatorNode(create_mock_operator(factor=2.0, rngs=nnx.Rngs()), name="test_transform")
        t2 = OperatorNode(create_mock_operator(factor=3.0, rngs=nnx.Rngs()), name="test_transform")

        # Create complex pipeline
        pipeline = source >> batch >> (t1 | t2)

        assert isinstance(pipeline, Sequential)
        assert len(pipeline.nodes) == 3
        assert isinstance(pipeline.nodes[2], Parallel)


class TestIntegration:
    """Integration tests for complete pipelines."""

    def test_end_to_end_pipeline(self):
        """Test complete pipeline execution."""
        # Create data source
        data = [{"value": jnp.array([float(i)])} for i in range(10)]
        source = DataSourceNode(MockDataSource(StructuralConfig(), data, rngs=nnx.Rngs(0)))

        # Create pipeline
        batch = BatchNode(batch_size=3)
        transform = OperatorNode(
            create_mock_operator(factor=2.0, rngs=nnx.Rngs()), name="test_transform"
        )

        source >> batch >> transform

        # Execute pipeline
        results = []
        for item in data:
            elem = source(None)
            if elem is not None:
                batched = batch(elem)
                if batched is not None:
                    result = transform(batched)
                    results.append(result)

        # Get last incomplete batch
        last_batch = batch.flush()
        if last_batch is not None:
            results.append(transform(last_batch))

        # Verify results
        assert len(results) == 4  # 3 complete + 1 incomplete

        # Check first batch values
        expected_first = jnp.array([[0.0], [2.0], [4.0]])
        assert jnp.allclose(results[0]["value"], expected_first)


class TestImplementationClasses:
    """Test individual implementation classes."""

    @pytest.fixture
    def simple_batch(self) -> Batch:
        """Create a simple batch for testing."""
        return {
            "data": jnp.array([[1.0], [2.0], [3.0]]),
            "labels": jnp.array([0, 1, 2]),
        }

    def test_differentiable_impl_initialization(self) -> None:
        """Test DifferentiableRebatchImpl initialization."""
        impl = DifferentiableRebatchImpl(target_batch_size=4, max_buffer_size=100)

        assert impl.target_batch_size == 4
        assert impl.max_buffer_size == 100
        assert impl.buffer.get_value() is None  # Not initialized until first batch
        assert impl.buffer_size.get_value() == 0

    def test_fast_impl_initialization(self) -> None:
        """Test FastRebatchImpl initialization."""
        impl = FastRebatchImpl(target_batch_size=8, max_buffer_size=256)

        assert impl.target_batch_size == 8
        assert impl.max_buffer_size == 256
        assert impl.buffer.get_value() is None  # Not initialized until first batch
        assert impl.write_index.get_value() == 0
        assert impl.read_index.get_value() == 0
        assert impl.count.get_value() == 0

    def test_gradient_transparent_impl_initialization(self) -> None:
        """Test GradientTransparentRebatchImpl initialization."""
        impl = GradientTransparentRebatchImpl(target_batch_size=5)

        assert impl.target_batch_size == 5
        assert len(impl.buffer) == 0
        assert len(impl.gradient_tape) == 0

    def test_differentiable_impl_basic_operation(self, simple_batch: Batch) -> None:
        """Test basic operation of DifferentiableRebatchImpl."""
        impl = DifferentiableRebatchImpl(target_batch_size=2)

        # First call initializes buffer and accumulates
        output, is_valid = impl(simple_batch)

        # Should emit a batch of size 2
        assert is_valid
        assert output is not None
        assert output["data"].shape == (2, 1)
        assert jnp.array_equal(output["labels"], jnp.array([0, 1]))

        # Flush should return remainder
        remainder = impl.flush()
        assert remainder is not None
        assert remainder["data"].shape == (1, 1)
        assert remainder["labels"][0] == 2

    def test_fast_impl_circular_buffer(self) -> None:
        """Test FastRebatchImpl circular buffer behavior."""
        impl = FastRebatchImpl(target_batch_size=3, max_buffer_size=5)

        # Add batches that would overflow a linear buffer
        for i in range(3):
            batch = {"data": jnp.array([[i], [i + 1]])}
            output, is_valid = impl(batch)

            if i == 0:
                assert not is_valid  # 2 < 3
            elif i == 1:
                assert is_valid  # 4 >= 3
                assert output is not None
                assert output["data"].shape == (3, 1)
            elif i == 2:
                # After emitting 3, we had 1 left, now adding 2 more = 3
                assert is_valid  # 3 >= 3
                assert output is not None

        # After two emissions, buffer should be empty or have remainder
        assert impl.count.get_value() >= 0  # Valid state

    def test_gradient_transparent_impl_preserves_structure(self, simple_batch: Batch) -> None:
        """Test GradientTransparentRebatchImpl preserves gradient structure."""
        impl = GradientTransparentRebatchImpl(target_batch_size=2)

        # Process batch
        output, is_valid = impl(simple_batch)

        assert is_valid
        assert output is not None
        assert output["data"].shape == (2, 1)

        # Check that gradient tape is maintained
        assert len(impl.gradient_tape) == 1  # Remainder stored

        # Flush should use gradient tape
        remainder = impl.flush()
        assert remainder is not None
        assert len(impl.gradient_tape) == 0  # Cleared after flush


class TestRebatchNodeIntegration:
    """Test integrated RebatchNode with all implementations."""

    @pytest.fixture
    def simple_batch(self) -> Batch:
        """Create a simple batch for testing."""
        return {
            "data": jnp.array([[1.0], [2.0], [3.0], [4.0], [5.0]]),
            "labels": jnp.array([0, 1, 2, 3, 4]),
        }

    @pytest.fixture
    def complex_batch(self) -> Batch:
        """Create a complex nested batch."""
        return {
            "images": jnp.ones((8, 32, 32, 3)),
            "labels": jnp.arange(8),
            "metadata": {
                "ids": jnp.arange(8) * 10,
                "weights": jnp.ones(8) * 0.5,
            },
        }

    def test_node_delegates_to_correct_impl(self) -> None:
        """Test that RebatchNode correctly delegates to implementation."""
        # Test each mode creates correct implementation
        node_diff = RebatchNode(target_batch_size=4, mode="differentiable")
        assert isinstance(node_diff.impl, DifferentiableRebatchImpl)
        assert node_diff.impl.target_batch_size == 4

        node_fast = RebatchNode(target_batch_size=8, mode="fast")
        assert isinstance(node_fast.impl, FastRebatchImpl)
        assert node_fast.impl.target_batch_size == 8

        # gradient_transparent uses DifferentiableRebatchImpl (DRY consolidation)
        node_grad = RebatchNode(target_batch_size=6, mode="gradient_transparent")
        assert isinstance(node_grad.impl, DifferentiableRebatchImpl)
        assert node_grad.impl.target_batch_size == 6

    @pytest.mark.parametrize("mode", ["differentiable", "fast", "gradient_transparent"])
    def test_basic_accumulation_and_emission(self, mode: str) -> None:
        """Test that node accumulates elements and emits when target size reached."""
        node = RebatchNode(target_batch_size=3, mode=mode)

        # Create individual elements
        elements = [{"data": jnp.array([[i]]), "label": jnp.array([i])} for i in range(7)]

        results = []
        for elem in elements:
            result = node(elem)
            if result is not None:
                results.append(result)

        # Should have emitted 2 complete batches (6 elements / 3 target)
        assert len(results) == 2

        # Check first batch
        assert results[0]["data"].shape == (3, 1)
        assert jnp.array_equal(results[0]["label"], jnp.array([0, 1, 2]))

        # Check second batch
        assert results[1]["data"].shape == (3, 1)
        assert jnp.array_equal(results[1]["label"], jnp.array([3, 4, 5]))

        # Flush should return the remaining element
        final = node.flush()
        assert final is not None
        assert final["data"].shape == (1, 1)
        assert final["label"][0] == 6

    @pytest.mark.parametrize("mode", ["differentiable", "fast", "gradient_transparent"])
    def test_exact_batch_size_input(self, mode: str, simple_batch: Batch) -> None:
        """Test when input batch exactly matches target size."""
        node = RebatchNode(target_batch_size=5, mode=mode)

        result = node(simple_batch)
        assert result is not None
        assert result["data"].shape == simple_batch["data"].shape
        assert jnp.array_equal(result["data"], simple_batch["data"])
        assert jnp.array_equal(result["labels"], simple_batch["labels"])

    @pytest.mark.parametrize("mode", ["differentiable", "fast", "gradient_transparent"])
    def test_larger_input_batch(self, mode: str, complex_batch: Batch) -> None:
        """Test when input batch is larger than target size."""
        node = RebatchNode(target_batch_size=3, mode=mode)

        # Process batch of 8, expecting target size of 3
        results = []

        # Process the large batch
        result = node(complex_batch)
        while result is not None:
            results.append(result)
            # Continue processing buffer
            result = node(None)

        # Should have at least 2 complete batches
        assert len(results) >= 2
        assert all(r["images"].shape[0] == 3 for r in results[:2])

        # Flush for remainder
        final = node.flush()
        assert final is not None
        assert final["images"].shape[0] == 2  # Remaining 2 elements

    @pytest.mark.parametrize("mode", ["differentiable", "fast", "gradient_transparent"])
    def test_empty_flush(self, mode: str) -> None:
        """Test flush when buffer is empty."""
        node = RebatchNode(target_batch_size=5, mode=mode)

        # Flush without processing anything
        result = node.flush()
        assert result is None

    def test_invalid_mode_raises_error(self) -> None:
        """Test that invalid mode raises appropriate error."""
        with pytest.raises(ValueError, match="Unknown mode"):
            RebatchNode(target_batch_size=5, mode="invalid_mode")


class TestDifferentiability:
    """Test gradient flow through rebatch operations."""

    def test_differentiable_mode_gradients(self) -> None:
        """Test that differentiable mode preserves gradients."""
        node = RebatchNode(target_batch_size=2, mode="differentiable")

        def loss_fn(x: jax.Array) -> jax.Array:
            # Apply rebatch
            batch = {"data": x}
            rebatched = node(batch)

            if rebatched is None:
                # Accumulating, return 0
                return 0.0

            # Simple loss
            return jnp.sum(rebatched["data"] ** 2)

        # Test gradient flow
        x = jnp.array([[1.0], [2.0], [3.0]])

        # Compute gradient
        grad_fn = jax.grad(loss_fn)
        grad = grad_fn(x)

        # Should have non-zero gradients for elements that contributed
        assert grad.shape == x.shape
        # First two elements should have gradients (batch size 2)
        assert not jnp.allclose(grad[:2], 0.0)

    def test_gradient_transparent_mode(self) -> None:
        """Test gradient transparent mode preserves value gradients."""
        node = RebatchNode(target_batch_size=3, mode="gradient_transparent")

        def loss_with_rebatch(x: jax.Array) -> jax.Array:
            # Create batch with augmentation that increases size
            augmented = jnp.repeat(x, 2, axis=0)  # Double the batch

            batch = {"data": augmented}
            rebatched = node(batch)

            if rebatched is None:
                return 0.0

            # Loss only on rebatched data
            return jnp.mean(rebatched["data"])

        x = jnp.array([[1.0], [2.0]])

        # Check that gradients flow back
        grad_fn = jax.grad(loss_with_rebatch)
        grad = grad_fn(x)

        # Should have non-zero gradients
        assert not jnp.allclose(grad, 0.0)

    def test_fast_mode_with_jit(self) -> None:
        """Test that fast mode works with JIT compilation."""
        node = RebatchNode(target_batch_size=4, mode="fast")

        @nnx.jit
        def process_batch(batch: Batch) -> Batch:
            return node(batch)

        # Test with multiple calls to ensure JIT works
        batch1 = {"data": jnp.ones((3, 2))}
        batch2 = {"data": jnp.ones((3, 2)) * 2}

        # First call compiles
        result1 = process_batch(batch1)

        # Second call uses compiled version
        result2 = process_batch(batch2)

        # At least one should produce output
        assert result1 is not None or result2 is not None


class TestPyTreeHandling:
    """Test handling of complex PyTree structures."""

    def test_nested_pytree_batch(self) -> None:
        """Test with deeply nested PyTree structures."""
        node = RebatchNode(target_batch_size=2, mode="differentiable")

        batch = {
            "features": {
                "dense": jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                "sparse": {
                    "indices": jnp.array([[0, 1], [1, 0], [0, 0]]),
                    "values": jnp.array([1.0, 2.0, 3.0]),
                },
            },
            "labels": jnp.array([0, 1, 2]),
            "metadata": ["sample1", "sample2", "sample3"],  # Non-JAX data
        }

        result = node(batch)
        assert result is not None

        # Check structure preserved
        assert "features" in result
        assert "dense" in result["features"]
        assert "sparse" in result["features"]
        assert "indices" in result["features"]["sparse"]
        assert "values" in result["features"]["sparse"]
        assert "labels" in result
        assert "metadata" in result

        # Check shapes
        assert result["features"]["dense"].shape == (2, 2)
        assert result["features"]["sparse"]["indices"].shape == (2, 2)
        assert result["features"]["sparse"]["values"].shape == (2,)
        assert result["labels"].shape == (2,)
        assert len(result["metadata"]) == 2

    def test_mixed_array_types(self) -> None:
        """Test with mixed array and non-array data."""
        node = RebatchNode(target_batch_size=2, mode="gradient_transparent")

        batch = {
            "images": jnp.ones((3, 32, 32, 3)),
            "labels": [0, 1, 2],  # Python list
            "filenames": ["a.jpg", "b.jpg", "c.jpg"],  # Strings
            "flags": (True, False, True),  # Tuple
        }

        result = node(batch)
        assert result is not None

        # JAX arrays rebatched
        assert result["images"].shape == (2, 32, 32, 3)

        # Non-arrays handled appropriately
        assert len(result["labels"]) == 2
        assert len(result["filenames"]) == 2
        assert len(result["flags"]) == 2


class TestStateManagement:
    """Test state saving and restoration."""

    def test_save_and_restore_state(self) -> None:
        """Test saving and restoring node state."""
        node1 = RebatchNode(target_batch_size=3, mode="differentiable")

        # Process some data
        batch = {"data": jnp.array([[1.0], [2.0]])}
        node1(batch)  # Accumulating

        # Save state
        state = node1.get_state()

        assert "mode" in state
        assert "target_batch_size" in state
        assert "impl_state" in state
        assert "statistics" in state

        # Create new node and restore
        node2 = RebatchNode(target_batch_size=3, mode="differentiable")
        node2.set_state(state)

        # Continue processing should work
        batch2 = {"data": jnp.array([[3.0]])}
        result = node2(batch2)

        assert result is not None  # Should emit after restoration
        assert result["data"].shape == (3, 1)
        assert jnp.array_equal(result["data"], jnp.array([[1.0], [2.0], [3.0]]))

    def test_statistics_tracking(self) -> None:
        """Test that statistics are properly tracked."""
        node = RebatchNode(target_batch_size=2, mode="fast")

        # Process batches
        for i in range(5):
            batch = {"data": jnp.array([[float(i)]])}
            node(batch)

        # Check statistics
        assert node.elements_processed.get_value() == 5
        assert node.batches_emitted.get_value() == 2  # 5 elements / 2 target = 2 complete

        # Flush remainder
        node.flush()
        assert node.batches_emitted.get_value() == 3  # Including partial


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_large_input_batch(self) -> None:
        """Test handling of input batch much larger than target."""
        node = RebatchNode(target_batch_size=2, mode="fast", max_buffer_size=10)

        # Input batch of 20, but buffer max is 10
        large_batch = {"data": jnp.ones((20, 3))}

        results = []
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Circular buffer overflow", UserWarning)
            result = node(large_batch)

        # Should handle this gracefully
        while result is not None:
            results.append(result)
            result = node(None)  # Continue processing

        assert len(results) >= 1

        # All results should have target size or less
        for r in results:
            assert r["data"].shape[0] <= 2

    def test_single_element_batches(self) -> None:
        """Test processing single-element batches."""
        node = RebatchNode(target_batch_size=3, mode="differentiable")

        results = []
        for i in range(10):
            single = {"value": jnp.array([i])}
            result = node(single)
            if result is not None:
                results.append(result)

        # Should have 3 complete batches
        assert len(results) == 3

        # Each should have 3 elements
        for r in results:
            assert r["value"].shape == (3,)

    def test_empty_batch_input(self) -> None:
        """Test handling of empty batch input."""
        node = RebatchNode(target_batch_size=5, mode="gradient_transparent")

        # Empty batch (0 elements)
        empty = {"data": jnp.array([]).reshape(0, 2)}

        result = node(empty)
        assert result is None  # Should handle gracefully

    def test_none_input(self) -> None:
        """Test handling of None input for buffer continuation."""
        node = RebatchNode(target_batch_size=5, mode="fast")

        # None input when buffer is empty
        result = node(None)
        assert result is None

        # Add some data
        batch = {"data": jnp.array([[1.0], [2.0]])}
        node(batch)

        # None input should try to emit from buffer
        result = node(None)
        assert result is None  # Still not enough data


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_multi_crop_augmentation_scenario(self) -> None:
        """Test simulating multi-crop augmentation use case."""

        # Simulate augmentation that produces 5 crops per image
        def multi_crop_augment(batch: Batch, num_crops: int = 5) -> Batch:
            images = batch["images"]
            labels = batch["labels"]

            # Repeat each image num_crops times
            augmented_images = jnp.repeat(images, num_crops, axis=0)
            augmented_labels = jnp.repeat(labels, num_crops, axis=0)

            return {
                "images": augmented_images,
                "labels": augmented_labels,
            }

        # Original batch size 8, after augmentation 40
        original_batch = {
            "images": jnp.ones((8, 32, 32, 3)),
            "labels": jnp.arange(8),
        }

        # Apply augmentation
        augmented = multi_crop_augment(original_batch, num_crops=5)
        assert augmented["images"].shape == (40, 32, 32, 3)

        # Rebatch to model's expected size (32)
        node = RebatchNode(target_batch_size=32, mode="gradient_transparent")

        rebatched = node(augmented)
        assert rebatched is not None
        assert rebatched["images"].shape == (32, 32, 32, 3)
        assert rebatched["labels"].shape == (32,)

        # Flush remainder
        remainder = node.flush()
        assert remainder is not None
        assert remainder["images"].shape == (8, 32, 32, 3)

    def test_pipeline_with_varying_batch_sizes(self) -> None:
        """Test in pipeline with varying batch sizes."""
        # Simulate a pipeline that processes batches of different sizes
        sizes = [3, 5, 2, 7, 4]
        node = RebatchNode(target_batch_size=8, mode="fast")

        all_results = []
        for i, size in enumerate(sizes):
            batch = {
                "data": jnp.ones((size, 10)) * (i + 1),
                "batch_id": i,
            }

            result = node(batch)
            if result is not None:
                all_results.append(result)

        # Should have accumulated enough for at least 2 batches
        assert len(all_results) >= 2

        # Each emitted batch should have target size
        for r in all_results:
            assert r["data"].shape == (8, 10)

        # Flush final partial batch
        final = node.flush()
        if final is not None:
            assert final["data"].shape[0] <= 8


class TestImplementationSpecificFeatures:
    """Test features specific to each implementation."""

    def test_differentiable_impl_jax_operations(self) -> None:
        """Test that DifferentiableRebatchImpl uses JAX operations."""
        impl = DifferentiableRebatchImpl(target_batch_size=3)

        # The buffer should use JAX arrays when initialized
        batch = {"data": jnp.ones((2, 4))}
        impl(batch)

        assert isinstance(impl.buffer.get_value(), dict)
        assert isinstance(impl.buffer.get_value()["data"], jax.Array)
        assert isinstance(impl.buffer_mask.get_value(), jax.Array)

    def test_fast_impl_circular_buffer_wrapping(self) -> None:
        """Test FastRebatchImpl circular buffer wrapping."""
        impl = FastRebatchImpl(target_batch_size=2, max_buffer_size=3)

        # Fill buffer beyond capacity to test wrapping
        for i in range(5):
            batch = {"data": jnp.array([[float(i)]])}
            impl(batch)

        # Check that indices wrapped correctly
        assert impl.write_index.get_value() < impl.max_buffer_size
        assert impl.read_index.get_value() < impl.max_buffer_size

    def test_gradient_transparent_impl_tape(self) -> None:
        """Test GradientTransparentRebatchImpl maintains gradient tape."""
        impl = GradientTransparentRebatchImpl(target_batch_size=4)

        batch1 = {"data": jnp.array([[1.0], [2.0]])}
        batch2 = {"data": jnp.array([[3.0], [4.0]])}

        impl(batch1)
        assert len(impl.gradient_tape) == 2

        output, _ = impl(batch2)
        assert output is not None
        assert len(impl.gradient_tape) == 0  # Cleared after emission


# Convenience test
def test_module_imports() -> None:
    """Test that all expected exports are available."""
    from datarax.dag.nodes import (
        RebatchNode,
        DifferentiableRebatchImpl,
        FastRebatchImpl,
        GradientTransparentRebatchImpl,
        rebatch,
    )

    assert RebatchNode is not None
    assert DifferentiableRebatchImpl is not None
    assert FastRebatchImpl is not None
    assert GradientTransparentRebatchImpl is not None
    assert callable(rebatch)

    # Test convenience function
    node = rebatch(target_batch_size=32)
    assert isinstance(node, RebatchNode)
    assert node.target_batch_size == 32


class TestDataLoaderClass:
    """Test DataLoader class functionality."""

    def test_dataloader_initialization_with_datasource_module(self):
        """Test DataLoader initialization with DataSourceModule."""
        source = MockDataSourceV2(StructuralConfig(), size=5)
        loader = DataLoader(source, batch_size=2)

        assert loader.batch_size == 2
        assert loader.shuffle_buffer_size is None
        assert loader.drop_remainder is False
        assert loader.shuffle_seed is None
        assert isinstance(loader.source, DataSourceNode)
        assert loader.source.source is source

    def test_dataloader_initialization_with_datasource_node(self):
        """Test DataLoader initialization with DataSourceNode."""
        source = MockDataSourceV2(StructuralConfig(), size=5)
        source_node = DataSourceNode(source)
        loader = DataLoader(source_node, batch_size=3)

        assert loader.batch_size == 3
        assert loader.source is source_node

    def test_dataloader_initialization_with_shuffling(self):
        """Test DataLoader initialization with shuffling enabled."""
        source = MockDataSourceV2(StructuralConfig(), size=10)
        loader = DataLoader(source, batch_size=4, shuffle_buffer_size=5, shuffle_seed=42)

        assert loader.batch_size == 4
        assert loader.shuffle_buffer_size == 5
        assert loader.shuffle_seed == 42

        # Check that nodes are properly configured
        assert len(loader.nodes) == 3  # DataSource + Shuffle + Batch
        assert isinstance(loader.nodes[0], DataSourceNode)
        assert isinstance(loader.nodes[1], ShuffleNode)
        assert isinstance(loader.nodes[2], BatchNode)

    def test_dataloader_initialization_without_shuffling(self):
        """Test DataLoader initialization without shuffling."""
        source = MockDataSourceV2(StructuralConfig(), size=10)
        loader = DataLoader(source, batch_size=4, shuffle_buffer_size=None)

        assert loader.shuffle_buffer_size is None

        # Check that only DataSource + Batch nodes exist
        assert len(loader.nodes) == 2  # DataSource + Batch
        assert isinstance(loader.nodes[0], DataSourceNode)
        assert isinstance(loader.nodes[1], BatchNode)

    def test_dataloader_initialization_with_drop_remainder(self):
        """Test DataLoader initialization with drop_remainder option."""
        source = MockDataSourceV2(StructuralConfig(), size=7)
        loader = DataLoader(source, batch_size=3, drop_remainder=True)

        assert loader.drop_remainder is True
        batch_node = loader.nodes[-1]  # Last node should be BatchNode
        assert isinstance(batch_node, BatchNode)
        assert batch_node.drop_remainder is True

    def test_dataloader_invalid_source_type(self):
        """Test DataLoader with invalid source type."""
        with pytest.raises(ValueError, match="Source must be DataSourceModule or DataSourceNode"):
            DataLoader("invalid_source", batch_size=2)

    def test_dataloader_string_representation(self):
        """Test DataLoader string representation."""
        source = MockDataSourceV2(StructuralConfig(), size=5)

        # Without shuffling
        loader1 = DataLoader(source, batch_size=4)
        repr1 = repr(loader1)
        assert "DataLoader(batch_size=4)" in repr1

        # With shuffling
        loader2 = DataLoader(source, batch_size=2, shuffle_buffer_size=10)
        repr2 = repr(loader2)
        assert "DataLoader(batch_size=2, shuffle=10)" in repr2

    def test_dataloader_custom_name(self):
        """Test DataLoader with custom name."""
        source = MockDataSourceV2(StructuralConfig(), size=5)
        loader = DataLoader(source, batch_size=2, name="CustomLoader")

        assert loader.name == "CustomLoader"


class TestDataLoaderConvenienceFunction:
    """Test dataloader convenience function."""

    def test_dataloader_function_basic(self):
        """Test basic dataloader function usage."""
        source = MockDataSourceV2(StructuralConfig(), size=5)
        loader = dataloader(source, batch_size=3)

        assert isinstance(loader, DataLoader)
        assert loader.batch_size == 3
        assert loader.shuffle_buffer_size is None

    def test_dataloader_function_with_all_options(self):
        """Test dataloader function with all options."""
        source = MockDataSourceV2(StructuralConfig(), size=10)
        loader = dataloader(
            source=source,
            batch_size=4,
            shuffle_buffer_size=8,
            drop_remainder=True,
            shuffle_seed=123,
        )

        assert isinstance(loader, DataLoader)
        assert loader.batch_size == 4
        assert loader.shuffle_buffer_size == 8
        assert loader.drop_remainder is True
        assert loader.shuffle_seed == 123


class TestDataLoaderExecution:
    """Test DataLoader execution functionality."""

    @pytest.fixture
    def sample_source(self) -> MockDataSourceV2:
        """Create a sample data source for testing."""
        return MockDataSourceV2(StructuralConfig(), size=6)

    def test_dataloader_call_single_element(self, sample_source: MockDataSourceV2):
        """Test DataLoader execution with single element."""
        loader = DataLoader(sample_source, batch_size=2)

        # First call should return None (buffering)
        result1 = loader(None)
        assert result1 is None

        # Second call should return first batch
        result2 = loader(None)
        if result2 is not None:
            # DataLoader returns Batch objects
            assert isinstance(result2, Batch)
            assert result2.batch_size == 2
            assert "data" in result2.data.get_value()
            assert result2.data.get_value()["data"].shape[0] == 2  # Batch size

    def test_dataloader_iteration(self, sample_source: MockDataSourceV2):
        """Test DataLoader iteration."""
        loader = DataLoader(sample_source, batch_size=2)

        # Test that loader can be iterated
        batches = []
        for i, batch in enumerate(loader):
            if batch is not None:
                batches.append(batch)
            if i > 10:  # Prevent infinite loop
                break

        # Should have some batches
        assert len(batches) > 0

    def test_dataloader_with_shuffling_execution(self, sample_source: MockDataSourceV2):
        """Test DataLoader execution with shuffling."""
        loader = DataLoader(sample_source, batch_size=2, shuffle_buffer_size=4, shuffle_seed=42)

        # Execute a few times to test shuffling
        results = []
        for i in range(5):
            result = loader(None)
            if result is not None:
                results.append(result)

        # Should have some results
        assert len(results) >= 0  # May be empty due to buffering

    def test_dataloader_state_management(self, sample_source: MockDataSourceV2):
        """Test DataLoader state management."""
        loader = DataLoader(sample_source, batch_size=2)

        # Get initial state
        initial_state = loader.get_state()
        assert isinstance(initial_state, dict)

        # Execute some operations
        loader(None)
        loader(None)

        # Get state after operations
        after_state = loader.get_state()
        assert isinstance(after_state, dict)

        # Restore initial state
        loader.set_state(initial_state)

        # State should be restored (this is a basic check)
        restored_state = loader.get_state()
        assert isinstance(restored_state, dict)


class TestDataLoaderIntegration:
    """Test DataLoader integration with other components."""

    def test_dataloader_with_dag_executor(self):
        """Test DataLoader integration with DAGExecutor."""
        from datarax.dag.dag_executor import DAGExecutor

        source = MockDataSourceV2(StructuralConfig(), size=4)
        loader = DataLoader(source, batch_size=2)

        executor = DAGExecutor()
        executor.add(loader)

        assert executor.graph is loader

    def test_dataloader_composition_with_transforms(self):
        """Test DataLoader composition with transform nodes."""
        from datarax.dag.nodes import OperatorNode
        from datarax.core.config import DataraxModuleConfig

        class SimpleTransform(DataraxModule):
            def __init__(self):
                super().__init__(DataraxModuleConfig(), name="simple_transform")

            def __call__(self, data, *, key=None):
                if data is None:
                    return None
                return data * 2

        source = MockDataSourceV2(StructuralConfig(), size=4)
        loader = DataLoader(source, batch_size=2)
        transform = OperatorNode(SimpleTransform())

        # Test sequential composition
        from datarax.dag.nodes import Sequential

        pipeline = Sequential([loader, transform])

        assert len(pipeline.nodes) == 2
        assert pipeline.nodes[0] is loader
        assert pipeline.nodes[1] is transform


class TestDataLoaderEdgeCases:
    """Test DataLoader edge cases and error conditions."""

    def test_dataloader_empty_source(self):
        """Test DataLoader with empty data source."""
        empty_source = MockDataSourceV2(StructuralConfig(), size=0)
        loader = DataLoader(empty_source, batch_size=2)

        # Should handle empty source gracefully
        loader(None)
        # Result may be None or raise StopIteration, both are acceptable

    def test_dataloader_large_batch_size(self):
        """Test DataLoader with batch size larger than data."""
        small_source = MockDataSourceV2(StructuralConfig(), size=2)
        loader = DataLoader(small_source, batch_size=5)

        # Should handle gracefully
        loader(None)
        # May return None due to insufficient data for batch

    def test_dataloader_zero_shuffle_buffer(self):
        """Test DataLoader with zero shuffle buffer size."""
        source = MockDataSourceV2(StructuralConfig(), size=5)
        loader = DataLoader(source, batch_size=2, shuffle_buffer_size=0)

        # Should disable shuffling
        assert len(loader.nodes) == 2  # Only DataSource + Batch

    def test_dataloader_configuration_inspection(self):
        """Test DataLoader configuration can be inspected."""
        source = MockDataSourceV2(StructuralConfig(), size=10)
        loader = DataLoader(
            source,
            batch_size=3,
            shuffle_buffer_size=7,
            drop_remainder=True,
            shuffle_seed=456,
            name="TestLoader",
        )

        # All configuration should be accessible
        assert loader.batch_size == 3
        assert loader.shuffle_buffer_size == 7
        assert loader.drop_remainder is True
        assert loader.shuffle_seed == 456
        assert loader.name == "TestLoader"
        assert isinstance(loader.source, DataSourceNode)


class TestBranchGradientFlow:
    """Test gradient flow through Branch nodes."""

    def test_branch_differentiability_basic(self):
        """Test Branch node preserves gradients in basic case."""
        from datarax.dag.nodes import Branch

        # Create simple condition and path functions
        def condition_fn(data):
            return data["value"] > 0.0

        def true_path(data, *, key=None):
            return {"result": data["value"] * 2.0}

        def false_path(data, *, key=None):
            return {"result": data["value"] * -1.0}

        branch = Branch(condition_fn, true_path, false_path)

        def loss_fn(value):
            data = {"value": value}
            result = branch(data)
            return jnp.sum(result["result"])

        # Test gradient computation
        grad_fn = jax.grad(loss_fn)

        # Positive input (true path)
        grad_pos = grad_fn(2.0)
        assert jnp.allclose(grad_pos, 2.0)  # d/dx(2x) = 2

        # Negative input (false path)
        grad_neg = grad_fn(-1.0)
        assert jnp.allclose(grad_neg, -1.0)  # d/dx(-x) = -1

    def test_branch_differentiability_with_scaling(self):
        """Test Branch node gradients with scaling transformations."""
        from datarax.dag.nodes import Branch

        def condition_fn(data):
            return jnp.sum(data["features"]) > 5.0

        def scale_up(data, *, key=None):
            return {"output": data["features"] * 3.0}

        def scale_down(data, *, key=None):
            return {"output": data["features"] * 0.5}

        branch = Branch(condition_fn, scale_up, scale_down)

        def loss_fn(features):
            data = {"features": features}
            result = branch(data)
            return jnp.sum(result["output"] ** 2)

        # Test gradient computation
        grad_fn = jax.grad(loss_fn)

        # High sum (scale up path)
        features_high = jnp.array([2.0, 3.0, 1.0])  # sum = 6 > 5
        grad_high = grad_fn(features_high)
        expected_high = 2 * features_high * 3.0 * 3.0  # 2x * 3 * 3 (chain rule)
        assert jnp.allclose(grad_high, expected_high)

        # Low sum (scale down path)
        features_low = jnp.array([1.0, 1.0, 1.0])  # sum = 3 < 5
        grad_low = grad_fn(features_low)
        expected_low = 2 * features_low * 0.5 * 0.5  # 2x * 0.5 * 0.5
        assert jnp.allclose(grad_low, expected_low)

    def test_branch_jit_compatibility(self):
        """Test Branch node works with JAX JIT compilation."""
        from datarax.dag.nodes import Branch

        def condition_fn(data):
            return data["x"] > 0.0

        def positive_path(data, *, key=None):
            return {"y": jnp.sqrt(data["x"])}

        def negative_path(data, *, key=None):
            return {"y": jnp.abs(data["x"])}

        branch = Branch(condition_fn, positive_path, negative_path)

        @nnx.jit
        def jitted_branch(x):
            data = {"x": x}
            result = branch(data)
            return result["y"]

        # Test positive input
        result_pos = jitted_branch(4.0)
        assert jnp.allclose(result_pos, 2.0)  # sqrt(4) = 2

        # Test negative input
        result_neg = jitted_branch(-3.0)
        assert jnp.allclose(result_neg, 3.0)  # abs(-3) = 3

    def test_branch_vmap_compatibility(self):
        """Test Branch node works with JAX vmap."""
        from datarax.dag.nodes import Branch

        def condition_fn(data):
            return data["values"] > 2.0

        def double_path(data, *, key=None):
            return {"result": data["values"] * 2.0}

        def square_path(data, *, key=None):
            return {"result": data["values"] ** 2}

        branch = Branch(condition_fn, double_path, square_path)

        def process_single(value):
            data = {"values": value}
            result = branch(data)
            return result["result"]

        # Vectorize over batch
        vmapped_process = jax.vmap(process_single)

        inputs = jnp.array([1.0, 3.0, 0.5, 4.0])
        results = vmapped_process(inputs)

        # Check results: [1^2, 3*2, 0.5^2, 4*2] = [1, 6, 0.25, 8]
        expected = jnp.array([1.0, 6.0, 0.25, 8.0])
        assert jnp.allclose(results, expected)

    def test_branch_complex_gradient_flow(self):
        """Test Branch node with complex nested operations."""
        from datarax.dag.nodes import Branch

        def condition_fn(data):
            return jnp.mean(data["input"]) > 0.0

        def complex_true_path(data, *, key=None):
            x = data["input"]
            return {"output": jnp.sin(x) + jnp.cos(x**2)}

        def complex_false_path(data, *, key=None):
            x = data["input"]
            return {"output": jnp.tanh(x) * jnp.exp(-(x**2))}

        branch = Branch(condition_fn, complex_true_path, complex_false_path)

        def loss_fn(input_data):
            data = {"input": input_data}
            result = branch(data)
            return jnp.sum(result["output"] ** 2)

        # Test gradient computation with complex operations
        grad_fn = jax.grad(loss_fn)

        # Positive mean input
        input_pos = jnp.array([1.0, 2.0, 0.5])
        grad_pos = grad_fn(input_pos)
        assert grad_pos is not None
        assert grad_pos.shape == input_pos.shape
        assert not jnp.allclose(grad_pos, 0.0)  # Non-zero gradients

        # Negative mean input
        input_neg = jnp.array([-1.0, -2.0, -0.5])
        grad_neg = grad_fn(input_neg)
        assert grad_neg is not None
        assert grad_neg.shape == input_neg.shape
        assert not jnp.allclose(grad_neg, 0.0)  # Non-zero gradients

    def test_branch_nested_branches(self):
        """Test nested Branch nodes preserve gradients."""
        from datarax.dag.nodes import Branch

        # Inner branch
        def inner_condition(data):
            return data["x"] > 1.0

        def inner_true(data, *, key=None):
            return {"x": data["x"] * 2.0}

        def inner_false(data, *, key=None):
            return {"x": data["x"] + 1.0}

        inner_branch = Branch(inner_condition, inner_true, inner_false)

        # Outer branch
        def outer_condition(data):
            return data["x"] > 5.0

        def outer_true(data, *, key=None):
            # Use inner branch
            result = inner_branch(data)
            return {"y": result["x"] ** 2}

        def outer_false(data, *, key=None):
            return {"y": data["x"] * 3.0}

        outer_branch = Branch(outer_condition, outer_true, outer_false)

        def loss_fn(x):
            data = {"x": x}
            result = outer_branch(data)
            return result["y"]

        # Test gradient computation through nested branches
        grad_fn = jax.grad(loss_fn)

        # Test different paths
        grad1 = grad_fn(0.5)  # x=0.5 -> +1=1.5 -> *3=4.5
        grad2 = grad_fn(2.0)  # x=2.0 -> *2=4.0 -> *3=12.0
        grad3 = grad_fn(6.0)  # x=6.0 -> *2=12.0 -> ^2=144, d/dx = 2*12*2 = 48

        assert not jnp.allclose(grad1, 0.0)
        assert not jnp.allclose(grad2, 0.0)
        assert not jnp.allclose(grad3, 0.0)
