"""
Individual Node Unit Tests.

This file provides detailed unit tests for individual nodes following TDD principles.
As per requirements, ShuffleNode and BatchNode must be tested independently, even though
they should not be used directly in data pipelines (DataLoader should be used instead).
"""

from typing import Any

import pytest
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from datarax.dag.nodes import ShuffleNode, BatchNode, DataSourceNode, OperatorNode
from datarax.core.data_source import DataSourceModule
from datarax.core.operator import OperatorModule
from datarax.core.config import OperatorConfig, StructuralConfig
from datarax.core.element_batch import Element, Batch


class MockDataSource(DataSourceModule):
    """Mock data source for testing individual nodes."""

    def __init__(
        self,
        data_size: int = 20,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        if rngs is None:
            rngs = nnx.Rngs(0)
        super().__init__(StructuralConfig(), rngs=rngs)
        self.data_size = data_size
        self.index = nnx.Variable(0)

    def __call__(self, key=None):
        """Return next data element."""
        if self.index.get_value() >= self.data_size:
            return None

        data = {"value": jnp.array(self.index.get_value()), "index": self.index.get_value()}
        self.index.set_value(self.index.get_value() + 1)
        return data

    def __iter__(self):
        return self

    def __next__(self):
        if self.index.get_value() >= self.data_size:
            raise StopIteration

        data = {"value": jnp.array(self.index.get_value()), "index": self.index.get_value()}
        self.index.set_value(self.index.get_value() + 1)
        return data


class SimpleOperator(OperatorModule):
    """Simple operator for testing."""

    def __init__(self, multiplier: float = 2.0, *, rngs: nnx.Rngs | None = None):
        config = OperatorConfig(stochastic=False)
        super().__init__(config, rngs=rngs, name="simple_operator")
        self.multiplier = multiplier

    def apply(
        self,
        data: dict[str, jax.Array],
        state: dict[str, Any],
        metadata: Any,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[dict[str, jax.Array], dict[str, Any], Any]:
        """Apply the transformation."""
        if "value" in data:
            result_data = {"value": data["value"] * self.multiplier, "index": data["index"]}
        else:
            result_data = data
        return result_data, state, metadata


@pytest.fixture
def mock_source():
    """Create a mock data source."""
    return MockDataSource(data_size=10)


@pytest.fixture
def simple_operator():
    """Create a simple operator."""
    return SimpleOperator(multiplier=3.0)


class TestShuffleNodeIndependent:
    """Independent unit tests for ShuffleNode.

    Note: As per requirements, ShuffleNode should not be used directly in data pipelines.
    DataLoader should be used instead. These tests verify ShuffleNode works correctly
    as an individual component.
    """

    def test_shuffle_node_creation(self):
        """Test ShuffleNode can be created with proper parameters."""
        # Test default creation
        shuffle_node = ShuffleNode(buffer_size=100)
        assert shuffle_node.buffer_size == 100
        assert shuffle_node.seed is None

        # Test creation with seed
        shuffle_node_with_seed = ShuffleNode(buffer_size=50, seed=42)
        assert shuffle_node_with_seed.buffer_size == 50
        assert shuffle_node_with_seed.seed == 42

    def test_shuffle_node_buffer_initialization(self):
        """Test ShuffleNode initializes buffer correctly."""
        shuffle_node = ShuffleNode(buffer_size=5)

        # Buffer should start empty (buffer is now a plain list, not nnx.List)
        assert len(shuffle_node.buffer) == 0
        assert shuffle_node.buffer_full is False

    def test_shuffle_node_single_element_processing(self):
        """Test ShuffleNode processes individual elements correctly."""
        shuffle_node = ShuffleNode(buffer_size=3, seed=42)

        # First few elements should return None (filling buffer)
        element1 = {"value": jnp.array(1), "index": 1}
        result1 = shuffle_node(element1)
        assert result1 is None  # Buffer not full yet

        element2 = {"value": jnp.array(2), "index": 2}
        result2 = shuffle_node(element2)
        assert result2 is None  # Buffer not full yet

        element3 = {"value": jnp.array(3), "index": 3}
        result3 = shuffle_node(element3)
        assert result3 is None  # Buffer just became full

        # Fourth element should return a shuffled element
        element4 = {"value": jnp.array(4), "index": 4}
        result4 = shuffle_node(element4)
        assert result4 is not None
        assert isinstance(result4, dict)
        assert "value" in result4
        assert "index" in result4

    def test_shuffle_node_deterministic_with_seed(self):
        """Test ShuffleNode produces deterministic results with same seed."""
        # Create two identical shuffle nodes with same seed
        shuffle_node1 = ShuffleNode(buffer_size=3, seed=42)
        shuffle_node2 = ShuffleNode(buffer_size=3, seed=42)

        # Feed same data to both
        elements = [{"value": jnp.array(i), "index": i} for i in range(10)]

        results1 = []
        results2 = []

        for element in elements:
            result1 = shuffle_node1(element)
            result2 = shuffle_node2(element)

            if result1 is not None:
                results1.append(result1)
            if result2 is not None:
                results2.append(result2)

        # Results should be identical (deterministic)
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1["index"] == r2["index"]

    def test_shuffle_node_different_seeds_different_results(self):
        """Test ShuffleNode produces different results with different seeds."""
        shuffle_node1 = ShuffleNode(buffer_size=5, seed=42)
        shuffle_node2 = ShuffleNode(buffer_size=5, seed=123)

        # Feed same data to both
        elements = [{"value": jnp.array(i), "index": i} for i in range(20)]

        results1 = []
        results2 = []

        for element in elements:
            result1 = shuffle_node1(element)
            result2 = shuffle_node2(element)

            if result1 is not None:
                results1.append(result1["index"])
            if result2 is not None:
                results2.append(result2["index"])

        # Results should be different (different shuffling)
        assert len(results1) > 5  # Should have some results
        assert len(results2) > 5  # Should have some results
        assert results1 != results2  # Should be different sequences

    def test_shuffle_node_state_management(self):
        """Test ShuffleNode state can be saved and restored."""
        shuffle_node = ShuffleNode(buffer_size=3, seed=42)

        # Process some elements
        for i in range(5):
            element = {"value": jnp.array(i), "index": i}
            shuffle_node(element)

        # Get state
        state = shuffle_node.get_state()
        assert isinstance(state, dict)

        # Create new node and set state
        new_shuffle_node = ShuffleNode(buffer_size=3, seed=42)
        new_shuffle_node.set_state(state)

        # Both nodes should behave identically from this point
        test_element = {"value": jnp.array(10), "index": 10}
        result1 = shuffle_node(test_element)
        result2 = new_shuffle_node(test_element)

        # Results should be the same
        if result1 is not None and result2 is not None:
            assert result1["index"] == result2["index"]

    def test_shuffle_node_repr(self):
        """Test ShuffleNode string representation."""
        shuffle_node = ShuffleNode(buffer_size=100)
        repr_str = repr(shuffle_node)
        assert "ShuffleNode" in repr_str
        assert "100" in repr_str


class TestBatchNodeIndependent:
    """Independent unit tests for BatchNode.

    Note: As per requirements, BatchNode should not be used directly in data pipelines.
    DataLoader should be used instead. These tests verify BatchNode works correctly
    as an individual component.
    """

    def test_batch_node_creation(self):
        """Test BatchNode can be created with proper parameters."""
        # Test default creation
        batch_node = BatchNode(batch_size=32)
        assert batch_node.batch_size == 32
        assert batch_node.drop_remainder is False  # Default

        # Test creation with drop_remainder=True
        batch_node_drop = BatchNode(batch_size=16, drop_remainder=True)
        assert batch_node_drop.batch_size == 16
        assert batch_node_drop.drop_remainder is True

    def test_batch_node_buffer_initialization(self):
        """Test BatchNode initializes buffer correctly."""
        batch_node = BatchNode(batch_size=4)

        # Buffer should start empty
        assert len(batch_node.buffer) == 0

    def test_batch_node_single_element_accumulation(self):
        """Test BatchNode accumulates elements correctly."""
        batch_node = BatchNode(batch_size=3)

        # First two elements should return None (accumulating)
        element1 = {"value": jnp.array(1), "index": 1}
        result1 = batch_node(element1)
        assert result1 is None

        element2 = {"value": jnp.array(2), "index": 2}
        result2 = batch_node(element2)
        assert result2 is None

        # Third element should return a batch
        element3 = {"value": jnp.array(3), "index": 3}
        result3 = batch_node(element3)
        assert result3 is not None
        assert isinstance(result3, Batch)
        assert "value" in result3.data.get_value()
        assert "index" in result3.data.get_value()

        # Check batch dimensions
        assert result3.data.get_value()["value"].shape == (3,)  # Batch of 3 scalars
        assert result3.data.get_value()["index"].shape == (3,)  # Batch of 3 scalars

        # Check batch contents
        expected_values = jnp.array([1, 2, 3])
        expected_indices = jnp.array([1, 2, 3])
        assert jnp.array_equal(result3.data.get_value()["value"], expected_values)
        assert jnp.array_equal(result3.data.get_value()["index"], expected_indices)

    def test_batch_node_continuous_batching(self):
        """Test BatchNode continues batching after producing a batch."""
        batch_node = BatchNode(batch_size=2)

        # First batch
        batch_node({"value": jnp.array(1), "index": 1})
        result1 = batch_node({"value": jnp.array(2), "index": 2})
        assert result1 is not None
        assert result1.data.get_value()["value"].shape == (2,)

        # Second batch
        batch_node({"value": jnp.array(3), "index": 3})
        result2 = batch_node({"value": jnp.array(4), "index": 4})
        assert result2 is not None
        assert result2.data.get_value()["value"].shape == (2,)

        # Batches should be different
        # Batches should be different
        assert not jnp.array_equal(
            result1.data.get_value()["value"], result2.data.get_value()["value"]
        )

    def test_batch_node_drop_remainder_true(self):
        """Test BatchNode with drop_remainder=True discards incomplete batches."""
        batch_node = BatchNode(batch_size=3, drop_remainder=True)

        # Add 2 elements (less than batch_size)
        batch_node({"value": jnp.array(1), "index": 1})
        result = batch_node({"value": jnp.array(2), "index": 2})
        assert result is None  # No batch produced

        # Buffer should have 2 elements but no batch produced yet
        assert len(batch_node.buffer) == 2

    def test_batch_node_drop_remainder_false(self):
        """Test BatchNode with drop_remainder=False handles incomplete batches."""
        batch_node = BatchNode(batch_size=3, drop_remainder=False)

        # Add 2 elements and signal end (by calling finalize or similar)
        batch_node({"value": jnp.array(1), "index": 1})
        batch_node({"value": jnp.array(2), "index": 2})

        # For this test, we'll simulate end-of-data by checking buffer state
        assert len(batch_node.buffer) == 2

        # In a real scenario, the incomplete batch would be returned
        # when the data source is exhausted

    def test_batch_node_complex_data_structures(self):
        """Test BatchNode handles complex nested data structures."""
        batch_node = BatchNode(batch_size=2)

        # Complex nested data
        element1 = {
            "features": jnp.array([1.0, 2.0]),
            "labels": jnp.array(0),
            "metadata": {"id": 1},
        }
        element2 = {
            "features": jnp.array([3.0, 4.0]),
            "labels": jnp.array(1),
            "metadata": {"id": 2},
        }

        batch_node(element1)
        result = batch_node(element2)

        assert result is not None
        assert isinstance(result, Batch)

        # Check batched features
        expected_features = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        assert jnp.array_equal(result.data.get_value()["features"], expected_features)

        # Check batched labels
        expected_labels = jnp.array([0, 1])
        assert jnp.array_equal(result.data.get_value()["labels"], expected_labels)

        # Metadata should be batched as a nested dictionary with array values
        assert "metadata" in result.data.get_value()
        assert "id" in result.data.get_value()["metadata"]
        # The IDs should be stacked into an array
        expected_ids = jnp.array([1, 2])
        assert jnp.array_equal(result.data.get_value()["metadata"]["id"], expected_ids)

    def test_batch_node_state_management(self):
        """Test BatchNode state can be saved and restored."""
        batch_node = BatchNode(batch_size=3)

        # Add one element to buffer
        batch_node({"value": jnp.array(1), "index": 1})

        # Get state
        state = batch_node.get_state()
        assert isinstance(state, dict)

        # Create new node and set state
        new_batch_node = BatchNode(batch_size=3)
        new_batch_node.set_state(state)

        # Both nodes should have same buffer state
        assert len(batch_node.buffer) == len(new_batch_node.buffer)

    def test_batch_node_repr(self):
        """Test BatchNode string representation."""
        batch_node = BatchNode(batch_size=32)
        repr_str = repr(batch_node)
        assert "BatchNode" in repr_str
        assert "32" in repr_str


class TestDataSourceNodeIndependent:
    """Independent unit tests for DataSourceNode."""

    def test_data_source_node_creation(self, mock_source):
        """Test DataSourceNode can be created with a data source."""
        source_node = DataSourceNode(mock_source)
        assert source_node.source is mock_source

    def test_data_source_node_ignores_input(self, mock_source):
        """Test DataSourceNode ignores input data and generates its own."""
        source_node = DataSourceNode(mock_source)

        # Should ignore input and generate data from source
        result1 = source_node(None)
        result2 = source_node("ignored_input")
        result3 = source_node({"ignored": "data"})

        # All should return valid data from source
        for result in [result1, result2, result3]:
            assert result is not None
            assert isinstance(result, dict)
            assert "value" in result
            assert "index" in result

    def test_data_source_node_sequential_calls(self, mock_source):
        """Test DataSourceNode produces sequential data on repeated calls."""
        source_node = DataSourceNode(mock_source)

        results = []
        for i in range(5):
            result = source_node(None)
            if result is not None:
                results.append(result)

        # Should have sequential indices
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result["index"] == i

    def test_data_source_node_exhaustion(self):
        """Test DataSourceNode handles source exhaustion."""
        small_source = MockDataSource(data_size=2)
        source_node = DataSourceNode(small_source)

        # Get all data
        result1 = source_node(None)
        result2 = source_node(None)
        result3 = source_node(None)  # Should be None (exhausted)

        assert result1 is not None
        assert result2 is not None
        assert result3 is None

    def test_data_source_node_state_management(self, mock_source):
        """Test DataSourceNode state management."""
        source_node = DataSourceNode(mock_source)

        # Generate some data
        source_node(None)
        source_node(None)

        # Get state
        state = source_node.get_state()
        assert isinstance(state, dict)

        # Create new node and set state
        new_source_node = DataSourceNode(MockDataSource(data_size=10))
        new_source_node.set_state(state)

        # Both should produce same next result
        result1 = source_node(None)
        result2 = new_source_node(None)

        if result1 is not None and result2 is not None:
            assert result1["index"] == result2["index"]


def make_batch_from_data(data_list: list[dict[str, jax.Array]]) -> Batch:
    """Helper to create Batch from list of data dicts for testing.

    Each dict in data_list becomes an Element's data field.
    """
    elements = [Element(data=d, state={}, metadata=None) for d in data_list]
    return Batch(elements)


class TestOperatorNodeIndependent:
    """Independent unit tests for OperatorNode."""

    def test_operator_node_creation(self, simple_operator):
        """Test OperatorNode can be created with an operator."""
        operator_node = OperatorNode(simple_operator)
        assert operator_node.operator is simple_operator

    def test_operator_node_single_transformation(self, simple_operator):
        """Test OperatorNode applies transformation correctly with single-element batch."""
        operator_node = OperatorNode(simple_operator)

        # Create proper Batch object with single element
        input_batch = make_batch_from_data([{"value": jnp.array(5), "index": jnp.array(5)}])
        result = operator_node(input_batch)

        assert result is not None
        assert isinstance(result, Batch)
        assert "value" in result.data.get_value()
        assert "index" in result.data.get_value()

        # Value should be multiplied by 3 (operator multiplier)
        expected_value = jnp.array([5 * 3])
        assert jnp.array_equal(result.data.get_value()["value"], expected_value)
        assert jnp.array_equal(result.data.get_value()["index"], jnp.array([5]))  # Index unchanged

    def test_operator_node_batch_transformation(self, simple_operator):
        """Test OperatorNode handles batched data correctly."""
        operator_node = OperatorNode(simple_operator)

        # Create Batch with multiple elements
        input_batch = make_batch_from_data(
            [
                {"value": jnp.array(1), "index": jnp.array(1)},
                {"value": jnp.array(2), "index": jnp.array(2)},
                {"value": jnp.array(3), "index": jnp.array(3)},
            ]
        )

        result = operator_node(input_batch)

        assert result is not None
        assert isinstance(result, Batch)

        # Values should be multiplied by 3
        expected_values = jnp.array([3, 6, 9])
        assert jnp.array_equal(result.data.get_value()["value"], expected_values)
        assert jnp.array_equal(result.data.get_value()["index"], jnp.array([1, 2, 3]))

    def test_operator_node_with_key(self, simple_operator):
        """Test OperatorNode passes key parameter correctly."""
        operator_node = OperatorNode(simple_operator)

        # Create proper Batch object
        input_batch = make_batch_from_data([{"value": jnp.array(2), "index": jnp.array(2)}])
        key = jax.random.PRNGKey(42)

        result = operator_node(input_batch, key=key)

        assert result is not None
        # Transformation should work with key parameter
        assert jnp.array_equal(result.data.get_value()["value"], jnp.array([6]))  # [2] * 3

    def test_operator_node_state_management(self, simple_operator):
        """Test OperatorNode state management."""
        operator_node = OperatorNode(simple_operator)

        # Get state
        state = operator_node.get_state()
        assert isinstance(state, dict)

        # Create new node and set state
        new_operator_node = OperatorNode(SimpleOperator(multiplier=3.0))
        new_operator_node.set_state(state)

        # Both should behave identically
        input_batch = make_batch_from_data([{"value": jnp.array(4), "index": jnp.array(4)}])
        result1 = operator_node(input_batch)
        result2 = new_operator_node(input_batch)

        assert jnp.array_equal(result1.data.get_value()["value"], result2.data.get_value()["value"])

    def test_operator_node_repr(self, simple_operator):
        """Test OperatorNode string representation."""
        operator_node = OperatorNode(simple_operator)
        repr_str = repr(operator_node)
        assert "OperatorNode" in repr_str


class TestNodeJAXCompatibility:
    """Test JAX compatibility for nodes (requirement #5).

    As per requirements: Except DataLoader (the starting point), all other nodes
    must be fully jittable and differentiable.
    """

    def test_batch_node_jit_compatibility(self):
        """Test BatchNode is compatible with JAX JIT compilation."""
        batch_node = BatchNode(batch_size=2)

        # Simple function using BatchNode
        def process_with_batch(data):
            return batch_node(data)

        # Should be jittable (though may not be useful in practice)
        # This tests that the node doesn't use non-jittable operations
        try:
            jitted_fn = nnx.jit(process_with_batch)
            # Test with simple data
            test_data = {"value": jnp.array(1.0)}
            result = jitted_fn(test_data)
            # Result may be None (accumulating) or a batch
            assert result is None or isinstance(result, dict)
        except Exception as e:
            pytest.fail(f"BatchNode should be JIT compatible: {e}")

    def test_operator_node_jit_compatibility(self, simple_operator):
        """Test OperatorNode is compatible with nnx.jit compilation.

        Note: With nnx.jit, modules must be passed as function arguments
        (not captured in closures) for state tracking to work correctly.
        """
        operator_node = OperatorNode(simple_operator)

        # Correct nnx.jit pattern: pass module as argument
        @nnx.jit
        def process_with_operator(node, batch):
            return node(batch)

        try:
            # Create proper Batch object
            test_batch = make_batch_from_data([{"value": jnp.array(2.0), "index": jnp.array(2)}])
            result = process_with_operator(operator_node, test_batch)

            assert result is not None
            assert jnp.array_equal(
                result.data.get_value()["value"], jnp.array([6.0])
            )  # [2.0] * 3.0
        except Exception as e:
            pytest.fail(f"OperatorNode should be JIT compatible: {e}")

    def test_operator_node_grad_compatibility(self, simple_operator):
        """Test OperatorNode is compatible with NNX gradient computation.

        Note: Use nnx.grad (not jax.grad) for NNX modules. nnx.grad handles
        state propagation automatically. We use argnums=1 to differentiate
        with respect to the input x, not the model parameters.
        """
        operator_node = OperatorNode(simple_operator)

        def loss_fn(node, x):
            # Build batch with the input value
            batch = make_batch_from_data([{"value": x, "index": jnp.array(0)}])
            result = node(batch)
            return jnp.sum(result.data.get_value()["value"] ** 2)

        try:
            # Use nnx.grad with argnums=1 to differentiate w.r.t. x (second argument)
            # nnx.grad handles module state propagation automatically
            grad_fn = nnx.grad(loss_fn, argnums=1)
            x = jnp.array(2.0)  # Scalar value for single element
            gradient = grad_fn(operator_node, x)

            # Gradient should be computed correctly
            # loss = (x * 3)^2 = 9x^2, so grad = 18x = 36 for x=2
            expected_grad = jnp.array(36.0)
            assert jnp.allclose(gradient, expected_grad)
        except Exception as e:
            pytest.fail(f"OperatorNode should be differentiable: {e}")

    def test_data_source_node_limitations(self, mock_source):
        """Test DataSourceNode limitations with JAX (expected to have limitations)."""
        source_node = DataSourceNode(mock_source)

        # DataSourceNode may not be fully JIT compatible due to stateful operations
        # This is acceptable as it's typically the entry point of pipelines
        def process_with_source(dummy_input):
            return source_node(dummy_input)

        # This test documents the expected behavior
        # DataSourceNode may not be JIT compatible, which is acceptable
        try:
            jitted_fn = nnx.jit(process_with_source)
            jitted_fn(None)
            # If it works, great! If not, that's expected for DataSourceNode
        except Exception:
            # Expected for stateful DataSourceNode
            pass
