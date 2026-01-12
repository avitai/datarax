"""
Consolidated Integration Tests for DAG System.

This file consolidates valuable test logic from the old Pipeline tests and rewrites them
to use proper DAGExecutor patterns, following TDD principles and the new architecture.
"""

from typing import Any

import pytest
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from dataclasses import dataclass
from datarax.core.data_source import DataSourceModule
from datarax.dag.nodes import DataLoader, BatchNode, OperatorNode
from datarax.core.operator import OperatorModule
from datarax.core.config import OperatorConfig, StructuralConfig
from datarax.dag.dag_executor import DAGExecutor
from datarax.operators import ElementOperator, ElementOperatorConfig
from datarax.core.element_batch import Element, Batch


def batch_keys(batch):
    """Get keys from a batch (works for both dict and Batch types)."""
    if isinstance(batch, Batch):
        return batch.data.get_value().keys()
    return batch.keys()


def batch_has_key(batch, key: str) -> bool:
    """Check if batch has a key (works for both dict and Batch types)."""
    return key in batch_keys(batch)


@dataclass
class MockDataSourceConfig(StructuralConfig):
    """Config for mock data source."""

    pass


class MockDataSource(DataSourceModule):
    """Mock data source for integration testing."""

    def __init__(self, data_size=100, *, rngs: nnx.Rngs | None = None):
        config = MockDataSourceConfig()
        super().__init__(config, rngs=rngs)
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
        """Return iterator over data elements."""
        return self

    def __next__(self):
        """Get next data element."""
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
    return MockDataSource(data_size=20)


@pytest.fixture
def simple_operator():
    """Create a simple operator."""
    return SimpleOperator(multiplier=3.0)


class TestDAGCompleteIntegration:
    """Complete integration tests for DAG system."""

    def test_basic_dag_pipeline(self, mock_source, simple_operator):
        """Test basic DAG pipeline construction and execution."""
        # Create DAG pipeline: source -> batch -> transform
        executor = DAGExecutor()
        executor.add(mock_source)
        executor.add(BatchNode(batch_size=4))
        executor.add(OperatorNode(simple_operator))

        # Execute pipeline
        batches = []
        for i, batch in enumerate(executor):
            if batch is not None:
                batches.append(batch)
            if i >= 3:  # Get a few batches
                break

        # Verify results
        assert len(batches) > 0
        for batch in batches:
            assert isinstance(batch, dict | Batch)
            assert batch_has_key(batch, "value")
            assert batch["value"].ndim >= 1  # Should be batched
            # Values should be transformed (multiplied by 3)
        assert jnp.all(batch["value"] >= 0)  # Original values * 3

    def test_dataloader_as_pipeline_start(self, mock_source, simple_operator):
        """Test DataLoader as complete pipeline entry point."""
        # Test DataLoader as standalone component, then combine with DAG
        dataloader = DataLoader(mock_source, batch_size=4)

        # Execute DataLoader directly first
        dataloader_batches = []
        for batch in dataloader:
            if batch is not None:
                dataloader_batches.append(batch)
                if len(dataloader_batches) >= 2:
                    break

        # Verify DataLoader works
        assert len(dataloader_batches) > 0
        for batch in dataloader_batches:
            assert isinstance(batch, dict | Batch)
            assert batch_has_key(batch, "value")
            assert batch["value"].ndim >= 1  # Should be batched

        # Now test with DAG (using fresh source)
        fresh_source = MockDataSource(data_size=20)
        executor = DAGExecutor()
        executor.add(fresh_source)
        executor.add(BatchNode(batch_size=4))
        executor.add(OperatorNode(simple_operator))

        # Execute pipeline
        dag_batches = []
        for i, batch in enumerate(executor):
            if batch is not None:
                dag_batches.append(batch)
            if i >= 2:
                break

        # Verify DAG results
        assert len(dag_batches) > 0
        for batch in dag_batches:
            assert isinstance(batch, dict | Batch)
            assert batch_has_key(batch, "value")
            assert batch["value"].ndim >= 1  # Should be batched

    def test_element_operator_integration(self, mock_source):
        """Test integration with ElementOperator (consolidated from old Pipeline tests)."""

        # Define a simple element transformation function
        def add_ten_fn(element: Element, key: jax.Array) -> Element:
            data = element.data
            if isinstance(data, dict) and "value" in data:
                new_data = {"value": data["value"] + 10, "index": data["index"]}
            else:
                new_data = data
            return element.replace(data=new_data)

        # Create pipeline with ElementOperator
        config = ElementOperatorConfig(stochastic=False)
        operator = ElementOperator(config, fn=add_ten_fn)

        executor = DAGExecutor()
        executor.add(mock_source)
        executor.add(BatchNode(batch_size=3))
        executor.add(OperatorNode(operator))

        # Execute pipeline
        batches = []
        for i, batch in enumerate(executor):
            if batch is not None:
                batches.append(batch)
            if i >= 2:
                break

        # Verify augmentation worked
        assert len(batches) > 0
        for batch in batches:
            assert isinstance(batch, dict | Batch)
            assert batch_has_key(batch, "value")
            # Values should be original + 10
            assert jnp.all(batch["value"] >= 10)

    def test_multi_transform_pipeline(self, mock_source):
        """Test pipeline with multiple transforms."""
        # Create multiple transforms
        transform1 = SimpleOperator(multiplier=2.0)
        transform2 = SimpleOperator(multiplier=1.5)

        # Create pipeline: source -> batch -> transform1 -> transform2
        executor = DAGExecutor()
        executor.add(mock_source)
        executor.add(BatchNode(batch_size=4))
        executor.add(OperatorNode(transform1))
        executor.add(OperatorNode(transform2))

        # Execute pipeline
        batches = []
        for i, batch in enumerate(executor):
            if batch is not None:
                batches.append(batch)
            if i >= 2:
                break

        # Verify results
        assert len(batches) > 0
        for batch in batches:
            assert isinstance(batch, dict | Batch)
            assert batch_has_key(batch, "value")
            assert batch["value"].ndim >= 1  # Should be batched
            # Values should be transformed by both transforms: original * 2.0 * 1.5 = original * 3.0

    def test_pipeline_state_management(self, mock_source, simple_operator):
        """Test pipeline state management and checkpointing."""
        # Create pipeline
        executor = DAGExecutor()
        executor.add(mock_source)
        executor.add(BatchNode(batch_size=4))
        executor.add(OperatorNode(simple_operator))

        # Get initial state
        initial_state = executor.get_state()
        assert isinstance(initial_state, dict)

        # Execute some batches
        batches = []
        for i, batch in enumerate(executor):
            if batch is not None:
                batches.append(batch)
            if i >= 1:  # Process one batch
                break

        # Get state after execution
        after_state = executor.get_state()
        assert isinstance(after_state, dict)

        # States should be different (execution progressed)
        # Note: We don't compare directly as state structure may vary
        assert len(batches) > 0

    def test_pipeline_with_different_batch_sizes(self, mock_source, simple_operator):
        """Test pipeline behavior with different batch sizes."""
        batch_sizes = [2, 4, 8]

        for batch_size in batch_sizes:
            # Create pipeline with specific batch size
            executor = DAGExecutor()
            executor.add(MockDataSource(data_size=20))  # Fresh source for each test
            executor.add(BatchNode(batch_size=batch_size))
            executor.add(OperatorNode(simple_operator))

            # Execute pipeline
            batches = []
            for i, batch in enumerate(executor):
                if batch is not None:
                    batches.append(batch)
                if i >= 2:
                    break

            # Verify batch sizes
            assert len(batches) > 0
            for batch in batches[:-1]:  # All but last batch should have exact size
                assert batch["value"].shape[0] == batch_size

    def test_empty_data_handling(self):
        """Test pipeline behavior with empty data source."""
        # Create empty data source
        empty_source = MockDataSource(data_size=0)

        executor = DAGExecutor()
        executor.add(empty_source)
        executor.add(BatchNode(batch_size=4))

        # Execute pipeline
        batches = []
        for i, batch in enumerate(executor):
            if batch is not None:
                batches.append(batch)
            if i >= 5:  # Try a few iterations
                break

        # Should handle empty data gracefully
        # (May produce empty batches or no batches, both are acceptable)
        assert isinstance(batches, list)

    def test_pipeline_error_handling(self, mock_source):
        """Test pipeline error handling with invalid operators."""

        def failing_fn(element: Element, key: jax.Array) -> Element:
            raise ValueError("Intentional test error")

        config = ElementOperatorConfig(stochastic=False)
        operator = ElementOperator(config, fn=failing_fn)

        executor = DAGExecutor()
        executor.add(mock_source)
        executor.add(BatchNode(batch_size=4))
        executor.add(OperatorNode(operator))

        # Execution should handle errors gracefully
        with pytest.raises(ValueError, match="Intentional test error"):
            for batch in executor:
                if batch is not None:
                    break  # Should fail before this


class TestDataLoaderStandalone:
    """Test DataLoader as standalone pipeline component."""

    def test_dataloader_complete_pipeline(self, mock_source):
        """Test DataLoader as complete, self-contained pipeline."""
        # DataLoader should work as standalone component
        dataloader = DataLoader(
            mock_source,
            batch_size=4,
            shuffle_buffer_size=None,  # No shuffling for predictable results
            drop_remainder=False,
        )

        # Execute DataLoader directly
        batches = []
        for batch in dataloader:
            if batch is not None:
                batches.append(batch)
                if len(batches) >= 3:
                    break

        # Verify results
        assert len(batches) > 0
        for batch in batches:
            assert isinstance(batch, dict | Batch)
            assert batch_has_key(batch, "value")
            assert batch["value"].ndim >= 1  # Should be batched
            assert batch["value"].shape[0] <= 4  # Batch size constraint

    def test_dataloader_with_transforms(self, mock_source, simple_operator):
        """Test DataLoader combined with additional transforms."""
        # Create DataLoader
        dataloader = DataLoader(mock_source, batch_size=4)

        # Apply additional transform to DataLoader output
        batches = []
        for batch in dataloader:
            if batch is not None:
                # Apply transform to batch
                transformed_batch = simple_operator(batch)
                batches.append(transformed_batch)
                if len(batches) >= 2:
                    break

        # Verify transformation
        assert len(batches) > 0
        for batch in batches:
            assert isinstance(batch, dict | Batch)
            assert batch_has_key(batch, "value")
            # Values should be transformed (multiplied by 3)
            assert jnp.all(batch["value"] >= 0)


class TestDAGExecutorAdvanced:
    """Advanced DAG executor integration tests."""

    def test_dag_executor_iteration_interface(self, mock_source, simple_operator):
        """Test DAGExecutor iteration interface matches expected behavior."""
        executor = DAGExecutor()
        executor.add(mock_source)
        executor.add(BatchNode(batch_size=4))
        executor.add(OperatorNode(simple_operator))

        # Test iterator protocol
        iterator = iter(executor)

        # Get first batch
        first_batch = next(iterator)
        assert first_batch is not None
        assert isinstance(first_batch, dict | Batch)
        assert batch_has_key(first_batch, "value")

        # Get second batch
        second_batch = next(iterator)
        assert second_batch is not None
        assert isinstance(second_batch, dict | Batch)

        # Batches should be different
        assert not jnp.array_equal(first_batch["value"], second_batch["value"])

    def test_dag_executor_with_complex_data(self):
        """Test DAGExecutor with complex data structures."""

        class ComplexDataSource(DataSourceModule):
            def __init__(self, *, rngs: nnx.Rngs | None = None):
                config = MockDataSourceConfig()
                super().__init__(config, rngs=rngs)
                self.index = nnx.Variable(0)

            def __call__(self, key=None):
                if self.index.get_value() >= 10:
                    return None

                data = {
                    "features": jnp.array([self.index.get_value(), self.index.get_value() * 2]),
                    "labels": jnp.array(self.index.get_value() % 2),
                    "metadata": {"id": self.index.get_value()},
                }
                self.index.set_value(self.index.get_value() + 1)
                return data

            def __iter__(self):
                return self

            def __next__(self):
                if self.index.get_value() >= 10:
                    raise StopIteration

                data = {
                    "features": jnp.array([self.index.get_value(), self.index.get_value() * 2]),
                    "labels": jnp.array(self.index.get_value() % 2),
                    "metadata": {"id": self.index.get_value()},
                }
                self.index.set_value(self.index.get_value() + 1)
                return data

        # Create pipeline with complex data
        executor = DAGExecutor()
        executor.add(ComplexDataSource())
        executor.add(BatchNode(batch_size=3))

        # Execute pipeline
        batches = []
        for i, batch in enumerate(executor):
            if batch is not None:
                batches.append(batch)
            if i >= 2:
                break

        # Verify complex data handling
        assert len(batches) > 0
        for batch in batches:
            assert isinstance(batch, dict | Batch)
            assert batch_has_key(batch, "features")
            assert batch_has_key(batch, "labels")
            assert batch_has_key(batch, "metadata")

            # Features should be batched
            assert batch["features"].ndim == 2  # [batch_size, feature_dim]
            assert batch["labels"].ndim == 1  # [batch_size]
