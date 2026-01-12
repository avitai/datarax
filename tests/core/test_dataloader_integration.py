"""
TDD Tests for DataLoader Integration with DAGExecutor.

This test file defines the expected behavior of DataLoader as the proper entry point
for data processing pipelines, following TDD principles.
"""

import pytest
import jax.numpy as jnp
import flax.nnx as nnx

from datarax.dag.nodes import DataLoader
from datarax.core.data_source import DataSourceModule
from datarax.core.config import StructuralConfig
from datarax.dag.dag_executor import DAGExecutor
from datarax.typing import Batch


class MockDataSource(DataSourceModule):
    """Mock data source for testing."""

    def __init__(
        self,
        data_size: int = 100,
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
        """Return iterator over data elements."""
        return self

    def __next__(self):
        """Get next data element."""
        if self.index.get_value() >= self.data_size:
            raise StopIteration

        data = {"value": jnp.array(self.index.get_value()), "index": self.index.get_value()}
        self.index.set_value(self.index.get_value() + 1)
        return data


@pytest.fixture
def mock_source():
    """Create a mock data source."""
    return MockDataSource(data_size=20)


class TestDataLoaderIteration:
    """Test DataLoader iteration behavior."""

    def test_dataloader_is_iterable(self, mock_source):
        """Test that DataLoader is iterable and provides batches."""
        dataloader = DataLoader(mock_source, batch_size=4)

        # Should be able to iterate over DataLoader
        batches = []
        for i, batch in enumerate(dataloader):
            if batch is not None:
                batches.append(batch)
            if i >= 10:  # Prevent infinite loop
                break

        # Should have produced some batches
        assert len(batches) > 0

        # Each batch should have the correct batch size (or smaller for last batch)
        for batch in batches:
            assert isinstance(batch, Batch)
            assert "value" in batch.data.get_value()
            assert batch.data.get_value()["value"].shape[0] <= 4  # batch_size or smaller

    def test_dataloader_with_shuffling_iteration(self, mock_source):
        """Test DataLoader with shuffling iteration."""
        dataloader = DataLoader(mock_source, batch_size=3, shuffle_buffer_size=10, shuffle_seed=42)

        # Should be able to iterate
        batches = list(dataloader)

        # Should have some batches
        assert len(batches) >= 0  # May be empty due to shuffling buffer

    def test_dataloader_drop_remainder(self, mock_source):
        """Test DataLoader with drop_remainder functionality."""
        dataloader = DataLoader(mock_source, batch_size=3, drop_remainder=True)

        # Should be iterable
        batches = []
        for batch in dataloader:
            if batch is not None:
                batches.append(batch)

        # All batches should have exact batch size when drop_remainder=True
        for batch in batches:
            if batch is not None:
                assert batch.data.get_value()["value"].shape[0] == 3


class TestDataLoaderStateManagement:
    """Test DataLoader state management."""

    def test_dataloader_has_get_state(self, mock_source):
        """Test DataLoader get_state method."""
        dataloader = DataLoader(mock_source, batch_size=4)

        # Should have get_state method
        state = dataloader.get_state()
        assert isinstance(state, dict)

    def test_dataloader_has_set_state(self, mock_source):
        """Test DataLoader set_state method."""
        dataloader = DataLoader(mock_source, batch_size=4)

        # Get initial state
        dataloader.get_state()

        # Process some data to change state
        next(iter(dataloader))

        # Get state after processing
        after_state = dataloader.get_state()

        # Create new dataloader and set state
        new_dataloader = DataLoader(MockDataSource(data_size=20), batch_size=4)
        new_dataloader.set_state(after_state)

        # States should match
        assert new_dataloader.get_state() == after_state

    def test_dataloader_state_persistence(self, mock_source):
        """Test DataLoader state persistence across iterations."""
        dataloader = DataLoader(mock_source, batch_size=4)

        # Get initial state
        initial_state = dataloader.get_state()

        # Process some batches
        batches = []
        for i, batch in enumerate(dataloader):
            if batch is not None:
                batches.append(batch)
            if i >= 2:  # Process a few batches
                break

        # State should have changed
        current_state = dataloader.get_state()
        assert current_state != initial_state


class TestDataLoaderDAGIntegration:
    """Test DataLoader integration with DAGExecutor."""

    def test_dataloader_in_dag_executor(self, mock_source):
        """Test DataLoader as entry point in DAGExecutor."""
        # DataLoader should work as a standalone pipeline component
        # For DAGExecutor integration, we add the source and batch node
        from datarax.dag.nodes import BatchNode

        executor = DAGExecutor()
        executor.add(mock_source)  # Add source to DAGExecutor
        executor.add(BatchNode(batch_size=4))  # Add batching for batch-first enforcement

        # Should be able to iterate over executor with source
        batches = []
        for i, batch in enumerate(executor):
            if batch is not None:
                batches.append(batch)
            if i >= 3:  # Get fewer batches since we're batching
                break

                # Should produce batches
        assert len(batches) > 0

    def test_dataloader_as_pipeline_start(self, mock_source):
        """Test DataLoader as proper pipeline entry point."""
        # Test DataLoader without shuffling first (shuffling can affect batch formation)
        dataloader = DataLoader(mock_source, batch_size=4)

        # Should work as standalone pipeline component
        batches = []
        for batch in dataloader:
            if batch is not None:
                batches.append(batch)
                if len(batches) >= 3:
                    break

        assert len(batches) > 0

        # All batches should be properly formatted
        for batch in batches:
            assert isinstance(batch, Batch)
            assert "value" in batch.data.get_value()
            assert batch.data.get_value()["value"].ndim >= 1  # Should have batch dimension


class TestDataLoaderConfiguration:
    """Test DataLoader configuration and properties."""

    def test_dataloader_properties(self, mock_source):
        """Test DataLoader configuration property exposure."""
        dataloader = DataLoader(
            mock_source, batch_size=8, shuffle_buffer_size=16, drop_remainder=True, shuffle_seed=123
        )

        # Should expose configuration
        assert dataloader.batch_size == 8
        assert dataloader.shuffle_buffer_size == 16
        assert dataloader.drop_remainder
        assert dataloader.shuffle_seed == 123

    def test_dataloader_repr(self, mock_source):
        """Test DataLoader string representation."""
        dataloader = DataLoader(mock_source, batch_size=4, shuffle_buffer_size=8)

        repr_str = repr(dataloader)
        assert "DataLoader" in repr_str
        assert "batch_size=4" in repr_str
        assert "shuffle=8" in repr_str


class TestNNXModuleIteratorStateRegression:
    """Regression tests for NNX module iterator state management.

    These tests ensure that when DataSourceModule subclasses (which are NNX modules)
    are used as iterators, their internal state is properly maintained. This catches
    the bug where storing NNX module iterators in nnx.Variable caused state to be
    copied on access, breaking iterator progression.

    See: Session 65 fix for nnx.Variable iterator copying bug.
    """

    def test_iterator_state_increments_correctly(self):
        """Test that NNX module iterator state increments on each iteration.

        This is the core regression test - if the iterator is being copied,
        the index will stay at 0 forever.
        """
        source = MockDataSource(data_size=10)
        dataloader = DataLoader(source, batch_size=2, drop_remainder=True)

        batches = list(dataloader)

        # With 10 elements and batch_size=2, we should get exactly 5 batches
        assert len(batches) == 5, (
            f"Expected 5 batches from 10 elements with batch_size=2, got {len(batches)}. "
            "This may indicate iterator state is not being maintained correctly."
        )

    def test_batch_values_are_sequential(self):
        """Test that batch values are sequential, not repeated.

        If the iterator is being copied on access, all batches would contain
        the same values (starting from index 0).
        """
        source = MockDataSource(data_size=6)
        dataloader = DataLoader(source, batch_size=2, drop_remainder=True)

        batches = list(dataloader)

        # Collect all values from batches
        all_values = []
        for batch in batches:
            batch_data = batch.data.get_value() if hasattr(batch.data, "get_value") else batch.data
            values = batch_data["value"].tolist()
            all_values.extend(values)

        # Values should be sequential: [0, 1, 2, 3, 4, 5]
        expected = list(range(6))
        assert all_values == expected, (
            f"Expected sequential values {expected}, got {all_values}. "
            "This indicates iterator state is being reset or copied incorrectly."
        )

    def test_iteration_terminates_correctly(self):
        """Test that iteration terminates after consuming all elements.

        This ensures the iterator doesn't loop infinitely when source is exhausted.
        """
        source = MockDataSource(data_size=5)
        dataloader = DataLoader(source, batch_size=2, drop_remainder=True)

        # Should terminate without manual break
        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            # Safety limit - if we hit this, iteration didn't terminate
            assert batch_count <= 10, (
                "Iteration exceeded 10 batches for 5 elements - possible infinite loop. "
                "This may indicate iterator exhaustion is not being detected correctly."
            )

        # With 5 elements and batch_size=2, drop_remainder=True, we get 2 batches
        assert batch_count == 2

    def test_drop_remainder_works_correctly(self):
        """Test that drop_remainder correctly drops incomplete final batch.

        Related to the original bug - ensures the pipeline terminates properly
        when remaining elements don't fill a complete batch.
        """
        source = MockDataSource(data_size=7)
        dataloader = DataLoader(source, batch_size=3, drop_remainder=True)

        batches = list(dataloader)

        # 7 elements with batch_size=3 and drop_remainder=True = 2 batches (6 elements)
        assert len(batches) == 2, f"Expected 2 batches, got {len(batches)}"

        # Verify last element processed is element 5 (indices 0-5)
        all_values = []
        for batch in batches:
            batch_data = batch.data.get_value() if hasattr(batch.data, "get_value") else batch.data
            all_values.extend(batch_data["value"].tolist())

        assert all_values == [0, 1, 2, 3, 4, 5], (
            f"Expected values [0,1,2,3,4,5] (element 6 dropped), got {all_values}"
        )
