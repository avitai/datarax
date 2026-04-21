"""Tests for the iterator checkpoint functionality."""

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

import jax.numpy as jnp

from datarax.checkpoint.iterators import (
    CheckpointableIterator,
    IteratorCheckpoint,
    PipelineCheckpoint,
)


class SimpleIterator(CheckpointableIterator):
    """A simple iterator that can be checkpointed for testing."""

    def __init__(self, data, start_idx=0, max_idx=None):
        """Initialize the iterator.

        Args:
            data: The data to iterate over.
            start_idx: Starting index for iteration.
            max_idx: Maximum index (exclusive) to iterate to.
        """
        self.data = data
        self.idx = start_idx
        self.max_idx = max_idx or len(data)
        # Adding required protocol attributes
        self.epoch = 0
        self.position = 0
        self.current = None
        self.iterator = None

    def __next__(self):
        """Get the next item in the iterator."""
        if self.idx >= self.max_idx:
            raise StopIteration

        value = self.data[self.idx]
        self.idx += 1
        self.position += 1
        self.current = value
        return value

    def __iter__(self):
        """Return self as an iterator."""
        return self

    def get_state(self) -> dict[str, Any]:
        """Get the iterator state for checkpointing."""
        return {
            "data": self.data,
            "idx": self.idx,
            "max_idx": self.max_idx,
            "epoch": self.epoch,
            "position": self.position,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore the iterator state from a checkpoint."""
        self.data = state["data"]
        self.idx = state["idx"]
        self.max_idx = state["max_idx"]
        self.epoch = state.get("epoch", 0)
        self.position = state.get("position", 0)


class Counter(CheckpointableIterator[int]):
    """A simple counter that implements CheckpointableIterator."""

    def __init__(self, start=0, step=1):
        """Initialize the counter."""
        self.value = start
        self.step = step
        # Add idx attribute used by tests
        self.idx = 0

    def __next__(self) -> int:
        """Get the next value."""
        self.value += self.step
        self.idx += 1
        return self.value

    def get_state(self) -> dict[str, Any]:
        """Get the counter state."""
        return {"value": self.value, "step": self.step, "idx": self.idx}

    def set_state(self, state: dict[str, Any]) -> None:
        """Set the counter state."""
        self.value = state["value"]
        self.step = state["step"]
        self.idx = state["idx"]


class TestIteratorCheckpoint(unittest.TestCase):
    """Tests for the IteratorCheckpoint class."""

    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for checkpoints
        self.temp_dir = tempfile.mkdtemp()

        # Create test data
        self.test_data = jnp.arange(100)

        # Create a simple iterator
        self.iterator = SimpleIterator(self.test_data)

        # Create a checkpoint handler
        self.checkpoint = IteratorCheckpoint(self.temp_dir)

    def tearDown(self):
        """Clean up the test environment."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_save_and_restore(self):
        """Test saving and restoring an iterator."""
        # Iterate a bit
        for _ in range(10):
            next(self.iterator)

        # Save the iterator state
        path = self.checkpoint.save_to_directory(self.iterator, step=1)

        # Check that the path exists
        self.assertTrue(Path(path).exists())

        # Create a new iterator
        new_iterator = SimpleIterator(self.test_data)

        # Restore with the iterator checkpoint
        restored = self.checkpoint.restore(new_iterator, step=1)

        # Check that it's the same object
        self.assertIs(restored, new_iterator)

        # Check that the state was restored correctly
        restored_idx = restored.idx  # type: ignore[reportAttributeAccessIssue]
        self.assertEqual(restored_idx, 10)

        # Check that iteration continues correctly
        next_value = next(restored)
        self.assertEqual(next_value, self.test_data[10])

    def test_restore_with_version(self):
        """Test restoring with different versions."""
        # Save at different states
        for i in range(3):
            # Iterate a bit
            for _ in range(10):
                next(self.iterator)

            # Save the iterator state with a high keep value to preserve all checkpoints
            self.checkpoint.save_to_directory(self.iterator, step=i + 1, keep=10)

        # Test restoring from step 2
        new_iterator = SimpleIterator(self.test_data)
        restored = self.checkpoint.restore(new_iterator, step=2)

        # Check that the state was restored correctly
        restored_idx = restored.idx  # type: ignore[reportAttributeAccessIssue]
        self.assertEqual(restored_idx, 20)

        # Test restoring the latest (step 3)
        new_iterator = SimpleIterator(self.test_data)
        restored = self.checkpoint.restore(new_iterator)

        # Check that the state was restored correctly
        restored_idx = restored.idx  # type: ignore[reportAttributeAccessIssue]
        self.assertEqual(restored_idx, 30)


class TestPipelineCheckpoint(unittest.TestCase):
    """Tests for the PipelineCheckpoint class."""

    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for checkpoints
        self.temp_dir = tempfile.mkdtemp()

        # Create test data
        self.test_data = jnp.arange(100)

        # Create a simple iterator
        self.iterator = SimpleIterator(self.test_data)

        # Create a checkpoint handler
        self.checkpoint = PipelineCheckpoint(self.temp_dir)

    def tearDown(self):
        """Clean up the test environment."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_save_at_step(self):
        """Test saving at specified steps."""
        # Should save at step 1000
        result = self.checkpoint.save_to_step(self.iterator, step=1000)
        self.assertIsNotNone(result)

        # Should not save at step 1001
        result = self.checkpoint.save_to_step(self.iterator, step=1001)
        self.assertIsNone(result)

        # Should save at step 2000
        result = self.checkpoint.save_to_step(self.iterator, step=2000)
        self.assertIsNotNone(result)

        # Should save at step 500 with custom interval
        result = self.checkpoint.save_to_step(self.iterator, step=500, interval=500)
        self.assertIsNotNone(result)

    def test_restore_latest(self):
        """Test restoring the latest checkpoint."""
        # Iterate a bit
        for _ in range(10):
            next(self.iterator)

        # Save the checkpoint - use an explicit step to ensure it's saved
        checkpoint_path = self.checkpoint.save_to_directory(self.iterator, step=1)
        # Verify the checkpoint was created
        self.assertTrue(Path(checkpoint_path).exists())

        # Create a new iterator
        new_iterator = SimpleIterator(self.test_data)

        # Restore the latest checkpoint
        restored = self.checkpoint.restore_latest(new_iterator)

        # Check that it's the same object
        self.assertIs(restored, new_iterator)

        # Check that the state was restored correctly
        restored_idx = restored.idx  # type: ignore[reportAttributeAccessIssue]
        self.assertEqual(restored_idx, 10)

        # Create a no-checkpoint scenario
        with tempfile.TemporaryDirectory() as empty_dir:
            empty_checkpoint = PipelineCheckpoint(empty_dir)
            with self.assertRaises(ValueError):
                empty_checkpoint.restore_latest(new_iterator)
