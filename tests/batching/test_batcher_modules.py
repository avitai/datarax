"""Tests for NNX-based batcher modules.

This module contains tests for the NNX-based batcher module implementations.
"""

import numpy as np
from flax import nnx

from datarax.batching import DefaultBatcher, DefaultBatcherConfig


def test_default_batcher_basic():
    """Test the basic functionality of DefaultBatcher."""
    # Create a default batcher module
    config = DefaultBatcherConfig(stochastic=False)
    batcher = DefaultBatcher(config, rngs=nnx.Rngs(0))

    # Create some elements to batch
    elements = [{"x": np.array([i]), "y": np.array([i * 2])} for i in range(5)]

    # Test batching with batch_size=2
    batched_elements = list(batcher(iter(elements), batch_size=2))

    # Should have 3 batches (2, 2, 1)
    assert len(batched_elements) == 3

    # First batch should have 2 elements
    assert batched_elements[0]["x"].shape == (2, 1)
    assert batched_elements[0]["y"].shape == (2, 1)

    # Last batch should have 1 element
    assert batched_elements[2]["x"].shape == (1, 1)
    assert batched_elements[2]["y"].shape == (1, 1)

    # Check the values
    assert np.array_equal(batched_elements[0]["x"], np.array([[0], [1]]))
    assert np.array_equal(batched_elements[0]["y"], np.array([[0], [2]]))
    assert np.array_equal(batched_elements[1]["x"], np.array([[2], [3]]))
    assert np.array_equal(batched_elements[1]["y"], np.array([[4], [6]]))
    assert np.array_equal(batched_elements[2]["x"], np.array([[4]]))
    assert np.array_equal(batched_elements[2]["y"], np.array([[8]]))


def test_default_batcher_drop_remainder():
    """Test batching with drop_remainder=True."""
    # Create a default batcher module
    config = DefaultBatcherConfig(stochastic=False)
    batcher = DefaultBatcher(config, rngs=nnx.Rngs(0))

    # Create some elements to batch
    elements = [{"x": np.array([i]), "y": np.array([i * 2])} for i in range(5)]

    # Test batching with batch_size=2 and drop_remainder=True
    batched_elements = list(batcher(iter(elements), batch_size=2, drop_remainder=True))

    # Should have 2 batches (2, 2) and drop the remainder
    assert len(batched_elements) == 2

    # Check the values
    assert np.array_equal(batched_elements[0]["x"], np.array([[0], [1]]))
    assert np.array_equal(batched_elements[0]["y"], np.array([[0], [2]]))
    assert np.array_equal(batched_elements[1]["x"], np.array([[2], [3]]))
    assert np.array_equal(batched_elements[1]["y"], np.array([[4], [6]]))


def test_default_batcher_custom_collate():
    """Test the DefaultBatcher with a custom collate function."""

    # Create a custom collate function that concatenates instead of stacking
    def custom_collate(elements):
        result = {}
        for key in elements[0]:
            result[key] = np.concatenate([element[key] for element in elements])
        return result

    # Create a default batcher module with the custom collate function
    config = DefaultBatcherConfig(stochastic=False)
    batcher = DefaultBatcher(config, collate_fn=custom_collate, rngs=nnx.Rngs(0))

    # Create some elements to batch
    elements = [{"x": np.array([i, i + 1]), "y": np.array([i * 2])} for i in range(3)]

    # Test batching with batch_size=3
    batched_elements = list(batcher(iter(elements), batch_size=3))

    # Should have 1 batch with all elements
    assert len(batched_elements) == 1

    # Check the shapes
    assert batched_elements[0]["x"].shape == (6,)  # 3 elements * 2 values
    assert batched_elements[0]["y"].shape == (3,)  # 3 elements * 1 value

    # Check the values
    expected_x = np.array([0, 1, 1, 2, 2, 3])
    expected_y = np.array([0, 2, 4])
    assert np.array_equal(batched_elements[0]["x"], expected_x)
    assert np.array_equal(batched_elements[0]["y"], expected_y)
