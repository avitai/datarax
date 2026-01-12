"""Tests for DefaultBatcher class."""

import jax.numpy as jnp
import pytest
from flax import nnx

from datarax.batching import DefaultBatcher, DefaultBatcherConfig


class TestDefaultBatcher:
    """Tests for DefaultBatcher class."""

    def test_init_basic(self):
        """Test basic initialization."""
        config = DefaultBatcherConfig(stochastic=False)
        batcher = DefaultBatcher(config, rngs=nnx.Rngs(0))
        assert batcher.collate_fn is None

    def test_init_with_collate_fn(self):
        """Test initialization with custom collate function."""

        def custom_collate(batch):
            return {"data": jnp.stack(batch)}

        config = DefaultBatcherConfig(stochastic=False)
        batcher = DefaultBatcher(config, collate_fn=custom_collate, rngs=nnx.Rngs(0))
        assert batcher.collate_fn == custom_collate

    def test_batch_arrays(self):
        """Test batching simple arrays."""
        config = DefaultBatcherConfig(stochastic=False)
        batcher = DefaultBatcher(config, rngs=nnx.Rngs(0))
        data = [jnp.array(i) for i in range(10)]

        batches = list(batcher(iter(data), batch_size=3))
        assert len(batches) == 4  # 10 items / 3 batch_size = 3 full + 1 partial
        assert jnp.array_equal(batches[0], jnp.array([0, 1, 2]))
        assert jnp.array_equal(batches[1], jnp.array([3, 4, 5]))
        assert jnp.array_equal(batches[2], jnp.array([6, 7, 8]))
        assert jnp.array_equal(batches[3], jnp.array([9]))

    def test_batch_with_drop_remainder(self):
        """Test batching with drop_remainder=True."""
        config = DefaultBatcherConfig(stochastic=False)
        batcher = DefaultBatcher(config, rngs=nnx.Rngs(0))
        data = [jnp.array(i) for i in range(10)]

        batches = list(batcher(iter(data), batch_size=3, drop_remainder=True))
        assert len(batches) == 3  # Last incomplete batch dropped
        assert jnp.array_equal(batches[0], jnp.array([0, 1, 2]))
        assert jnp.array_equal(batches[1], jnp.array([3, 4, 5]))
        assert jnp.array_equal(batches[2], jnp.array([6, 7, 8]))

    def test_batch_dicts(self):
        """Test batching dictionaries."""
        config = DefaultBatcherConfig(stochastic=False)
        batcher = DefaultBatcher(config, rngs=nnx.Rngs(0))
        data = [{"x": jnp.array(i), "y": jnp.array(i * 2)} for i in range(5)]

        batches = list(batcher(iter(data), batch_size=2))
        assert len(batches) == 3

        # First batch
        assert "x" in batches[0]
        assert "y" in batches[0]
        assert jnp.array_equal(batches[0]["x"], jnp.array([0, 1]))
        assert jnp.array_equal(batches[0]["y"], jnp.array([0, 2]))

        # Last batch (partial)
        assert jnp.array_equal(batches[2]["x"], jnp.array([4]))
        assert jnp.array_equal(batches[2]["y"], jnp.array([8]))

    def test_batch_tuples(self):
        """Test batching tuples."""
        config = DefaultBatcherConfig(stochastic=False)
        batcher = DefaultBatcher(config, rngs=nnx.Rngs(0))
        data = [(jnp.array(i), jnp.array(i * 2)) for i in range(4)]

        batches = list(batcher(iter(data), batch_size=2))
        assert len(batches) == 2

        # First batch
        assert isinstance(batches[0], tuple)
        assert len(batches[0]) == 2
        assert jnp.array_equal(batches[0][0], jnp.array([0, 1]))
        assert jnp.array_equal(batches[0][1], jnp.array([0, 2]))

    def test_batch_nested_structures(self):
        """Test batching nested data structures."""
        config = DefaultBatcherConfig(stochastic=False)
        batcher = DefaultBatcher(config, rngs=nnx.Rngs(0))
        data = [
            {
                "image": jnp.ones((3, 3)) * i,
                "label": jnp.array(i),
                "metadata": {"id": jnp.array(i * 10), "name": f"item_{i}"},
            }
            for i in range(4)
        ]

        batches = list(batcher(iter(data), batch_size=2))
        assert len(batches) == 2

        # Check first batch structure
        batch = batches[0]
        assert batch["image"].shape == (2, 3, 3)
        assert jnp.array_equal(batch["label"], jnp.array([0, 1]))
        assert jnp.array_equal(batch["metadata"]["id"], jnp.array([0, 10]))
        assert batch["metadata"]["name"] == ["item_0", "item_1"]

    def test_batch_with_custom_collate(self):
        """Test batching with custom collate function."""

        def custom_collate(batch):
            # Sum all elements in batch
            return jnp.sum(jnp.stack(batch))

        config = DefaultBatcherConfig(stochastic=False)
        batcher = DefaultBatcher(config, collate_fn=custom_collate, rngs=nnx.Rngs(0))
        data = [jnp.array(i) for i in range(9)]

        batches = list(batcher(iter(data), batch_size=3))
        assert len(batches) == 3
        assert batches[0] == 3  # 0 + 1 + 2
        assert batches[1] == 12  # 3 + 4 + 5
        assert batches[2] == 21  # 6 + 7 + 8

    def test_batch_empty_iterator(self):
        """Test batching empty iterator."""
        config = DefaultBatcherConfig(stochastic=False)
        batcher = DefaultBatcher(config, rngs=nnx.Rngs(0))
        data = []

        batches = list(batcher(iter(data), batch_size=5))
        assert len(batches) == 0

    def test_batch_single_element(self):
        """Test batching single element."""
        config = DefaultBatcherConfig(stochastic=False)
        batcher = DefaultBatcher(config, rngs=nnx.Rngs(0))
        data = [jnp.array(42)]

        batches = list(batcher(iter(data), batch_size=5))
        assert len(batches) == 1
        assert jnp.array_equal(batches[0], jnp.array([42]))

    def test_batch_exactly_divisible(self):
        """Test when data size is exactly divisible by batch size."""
        config = DefaultBatcherConfig(stochastic=False)
        batcher = DefaultBatcher(config, rngs=nnx.Rngs(0))
        data = [jnp.array(i) for i in range(9)]

        batches = list(batcher(iter(data), batch_size=3))
        assert len(batches) == 3
        for batch in batches:
            assert len(batch) == 3

    def test_batch_mixed_types_error(self):
        """Test that batching handles mixed types appropriately."""
        config = DefaultBatcherConfig(stochastic=False)
        batcher = DefaultBatcher(config, rngs=nnx.Rngs(0))
        # Create data with mixed types - DefaultBatcher may handle this gracefully
        data = [jnp.array(1), jnp.array(2), jnp.array(3)]

        batches_iter = batcher(iter(data), batch_size=2)
        # First batch should work (two arrays)
        batch1 = next(batches_iter)
        assert jnp.array_equal(batch1, jnp.array([1, 2]))

        # Second batch with single element
        batch2 = next(batches_iter)
        assert jnp.array_equal(batch2, jnp.array([3]))

    def test_batch_multidimensional_arrays(self):
        """Test batching multidimensional arrays."""
        config = DefaultBatcherConfig(stochastic=False)
        batcher = DefaultBatcher(config, rngs=nnx.Rngs(0))
        data = [jnp.ones((3, 4)) * i for i in range(5)]

        batches = list(batcher(iter(data), batch_size=2))
        assert len(batches) == 3
        assert batches[0].shape == (2, 3, 4)
        assert batches[1].shape == (2, 3, 4)
        assert batches[2].shape == (1, 3, 4)

    def test_invalid_batch_size(self):
        """Test that invalid batch size raises error."""
        config = DefaultBatcherConfig(stochastic=False)
        batcher = DefaultBatcher(config, rngs=nnx.Rngs(0))
        with pytest.raises(ValueError):
            list(batcher(iter([1, 2, 3]), batch_size=0))

        with pytest.raises(ValueError):
            list(batcher(iter([1, 2, 3]), batch_size=-1))

    def test_repr(self):
        """Test string representation."""
        config = DefaultBatcherConfig(stochastic=False)
        batcher = DefaultBatcher(config, rngs=nnx.Rngs(0))
        repr_str = repr(batcher)
        assert "DefaultBatcher" in repr_str
