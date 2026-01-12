# File: tests/integration/test_distributed_sharding.py

from unittest.mock import patch

import numpy as np

from datarax.sharding.jax_process_sharder import JaxProcessSharderModule, JaxProcessSharderConfig


class TestDistributedSharding:
    """Tests for distributed sharding functionality."""

    @patch("jax.process_count", return_value=4)
    @patch("jax.process_index", return_value=0)
    def test_shard_by_process(self, mock_index, mock_count):
        """Test sharding across JAX processes."""
        # Config-first pattern: create config, then module
        config = JaxProcessSharderConfig(drop_remainder=True)
        sharder = JaxProcessSharderModule(config)

        # Test list sharding
        data = list(range(100))
        sharded = sharder.shard_data(data)

        # Each process should get 25 elements
        assert len(sharded) == 25
        assert sharded == list(range(0, 25))

    @patch("jax.process_count", return_value=4)
    @patch("jax.process_index", return_value=3)
    def test_last_process_sharding(self, mock_index, mock_count):
        """Test sharding for last process."""
        # Config-first pattern
        config = JaxProcessSharderConfig(drop_remainder=False)
        sharder = JaxProcessSharderModule(config)

        # Test with uneven division
        data = list(range(102))  # Not divisible by 4
        sharded = sharder.shard_data(data)

        # Last process gets remainder
        assert len(sharded) == 27  # 25 + 2 remainder

    def test_array_sharding(self):
        """Test sharding of numpy arrays."""
        with patch("jax.process_count", return_value=2):
            with patch("jax.process_index", return_value=0):
                # Config-first pattern
                config = JaxProcessSharderConfig(drop_remainder=True)
                sharder = JaxProcessSharderModule(config)

                array = np.arange(100).reshape(100, 1)
                sharded = sharder.shard_data(array)

                assert sharded.shape == (50, 1)
                np.testing.assert_array_equal(sharded, array[:50])
