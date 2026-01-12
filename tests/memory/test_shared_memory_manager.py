# File: tests/unit/memory/test_shared_memory_manager.py


import flax.nnx as nnx
import numpy as np
import pytest

from datarax.memory.shared_memory_manager import SharedMemoryManager


class TestSharedMemoryManager:
    """Test suite for SharedMemoryManager."""

    @pytest.fixture
    def manager(self):
        """Create SharedMemoryManager instance."""
        manager = SharedMemoryManager()
        yield manager
        # Cleanup
        manager.cleanup()

    def test_initialization(self, manager):
        """Test SharedMemoryManager initialization."""
        assert isinstance(manager, nnx.Module)
        assert isinstance(manager.shared_blocks.get_value(), dict)
        assert isinstance(manager.array_metadata.get_value(), dict)

    def test_make_shared_small_array(self, manager):
        """Test that small arrays are not made shared."""
        small_array = np.ones((10, 10))  # 800 bytes < 1MB

        result = manager.make_shared("small", small_array)

        # Should return original array (not shared)
        assert result is small_array
        assert "small" not in manager.shared_blocks.get_value()

    def test_make_shared_large_array(self, manager):
        """Test making large array shared."""
        large_array = np.ones((1024, 1024))  # 8MB > 1MB

        result = manager.make_shared("large", large_array)

        # Should create shared memory
        assert "large" in manager.shared_blocks.get_value()
        assert "large" in manager.array_metadata.get_value()

        # Verify metadata
        metadata = manager.array_metadata.get_value()["large"]
        assert metadata["shape"] == (1024, 1024)
        assert metadata["dtype"] == np.float64

        # Verify data is preserved
        np.testing.assert_array_equal(result, large_array)

    def test_get_shared(self, manager):
        """Test retrieving shared array."""
        original = np.arange(1024 * 1024).reshape(1024, 1024)

        # Make shared
        manager.make_shared("test", original)

        # Retrieve
        retrieved = manager.get_shared("test")

        assert retrieved is not None
        np.testing.assert_array_equal(retrieved, original)

    def test_get_shared_nonexistent(self, manager):
        """Test retrieving non-existent shared array."""
        result = manager.get_shared("nonexistent")
        assert result is None

    def test_cleanup(self, manager):
        """Test cleanup of shared memory."""
        array = np.ones((1024, 1024))
        manager.make_shared("test", array)

        assert len(manager.shared_blocks.get_value()) == 1

        manager.cleanup()

        assert len(manager.shared_blocks.get_value()) == 0
        assert len(manager.array_metadata.get_value()) == 0

    def test_multiple_arrays(self, manager):
        """Test managing multiple shared arrays."""
        array1 = np.ones((1024, 1024))
        array2 = np.zeros((2048, 512))
        array3 = np.full((512, 2048), 42)

        manager.make_shared("array1", array1)
        manager.make_shared("array2", array2)
        manager.make_shared("array3", array3)

        assert len(manager.shared_blocks.get_value()) == 3

        # Verify all can be retrieved
        np.testing.assert_array_equal(manager.get_shared("array1"), array1)
        np.testing.assert_array_equal(manager.get_shared("array2"), array2)
        np.testing.assert_array_equal(manager.get_shared("array3"), array3)

    def test_dtype_preservation(self, manager):
        """Test that dtypes are preserved."""
        int_array = np.ones((1024, 512), dtype=np.int32)
        float_array = np.ones((1024, 512), dtype=np.float32)

        manager.make_shared("int", int_array)
        manager.make_shared("float", float_array)

        retrieved_int = manager.get_shared("int")
        retrieved_float = manager.get_shared("float")

        assert retrieved_int.dtype == np.int32
        assert retrieved_float.dtype == np.float32
