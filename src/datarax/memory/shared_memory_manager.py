# File: src/datarax/memory/shared_memory_manager.py

from multiprocessing import shared_memory

import flax.nnx as nnx
import jax


class SharedMemoryManager(nnx.Module):
    """Manage shared memory arrays for multi-worker scenarios.

    Automatically converts large numpy arrays to shared memory
    to avoid duplication across workers.
    """

    def __init__(self):
        super().__init__()
        self.shared_blocks = nnx.Variable({})
        self.array_metadata = nnx.Variable({})

    def make_shared(self, name: str, array: jax.Array, force: bool = False) -> jax.Array:
        """Convert array to shared memory.

        Args:
            name: Name for the shared memory block
            array: Array to store in shared memory
            force: If True, always use shared memory regardless of size

        Returns:
            The original array (shared memory is accessed via get_shared)
        """
        import numpy as np

        # Convert to numpy if it's a JAX array
        if hasattr(array, "__array__"):
            np_array = np.array(array)
        else:
            np_array = array

        # Get current dict values using new NNX API (get_value for non-array Variables)
        metadata_dict = self.array_metadata.get_value()
        blocks_dict = self.shared_blocks.get_value()

        # Check size threshold (arrays > 1MB) unless forced
        if not force and np_array.nbytes < 1024 * 1024:
            # For small arrays, just store in metadata without shared memory
            metadata_dict[name] = {
                "shape": np_array.shape,
                "dtype": np_array.dtype,
                "name": None,
                "data": np_array,
            }
            self.array_metadata.set_value(metadata_dict)
            return array

        # Create shared memory block
        shm = shared_memory.SharedMemory(create=True, size=np_array.nbytes)
        blocks_dict[name] = shm
        self.shared_blocks.set_value(blocks_dict)

        # Store metadata
        metadata_dict[name] = {
            "shape": np_array.shape,
            "dtype": np_array.dtype,
            "name": shm.name,
            "data": None,
        }
        self.array_metadata.set_value(metadata_dict)

        # Create numpy array from shared memory and copy data
        shared_np_array = np.ndarray(np_array.shape, dtype=np_array.dtype, buffer=shm.buf)
        shared_np_array[:] = np_array[:]

        return array  # Return original for now, get_shared will retrieve from shared memory

    def get_shared(self, name: str) -> jax.Array | None:
        """Get shared array by name."""
        import numpy as np

        # Use get_value() for non-array Variables (new NNX API)
        metadata_dict = self.array_metadata.get_value()

        if name not in metadata_dict:
            return None

        metadata = metadata_dict[name]

        # Check if it's stored in shared memory or directly
        if metadata["name"] is None:
            # Small array stored directly
            return jax.numpy.array(metadata["data"])
        else:
            # Large array in shared memory
            shm = shared_memory.SharedMemory(name=metadata["name"])

            # Create numpy array from shared memory
            shared_np_array = np.ndarray(metadata["shape"], dtype=metadata["dtype"], buffer=shm.buf)

            # Convert to JAX array
            return jax.numpy.array(shared_np_array)

    def cleanup(self):
        """Clean up shared memory blocks."""
        # Use get_value() for non-array Variables (new NNX API)
        blocks_dict = self.shared_blocks.get_value()
        for shm in blocks_dict.values():
            shm.close()
            shm.unlink()
        # Set empty dicts to clear
        self.shared_blocks.set_value({})
        self.array_metadata.set_value({})
