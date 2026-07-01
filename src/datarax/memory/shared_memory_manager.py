"""Shared memory manager for multi-worker data pipeline scenarios."""

import logging
from contextlib import suppress
from multiprocessing import shared_memory
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


logger = logging.getLogger(__name__)

# Arrays at least this large (bytes) are placed in shared memory; smaller arrays
# are kept inline to avoid the shared-memory block overhead.
_SHARED_MEMORY_MIN_BYTES = 1024 * 1024


class SharedMemoryManager:
    """Manage shared memory arrays for multi-worker scenarios.

    Automatically converts large numpy arrays to shared memory to avoid
    duplication across worker processes.

    This is a plain resource manager, **not** a Flax NNX module: it owns
    ``multiprocessing.shared_memory`` blocks and plain metadata, none of which
    are traced JAX state. (An earlier version subclassed ``nnx.Module`` and
    stored numpy arrays / Python dicts inside ``nnx.Variable``; that broke
    ``nnx.split``/checkpointing and is deliberately avoided here.) Use it as a
    context manager, or call ``cleanup()`` explicitly, to release blocks.
    """

    def __init__(self) -> None:
        """Initialize with empty block and metadata tracking."""
        self.shared_blocks: dict[str, shared_memory.SharedMemory] = {}
        self.array_metadata: dict[str, dict[str, Any]] = {}

    def make_shared(self, name: str, array: jax.Array, force: bool = False) -> jax.Array:
        """Convert array to shared memory.

        Args:
            name: Name for the shared memory block.
            array: Array to store in shared memory.
            force: If True, always use shared memory regardless of size.

        Returns:
            The original array (shared memory is accessed via ``get_shared``).
        """
        # Convert to numpy if it's a JAX array.
        np_array = np.array(array) if hasattr(array, "__array__") else array

        # Small arrays are kept inline (no shared-memory block) unless forced.
        if not force and np_array.nbytes < _SHARED_MEMORY_MIN_BYTES:
            self.array_metadata[name] = {
                "shape": np_array.shape,
                "dtype": np_array.dtype,
                "name": None,
                "data": np_array,
            }
            return array

        # Create shared memory block and copy data into it.
        shm = shared_memory.SharedMemory(create=True, size=np_array.nbytes)
        self.shared_blocks[name] = shm
        self.array_metadata[name] = {
            "shape": np_array.shape,
            "dtype": np_array.dtype,
            "name": shm.name,
            "data": None,
        }
        shared_np_array = np.ndarray(np_array.shape, dtype=np_array.dtype, buffer=shm.buf)
        shared_np_array[:] = np_array[:]

        return array

    def get_shared(self, name: str) -> jax.Array | None:
        """Get shared array by name, or None if it was never stored."""
        metadata = self.array_metadata.get(name)
        if metadata is None:
            return None

        # Small arrays are stored inline.
        if metadata["name"] is None:
            return jnp.array(metadata["data"])

        # Large arrays live in a shared-memory block.
        shm = shared_memory.SharedMemory(name=metadata["name"])
        try:
            shared_np_array = np.ndarray(metadata["shape"], dtype=metadata["dtype"], buffer=shm.buf)
            return jnp.array(np.array(shared_np_array, copy=True))
        finally:
            shm.close()

    def cleanup(self) -> None:
        """Close and unlink all shared memory blocks, then clear tracking."""
        for shm in self.shared_blocks.values():
            with suppress(FileNotFoundError, OSError):
                shm.close()
            with suppress(FileNotFoundError, OSError):
                shm.unlink()
        self.shared_blocks = {}
        self.array_metadata = {}

    def __enter__(self) -> "SharedMemoryManager":
        """Enter context manager."""
        return self

    def __exit__(self, _exc_type: object, _exc_val: object, _exc_tb: object) -> None:
        """Exit context manager and release resources."""
        self.cleanup()

    def __del__(self) -> None:
        """Safety net — prefer using as a context manager."""
        with suppress(AttributeError, FileNotFoundError, OSError):
            self.cleanup()
