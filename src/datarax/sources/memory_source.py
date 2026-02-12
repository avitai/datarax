"""
In-memory data source implementation for Datarax.

This module provides a data source that serves data from in-memory collections
with support for both stateless and stateful operation modes.
"""

from dataclasses import dataclass
from typing import Any, Union
from collections.abc import Iterator, Sequence
import jax
import flax.nnx as nnx

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule
from datarax.core.metadata import MetadataManager, RecordMetadata
from datarax.config.registry import register_component
from datarax.samplers.index_shuffle import index_shuffle


@dataclass
class MemorySourceConfig(StructuralConfig):
    """Configuration for MemorySource (in-memory data source).

    Args:
        shuffle: Whether to shuffle data on each epoch
        cache_size: Number of batches to cache (0 = no caching)
        prefetch_size: Number of items to prefetch (0 = no prefetching)
        track_metadata: Whether to track metadata for each record
        shard_id: Optional shard identifier for distributed processing
        num_workers: Number of parallel workers (default 1). When > 1,
            each worker (identified by shard_id) receives a disjoint
            partition of the globally-shuffled elements. Worker k
            gets elements at global positions [k::num_workers].
    """

    # Optional parameters with defaults
    shuffle: bool = False
    cache_size: int = 0
    prefetch_size: int = 0
    track_metadata: bool = False
    shard_id: int | None = None
    num_workers: int = 1

    def __post_init__(self):
        """Validate configuration after initialization."""
        # MemorySource is deterministic unless shuffle is enabled
        if self.shuffle:
            object.__setattr__(self, "stochastic", True)
            if self.stream_name is None:
                object.__setattr__(self, "stream_name", "shuffle")
        else:
            object.__setattr__(self, "stochastic", False)

        if self.num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {self.num_workers}")
        if self.num_workers > 1 and self.shard_id is None:
            raise ValueError("shard_id is required when num_workers > 1")
        if self.num_workers > 1 and self.shard_id is not None and self.shard_id >= self.num_workers:
            raise ValueError(
                f"shard_id ({self.shard_id}) must be < num_workers ({self.num_workers})"
            )

        # Call parent validation
        super().__post_init__()


@register_component("source", "MemorySource")
class MemorySource(DataSourceModule):
    """In-memory data source for Datarax.

    This data source serves data from in-memory collections and supports
    both stateless and stateful operation modes.

    Key Features:

        - Dual-mode operation (stateless iteration and stateful with internal index)
        - Random access via __getitem__
        - Optional shuffling with RNG support
        - Batch retrieval with get_batch method
        - Support for dictionary and list/sequence data
        - Batch-first design for efficient processing

    Examples:
        Create source with list data:

        ```python
        # Create source with list data
        data = [{'x': i, 'y': i*2} for i in range(100)]
        config = MemorySourceConfig(shuffle=False)
        source = MemorySource(config, data, rngs=nnx.Rngs(0))

        # Stateless iteration
        for item in source:
            process(item)

        # Stateful iteration with internal index
        batch = source.get_batch(32)  # Gets next 32 items

        # With shuffling
        config = MemorySourceConfig(shuffle=True)
        source = MemorySource(config, data, rngs=nnx.Rngs(0))
        ```
    """

    # Static data container (not trainable parameters)
    data: Union[dict[str, Any], list[Any], Sequence[Any]] = nnx.data()

    def __init__(
        self,
        config: MemorySourceConfig,
        data: Union[dict[str, Any], list[Any], Sequence[Any]],
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize memory source with config.

        Args:
            config: Configuration for the MemorySource
            data: Either a dictionary mapping keys to data arrays or a
                list/sequence of elements. If a dictionary is provided,
                all values must have the same first dimension size.
            rngs: Optional RNG state for shuffling and stateful iteration
            name: Optional name for the module (defaults to "MemorySource")

        Raises:
            ValueError: If dictionary data has inconsistent lengths
            TypeError: If data is a string
        """
        # Set default name if not provided
        if name is None:
            name = "MemorySource"

        super().__init__(config, rngs=rngs, name=name)

        # Validate input data type
        if isinstance(data, str):
            raise TypeError(
                f"MemorySource expects a list, sequence, or dictionary, got string: {type(data)}"
            )

        # Store data and config values
        self.data = data
        self.shuffle = config.shuffle
        self.prefetch_size = config.prefetch_size

        # Calculate and validate length
        if isinstance(data, dict):
            # Verify all arrays have the same first dimension size
            lengths = []
            for key, value in data.items():
                if hasattr(value, "__len__"):
                    lengths.append(len(value))

            if not lengths:
                raise ValueError("Data dictionary must contain at least one array-like value")
            if not all(length == lengths[0] for length in lengths):
                raise ValueError(
                    f"All arrays in data dictionary must have the same length. "
                    f"Got lengths: {dict(zip(data.keys(), lengths))}"
                )
            self.length = lengths[0]
        else:
            self.length = len(data)

        # State variables for stateful iteration
        self.index = nnx.Variable(0)
        self.epoch = nnx.Variable(0)

        # Shuffle state (computed lazily per epoch)
        self._shuffle_seed: int | None = None  # Feistel seed, derived from RNG
        self._shuffled_indices = nnx.Variable(None)  # Materialized only for get_batch
        self._last_shuffle_epoch = nnx.Variable(-1)

        # Optional metadata tracking
        if config.track_metadata:
            self.metadata_manager = MetadataManager(
                rngs=rngs,
                track_batches=True,
                shard_id=config.shard_id,
            )
        else:
            self.metadata_manager = None

    def __len__(self) -> int:
        """Return the total number of data elements.

        Returns:
            Total number of elements in the source
        """
        return self.length

    def __iter__(self) -> Iterator[Any]:
        """Iterate over data elements.

        For stateless iteration, yields elements in order.
        For stateful iteration with shuffling, uses internal RNG.
        When prefetch_size > 0, wraps iteration with threaded prefetching.

        Returns:
            Iterator over data elements
        """
        raw = self._raw_iter()

        if self.prefetch_size > 0:
            from datarax.control.prefetcher import Prefetcher

            return Prefetcher(buffer_size=self.prefetch_size).prefetch(raw)
        return raw

    def _raw_iter(self) -> Iterator[Any]:
        """Synchronous element iteration (no prefetching).

        Uses lazy index computation via index_shuffle — O(1) memory per
        element, no full permutation array materialized.

        When num_workers > 1, yields only this worker's partition of the
        global order: worker k gets global positions [k::num_workers].

        Returns:
            Iterator over data elements
        """
        # Reset for new iteration
        self.index.set_value(0)
        self.epoch.set_value(self.epoch.get_value() + 1)

        num_workers = self.config.num_workers
        shard_id = self.config.shard_id or 0

        if self.shuffle and self.rngs is not None:
            # Lazy shuffle: compute each index on-the-fly via Feistel cipher
            seed = self._derive_shuffle_seed()
            if num_workers > 1:
                # Partition: yield elements at global positions [shard_id::num_workers]
                for i in range(shard_id, self.length, num_workers):
                    yield self._get_element(index_shuffle(i, seed, self.length))
            else:
                for i in range(self.length):
                    yield self._get_element(index_shuffle(i, seed, self.length))
        else:
            if num_workers > 1:
                for i in range(shard_id, self.length, num_workers):
                    yield self._get_element(i)
            else:
                for i in range(self.length):
                    yield self._get_element(i)

    def __getitem__(self, index: int) -> Any:
        """Get element at specific index.

        Args:
            index: Index of element to retrieve

        Returns:
            Data element at the specified index

        Raises:
            IndexError: If index is out of bounds
        """
        if index < 0:
            index = self.length + index

        if index < 0 or index >= self.length:
            raise IndexError(f"Index {index} out of range for source with {self.length} elements")

        return self._get_element(index)

    def get_batch(self, batch_size: int, key: jax.Array | None = None) -> Any:
        """Get next batch of data.

        This method supports both stateless (with explicit key) and
        stateful (with internal index tracking) operation.

        Args:
            batch_size: Number of elements in the batch
            key: Optional RNG key for shuffling (stateless mode)

        Returns:
            Batch of data with shape (batch_size, ...)
        """
        # Get indices for this batch
        if key is not None:
            # Stateless mode — derive seed from the provided key
            if self.shuffle:
                seed = int(jax.random.bits(key))
                batch_indices = [
                    index_shuffle(i, seed, self.length) for i in range(min(batch_size, self.length))
                ]
                return self._gather_batch(batch_indices)
            else:
                # Sequential: use slicing (zero-copy for arrays)
                return self._gather_batch_slice(0, min(batch_size, self.length))
        else:
            # Stateful mode - use internal index
            start = self.index.get_value()
            end = min(start + batch_size, self.length)

            # Update index for next call
            new_index = end % self.length
            self.index.set_value(new_index)
            if new_index == 0:
                self.epoch.set_value(self.epoch.get_value() + 1)
                # Force reshuffle on next epoch
                self._shuffle_seed = None
                self._shuffled_indices.set_value(None)

            if not self.shuffle:
                # Sequential: use slicing (zero-copy for arrays)
                return self._gather_batch_slice(start, end)

            # Shuffled: gather by the shuffled indices for this range
            indices = self._get_indices()
            return self._gather_batch(indices[start:end])

    def _apply_transform(
        self, batch_size: int, key: jax.Array | None, stats: Any | None = None
    ) -> Any:
        """Apply transform (get batch) - for compatibility with TransformBase.

        Args:
            batch_size: Size of batch to retrieve
            key: Optional RNG key
            stats: Unused (for compatibility)

        Returns:
            Batch of data
        """
        return self.get_batch(batch_size, key)

    def _derive_shuffle_seed(self) -> int:
        """Derive an integer seed from the JAX RNG stream for the current epoch.

        The seed is cached per epoch so that _raw_iter() (lazy) and
        _get_indices() (materialized) produce the same permutation.

        Returns:
            Integer seed for index_shuffle.
        """
        current_epoch = self.epoch.get_value()
        last_shuffle_epoch = self._last_shuffle_epoch.get_value()

        if self._shuffle_seed is None or last_shuffle_epoch != current_epoch:
            stream_name = self.config.stream_name or "shuffle"
            rng_stream = getattr(self.rngs, stream_name, self.rngs.default)
            key = rng_stream()
            self._shuffle_seed = int(jax.random.bits(key))
            self._last_shuffle_epoch.set_value(current_epoch)
            # Invalidate cached materialized indices
            self._shuffled_indices.set_value(None)

        return self._shuffle_seed

    def _get_indices(self) -> list[int]:
        """Get indices for iteration (possibly shuffled).

        Uses Feistel cipher index_shuffle for O(1)-per-element, worker-count
        invariant permutations. The full list is materialized here for
        get_batch() slicing; _raw_iter() uses lazy per-element computation.

        Returns:
            List of indices in iteration order.
        """
        if self.shuffle and self.rngs is not None:
            self.epoch.get_value()
            shuffled_indices = self._shuffled_indices.get_value()
            if shuffled_indices is None:
                seed = self._derive_shuffle_seed()
                shuffled_indices = [index_shuffle(i, seed, self.length) for i in range(self.length)]
                self._shuffled_indices.set_value(shuffled_indices)
            return shuffled_indices
        else:
            return list(range(self.length))

    def _get_element(self, index: int) -> Any:
        """Get single element at index.

        Args:
            index: Index of element

        Returns:
            Element at index
        """
        data = self.data
        if isinstance(data, dict):
            # Build dictionary element
            element = {}
            for key, value in data.items():
                if hasattr(value, "__getitem__"):
                    element[key] = value[index]
                else:
                    element[key] = value
            return element
        else:
            # Return list/sequence element
            return data[index]

    def _gather_batch_slice(self, start: int, end: int) -> Any:
        """Gather a contiguous batch using slice-based access.

        For numpy/JAX arrays, ``array[start:end]`` returns a view (no copy),
        which is the key memory optimization for sequential (non-shuffle) paths.

        Args:
            start: Start index (inclusive).
            end: End index (exclusive).

        Returns:
            Batch of elements from data[start:end].
        """
        data = self.data
        if isinstance(data, dict):
            return {
                key: value[start:end] if hasattr(value, "__getitem__") else [value] * (end - start)
                for key, value in data.items()
            }
        else:
            return data[start:end]

    def _gather_batch(self, indices: list[int]) -> Any:
        """Gather batch of elements at arbitrary indices.

        For non-contiguous access (shuffled), uses fancy indexing which
        creates copies. For contiguous ranges, prefer _gather_batch_slice.

        Args:
            indices: List of indices to gather.

        Returns:
            Batch of elements.
        """
        import jax.numpy as jnp

        idx_array = jnp.array(indices)
        data = self.data
        if isinstance(data, dict):
            batch = {}
            for key, value in data.items():
                if hasattr(value, "__getitem__"):
                    if isinstance(value, list | tuple):
                        batch[key] = [value[i] for i in indices]
                    else:
                        # Array-like: fancy indexing (JAX requires jnp.array, not list)
                        batch[key] = value[idx_array]
                else:
                    batch[key] = [value] * len(indices)
            return batch
        else:
            if isinstance(data, list | tuple):
                return [data[i] for i in indices]
            else:
                return data[idx_array]

    def reset(self, seed: int | None = None) -> None:
        """Reset the source to the beginning.

        Args:
            seed: Optional seed for reproducibility (ignored for MemorySource)
        """
        self.index.set_value(0)
        self.epoch.set_value(0)
        self._shuffle_seed = None
        self._shuffled_indices.set_value(None)
        self._last_shuffle_epoch.set_value(-1)
        if self._cache is not None:
            self._cache.clear()
        if self.metadata_manager is not None:
            self.metadata_manager.reset()

    def set_shuffle(self, shuffle: bool) -> None:
        """Enable or disable shuffling.

        Args:
            shuffle: Whether to shuffle data
        """
        self.shuffle = shuffle
        if not shuffle:
            self._shuffle_seed = None
            self._shuffled_indices.set_value(None)

    def get_with_metadata(self, index: int) -> tuple[Any, RecordMetadata]:
        """Get element at specific index with its metadata.

        This method is only available when track_metadata=True was set
        during initialization.

        Args:
            index: Index of element to retrieve

        Returns:
            Tuple of (data_element, metadata)

        Raises:
            RuntimeError: If metadata tracking is not enabled
            IndexError: If index is out of bounds
        """
        if self.metadata_manager is None:
            raise RuntimeError(
                "Metadata tracking is not enabled. "
                "Initialize with track_metadata=True to use this method."
            )

        # Get the data element
        data = self[index]

        # Create metadata for this record
        source_info = {
            "source": "memory",
            "index": index,
            "shuffle_enabled": self.shuffle,
        }
        metadata = self.metadata_manager.create_metadata(
            record_key=index,
            source_info=source_info,
        )

        return data, metadata

    def get_batch_with_metadata(
        self, batch_size: int, key: jax.Array | None = None
    ) -> tuple[Any, list[RecordMetadata]]:
        """Get next batch of data with metadata for each element.

        This method is only available when track_metadata=True was set
        during initialization.

        Args:
            batch_size: Number of elements in the batch
            key: Optional RNG key for shuffling (stateless mode)

        Returns:
            Tuple of (batch_data, list_of_metadata)

        Raises:
            RuntimeError: If metadata tracking is not enabled
        """
        if self.metadata_manager is None:
            raise RuntimeError(
                "Metadata tracking is not enabled. "
                "Initialize with track_metadata=True to use this method."
            )

        # Get the batch data
        batch = self.get_batch(batch_size, key)

        # Create metadata for each element in the batch
        metadata_list = []
        for i in range(
            min(batch_size, len(batch) if isinstance(batch, list | tuple) else batch_size)
        ):
            source_info = {
                "source": "memory",
                "batch_position": i,
                "shuffle_enabled": self.shuffle,
            }
            metadata = self.metadata_manager.create_metadata(
                record_key=f"batch_{self.metadata_manager.state.get_value()['batch_idx']}_{i}",
                source_info=source_info,
            )
            metadata_list.append(metadata)

        # Advance batch counter
        self.metadata_manager.next_batch()

        return batch, metadata_list

    @property
    def has_metadata(self) -> bool:
        """Check if this source is tracking metadata.

        Returns:
            True if metadata tracking is enabled, False otherwise
        """
        return self.metadata_manager is not None

    # get_state/set_state inherited from TransformBase - handles all nnx.Variables automatically

    def __repr__(self) -> str:
        """String representation."""
        data = self.data
        data_type = "dict" if isinstance(data, dict) else "list"
        return (
            f"MemorySource("
            f"type={data_type}, "
            f"length={self.length}, "
            f"shuffle={self.shuffle}, "
            f"index={self.index.get_value()}/{self.length}, "
            f"epoch={self.epoch.get_value()})"
        )
