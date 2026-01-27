"""TensorFlow Datasets (TFDS) data source implementation for Datarax.

This module provides two distinct source types optimized for different use cases:

**TFDSEagerSource**: For small/medium datasets that fit in memory (~10% VRAM)
- Loads ALL data to JAX arrays at initialization
- Pure JAX iteration after init (no TensorFlow overhead during training)
- O(1) memory shuffling via Grain's index_shuffle (Feistel cipher)
- Fully checkpointable (just indices, no external state)
- Ideal for: MNIST, CIFAR-10, Fashion-MNIST, small custom datasets

**TFDSStreamingSource**: For large datasets that don't fit in memory
- Thin wrapper around TF dataset iterator
- DLPack zero-copy conversion for each batch
- Fixed prefetch buffer (no AUTOTUNE thread storms)
- Trade-offs: External iterator state, can't checkpoint mid-epoch
- Ideal for: ImageNet, large-scale datasets, memory-constrained environments

Architecture Insight:
    The ~0.4s delay at epoch 2 in previous implementations was caused by
    TensorFlow's AUTOTUNE prefetch spawning background threads during epoch
    transitions. TFDSEagerSource eliminates this entirely by loading all data
    upfront. TFDSStreamingSource uses fixed prefetch to prevent thread storms.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator

# Set protobuf implementation to avoid version conflicts
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule
from datarax.sources._conversion import tf_to_jax

if TYPE_CHECKING:
    import tensorflow_datasets as tfds


# =============================================================================
# Configuration Classes
# =============================================================================


@dataclass
class TFDSEagerConfig(StructuralConfig):
    """Configuration for TFDSEagerSource (loads all data to JAX at init).

    All original TfdsDataSourceConfig options are preserved for backward
    compatibility while adding the eager-loading behavior.

    Args:
        name: Name of the dataset in TFDS (required)
        split: Split of the dataset to load, e.g., "train", "test" (required)
        data_dir: Optional directory where the dataset is stored/downloaded
        shuffle: Whether to shuffle the dataset during iteration
        seed: Integer seed for Grain's index_shuffle (default: 42)
        as_supervised: If True, returns 'image'/'label' keys instead of original features
        download_and_prepare_kwargs: Optional keyword arguments for download_and_prepare
        include_keys: Optional set of keys to include in output (exclusive with exclude_keys)
        exclude_keys: Optional set of keys to exclude from output (exclusive with include_keys)

    Note:
        The seed parameter is an integer (not JAX RNG key) for Grain's index_shuffle.
        This ensures O(1) memory shuffling and reproducible per-epoch seeds.
    """

    # Required parameters use None sentinel for frozen dataclass
    name: str | None = None
    split: str | None = None

    # Optional parameters with defaults
    data_dir: str | None = None
    shuffle: bool = False
    seed: int = 42  # Integer seed for Grain's index_shuffle
    as_supervised: bool = False
    download_and_prepare_kwargs: dict[str, Any] | None = None
    include_keys: set[str] | None = None
    exclude_keys: set[str] | None = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # TFDSEagerSource is deterministic unless shuffle is enabled
        if self.shuffle:
            object.__setattr__(self, "stochastic", True)
            if self.stream_name is None:
                object.__setattr__(self, "stream_name", "shuffle")
            # Validate seed range for Grain's index_shuffle
            if self.seed < 0 or self.seed >= 2**32:
                raise ValueError("seed must be in [0, 2**32)")
        else:
            object.__setattr__(self, "stochastic", False)

        # Call parent validation
        super().__post_init__()

        # Validate required parameters
        if self.name is None:
            raise ValueError("name is required for TFDSEagerConfig")
        if self.split is None:
            raise ValueError("split is required for TFDSEagerConfig")

        # Validate key filtering
        if self.include_keys is not None and self.exclude_keys is not None:
            raise ValueError("Cannot specify both include_keys and exclude_keys")


@dataclass
class TFDSStreamingConfig(StructuralConfig):
    """Configuration for TFDSStreamingSource (streams data from TF dataset).

    Use this for datasets too large to fit in memory. The streaming source
    uses fixed prefetch buffers to avoid AUTOTUNE thread storms.

    Args:
        name: Name of the dataset in TFDS (required)
        split: Split of the dataset to load, e.g., "train", "test" (required)
        data_dir: Optional directory where the dataset is stored/downloaded
        shuffle: Whether to shuffle the dataset
        shuffle_buffer_size: TF shuffle buffer size (default: 1000)
        as_supervised: If True, returns 'image'/'label' keys
        download_and_prepare_kwargs: Optional keyword arguments for download_and_prepare
        include_keys: Optional set of keys to include in output
        exclude_keys: Optional set of keys to exclude from output
        prefetch_buffer: Fixed prefetch buffer size (default: 2, NOT AUTOTUNE)

    Note:
        The prefetch_buffer uses a fixed size instead of TF AUTOTUNE to prevent
        background thread storms that cause delays during epoch transitions.
    """

    # Required parameters
    name: str | None = None
    split: str | None = None

    # Streaming-specific options
    data_dir: str | None = None
    shuffle: bool = False
    shuffle_buffer_size: int = 1000
    as_supervised: bool = False
    download_and_prepare_kwargs: dict[str, Any] | None = None
    include_keys: set[str] | None = None
    exclude_keys: set[str] | None = None
    prefetch_buffer: int = 2  # Fixed, NOT AUTOTUNE

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.shuffle:
            object.__setattr__(self, "stochastic", True)
            if self.stream_name is None:
                object.__setattr__(self, "stream_name", "shuffle")
        else:
            object.__setattr__(self, "stochastic", False)

        super().__post_init__()

        if self.name is None:
            raise ValueError("name is required for TFDSStreamingConfig")
        if self.split is None:
            raise ValueError("split is required for TFDSStreamingConfig")

        if self.include_keys is not None and self.exclude_keys is not None:
            raise ValueError("Cannot specify both include_keys and exclude_keys")


# =============================================================================
# TFDSEagerSource - Loads All Data to JAX at Init
# =============================================================================


class TFDSEagerSource(DataSourceModule):
    """Eager-loading TFDS source for small/medium datasets.

    Loads ALL data to JAX arrays at initialization, then operates like a
    MemorySource with pure JAX operations. Use for datasets that fit in
    ~10% of device VRAM.

    Key Features:
        - One-time TF→JAX conversion at init (DLPack zero-copy when possible)
        - Pure JAX iteration after init (no TF threads during training)
        - O(1) memory shuffling via Grain's index_shuffle (Feistel cipher)
        - Full checkpointing support (indices only, no external state)
        - All original TFDSSource features preserved

    Performance:
        - Eliminates ~0.4s epoch 2 delay from TF AUTOTUNE threads
        - Training loops can use lax.fori_loop for 100-500x speedup
        - Device placement via collect_to_array() for staged training

    Example:
        ```python
        # Create eager source for MNIST
        config = TFDSEagerConfig(name="mnist", split="train", shuffle=True)
        source = TFDSEagerSource(config, rngs=nnx.Rngs(0))

        # Iterate - pure JAX, no TF overhead
        for item in source:
            process(item["image"])

        # Get batch (stateless with key, or stateful without)
        batch = source.get_batch(32)  # Stateful
        batch = source.get_batch(32, key=jax.random.key(0))  # Stateless
        ```
    """

    # Store data as JAX arrays (annotated for NNX to prevent parameter tracking)
    data: dict[str, jax.Array] = nnx.data()

    def __init__(
        self,
        config: TFDSEagerConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize TFDSEagerSource by loading all data to JAX arrays.

        Args:
            config: Configuration for the source
            rngs: Optional RNG state for shuffling
            name: Optional name (defaults to TFDSEagerSource(dataset:split))
        """
        if name is None:
            name = f"TFDSEagerSource({config.name}:{config.split})"
        super().__init__(config, rngs=rngs, name=name)

        # Store config for feature access
        self.dataset_name = config.name
        self.split_name = config.split
        self.shuffle = config.shuffle
        self._seed = config.seed
        self.as_supervised = config.as_supervised
        self.include_keys = config.include_keys
        self.exclude_keys = config.exclude_keys

        # Load dataset info BEFORE loading data (for get_dataset_info)
        self._dataset_info = self._load_dataset_info(config)

        # Load ALL data to JAX arrays at init
        self.data = self._load_all_to_jax(config)

        # Clean up TF resources completely
        self._cleanup_tf()

        # State for iteration (like MemorySource)
        first_key = next(iter(self.data.keys()))
        self.length = self.data[first_key].shape[0]
        self.index = nnx.Variable(0)
        self.epoch = nnx.Variable(0)

    def _load_dataset_info(self, config: TFDSEagerConfig) -> tfds.core.DatasetInfo:
        """Load and cache dataset info before cleanup.

        Args:
            config: Source configuration

        Returns:
            TFDS DatasetInfo object
        """
        import tensorflow_datasets as tfds

        builder = tfds.builder(config.name, data_dir=config.data_dir)
        download_kwargs = config.download_and_prepare_kwargs or {}
        builder.download_and_prepare(**download_kwargs)
        return builder.info

    def _load_all_to_jax(self, config: TFDSEagerConfig) -> dict[str, jax.Array]:
        """Load entire dataset to JAX arrays using DLPack.

        This is the core of the eager-loading strategy. All TF operations
        happen here at init time, so training loops are pure JAX.

        Args:
            config: Source configuration

        Returns:
            Dictionary mapping keys to JAX arrays
        """
        import tensorflow_datasets as tfds

        ds = tfds.load(
            config.name,
            split=config.split,
            data_dir=config.data_dir,
            as_supervised=config.as_supervised,
        )

        # Collect all data
        arrays: dict[str, list] = {}
        for tf_element in ds:
            # Handle as_supervised tuple format
            if config.as_supervised and isinstance(tf_element, tuple):
                tf_element = {"image": tf_element[0], "label": tf_element[1]}

            for k, v in tf_element.items():
                # Apply key filtering
                if config.include_keys and k not in config.include_keys:
                    continue
                if config.exclude_keys and k in config.exclude_keys:
                    continue

                if k not in arrays:
                    arrays[k] = []
                arrays[k].append(tf_to_jax(v))

        # Stack to single arrays
        return {k: jnp.stack(v) for k, v in arrays.items()}

    def _cleanup_tf(self) -> None:
        """Release all TensorFlow resources.

        This ensures no TF threads remain active after init,
        eliminating the epoch 2 delay problem.
        """
        try:
            import tensorflow as tf

            tf.keras.backend.clear_session()
        except ImportError:
            pass
        gc.collect()

    # === Core iteration (matches MemorySource pattern) ===

    def __len__(self) -> int:
        """Return the total number of data elements."""
        return self.length

    def __iter__(self) -> Iterator[dict[str, jax.Array]]:
        """Iterate over data elements with O(1) memory shuffling.

        Uses Grain's index_shuffle for shuffling, which computes
        shuffled indices on-the-fly without storing a permutation array.
        """
        self.index.set_value(0)
        self.epoch.set_value(self.epoch.get_value() + 1)

        for i in range(self.length):
            idx = self._get_shuffled_index(i)
            yield {k: v[idx] for k, v in self.data.items()}

    def __getitem__(self, index: int) -> dict[str, jax.Array]:
        """Get element at specific index.

        Args:
            index: Index of element to retrieve (supports negative indexing)

        Returns:
            Data element at the specified index

        Raises:
            IndexError: If index is out of bounds
        """
        if index < 0:
            index = self.length + index
        if index < 0 or index >= self.length:
            raise IndexError(f"Index {index} out of range for {self.length} elements")
        return {k: v[index] for k, v in self.data.items()}

    def _get_shuffled_index(self, index: int) -> int:
        """Get shuffled index using Grain's O(1) memory Feistel cipher.

        Uses grain.index_shuffle instead of jax.random.permutation:
        - O(1) memory vs O(n) for permutation array
        - Deterministic and reproducible
        - Per-epoch seeding for different shuffles each epoch

        Args:
            index: Original sequential index

        Returns:
            Shuffled index for current epoch
        """
        if not self.shuffle:
            return index

        try:
            from grain._src.python.experimental.index_shuffle.python import (
                index_shuffle_module as index_shuffle,
            )

            # Different seed per epoch for varied shuffles
            epoch = self.epoch.get_value()
            per_epoch_seed = (self._seed + epoch) % (2**32)

            return index_shuffle.index_shuffle(
                index, max_index=self.length - 1, seed=per_epoch_seed, rounds=4
            )
        except ImportError:
            # Fallback: use JAX random for this index
            # This is less efficient but works without Grain
            key = jax.random.key(self._seed + self.epoch.get_value())
            perm = jax.random.permutation(key, self.length)
            return int(perm[index])

    # === Batch retrieval (stateless and stateful) ===

    def get_batch(self, batch_size: int, key: jax.Array | None = None) -> dict[str, jax.Array]:
        """Get batch - stateless (with key) or stateful (without key).

        Args:
            batch_size: Number of elements in the batch
            key: Optional RNG key for stateless random sampling

        Returns:
            Batch of data as dictionary with batched arrays
        """
        if key is not None:
            # Stateless mode: use JAX random for one-off random sampling
            if self.shuffle:
                indices = jax.random.permutation(key, self.length)[:batch_size]
            else:
                indices = jnp.arange(batch_size)
            return {k: v[indices] for k, v in self.data.items()}

        # Stateful mode: sequential iteration through shuffled indices
        start = self.index.get_value()
        end = min(start + batch_size, self.length)

        # Build indices (using shuffled mapping if enabled)
        indices = jnp.array([self._get_shuffled_index(i) for i in range(start, end)])

        self.index.set_value(end % self.length)
        if end >= self.length:
            self.epoch.set_value(self.epoch.get_value() + 1)

        return {k: v[indices] for k, v in self.data.items()}

    # === Dataset info access ===

    def get_dataset_info(self) -> tfds.core.DatasetInfo:
        """Get TFDS dataset info (cached from init)."""
        return self._dataset_info

    # === State management ===

    def reset(self, seed: int | None = None) -> None:
        """Reset the source to the beginning.

        Args:
            seed: Optional new seed (ignored, use config seed)
        """
        del seed  # Use config seed
        self.index.set_value(0)
        self.epoch.set_value(0)
        if self._cache is not None:
            self._cache.clear()

    def set_shuffle(self, shuffle: bool) -> None:
        """Enable or disable shuffling.

        Args:
            shuffle: Whether to shuffle data
        """
        self.shuffle = shuffle

    def _apply_transform(
        self, batch_size: int, key: jax.Array | None, stats: Any | None = None
    ) -> dict[str, jax.Array]:
        """Apply transform (get batch) - for compatibility with TransformBase.

        Args:
            batch_size: Size of batch to retrieve
            key: Optional RNG key
            stats: Unused (for compatibility)

        Returns:
            Batch of data
        """
        del stats
        return self.get_batch(batch_size, key)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TFDSEagerSource("
            f"dataset={self.dataset_name}:{self.split_name}, "
            f"length={self.length}, "
            f"shuffle={self.shuffle}, "
            f"epoch={self.epoch.get_value()})"
        )


# =============================================================================
# TFDSStreamingSource - Thin Wrapper for Large Datasets
# =============================================================================


class TFDSStreamingSource(DataSourceModule):
    """Streaming TFDS source for large datasets.

    Thin wrapper around TF dataset for data that can't fit in memory.
    Uses DLPack for efficient conversion and fixed prefetch buffer.

    Key Features:
        - DLPack zero-copy for TF→JAX conversion
        - Fixed prefetch buffer (no AUTOTUNE thread storms)
        - Supports all TFDS datasets
        - include_keys/exclude_keys filtering

    Trade-offs vs Eager:
        - Cannot checkpoint mid-epoch (external iterator state)
        - Some TF thread overhead (minimized with fixed prefetch)
        - Use with Artifex train_epoch_streaming() for best results

    Example:
        ```python
        # Create streaming source for large dataset
        config = TFDSStreamingConfig(name="imagenet2012", split="train")
        source = TFDSStreamingSource(config, rngs=nnx.Rngs(0))

        # Iterate with prefetching
        for batch in prefetch_to_device(source, size=2):
            train_step(batch)
        ```
    """

    def __init__(
        self,
        config: TFDSStreamingConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize TFDSStreamingSource.

        Args:
            config: Configuration for the source
            rngs: Optional RNG state
            name: Optional name (defaults to TFDSStreamingSource(dataset:split))
        """
        if name is None:
            name = f"TFDSStreamingSource({config.name}:{config.split})"
        super().__init__(config, rngs=rngs, name=name)

        import tensorflow_datasets as tfds

        self.dataset_name = config.name
        self.split_name = config.split
        self.shuffle = config.shuffle
        self.as_supervised = config.as_supervised
        self.include_keys = config.include_keys
        self.exclude_keys = config.exclude_keys

        # Load builder and info
        builder = tfds.builder(config.name, data_dir=config.data_dir)
        download_kwargs = config.download_and_prepare_kwargs or {}
        builder.download_and_prepare(**download_kwargs)
        self._dataset_info = builder.info

        # Build TF dataset with optimizations
        self._tf_dataset = builder.as_dataset(
            split=config.split,
            as_supervised=config.as_supervised,
        )

        if config.shuffle:
            self._tf_dataset = self._tf_dataset.shuffle(
                buffer_size=config.shuffle_buffer_size,
                reshuffle_each_iteration=True,
            )

        # CRITICAL: Fixed prefetch, NOT AUTOTUNE
        # This prevents thread storms during epoch transitions
        self._tf_dataset = self._tf_dataset.prefetch(config.prefetch_buffer)

        # Try to get length from split info
        try:
            split_base = config.split.split("[")[0]  # Handle splits like "train[:1000]"
            self.length: int | None = self._dataset_info.splits[split_base].num_examples
        except (AttributeError, KeyError):
            self.length = None

        self._iterator: Iterator | None = None
        self.epoch = nnx.Variable(0)

    def __len__(self) -> int:
        """Return the total number of data elements if known.

        Returns:
            Total number of elements or raises NotImplementedError if unknown
        """
        if self.length is None:
            raise NotImplementedError("Length unknown for this dataset split")
        return self.length

    def __iter__(self) -> Iterator[dict[str, jax.Array]]:
        """Start iteration over the dataset."""
        self.epoch.set_value(self.epoch.get_value() + 1)
        self._iterator = iter(self._tf_dataset)
        return self

    def __next__(self) -> dict[str, jax.Array]:
        """Get next element from the dataset.

        Returns:
            Dictionary of JAX arrays

        Raises:
            StopIteration: When dataset is exhausted
        """
        if self._iterator is None:
            self._iterator = iter(self._tf_dataset)

        tf_element = next(self._iterator)

        # Handle as_supervised tuple format
        if self.as_supervised and isinstance(tf_element, tuple):
            tf_element = {"image": tf_element[0], "label": tf_element[1]}

        # Convert with DLPack and apply filtering
        result = {}
        for k, v in tf_element.items():
            if self.include_keys and k not in self.include_keys:
                continue
            if self.exclude_keys and k in self.exclude_keys:
                continue
            result[k] = tf_to_jax(v)
        return result

    def get_dataset_info(self) -> tfds.core.DatasetInfo:
        """Get TFDS dataset info."""
        return self._dataset_info

    def reset(self, seed: int | None = None) -> None:
        """Reset the source to the beginning.

        Args:
            seed: Unused (for interface compatibility)
        """
        del seed
        self._iterator = None
        self.epoch.set_value(0)
        if self._cache is not None:
            self._cache.clear()

    def _apply_transform(
        self, batch_size: int, key: jax.Array | None, stats: Any | None = None
    ) -> dict[str, jax.Array]:
        """Apply transform - for compatibility with TransformBase.

        Note: For streaming sources, use pipeline batching instead.
        """
        del stats, key
        # Collect batch_size elements
        batch_elements = []
        for _ in range(batch_size):
            try:
                batch_elements.append(next(self))
            except StopIteration:
                break

        if not batch_elements:
            return {}

        # Stack into batch
        keys = batch_elements[0].keys()
        return {k: jnp.stack([elem[k] for elem in batch_elements]) for k in keys}

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TFDSStreamingSource("
            f"dataset={self.dataset_name}:{self.split_name}, "
            f"length={self.length}, "
            f"shuffle={self.shuffle}, "
            f"epoch={self.epoch.get_value()})"
        )
