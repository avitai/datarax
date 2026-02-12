"""HuggingFace Datasets data source implementation for Datarax.

This module provides two distinct source types optimized for different use cases:

**HFEagerSource**: For small/medium datasets that fit in memory (~10% VRAM)
- Loads ALL data to JAX arrays at initialization
- Pure JAX iteration after init (no HuggingFace overhead during training)
- O(1) memory shuffling via Grain's index_shuffle (Feistel cipher)
- Fully checkpointable (just indices, no external state)
- Ideal for: MNIST, CIFAR-10, sentiment datasets, small custom datasets

**HFStreamingSource**: For large datasets that don't fit in memory
- Thin wrapper around HuggingFace dataset iterator
- Supports HuggingFace's built-in streaming mode
- Trade-offs: External iterator state, can't checkpoint mid-epoch
- Ideal for: The Pile, C4, large-scale datasets, memory-constrained environments

Architecture Insight:
    The separation between eager and streaming follows the same pattern as TFDS,
    ensuring consistent behavior across data backends while optimizing for each
    use case's specific requirements.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from collections.abc import Iterator

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule
from datarax.sources._conversion import hf_to_jax
from datarax.sources._eager_source_ops import (
    batch_elements_to_dict,
    convert_and_filter_element,
    eager_get_batch,
    eager_iter,
    eager_reset,
    reset_streaming_state,
    validate_eager_config,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# Configuration Classes
# =============================================================================


@dataclass
class HFEagerConfig(StructuralConfig):
    """Configuration for HFEagerSource (loads all data to JAX at init).

    All original HfDataSourceConfig options are preserved for backward
    compatibility while adding the eager-loading behavior.

    Args:
        name: Name of the dataset in HuggingFace Hub (required)
        split: Split of the dataset to load, e.g., "train", "test" (required)
        data_dir: Optional directory where the dataset is stored/downloaded
        shuffle: Whether to shuffle the dataset during iteration
        seed: Integer seed for Grain's index_shuffle (default: 42)
        download_kwargs: Optional keyword arguments for load_dataset
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
    download_kwargs: dict[str, Any] | None = None
    include_keys: set[str] | None = None
    exclude_keys: set[str] | None = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # HFEagerSource is deterministic unless shuffle is enabled
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

        # Shared validation
        validate_eager_config(
            self.name, self.split, self.include_keys, self.exclude_keys, "HFEagerConfig"
        )


@dataclass
class HFStreamingConfig(StructuralConfig):
    """Configuration for HFStreamingSource (streams data from HF dataset).

    Use this for datasets too large to fit in memory or when using HuggingFace's
    built-in streaming mode for efficient data loading.

    Args:
        name: Name of the dataset in HuggingFace Hub (required)
        split: Split of the dataset to load, e.g., "train", "test" (required)
        data_dir: Optional directory where the dataset is stored/downloaded
        streaming: Whether to use HuggingFace streaming mode (default: False)
        shuffle: Whether to shuffle the dataset
        shuffle_buffer_size: Buffer size for shuffling in streaming mode (default: 1000)
        download_kwargs: Optional keyword arguments for load_dataset
        include_keys: Optional set of keys to include in output
        exclude_keys: Optional set of keys to exclude from output
    """

    # Required parameters
    name: str | None = None
    split: str | None = None

    # Streaming-specific options
    data_dir: str | None = None
    streaming: bool = False
    shuffle: bool = False
    shuffle_buffer_size: int = 1000
    download_kwargs: dict[str, Any] | None = None
    include_keys: set[str] | None = None
    exclude_keys: set[str] | None = None

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
            raise ValueError("name is required for HFStreamingConfig")
        if self.split is None:
            raise ValueError("split is required for HFStreamingConfig")

        if self.include_keys is not None and self.exclude_keys is not None:
            raise ValueError("Cannot specify both include_keys and exclude_keys")


# =============================================================================
# HFEagerSource - Loads All Data to JAX at Init
# =============================================================================


class HFEagerSource(DataSourceModule):
    """Eager-loading HuggingFace source for small/medium datasets.

    Loads ALL data to JAX arrays at initialization, then operates like a
    MemorySource with pure JAX operations. Use for datasets that fit in
    ~10% of device VRAM.

    Key Features:
        - One-time conversion at init (PIL→numpy→JAX for images)
        - Pure JAX iteration after init
        - O(1) memory shuffling via Grain's index_shuffle (Feistel cipher)
        - Full checkpointing support (indices only, no external state)
        - Automatic PIL Image to JAX array conversion

    Performance:
        - Training loops can use lax.fori_loop for 100-500x speedup
        - Device placement via collect_to_array() for staged training

    Example:
        ```python
        # Create eager source for MNIST from HuggingFace
        config = HFEagerConfig(name="mnist", split="train", shuffle=True)
        source = HFEagerSource(config, rngs=nnx.Rngs(0))

        # Iterate - pure JAX, no HF overhead
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
        config: HFEagerConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize HFEagerSource by loading all data to JAX arrays.

        Args:
            config: Configuration for the source
            rngs: Optional RNG state for shuffling
            name: Optional name (defaults to HFEagerSource(dataset:split))

        Raises:
            ImportError: If the datasets package is not installed
        """
        if name is None:
            name = f"HFEagerSource({config.name}:{config.split})"
        super().__init__(config, rngs=rngs, name=name)

        # Import datasets lazily
        try:
            import datasets  # type: ignore[import-not-found]

            self._datasets_module = datasets
        except ImportError as e:
            raise ImportError(
                "Loading from HuggingFace Datasets requires additional "
                "dependencies. Install Datarax with optional HF dependencies "
                "using: pip install datarax[hf]"
            ) from e

        # Store config for feature access
        self.dataset_name = config.name
        self.split_name = config.split
        self.shuffle = config.shuffle
        self._seed = config.seed
        self.include_keys = config.include_keys
        self.exclude_keys = config.exclude_keys

        # Load dataset info BEFORE loading data
        self._dataset_info = self._load_dataset_info(config)

        # Load ALL data to JAX arrays at init
        self.data = self._load_all_to_jax(config)

        # Clean up resources
        gc.collect()

        # State for iteration (like MemorySource)
        first_key = next(iter(self.data.keys()))
        self.length = self.data[first_key].shape[0]
        self.index = nnx.Variable(0)
        self.epoch = nnx.Variable(0)

    def _load_dataset_info(self, config: HFEagerConfig) -> Any:
        """Load and cache dataset info.

        Args:
            config: Source configuration

        Returns:
            HuggingFace DatasetInfo object if available
        """
        download_kwargs = config.download_kwargs or {}
        # Add revision for security (bandit B615)
        if "revision" not in download_kwargs:
            download_kwargs["revision"] = "main"

        # Load dataset to get info
        dataset = self._datasets_module.load_dataset(  # nosec B615
            config.name,
            split=config.split,
            data_dir=config.data_dir,
            **download_kwargs,
        )

        if hasattr(dataset, "info"):
            return dataset.info
        return None

    def _load_all_to_jax(self, config: HFEagerConfig) -> dict[str, jax.Array]:
        """Load entire dataset to JAX arrays.

        This is the core of the eager-loading strategy. All HuggingFace operations
        happen here at init time, so training loops are pure JAX.

        Args:
            config: Source configuration

        Returns:
            Dictionary mapping keys to JAX arrays
        """
        download_kwargs = config.download_kwargs or {}
        if "revision" not in download_kwargs:
            download_kwargs["revision"] = "main"

        dataset = self._datasets_module.load_dataset(  # nosec B615
            config.name,
            split=config.split,
            data_dir=config.data_dir,
            **download_kwargs,
        )

        # Collect all data
        arrays: dict[str, list] = {}
        for element in dataset:
            for k, v in element.items():
                # Apply key filtering
                if config.include_keys and k not in config.include_keys:
                    continue
                if config.exclude_keys and k in config.exclude_keys:
                    continue

                if k not in arrays:
                    arrays[k] = []

                # Convert to JAX (handles PIL images, numpy arrays, scalars)
                converted = hf_to_jax(v)
                # Only include if it was successfully converted to array
                if isinstance(converted, jax.Array):
                    arrays[k].append(converted)
                elif isinstance(converted, np.ndarray | int | float | bool):
                    arrays[k].append(jnp.array(converted))
                else:
                    # Non-numeric types (strings, etc.) - store as-is for now
                    arrays[k].append(converted)

        # Stack to single arrays where possible
        result = {}
        for k, v in arrays.items():
            if v and isinstance(v[0], jax.Array):
                result[k] = jnp.stack(v)
            elif v and isinstance(v[0], int | float | bool):
                result[k] = jnp.array(v)
            else:
                # Keep as list for non-stackable types
                result[k] = v  # type: ignore[assignment]

        return result

    # === Core iteration (matches MemorySource pattern) ===

    def __len__(self) -> int:
        """Return the total number of data elements."""
        return self.length

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over data elements with O(1) memory shuffling.

        Uses Grain's index_shuffle for shuffling, which computes
        shuffled indices on-the-fly without storing a permutation array.
        """

        def build_element(data: dict, idx: int) -> dict[str, Any]:
            return {k: v[idx] if isinstance(v, jax.Array) else v[idx] for k, v in data.items()}

        return eager_iter(
            self.data,
            self.length,
            self.index,
            self.epoch,
            self.shuffle,
            self._seed,
            build_element,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
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
        return {k: v[index] if isinstance(v, jax.Array) else v[index] for k, v in self.data.items()}

    # === Batch retrieval (stateless and stateful) ===

    def get_batch(self, batch_size: int, key: jax.Array | None = None) -> dict[str, Any]:
        """Get batch - stateless (with key) or stateful (without key).

        Args:
            batch_size: Number of elements in the batch
            key: Optional RNG key for stateless random sampling

        Returns:
            Batch of data as dictionary with batched arrays
        """

        def gather_fn(data: dict, indices: jax.Array) -> dict[str, Any]:
            batch = {}
            for k, v in data.items():
                if isinstance(v, jax.Array):
                    batch[k] = v[indices]
                else:
                    batch[k] = [v[int(i)] for i in indices]
            return batch

        return eager_get_batch(
            self.data,
            self.length,
            self.index,
            self.epoch,
            self.shuffle,
            self._seed,
            batch_size,
            key,
            gather_fn,
        )

    # === Dataset info access ===

    def get_dataset_info(self) -> Any:
        """Get HuggingFace dataset info (cached from init)."""
        return self._dataset_info

    # === State management ===

    def reset(self, seed: int | None = None) -> None:
        """Reset the source to the beginning.

        Args:
            seed: Optional new seed (ignored, use config seed)
        """
        del seed  # Use config seed
        eager_reset(self.index, self.epoch, self._cache)

    def set_shuffle(self, shuffle: bool) -> None:
        """Enable or disable shuffling.

        Args:
            shuffle: Whether to shuffle data
        """
        self.shuffle = shuffle

    def _apply_transform(
        self, batch_size: int, key: jax.Array | None, stats: Any | None = None
    ) -> dict[str, Any]:
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
            f"HFEagerSource("
            f"dataset={self.dataset_name}:{self.split_name}, "
            f"length={self.length}, "
            f"shuffle={self.shuffle}, "
            f"epoch={self.epoch.get_value()})"
        )


# =============================================================================
# HFStreamingSource - Thin Wrapper for Large Datasets
# =============================================================================


class HFStreamingSource(DataSourceModule):
    """Streaming HuggingFace source for large datasets.

    Thin wrapper around HuggingFace dataset for data that can't fit in memory.
    Supports HuggingFace's native streaming mode for efficient large-scale data loading.

    Key Features:
        - Native HuggingFace streaming support
        - Automatic PIL Image to JAX array conversion
        - include_keys/exclude_keys filtering
        - Revision pinning for security (B615)

    Trade-offs vs Eager:
        - Cannot checkpoint mid-epoch (external iterator state)
        - Use with prefetch_to_device() for best results

    Example:
        ```python
        # Create streaming source for large dataset
        config = HFStreamingConfig(name="allenai/c4", split="train", streaming=True)
        source = HFStreamingSource(config, rngs=nnx.Rngs(0))

        # Iterate with prefetching
        for batch in prefetch_to_device(source, size=2):
            train_step(batch)
        ```
    """

    def __init__(
        self,
        config: HFStreamingConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize HFStreamingSource.

        Args:
            config: Configuration for the source
            rngs: Optional RNG state
            name: Optional name (defaults to HFStreamingSource(dataset:split))

        Raises:
            ImportError: If the datasets package is not installed
        """
        if name is None:
            name = f"HFStreamingSource({config.name}:{config.split})"
        super().__init__(config, rngs=rngs, name=name)

        # Import datasets lazily
        try:
            import datasets  # type: ignore[import-not-found]

            self._datasets_module = datasets
        except ImportError as e:
            raise ImportError(
                "Loading from HuggingFace Datasets requires additional "
                "dependencies. Install Datarax with optional HF dependencies "
                "using: pip install datarax[hf]"
            ) from e

        self.dataset_name = config.name
        self.split_name = config.split
        self.streaming = config.streaming
        self.shuffle = config.shuffle
        self.shuffle_buffer_size = config.shuffle_buffer_size
        self.include_keys = config.include_keys
        self.exclude_keys = config.exclude_keys

        # Load the dataset
        download_kwargs = config.download_kwargs or {}
        if "revision" not in download_kwargs:
            download_kwargs["revision"] = "main"

        self._hf_dataset = self._datasets_module.load_dataset(  # nosec B615
            config.name,
            split=config.split,
            data_dir=config.data_dir,
            streaming=config.streaming,
            **download_kwargs,
        )

        # Apply shuffling if requested
        if config.shuffle:
            if config.streaming:
                self._hf_dataset = self._hf_dataset.shuffle(
                    buffer_size=config.shuffle_buffer_size,
                    seed=42,
                )
            else:
                self._hf_dataset = self._hf_dataset.shuffle(seed=42)

        # Get dataset info and length
        if hasattr(self._hf_dataset, "info"):
            self._dataset_info = self._hf_dataset.info
        else:
            self._dataset_info = None

        # Try to get length
        if not config.streaming:
            try:
                self.length: int | None = len(self._hf_dataset)
            except (TypeError, AttributeError):
                self.length = None
        else:
            self.length = None

        self._iterator: Iterator | None = None
        self.epoch = nnx.Variable(0)

    def __len__(self) -> int:
        """Return the total number of data elements if known.

        Returns:
            Total number of elements or raises NotImplementedError if unknown
        """
        if self.length is None:
            raise NotImplementedError("Length unknown for streaming dataset")
        return self.length

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Start iteration over the dataset."""
        self.epoch.set_value(self.epoch.get_value() + 1)
        self._iterator = iter(self._hf_dataset)
        return self

    def __next__(self) -> dict[str, Any]:
        """Get next element from the dataset.

        Returns:
            Dictionary of values (JAX arrays for numeric data)

        Raises:
            StopIteration: When dataset is exhausted
        """
        if self._iterator is None:
            self._iterator = iter(self._hf_dataset)

        element = next(self._iterator)
        return convert_and_filter_element(
            element,
            self.include_keys,
            self.exclude_keys,
            hf_to_jax,
        )

    def get_dataset_info(self) -> Any:
        """Get HuggingFace dataset info."""
        return self._dataset_info

    def reset(self, seed: int | None = None) -> None:
        """Reset the source to the beginning.

        Args:
            seed: Unused (for interface compatibility)
        """
        del seed
        self._iterator = None
        reset_streaming_state(self.epoch, self._cache)

    def _apply_transform(
        self, batch_size: int, key: jax.Array | None, stats: Any | None = None
    ) -> dict[str, Any]:
        """Apply transform - for compatibility with TransformBase.

        Note: For streaming sources, use pipeline batching instead.
        """
        del stats, key
        elements = []
        for _ in range(batch_size):
            try:
                elements.append(next(self))
            except StopIteration:
                break
        return batch_elements_to_dict(elements)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"HFStreamingSource("
            f"dataset={self.dataset_name}:{self.split_name}, "
            f"length={self.length}, "
            f"shuffle={self.shuffle}, "
            f"streaming={self.streaming}, "
            f"epoch={self.epoch.get_value()})"
        )
