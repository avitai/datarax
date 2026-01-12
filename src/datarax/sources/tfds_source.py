"""
TensorFlow Datasets (TFDS) data source implementation for Datarax.

This unified module provides a data source that loads data from TensorFlow Datasets
(TFDS) and converts it to JAX arrays with support for both stateless and stateful modes.
"""

import os
from dataclasses import dataclass
from typing import Any, Iterator

# Set protobuf implementation to avoid version conflicts
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import tensorflow as tf
import tensorflow_datasets as tfds

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule


@dataclass
class TfdsDataSourceConfig(StructuralConfig):
    """Configuration for TFDSSource (TensorFlow Datasets data source).

    Args:
        name: Name of the dataset in TFDS (required)
        split: Split of the dataset to load, e.g., "train", "test" (required)
        data_dir: Optional directory where the dataset is stored/downloaded
        streaming: Whether to use TFDS streaming mode
        shuffle: Whether to shuffle the dataset
        shuffle_buffer_size: Buffer size for shuffling
        as_supervised: If True, returns 'image'/'label' keys instead of original features
        download_and_prepare_kwargs: Optional keyword arguments for download_and_prepare
        include_keys: Optional set of keys to include in output (exclusive with exclude_keys)
        exclude_keys: Optional set of keys to exclude from output (exclusive with include_keys)
    """

    # Required parameters use None sentinel for frozen dataclass
    name: str | None = None
    split: str | None = None

    # Optional parameters with defaults
    data_dir: str | None = None
    streaming: bool = False
    shuffle: bool = False
    shuffle_buffer_size: int = 1000
    as_supervised: bool = False
    download_and_prepare_kwargs: dict[str, Any] | None = None
    include_keys: set[str] | None = None
    exclude_keys: set[str] | None = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        # TFDSSource is deterministic unless shuffle is enabled
        if self.shuffle:
            object.__setattr__(self, "stochastic", True)
            if self.stream_name is None:
                object.__setattr__(self, "stream_name", "shuffle")
        else:
            object.__setattr__(self, "stochastic", False)

        # Call parent validation
        super().__post_init__()

        # Validate required parameters
        if self.name is None:
            raise ValueError("name is required for TfdsDataSourceConfig")
        if self.split is None:
            raise ValueError("split is required for TfdsDataSourceConfig")

        # Validate key filtering
        if self.include_keys is not None and self.exclude_keys is not None:
            raise ValueError("Cannot specify both include_keys and exclude_keys")


class TFDSSource(DataSourceModule):
    """TensorFlow Datasets (TFDS) data source for Datarax.

    This unified data source loads data from TFDS and converts TensorFlow tensors
    to JAX arrays. It supports both downloaded and streaming modes of TFDS.

    Key Features:
    - Dual-mode operation (stateless iteration and stateful with internal state)
    - Automatic TensorFlow to JAX conversion
    - Support for streaming and downloaded datasets
    - Optional shuffling with configurable buffer size
    - Batch retrieval with get_batch method
    - Include/exclude key filtering

    Examples:
        Create source for MNIST dataset:

        ```python
        # Create source for MNIST dataset
        config = TfdsDataSourceConfig(name="mnist", split="train")
        source = TFDSSource(config, rngs=nnx.Rngs(0))

        # Stateless iteration
        for item in source:
            process(item)

        # Stateful batch retrieval
        batch = source.get_batch(32)
        ```
    """

    def __init__(
        self,
        config: TfdsDataSourceConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize a TFDSSource with config.

        Args:
            config: Configuration for the TFDSSource
            rngs: Optional RNG state for shuffling and stateful iteration
            name: Optional name for the module (defaults to TFDSSource(dataset:split))
        """
        # Set default name if not provided
        if name is None:
            name = f"TFDSSource({config.name}:{config.split})"

        super().__init__(config, rngs=rngs, name=name)

        # Store configuration values
        self.dataset_name = config.name
        self.split = config.split
        self.data_dir = config.data_dir
        self.streaming = config.streaming
        self.shuffle = config.shuffle
        self.shuffle_buffer_size = config.shuffle_buffer_size
        self.as_supervised = config.as_supervised
        self.include_keys = config.include_keys
        self.exclude_keys = config.exclude_keys

        # Load the dataset and info
        download_and_prepare_kwargs = config.download_and_prepare_kwargs or {}
        self.dataset_builder = tfds.builder(config.name, data_dir=config.data_dir)

        if not config.streaming:
            self.dataset_builder.download_and_prepare(**download_and_prepare_kwargs)

        self.dataset_info = self.dataset_builder.info

        # Load the dataset
        read_config = tfds.ReadConfig(
            shuffle_seed=42 if config.shuffle else None,
        )

        self.dataset = self.dataset_builder.as_dataset(
            split=config.split,
            as_supervised=config.as_supervised,
            read_config=read_config,
            shuffle_files=config.shuffle,
        )

        # Apply shuffling if requested
        if config.shuffle:
            self.dataset = self.dataset.shuffle(
                buffer_size=config.shuffle_buffer_size,
                reshuffle_each_iteration=True,
            )

        # Cache if requested
        if config.cacheable:
            self.dataset = self.dataset.cache()

        # Prefetch for performance
        self.dataset = self.dataset.prefetch(tf.data.experimental.AUTOTUNE)

        # Try to get the dataset length
        try:
            split_info = self.dataset_info.splits[self.split]
            self.length = split_info.num_examples
        except (AttributeError, KeyError):
            # Some datasets may not have length information
            self.length = None

        # State variables for stateful iteration
        self.iterator = nnx.Variable(None)
        self.epoch = nnx.Variable(0)
        self.index = nnx.Variable(0)

    def __len__(self) -> int | None:
        """Return the total number of data elements if known.

        Returns:
            Total number of elements in the dataset or None if unknown.
        """
        return self.length

    def __iter__(self) -> Iterator[dict[str, jax.Array]]:
        """Iterate over data elements.

        Returns:
            Iterator over data elements as dictionaries of JAX arrays.
        """
        # Reset iterator for new epoch
        self.epoch.set_value(self.epoch.get_value() + 1)
        self.index.set_value(0)

        # Create new iterator
        tf_iterator = iter(self.dataset)

        # Yield elements
        for tf_element in tf_iterator:
            element = self._convert_element(tf_element)
            self.index.set_value(self.index.get_value() + 1)
            yield element

    def __getitem__(self, index: int) -> dict[str, jax.Array]:
        """Get element at specific index.

        Note: This may be inefficient for large datasets as it needs to
        iterate through the dataset to reach the desired index.

        Args:
            index: Index of element to retrieve.

        Returns:
            Data element at the specified index.

        Raises:
            IndexError: If index is out of bounds.
        """
        if self.length is not None:
            if index < 0:
                index = self.length + index
            if index < 0 or index >= self.length:
                raise IndexError(
                    f"Index {index} out of range for dataset with {self.length} elements"
                )

        # Skip to the desired index
        tf_iterator = iter(self.dataset.skip(index).take(1))
        try:
            tf_element = next(tf_iterator)
            return self._convert_element(tf_element)
        except StopIteration as e:
            raise IndexError(f"Index {index} out of range for dataset") from e

    def get_batch(self, batch_size: int, key: jax.Array | None = None) -> dict[str, jax.Array]:
        """Get next batch of data.

        Args:
            batch_size: Number of elements in the batch.
            key: Optional RNG key (not used for TFDS as shuffling is handled by TF).

        Returns:
            Batch of data as dictionary with arrays of shape (batch_size, ...).
        """
        # For TFDS, we use TensorFlow's batching
        current_iterator = self.iterator.get_value()
        if current_iterator is None:
            # Create batched dataset
            batched_dataset = self.dataset.batch(batch_size, drop_remainder=False)
            current_iterator = iter(batched_dataset)
            self.iterator.set_value(current_iterator)

        try:
            tf_batch = next(current_iterator)
            batch = self._convert_element(tf_batch)
            self.index.set_value(self.index.get_value() + batch_size)
            return batch
        except StopIteration:
            # Reset iterator for next epoch
            self.epoch.set_value(self.epoch.get_value() + 1)
            self.index.set_value(0)
            batched_dataset = self.dataset.batch(batch_size, drop_remainder=False)
            current_iterator = iter(batched_dataset)
            self.iterator.set_value(current_iterator)
            tf_batch = next(current_iterator)
            batch = self._convert_element(tf_batch)
            self.index.set_value(self.index.get_value() + batch_size)
            return batch

    def _convert_element(self, tf_element: Any) -> dict[str, jax.Array]:
        """Convert TensorFlow element to JAX arrays.

        Args:
            tf_element: TensorFlow element (tensor or dict of tensors).

        Returns:
            Dictionary of JAX arrays.
        """
        # Convert to dictionary if supervised
        if self.as_supervised and isinstance(tf_element, tuple):
            tf_element = {"image": tf_element[0], "label": tf_element[1]}

        # Convert TensorFlow tensors to JAX arrays
        if isinstance(tf_element, dict):
            jax_element = {}
            for key, value in tf_element.items():
                # Apply key filtering
                if self.include_keys and key not in self.include_keys:
                    continue
                if self.exclude_keys and key in self.exclude_keys:
                    continue

                # Convert to numpy then JAX
                if tf.is_tensor(value):
                    jax_element[key] = jnp.array(value.numpy())
                else:
                    jax_element[key] = jnp.array(value)
            return jax_element
        else:
            # Single tensor
            if tf.is_tensor(tf_element):
                return {"data": jnp.array(tf_element.numpy())}
            else:
                return {"data": jnp.array(tf_element)}

    def _apply_transform(
        self, batch_size: int, key: jax.Array | None, stats: Any | None = None
    ) -> dict[str, jax.Array]:
        """Apply transform (get batch) - for compatibility with TransformBase.

        Args:
            batch_size: Size of batch to retrieve.
            key: Optional RNG key.
            stats: Unused (for compatibility).

        Returns:
            Batch of data.
        """
        return self.get_batch(batch_size, key)

    def reset(self) -> None:
        """Reset the source to the beginning."""
        self.iterator.set_value(None)
        self.epoch.set_value(0)
        self.index.set_value(0)
        if self._cache is not None:
            self._cache.clear()

    def get_dataset_info(self) -> tfds.core.DatasetInfo:
        """Get information about the dataset.

        Returns:
            TFDS DatasetInfo object with metadata about the dataset.
        """
        return self.dataset_info

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TFDSSource("
            f"dataset={self.dataset_name}:{self.split}, "
            f"length={self.length}, "
            f"shuffle={self.shuffle}, "
            f"streaming={self.streaming}, "
            f"index={self.index.get_value()}, "
            f"epoch={self.epoch.get_value()})"
        )
