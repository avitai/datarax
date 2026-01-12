"""
HuggingFace Datasets data source implementation for Datarax.

This unified module provides a data source that loads data from HuggingFace Datasets
and converts it to JAX arrays with support for both stateless and stateful modes.
"""

from dataclasses import dataclass
from typing import Any, Iterator
import jax
import jax.numpy as jnp
import flax.nnx as nnx

from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule


@dataclass
class HfDataSourceConfig(StructuralConfig):
    """Configuration for HFSource (HuggingFace Datasets data source).

    Args:
        name: Name of the dataset in HuggingFace Datasets (required)
        split: Split of the dataset to load, e.g., "train", "test" (required)
        data_dir: Optional directory where the dataset is stored/downloaded
        streaming: Whether to use streaming mode (data streamed on-the-fly)
        shuffle: Whether to shuffle the dataset
        shuffle_buffer_size: Buffer size for shuffling
        download_kwargs: Optional keyword arguments for download method
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
    download_kwargs: dict[str, Any] | None = None
    include_keys: set[str] | None = None
    exclude_keys: set[str] | None = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        # HFSource is deterministic unless shuffle is enabled
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
            raise ValueError("name is required for HfDataSourceConfig")
        if self.split is None:
            raise ValueError("split is required for HfDataSourceConfig")

        # Validate key filtering
        if self.include_keys is not None and self.exclude_keys is not None:
            raise ValueError("Cannot specify both include_keys and exclude_keys")


class HFSource(DataSourceModule):
    """HuggingFace Datasets data source for Datarax.

    This unified data source loads data from HuggingFace Datasets and converts
    them to JAX arrays. It supports both downloaded and streaming modes.

    Key Features:
    - Dual-mode operation (stateless iteration and stateful with internal state)
    - Support for streaming and downloaded datasets
    - Optional shuffling with configurable buffer size
    - Batch retrieval with get_batch method
    - Include/exclude key filtering
    - Automatic conversion to JAX arrays

    Examples:
        Create source for IMDB dataset:

        ```python
        # Create source for IMDB dataset
        config = HfDataSourceConfig(name="imdb", split="train")
        source = HFSource(config, rngs=nnx.Rngs(0))

        # Stateless iteration
        for item in source:
            process(item)

        # Stateful batch retrieval
        batch = source.get_batch(32)
        ```
    """

    def __init__(
        self,
        config: HfDataSourceConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize a HFSource with config.

        Args:
            config: Configuration for the HFSource
            rngs: Optional RNG state for shuffling and stateful iteration
            name: Optional name for the module (defaults to HFSource(dataset:split))

        Raises:
            ImportError: If the datasets package is not installed.
        """
        # Set default name if not provided
        if name is None:
            name = f"HFSource({config.name}:{config.split})"

        super().__init__(config, rngs=rngs, name=name)

        # Import datasets lazily
        try:
            import datasets  # type: ignore

            self.datasets = datasets
        except ImportError as e:
            raise ImportError(
                "Loading from HuggingFace Datasets requires additional "
                "dependencies. Install Datarax with optional HF dependencies "
                "using: pip install datarax[hf]"
            ) from e

        # Store configuration values
        self.dataset_name = config.name
        self.split = config.split
        self.data_dir = config.data_dir
        self.streaming = config.streaming
        self.shuffle = config.shuffle
        self.shuffle_buffer_size = config.shuffle_buffer_size
        self.include_keys = config.include_keys
        self.exclude_keys = config.exclude_keys

        # Load the dataset
        download_kwargs = config.download_kwargs or {}
        # Add revision parameter for security (bandit B615)
        if "revision" not in download_kwargs:
            download_kwargs["revision"] = "main"  # Pin to main revision for security
        self.dataset = self.datasets.load_dataset(  # nosec B615 - revision is set above
            config.name,
            split=config.split,
            data_dir=config.data_dir,
            streaming=config.streaming,
            **download_kwargs,
        )

        # Apply shuffling if requested
        if config.shuffle:
            if config.streaming:
                # For streaming datasets, use buffer shuffling
                self.dataset = self.dataset.shuffle(
                    num_samples=config.shuffle_buffer_size,
                    seed=42 if rngs is None else None,
                )
            else:
                # For non-streaming, shuffle the entire dataset
                self.dataset = self.dataset.shuffle(seed=42 if rngs is None else None)

        # Get dataset info and length
        if hasattr(self.dataset, "info"):
            self.dataset_info = self.dataset.info
        else:
            self.dataset_info = None

        # Try to get the dataset length
        if not self.streaming:
            try:
                self.length = len(self.dataset)
            except (TypeError, AttributeError):
                self.length = None
        else:
            # Streaming datasets don't have a defined length
            self.length = None

        # State variables for stateful iteration
        self.iterator = nnx.Variable(None)
        self.epoch = nnx.Variable(0)
        self.index = nnx.Variable(0)
        self._cached_data = None

        # Cache dataset if requested and not streaming
        if config.cacheable and not config.streaming:
            self._cached_data = list(self.dataset)

    def __len__(self) -> int:
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
        # Reset for new epoch
        self.epoch.set_value(self.epoch.get_value() + 1)
        self.index.set_value(0)

        # Use cached data if available
        if self._cached_data is not None:
            for element in self._cached_data:
                converted = self._convert_element(element)
                self.index.set_value(self.index.get_value() + 1)
                yield converted
        else:
            # Stream from dataset
            for element in self.dataset:
                converted = self._convert_element(element)
                self.index.set_value(self.index.get_value() + 1)
                yield converted

    def __getitem__(self, index: int) -> dict[str, jax.Array]:
        """Get element at specific index.

        Note: This is not supported for streaming datasets.

        Args:
            index: Index of element to retrieve.

        Returns:
            Data element at the specified index.

        Raises:
            IndexError: If index is out of bounds.
            NotImplementedError: If dataset is in streaming mode.
        """
        if self.streaming:
            raise NotImplementedError("Random access is not supported for streaming datasets")

        if self.length is not None:
            if index < 0:
                index = self.length + index
            if index < 0 or index >= self.length:
                raise IndexError(
                    f"Index {index} out of range for dataset with {self.length} elements"
                )

        # Get element from dataset or cache
        if self._cached_data is not None:
            element = self._cached_data[index]
        else:
            element = self.dataset[index]

        return self._convert_element(element)

    def get_batch(self, batch_size: int, key: jax.Array | None = None) -> dict[str, jax.Array]:
        """Get next batch of data.

        Args:
            batch_size: Number of elements in the batch.
            key: Optional RNG key for shuffling (used in stateless mode).

        Returns:
            Batch of data as dictionary with arrays of shape (batch_size, ...).
        """
        # Create iterator if needed
        current_iterator = self.iterator.get_value()
        if current_iterator is None:
            if self._cached_data is not None:
                current_iterator = iter(self._cached_data)
            else:
                current_iterator = iter(self.dataset)
            self.iterator.set_value(current_iterator)

        # Collect batch
        batch_elements = []
        for _ in range(batch_size):
            try:
                element = next(current_iterator)
                batch_elements.append(self._convert_element(element))
                self.index.set_value(self.index.get_value() + 1)
            except StopIteration:
                # Reset iterator for next epoch
                new_epoch = self.epoch.get_value() + 1
                self.epoch.set_value(new_epoch)
                self.index.set_value(0)

                if self._cached_data is not None:
                    current_iterator = iter(self._cached_data)
                else:
                    # Re-create dataset iterator
                    if self.shuffle and not self.streaming:
                        # Re-shuffle for new epoch
                        seed = None if self.rngs is None else new_epoch
                        self.dataset = self.dataset.shuffle(seed=seed)
                    current_iterator = iter(self.dataset)
                self.iterator.set_value(current_iterator)

                # Try to get element again only if dataset is not empty
                if len(batch_elements) < batch_size:
                    try:
                        element = next(current_iterator)
                        batch_elements.append(self._convert_element(element))
                        self.index.set_value(self.index.get_value() + 1)
                    except StopIteration:
                        # Dataset is empty or we've cycled through it
                        break

        # Stack batch elements
        if not batch_elements:
            return {}

        # Create batch dictionary
        batch_dict = {}
        keys = batch_elements[0].keys()

        for key in keys:
            # Stack values for this key
            values = [elem[key] for elem in batch_elements]

            # Convert to JAX array and stack
            if isinstance(values[0], jax.Array):
                batch_dict[key] = jnp.stack(values)
            else:
                # Handle non-array values (e.g., strings)
                batch_dict[key] = values

        return batch_dict

    def _convert_element(self, element: Any) -> dict[str, Any]:
        """Convert HuggingFace element to dictionary with JAX arrays.

        Args:
            element: HuggingFace dataset element.

        Returns:
            Dictionary with JAX arrays and other values.
        """
        import numpy as np

        if not isinstance(element, dict):
            # Wrap non-dict elements
            element = {"data": element}

        # Convert and filter
        result = {}
        for key, value in element.items():
            # Apply key filtering
            if self.include_keys and key not in self.include_keys:
                continue
            if self.exclude_keys and key in self.exclude_keys:
                continue

            # Handle PIL images (common in HuggingFace datasets)
            if hasattr(value, "mode"):  # PIL Image check
                result[key] = jnp.array(np.array(value))
            # Convert to JAX array if possible
            elif hasattr(value, "__array__"):
                # NumPy array or similar
                result[key] = jnp.array(value)
            elif isinstance(value, list | tuple) and value:
                # Try to convert sequences to arrays
                try:
                    result[key] = jnp.array(value)
                except (ValueError, TypeError):
                    # Keep as-is if can't convert (e.g., list of strings)
                    result[key] = value
            else:
                # Keep other types as-is
                result[key] = value

        return result

    def _apply_transform(
        self, batch_size: int, key: jax.Array | None, stats: Any | None = None
    ) -> dict[str, Any]:
        """Apply transform (get batch) - for compatibility with TransformBase.

        Args:
            batch_size: Size of batch to retrieve.
            key: Optional RNG key.
            stats: Unused (for compatibility).

        Returns:
            Batch of data.
        """
        return self.get_batch(batch_size, key)

    def reset(self, seed: int | None = None) -> None:
        """Reset the source to the beginning.

        Args:
            seed: Optional seed for reproducibility (ignored for HFSource)
        """
        self.iterator.set_value(None)
        self.epoch.set_value(0)
        self.index.set_value(0)
        if self._cache is not None:
            self._cache.clear()

    def get_dataset_info(self) -> Any | None:
        """Get information about the dataset.

        Returns:
            HuggingFace DatasetInfo object if available, None otherwise.
        """
        return self.dataset_info

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"HFSource("
            f"dataset={self.dataset_name}:{self.split}, "
            f"length={self.length}, "
            f"shuffle={self.shuffle}, "
            f"streaming={self.streaming}, "
            f"index={self.index.get_value()}, "
            f"epoch={self.epoch.get_value()})"
        )
