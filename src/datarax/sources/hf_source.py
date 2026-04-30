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
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from datarax.sources._config_base import SourceConfigBase
from datarax.sources._conversion import hf_to_jax
from datarax.sources._eager_source_ops import (
    converted_filtered_record,
    validate_eager_source_settings,
    validate_streaming_source_settings,
)
from datarax.sources._source_base import EagerSourceBase, StreamingSourceBase


logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Classes
# =============================================================================


def _resolve_hf_download_options(
    download_kwargs: dict[str, Any] | None,
    *,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Prepare kwargs for ``datasets.load_dataset``.

    The ``datasets`` 4.x API no longer accepts ``local_files_only`` as a
    top-level kwarg (it is forwarded to ``BuilderConfig`` and rejected).
    The flag now lives on ``DownloadConfig`` and is passed via
    ``download_config=``.
    """
    resolved = dict(download_kwargs or {})
    # datasets.load_dataset no longer accepts this transformers-only flag.
    resolved.pop("trust_remote_code", None)
    resolved.setdefault("revision", "main")
    if local_files_only:
        from datasets import DownloadConfig  # noqa: PLC0415  (lazy: optional dep)

        existing = resolved.get("download_config")
        if isinstance(existing, DownloadConfig):
            existing.local_files_only = True
        else:
            resolved["download_config"] = DownloadConfig(local_files_only=True)
    return resolved


def _append_hf_converted_value(arrays: dict[str, list[Any]], key: str, value: Any) -> None:
    """Convert a HF value and append it to per-key buffers."""
    converted = hf_to_jax(value)
    if isinstance(converted, jax.Array):
        arrays.setdefault(key, []).append(converted)
    elif isinstance(converted, np.ndarray | int | float | bool):
        arrays.setdefault(key, []).append(jnp.array(converted))
    else:
        arrays.setdefault(key, []).append(converted)


def _stack_hf_array_columns(arrays: dict[str, list[Any]]) -> dict[str, Any]:
    """Stack buffered HF values into batched outputs where possible."""
    result: dict[str, Any] = {}
    for key, values in arrays.items():
        if values and isinstance(values[0], jax.Array):
            result[key] = jnp.stack(values)
        elif values and isinstance(values[0], int | float | bool):
            result[key] = jnp.array(values)
        else:
            result[key] = values
    return result


def _infer_hf_column_length(data: dict[str, Any]) -> int:
    """Infer row count from a loaded eager HF column mapping."""
    first_value = next(iter(data.values()))
    if hasattr(first_value, "shape"):
        return int(first_value.shape[0])
    return len(first_value)


@dataclass(frozen=True)
class HFEagerConfig(SourceConfigBase):
    """Configuration for HFEagerSource (loads all data to JAX at init).

    Configuration for eager-loading HuggingFace datasets into JAX arrays.

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

    shuffle: bool = False
    seed: int = 42  # Integer seed for Grain's index_shuffle
    download_kwargs: dict[str, Any] | None = None
    local_files_only: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        validate_eager_source_settings(
            self,
            "HFEagerConfig",
            seed=self.seed,
        )


@dataclass(frozen=True)
class HFStreamingConfig(SourceConfigBase):
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

    streaming: bool = False
    shuffle: bool = False
    shuffle_buffer_size: int = 1000
    download_kwargs: dict[str, Any] | None = None
    local_files_only: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        validate_streaming_source_settings(self, "HFStreamingConfig")


# =============================================================================
# HFEagerSource - Loads All Data to JAX at Init
# =============================================================================


class HFEagerSource(EagerSourceBase):
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

    # Store loaded columns as JAX arrays or Python lists for non-array columns.
    data: dict[str, Any]

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
        self._is_random_order = config.shuffle
        self._seed = config.seed
        self.include_keys = config.include_keys
        self.exclude_keys = config.exclude_keys

        # Load dataset info BEFORE loading data
        self._dataset_info = self._load_dataset_info_from_backend(config)

        # Load ALL data to JAX arrays at init
        self.data = nnx.data(self._load_all_from_backend_to_jax(config))

        # Clean up resources
        gc.collect()

        # State for iteration (like MemorySource)
        self.length = _infer_hf_column_length(self.data)
        self.index = nnx.Variable(0)
        self.epoch = nnx.Variable(0)

    def _load_dataset_info_from_backend(self, config: HFEagerConfig) -> Any:
        """Load and cache dataset info.

        Args:
            config: Source configuration

        Returns:
            HuggingFace DatasetInfo object if available
        """
        download_kwargs = _resolve_hf_download_options(
            config.download_kwargs, local_files_only=config.local_files_only
        )

        # name/split validated non-None by config __post_init__
        name = config.name
        split = config.split
        assert name is not None  # noqa: S101 (invariant, not control flow)
        assert split is not None  # noqa: S101 (invariant, not control flow)

        # Load dataset to get info
        dataset = self._datasets_module.load_dataset(  # nosec B615
            name,
            split=split,
            data_dir=config.data_dir,
            **download_kwargs,
        )

        if hasattr(dataset, "info"):
            return dataset.info
        return None

    def _load_all_from_backend_to_jax(self, config: HFEagerConfig) -> dict[str, jax.Array]:
        """Load entire dataset to JAX arrays.

        This is the core of the eager-loading strategy. All HuggingFace operations
        happen here at init time, so training loops are pure JAX.

        Args:
            config: Source configuration

        Returns:
            Dictionary mapping keys to JAX arrays
        """
        download_kwargs = _resolve_hf_download_options(
            config.download_kwargs, local_files_only=config.local_files_only
        )

        # name/split validated non-None by config __post_init__
        name = config.name
        split = config.split
        assert name is not None  # noqa: S101 (invariant, not control flow)
        assert split is not None  # noqa: S101 (invariant, not control flow)

        dataset = self._datasets_module.load_dataset(  # nosec B615
            name,
            split=split,
            data_dir=config.data_dir,
            **download_kwargs,
        )

        arrays: dict[str, list[Any]] = {}
        for element in dataset:
            for k, v in element.items():
                # Apply key filtering
                if config.include_keys and k not in config.include_keys:
                    continue
                if config.exclude_keys and k in config.exclude_keys:
                    continue
                _append_hf_converted_value(arrays, k, v)

        if not arrays:
            raise ValueError(
                "Dataset produced no elements after loading/filtering. "
                "Check split selection and include/exclude key filters."
            )

        return _stack_hf_array_columns(arrays)


# =============================================================================
# HFStreamingSource - Thin Wrapper for Large Datasets
# =============================================================================


class HFStreamingSource(StreamingSourceBase):
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

    # Narrow config type for pyright (base stores via nnx.static)
    config: HFStreamingConfig  # pyright: ignore[reportIncompatibleVariableOverride]

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
        self._is_random_order = config.shuffle

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

        # Load the dataset
        download_kwargs = _resolve_hf_download_options(
            config.download_kwargs, local_files_only=config.local_files_only
        )

        # name/split validated non-None by config __post_init__
        name = config.name
        split = config.split
        assert name is not None  # noqa: S101 (invariant, not control flow)
        assert split is not None  # noqa: S101 (invariant, not control flow)

        self._hf_dataset = self._datasets_module.load_dataset(  # nosec B615
            name,
            split=split,
            data_dir=config.data_dir,
            streaming=config.streaming,  # type: ignore[arg-type]
            **download_kwargs,
        )

        # Apply shuffling if requested
        if config.shuffle:
            if config.streaming:
                # HF IterableDataset.shuffle accepts buffer_size (type stubs incomplete)
                shuffle_kwargs: dict[str, Any] = {
                    "buffer_size": config.shuffle_buffer_size,
                    "seed": 42,
                }
                self._hf_dataset = self._hf_dataset.shuffle(**shuffle_kwargs)
            else:
                self._hf_dataset = self._hf_dataset.shuffle(seed=42)

        # Get dataset info and length
        if hasattr(self._hf_dataset, "info"):
            self._dataset_info = self._hf_dataset.info
        else:
            self._dataset_info = None

        # Try to get length (non-streaming Dataset supports __len__)
        if not config.streaming:
            try:
                self._length: int | None = len(self._hf_dataset)  # type: ignore[arg-type]
            except (TypeError, AttributeError):
                self._length = None
        else:
            self._length = None

        self._iterator: Iterator | None = None
        self.epoch = nnx.Variable(0)

    @property
    def dataset_name(self) -> str | None:
        """Dataset name from source config."""
        return self.config.name

    @property
    def split_name(self) -> str | None:
        """Dataset split from source config."""
        return self.config.split

    @property
    def is_iterable_mode(self) -> bool:
        """Streaming mode flag from source config."""
        return self.config.streaming

    @property
    def random_order_buffer_depth(self) -> int:
        """Shuffle buffer size from source config."""
        return self.config.shuffle_buffer_size

    @property
    def selected_keys(self) -> set[str] | None:
        """Optional key-include filter from source config."""
        return self.config.include_keys

    @property
    def rejected_keys(self) -> set[str] | None:
        """Optional key-exclude filter from source config."""
        return self.config.exclude_keys

    @property
    def length(self) -> int | None:
        """Dataset length when known."""
        return self._length

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
        iterator = self._iterator
        if iterator is None:
            iterator = iter(self._hf_dataset)
            self._iterator = iterator

        element = next(iterator)
        return converted_filtered_record(
            element,
            self.selected_keys,
            self.rejected_keys,
            hf_to_jax,
        )

    def _repr_extra_fields(self) -> dict[str, Any]:
        """Add source-mode details to the shared representation."""
        return {"streaming": self.is_iterable_mode}

    def element_spec(self) -> Any:
        """Return per-element shape/dtype derived by peeking the backend.

        Streaming sources cannot strip a leading dataset-size dimension because
        each iteration yields one element. The spec is derived by peeking the
        first element from the underlying HuggingFace dataset (without
        consuming the iterator state for normal training) and converting each
        top-level value into a single ``ShapeDtypeStruct``.

        Top-level dict values are treated as single arrays (HuggingFace
        commonly emits Python lists for vector features; those become 1-D
        arrays, not nested per-element scalars).

        The peek operates on the cached ``self._hf_dataset`` (already loaded
        in ``__init__``) so it does not re-trigger downloads and is safe to
        call repeatedly.
        """
        from datarax.utils.spec import array_to_spec  # noqa: PLC0415

        first = next(iter(self._hf_dataset))
        if not isinstance(first, dict):
            return array_to_spec(first)
        return {key: array_to_spec(value) for key, value in first.items()}
