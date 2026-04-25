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
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from datarax.sources._config_base import SourceConfigBase
from datarax.sources._conversion import tf_to_jax
from datarax.sources._eager_source_ops import (
    converted_filtered_record,
    validate_eager_source_settings,
    validate_positive_optional_int,
    validate_streaming_source_settings,
)
from datarax.sources._source_base import EagerSourceBase, StreamingSourceBase


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import tensorflow_datasets as tfds


# =============================================================================
# TFDS Builder Helpers
# =============================================================================


def _is_read_only_tfds_source(builder: Any) -> bool:
    """Check if a TFDS builder is a ReadOnlyBuilder (from try_gcs or GCS cache).

    ReadOnlyBuilder reads pre-built TFRecords from GCS and needs no
    download_and_prepare() call. We use a string-based check to avoid
    importing the ReadOnlyBuilder class at module level (heavy import).
    """
    return type(builder).__name__ == "ReadOnlyBuilder"


def _configure_protobuf_runtime() -> None:
    """Configure protobuf runtime before importing TensorFlow ecosystem modules."""
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


def _prepare_tfds_builder(
    name: str,
    data_dir: str | None,
    try_gcs: bool,
    download_and_prepare_kwargs: dict[str, Any] | None,
    beam_num_workers: int | None = None,
) -> Any:
    """Create and prepare a TFDS builder, handling ReadOnlyBuilder from GCS.

    Args:
        name: TFDS dataset name
        data_dir: Optional local data directory
        try_gcs: Whether to try loading from GCS
        download_and_prepare_kwargs: Optional kwargs for download_and_prepare
        beam_num_workers: Optional number of Beam DirectRunner workers.
            When set, enables multi-processing mode for parallel dataset
            generation. Useful for large datasets that use Apache Beam
            (e.g., NSynth). None means single-threaded (Beam default).

    Returns:
        A prepared TFDS builder with dataset info available.
    """
    _configure_protobuf_runtime()
    import tensorflow_datasets as tfds

    builder = tfds.builder(name, data_dir=data_dir, try_gcs=try_gcs)

    # ReadOnlyBuilder (from try_gcs when dataset is on GCS) needs no preparation
    if not _is_read_only_tfds_source(builder):
        download_kwargs = download_and_prepare_kwargs or {}

        if beam_num_workers is not None:
            import apache_beam as beam

            beam_options = beam.options.pipeline_options.PipelineOptions(
                direct_num_workers=beam_num_workers,
                direct_running_mode="multi_processing",
            )
            download_config = tfds.download.DownloadConfig(
                beam_options=beam_options,
            )
            builder.download_and_prepare(download_config=download_config, **download_kwargs)
        else:
            builder.download_and_prepare(**download_kwargs)

    return builder


# =============================================================================
# Configuration Classes
# =============================================================================


@dataclass(frozen=True)
class TFDSEagerConfig(SourceConfigBase):
    """Configuration for TFDSEagerSource (loads all data to JAX at init).

    Configuration for eager-loading TensorFlow Datasets into JAX arrays.

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

    try_gcs: bool = False
    shuffle: bool = False
    seed: int = 42  # Integer seed for Grain's index_shuffle
    as_supervised: bool = False
    download_and_prepare_kwargs: dict[str, Any] | None = None
    beam_num_workers: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        validate_eager_source_settings(
            self,
            "TFDSEagerConfig",
            seed=self.seed,
            try_gcs=self.try_gcs,
            data_dir=self.data_dir,
        )

        validate_positive_optional_int(self.beam_num_workers, "beam_num_workers")


@dataclass(frozen=True)
class TFDSStreamingConfig(SourceConfigBase):
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

    try_gcs: bool = False
    shuffle: bool = False
    shuffle_buffer_size: int = 1000
    as_supervised: bool = False
    download_and_prepare_kwargs: dict[str, Any] | None = None
    beam_num_workers: int | None = None
    prefetch_buffer: int = 2  # Fixed, NOT AUTOTUNE

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        validate_streaming_source_settings(self, "TFDSStreamingConfig")

        if self.try_gcs and self.data_dir is not None:
            raise ValueError(
                f"Cannot specify both try_gcs=True and data_dir='{self.data_dir}' "
                "in TFDSStreamingConfig. "
                "try_gcs overrides data_dir to the public GCS bucket (gs://tfds-data/datasets/)."
            )

        validate_positive_optional_int(self.beam_num_workers, "beam_num_workers")


# =============================================================================
# TFDSEagerSource - Loads All Data to JAX at Init
# =============================================================================


class TFDSEagerSource(EagerSourceBase):
    """Eager-loading TFDS source for small/medium datasets.

    Loads ALL data to JAX arrays at initialization, then operates like a
    MemorySource with pure JAX operations. Use for datasets that fit in
    ~10% of device VRAM.

    Key Features:
        - One-time TF→JAX conversion at init (DLPack zero-copy when possible)
        - Pure JAX iteration after init (no TF threads during training)
        - O(1) memory shuffling via Grain's index_shuffle (Feistel cipher)
        - Full checkpointing support (indices only, no external state)
        - Supports `as_supervised` mode and key filtering

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
    data: dict[str, jax.Array]

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
        self._is_random_order = config.shuffle
        self._seed = config.seed
        self.as_supervised = config.as_supervised
        self.include_keys = config.include_keys
        self.exclude_keys = config.exclude_keys

        # Load dataset info BEFORE loading data (for get_dataset_info)
        self._dataset_info = self._load_dataset_info_from_backend(config)

        # Load ALL data to JAX arrays at init
        self.data = nnx.data(self._load_all_from_backend_to_jax(config))

        # Clean up TF resources completely
        self._cleanup_tf()

        # State for iteration (like MemorySource)
        first_key = next(iter(self.data.keys()))
        self.length = self.data[first_key].shape[0]
        self.index = nnx.Variable(0)
        self.epoch = nnx.Variable(0)

    def _load_dataset_info_from_backend(self, config: TFDSEagerConfig) -> tfds.core.DatasetInfo:
        """Load and cache dataset info before cleanup.

        Args:
            config: Source configuration

        Returns:
            TFDS DatasetInfo object
        """
        # name validated non-None by config __post_init__
        name = config.name
        assert name is not None  # noqa: S101 (invariant, not control flow)
        builder = _prepare_tfds_builder(
            name,
            config.data_dir,
            config.try_gcs,
            config.download_and_prepare_kwargs,
            beam_num_workers=config.beam_num_workers,
        )
        return builder.info

    def _load_all_from_backend_to_jax(self, config: TFDSEagerConfig) -> dict[str, jax.Array]:
        """Load entire dataset to JAX arrays using DLPack.

        This is the core of the eager-loading strategy. All TF operations
        happen here at init time, so training loops are pure JAX.

        Args:
            config: Source configuration

        Returns:
            Dictionary mapping keys to JAX arrays
        """
        _configure_protobuf_runtime()
        import tensorflow_datasets as tfds

        # name validated non-None by config __post_init__
        name = config.name
        assert name is not None  # noqa: S101 (invariant, not control flow)

        ds = tfds.load(
            name,
            split=config.split,
            data_dir=config.data_dir,
            as_supervised=config.as_supervised,
            try_gcs=config.try_gcs,
        )

        # Collect all data
        arrays: dict[str, list[Any]] = {}
        for tf_element in ds:  # type: ignore[union-attr]
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

            tf.keras.backend.clear_session()  # type: ignore[reportAttributeAccessIssue]
        except ImportError:
            pass
        gc.collect()


# =============================================================================
# TFDSStreamingSource - Thin Wrapper for Large Datasets
# =============================================================================


class TFDSStreamingSource(StreamingSourceBase):
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

        self.dataset_name = config.name
        self.split_name = config.split
        self._is_random_order = config.shuffle
        self.as_supervised = config.as_supervised
        self.include_keys = config.include_keys
        self.exclude_keys = config.exclude_keys

        # name/split validated non-None by config __post_init__
        name = config.name
        assert name is not None  # noqa: S101 (invariant, not control flow)

        # Load builder and info
        builder = _prepare_tfds_builder(
            name,
            config.data_dir,
            config.try_gcs,
            config.download_and_prepare_kwargs,
            beam_num_workers=config.beam_num_workers,
        )
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
        split = config.split
        try:
            assert split is not None  # noqa: S101 (invariant, not control flow)
            split_base = split.split("[")[0]  # Handle splits like "train[:1000]"
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
        iterator = self._iterator
        if iterator is None:
            iterator = iter(self._tf_dataset)
            self._iterator = iterator

        tf_element = next(iterator)

        # Handle as_supervised tuple format
        if self.as_supervised and isinstance(tf_element, tuple):
            tf_element = {"image": tf_element[0], "label": tf_element[1]}

        return converted_filtered_record(
            tf_element,
            self.include_keys,
            self.exclude_keys,
            tf_to_jax,
        )
