"""Datarax data source components.

This module provides data source components for loading data with a clean
architectural separation between **eager** and **streaming** sources:

**Eager Sources** (for small/medium datasets):
    - TFDSEagerSource, HFEagerSource
    - Load ALL data to JAX arrays at initialization
    - Pure JAX iteration (no external framework overhead during training)
    - O(1) memory shuffling via Grain's index_shuffle
    - Ideal for: MNIST, CIFAR-10, Fashion-MNIST, small custom datasets

**Streaming Sources** (for large datasets):
    - TFDSStreamingSource, HFStreamingSource
    - Thin wrapper around external iterators
    - DLPack zero-copy conversion for efficient data transfer
    - Ideal for: ImageNet, The Pile, large-scale datasets

**Factory Functions**:
    - from_tfds(): Auto-detect eager vs streaming based on dataset size
    - from_hf(): Auto-detect eager vs streaming based on dataset size
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from datarax.sources.mixed_source import MixDataSourcesConfig, MixDataSourcesNode
from datarax.sources.array_record_source import ArrayRecordSourceModule

# Type-checking imports for static analysis (not executed at runtime)
if TYPE_CHECKING:
    import flax.nnx as nnx

    from datarax.sources.tfds_source import (
        TFDSEagerConfig,
        TFDSEagerSource,
        TFDSStreamingConfig,
        TFDSStreamingSource,
    )
    from datarax.sources.hf_source import (
        HFEagerConfig,
        HFEagerSource,
        HFStreamingConfig,
        HFStreamingSource,
    )
    from datarax.core.data_source import DataSourceModule

# Lazy imports for modules with heavy dependencies (TensorFlow, HuggingFace)
# This prevents import hangs on macOS ARM64 and speeds up import time
_lazy_imports = {
    # TFDS sources
    "TFDSEagerSource": "datarax.sources.tfds_source",
    "TFDSEagerConfig": "datarax.sources.tfds_source",
    "TFDSStreamingSource": "datarax.sources.tfds_source",
    "TFDSStreamingConfig": "datarax.sources.tfds_source",
    # HF sources
    "HFEagerSource": "datarax.sources.hf_source",
    "HFEagerConfig": "datarax.sources.hf_source",
    "HFStreamingSource": "datarax.sources.hf_source",
    "HFStreamingConfig": "datarax.sources.hf_source",
}


def __getattr__(name: str):
    """Lazy import for TFDS and HF sources to avoid heavy dependency loading."""
    if name in _lazy_imports:
        import importlib

        module = importlib.import_module(_lazy_imports[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Include lazy imports in dir() for discoverability."""
    return list(__all__)


# =============================================================================
# Factory Functions
# =============================================================================


def from_tfds(
    name: str,
    split: str,
    *,
    eager: bool | None = None,
    shuffle: bool = False,
    seed: int = 42,
    rngs: nnx.Rngs | None = None,
    data_dir: str | None = None,
    try_gcs: bool = False,
    as_supervised: bool = False,
    download_and_prepare_kwargs: dict | None = None,
    beam_num_workers: int | None = None,
    include_keys: set[str] | None = None,
    exclude_keys: set[str] | None = None,
) -> DataSourceModule:
    """Create a TFDS source, choosing eager or streaming based on size.

    This factory function automatically selects the optimal source type:
    - TFDSEagerSource for datasets < 1GB (loads all to JAX at init)
    - TFDSStreamingSource for datasets >= 1GB (streams with fixed prefetch)

    Args:
        name: TFDS dataset name (e.g., "mnist", "cifar10", "imagenet2012")
        split: Dataset split (e.g., "train", "test", "train[:1000]")
        eager: Force eager (True) or streaming (False). None = auto-detect.
        shuffle: Whether to shuffle the dataset
        seed: Integer seed for shuffling (for Grain's index_shuffle)
        rngs: Optional Flax NNX RNG state
        data_dir: Optional directory for dataset storage
        try_gcs: If True, load pre-built data from Google Cloud Storage
            (gs://tfds-data/datasets/). Bypasses local download_and_prepare(),
            which avoids Apache Beam dependencies for datasets like NSynth.
            Mutually exclusive with data_dir.
        as_supervised: If True, returns {"image": ..., "label": ...}
        download_and_prepare_kwargs: Optional kwargs for download_and_prepare
        beam_num_workers: Number of Apache Beam DirectRunner workers for
            parallel dataset generation. Useful for large Beam-based datasets
            (e.g., NSynth). None uses Beam's default (single-threaded).
        include_keys: Optional set of keys to include
        exclude_keys: Optional set of keys to exclude

    Returns:
        TFDSEagerSource for small datasets, TFDSStreamingSource for large.

    Example:
        ```python
        from datarax.sources import from_tfds
        import flax.nnx as nnx

        # Auto-detect: MNIST is small, will use eager
        source = from_tfds("mnist", "train", shuffle=True, rngs=nnx.Rngs(0))

        # Load from GCS (bypasses Apache Beam for datasets like NSynth)
        source = from_tfds("nsynth/gansynth_subset", "train", try_gcs=True)

        # Parallel dataset generation for Beam-based datasets
        source = from_tfds("nsynth", "train", beam_num_workers=4)

        # Force streaming for memory-constrained environments
        source = from_tfds("mnist", "train", eager=False, rngs=nnx.Rngs(0))
        ```
    """
    from datarax.sources.tfds_source import (
        TFDSEagerConfig,
        TFDSEagerSource,
        TFDSStreamingConfig,
        TFDSStreamingSource,
    )

    # Auto-detect based on dataset size if not specified
    if eager is None:
        try:
            import tensorflow_datasets as tfds

            builder = tfds.builder(name, data_dir=data_dir, try_gcs=try_gcs)
            # Get size from dataset info
            split_base = split.split("[")[0]  # Handle "train[:1000]"
            if builder.info.splits and split_base in builder.info.splits:
                size_bytes = builder.info.splits[split_base].num_bytes
                size_mb = size_bytes / 1e6 if size_bytes else 0
                eager = size_mb < 1000  # < 1GB → eager
            else:
                eager = True  # Default to eager for unknown splits
        except Exception:
            eager = True  # Default to eager on errors

    if eager:
        config = TFDSEagerConfig(
            name=name,
            split=split,
            shuffle=shuffle,
            seed=seed,
            data_dir=data_dir,
            try_gcs=try_gcs,
            as_supervised=as_supervised,
            download_and_prepare_kwargs=download_and_prepare_kwargs,
            beam_num_workers=beam_num_workers,
            include_keys=include_keys,
            exclude_keys=exclude_keys,
        )
        return TFDSEagerSource(config, rngs=rngs)
    else:
        config = TFDSStreamingConfig(
            name=name,
            split=split,
            shuffle=shuffle,
            data_dir=data_dir,
            try_gcs=try_gcs,
            as_supervised=as_supervised,
            download_and_prepare_kwargs=download_and_prepare_kwargs,
            beam_num_workers=beam_num_workers,
            include_keys=include_keys,
            exclude_keys=exclude_keys,
        )
        return TFDSStreamingSource(config, rngs=rngs)


def from_hf(
    name: str,
    split: str,
    *,
    eager: bool | None = None,
    streaming: bool | None = None,
    shuffle: bool = False,
    seed: int = 42,
    rngs: nnx.Rngs | None = None,
    data_dir: str | None = None,
    include_keys: set[str] | None = None,
    exclude_keys: set[str] | None = None,
    download_kwargs: dict | None = None,
) -> DataSourceModule:
    """Create a HuggingFace source, choosing eager or streaming based on size.

    This factory function automatically selects the optimal source type:
    - HFEagerSource for datasets < 1GB (loads all to JAX at init)
    - HFStreamingSource for datasets >= 1GB or when streaming=True

    Args:
        name: HuggingFace dataset name (e.g., "mnist", "imdb", "allenai/c4")
        split: Dataset split (e.g., "train", "test")
        eager: Force eager (True) or streaming source (False). None = auto-detect.
        streaming: Use HuggingFace streaming mode (implies eager=False)
        shuffle: Whether to shuffle the dataset
        seed: Integer seed for shuffling (for Grain's index_shuffle)
        rngs: Optional Flax NNX RNG state
        data_dir: Optional directory for dataset storage
        include_keys: Optional set of keys to include
        exclude_keys: Optional set of keys to exclude
        download_kwargs: Optional kwargs for datasets.load_dataset

    Returns:
        HFEagerSource for small datasets, HFStreamingSource for large.

    Example:
        ```python
        from datarax.sources import from_hf
        import flax.nnx as nnx

        # Auto-detect: MNIST is small, will use eager
        source = from_hf("mnist", "train", shuffle=True, rngs=nnx.Rngs(0))

        # Force HuggingFace streaming for large datasets
        source = from_hf("allenai/c4", "train", streaming=True, rngs=nnx.Rngs(0))
        ```
    """
    from datarax.sources.hf_source import (
        HFEagerConfig,
        HFEagerSource,
        HFStreamingConfig,
        HFStreamingSource,
    )

    # If streaming explicitly requested, use streaming source
    if streaming:
        eager = False

    # Auto-detect based on whether HF streaming is commonly used for this dataset
    if eager is None:
        # Default to eager for most datasets
        # Users can explicitly set streaming=True for large datasets
        eager = True

    if eager:
        config = HFEagerConfig(
            name=name,
            split=split,
            shuffle=shuffle,
            seed=seed,
            data_dir=data_dir,
            include_keys=include_keys,
            exclude_keys=exclude_keys,
            download_kwargs=download_kwargs,
        )
        return HFEagerSource(config, rngs=rngs)
    else:
        hf_streaming = streaming if streaming is not None else False
        config = HFStreamingConfig(
            name=name,
            split=split,
            streaming=hf_streaming,
            shuffle=shuffle,
            data_dir=data_dir,
            include_keys=include_keys,
            exclude_keys=exclude_keys,
            download_kwargs=download_kwargs,
        )
        return HFStreamingSource(config, rngs=rngs)


__all__ = [
    # Memory source (always available)
    "MemorySource",
    "MemorySourceConfig",
    # Mixed source
    "MixDataSourcesNode",
    "MixDataSourcesConfig",
    # Array record source
    "ArrayRecordSourceModule",
    # TFDS sources
    "TFDSEagerSource",
    "TFDSEagerConfig",
    "TFDSStreamingSource",
    "TFDSStreamingConfig",
    # HF sources
    "HFEagerSource",
    "HFEagerConfig",
    "HFStreamingSource",
    "HFStreamingConfig",
    # Factory functions
    "from_tfds",
    "from_hf",
]
