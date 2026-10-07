"""Datarax data source components.

This module provides data source components for loading data with a clean
architectural separation between **eager** and **streaming** sources:

**Eager Sources** (for small/medium datasets):
    - EagerSource, the base of MemorySource, TFDSEagerSource and HFEagerSource
    - Load ALL data at initialization as host NumPy columns, with each record's
      strings and objects kept beside them as its provenance (TFDSEagerSource reads a
      copy prepared as ArrayRecord, without TensorFlow)
    - One stateless host read, ``get_batch(indices, epochs=...)``, returning a ``Batch``
    - Ideal for: MNIST, CIFAR-10, Fashion-MNIST, small custom datasets

**Streams** (for datasets too large to hold):
    - StreamingSourceBase, the public base of TFDSStreamingSource and HFStreamingSource
    - Read forward one pass after another, in an order the pipeline's key chooses
    - Return host ``Batch``es named by the stream: the ids it reports (TFDS's ``tfds_id``) or
      arrival ordinals (HuggingFace); strings and objects beside the batch, never in it
    - Ideal for: ImageNet, The Pile, large-scale datasets

**Factory Functions**:
    - from_tfds(): the eager source for a copy prepared as ArrayRecord, the stream for TFRecord
    - from_hf(): the eager source, or the stream with ``streaming=True``
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from datarax.sources._source_base import StreamChunk, StreamingSourceBase
from datarax.sources.eager_source import EagerSource
from datarax.sources.memory_source import MemorySource, MemorySourceConfig
from datarax.sources.mixed_source import MixDataSourcesConfig, MixDataSourcesNode
from datarax.sources.source_ops import resolve_wrapped_indices
from datarax.sources.streaming_disk_source import StreamingDiskSource, StreamingDiskSourceConfig


# Type-checking imports for static analysis (not executed at runtime)
if TYPE_CHECKING:
    from datarax.core.data_source import DataSourceModule
    from datarax.sources.array_record_source import (
        ArrayRecordSourceConfig,
        ArrayRecordSourceModule,
    )
    from datarax.sources.hf_source import (
        HFEagerConfig,
        HFEagerSource,
        HFStreamingConfig,
        HFStreamingSource,
    )
    from datarax.sources.tfds_source import (
        TFDSEagerConfig,
        TFDSEagerSource,
        TFDSStreamingConfig,
        TFDSStreamingSource,
    )

# Lazy imports for modules with heavy dependencies (TensorFlow, HuggingFace)
# This prevents import hangs on macOS ARM64 and speeds up import time
_lazy_imports = {
    # ArrayRecord source (requires grain)
    "ArrayRecordSourceModule": "datarax.sources.array_record_source",
    "ArrayRecordSourceConfig": "datarax.sources.array_record_source",
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


def __getattr__(name: str) -> Any:
    """Lazy import for TFDS and HF sources to avoid heavy dependency loading."""
    if name in _lazy_imports:
        import importlib

        module = importlib.import_module(_lazy_imports[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazy imports in dir() for discoverability."""
    return list(__all__)


# =============================================================================
# Factory Functions
# =============================================================================


def from_tfds(
    name: str,
    split: str,
    *,
    data_dir: str | None = None,
    in_memory: bool = True,
    as_supervised: bool = False,
    include_keys: set[str] | None = None,
    exclude_keys: set[str] | None = None,
) -> DataSourceModule:
    """Create the TFDS source that reads the copy prepared in ``data_dir``, by its format.

    - A copy prepared as ArrayRecord has random access. With ``in_memory=True`` (the default)
      ``TFDSEagerSource`` decodes it into host columns at init; with ``in_memory=False`` an
      ``ArrayRecordSourceModule`` reads and decodes each batch's records when the pipeline reads
      them, for a split larger than RAM, at the cost of decoding every record every epoch.
    - A copy prepared as TFRecord (TFDS's default format) is streamed by
      ``TFDSStreamingSource``, which holds no split in memory, whatever ``in_memory`` says.

    Both read without TensorFlow.

    Neither prepares a dataset; a split that is not prepared is refused, naming the call that
    prepares it as ArrayRecord. The order records are served in belongs to the pipeline
    (``Pipeline(shuffle=...)``).

    Args:
        name: TFDS dataset name (e.g., "mnist", "cifar10", "imagenet2012")
        split: Dataset split (e.g., "train", "test", "train[:1000]")
        data_dir: Optional directory where the dataset is prepared
        in_memory: Whether an ArrayRecord copy is decoded into memory at init (True) or read
            per batch (False)
        as_supervised: If True, keeps only the supervised features, under their own names
        include_keys: Optional set of keys to include
        exclude_keys: Optional set of keys to exclude

    Returns:
        TFDSEagerSource or ArrayRecordSourceModule for an ArrayRecord copy, TFDSStreamingSource
        for a TFRecord copy.

    Example:
        ```python
        from datarax.sources import from_tfds

        source = from_tfds("mnist", "train")  # prepared as ArrayRecord: the eager source
        ```
    """
    from datarax.sources.tfds_source import (
        per_batch_split,
        TFDSEagerConfig,
        TFDSEagerSource,
        TFDSStreamingConfig,
        TFDSStreamingSource,
        tfrecord_only,
    )

    if tfrecord_only(name, data_dir):
        return TFDSStreamingSource(
            TFDSStreamingConfig(
                name=name,
                split=split,
                data_dir=data_dir,
                as_supervised=as_supervised,
                include_keys=include_keys,
                exclude_keys=exclude_keys,
            )
        )
    if not in_memory:
        return per_batch_split(
            name,
            split,
            data_dir=data_dir,
            as_supervised=as_supervised,
            include_keys=include_keys,
            exclude_keys=exclude_keys,
        )
    return TFDSEagerSource(
        TFDSEagerConfig(
            name=name,
            split=split,
            data_dir=data_dir,
            as_supervised=as_supervised,
            include_keys=include_keys,
            exclude_keys=exclude_keys,
        )
    )


def from_hf(
    name: str,
    split: str,
    *,
    streaming: bool = False,
    data_dir: str | None = None,
    cache_dir: str | None = None,
    include_keys: set[str] | None = None,
    exclude_keys: set[str] | None = None,
    download_kwargs: dict | None = None,
) -> DataSourceModule:
    """Create a HuggingFace source: the eager source, or the stream when asked.

    ``HFEagerSource`` loads the dataset whole into host columns at init; with ``streaming=True``,
    ``HFStreamingSource`` streams it with HuggingFace's streaming mode, for datasets too large to
    hold. The order records are served in belongs to the pipeline (``Pipeline(shuffle=...)``).

    Args:
        name: HuggingFace dataset name (e.g., "mnist", "imdb", "allenai/c4")
        split: Dataset split (e.g., "train", "test")
        streaming: Whether to stream the dataset rather than load it whole
        data_dir: Optional folder inside the dataset's repository whose data files are
            loaded (``datasets.load_dataset``'s ``data_dir``), not a storage location
        cache_dir: Optional folder where downloaded files are cached
        include_keys: Optional set of keys to include
        exclude_keys: Optional set of keys to exclude
        download_kwargs: Optional kwargs for datasets.load_dataset

    Returns:
        HFEagerSource, or HFStreamingSource with ``streaming=True``.

    Example:
        ```python
        from datarax.sources import from_hf

        source = from_hf("ylecun/mnist", "train")  # loaded whole
        stream = from_hf("allenai/c4", "train", streaming=True, download_kwargs={"name": "en"})
        ```
    """
    from datarax.sources.hf_source import (
        HFEagerConfig,
        HFEagerSource,
        HFStreamingConfig,
        HFStreamingSource,
    )

    if streaming:
        return HFStreamingSource(
            HFStreamingConfig(
                name=name,
                split=split,
                data_dir=data_dir,
                cache_dir=cache_dir,
                include_keys=include_keys,
                exclude_keys=exclude_keys,
                download_kwargs=download_kwargs,
            )
        )
    return HFEagerSource(
        HFEagerConfig(
            name=name,
            split=split,
            data_dir=data_dir,
            cache_dir=cache_dir,
            include_keys=include_keys,
            exclude_keys=exclude_keys,
            download_kwargs=download_kwargs,
        )
    )


__all__ = [
    # The in-memory base and the memory source (always available)
    "EagerSource",
    # The base every stream builds on
    "StreamingSourceBase",
    "StreamChunk",
    "MemorySource",
    "MemorySourceConfig",
    # Mixed source
    "MixDataSourcesNode",
    "MixDataSourcesConfig",
    # Array record source (lazy — requires grain)
    "ArrayRecordSourceModule",
    "ArrayRecordSourceConfig",
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
    # The memory-mapped on-disk source
    "StreamingDiskSource",
    "StreamingDiskSourceConfig",
    # Factory functions
    "from_tfds",
    "from_hf",
    # Helpers a source is built from
    "resolve_wrapped_indices",
]
