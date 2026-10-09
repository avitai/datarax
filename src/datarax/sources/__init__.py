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

Every export is listed in ``__init__.pyi``, which type checkers read. The sources that need no
optional dependency are imported with the package, so their components are registered
(``datarax.config.registry``); the ArrayRecord, TFDS and HuggingFace names load their module on
first use (scientific-python SPEC 1), so ``import datarax.sources`` imports neither TFDS nor
HuggingFace ``datasets``.
"""

import lazy_loader

from datarax.sources._source_base import (
    StreamChunk as StreamChunk,
    StreamingSourceBase as StreamingSourceBase,
)
from datarax.sources.eager_source import EagerSource as EagerSource
from datarax.sources.memory_source import (
    MemorySource as MemorySource,
    MemorySourceConfig as MemorySourceConfig,
)
from datarax.sources.mixed_source import (
    MixDataSourcesConfig as MixDataSourcesConfig,
    MixDataSourcesNode as MixDataSourcesNode,
)
from datarax.sources.source_ops import resolve_wrapped_indices as resolve_wrapped_indices
from datarax.sources.streaming_disk_source import (
    StreamingDiskSource as StreamingDiskSource,
    StreamingDiskSourceConfig as StreamingDiskSourceConfig,
)


__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)
