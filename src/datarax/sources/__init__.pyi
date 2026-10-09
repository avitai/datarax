"""Datarax data sources; the ArrayRecord, TFDS and HuggingFace names load on first use."""

__all__: list[str]

from ._source_base import (
    StreamChunk as StreamChunk,
    StreamingSourceBase as StreamingSourceBase,
)
from .array_record_source import (
    ArrayRecordSourceConfig as ArrayRecordSourceConfig,
    ArrayRecordSourceModule as ArrayRecordSourceModule,
)
from .eager_source import EagerSource as EagerSource
from .hf_source import (
    from_hf as from_hf,
    HFEagerConfig as HFEagerConfig,
    HFEagerSource as HFEagerSource,
    HFStreamingConfig as HFStreamingConfig,
    HFStreamingSource as HFStreamingSource,
)
from .memory_source import (
    MemorySource as MemorySource,
    MemorySourceConfig as MemorySourceConfig,
)
from .mixed_source import (
    MixDataSourcesConfig as MixDataSourcesConfig,
    MixDataSourcesNode as MixDataSourcesNode,
)
from .source_ops import resolve_wrapped_indices as resolve_wrapped_indices
from .streaming_disk_source import (
    StreamingDiskSource as StreamingDiskSource,
    StreamingDiskSourceConfig as StreamingDiskSourceConfig,
)
from .tfds_source import (
    from_tfds as from_tfds,
    TFDSEagerConfig as TFDSEagerConfig,
    TFDSEagerSource as TFDSEagerSource,
    TFDSStreamingConfig as TFDSStreamingConfig,
    TFDSStreamingSource as TFDSStreamingSource,
)
