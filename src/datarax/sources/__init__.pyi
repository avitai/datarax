"""Datarax data sources; the ArrayRecord, TFDS and HuggingFace names load on first use."""

__all__: list[str]

from ._disk_provenance import DiskProvenance as DiskProvenance
from ._source_base import StreamingSourceBase as StreamingSourceBase
from ._tfrecord import DamagedRecordError as DamagedRecordError
from .array_record_source import (
    ArrayRecordSourceConfig as ArrayRecordSourceConfig,
    ArrayRecordSourceModule as ArrayRecordSourceModule,
)
from .decoded_copy import open_decoded_copy as open_decoded_copy
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
from .stream_iterator import StreamChunk as StreamChunk
from .streaming_disk_source import (
    DiskColumn as DiskColumn,
    DiskRagged as DiskRagged,
    StreamingDiskSource as StreamingDiskSource,
    StreamingDiskSourceConfig as StreamingDiskSourceConfig,
)
from .tfds_source import (
    from_tfds as from_tfds,
    TFDSEagerConfig as TFDSEagerConfig,
    TFDSEagerSource as TFDSEagerSource,
)
from .tfds_streaming_source import (
    TFDSStreamingConfig as TFDSStreamingConfig,
    TFDSStreamingSource as TFDSStreamingSource,
)
