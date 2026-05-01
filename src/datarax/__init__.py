"""Datarax: A high-performance data pipeline framework for JAX.

Datarax provides a JAX-native solution for constructing complex data pipelines
for machine learning with JAX, leveraging the full potential of JAX's
Just-In-Time (JIT) compilation, automatic differentiation, and hardware
acceleration capabilities.
"""

# Core modules
from datarax.core.batcher import BatcherModule
from datarax.core.data_source import DataSourceModule
from datarax.core.operator import OperatorModule
from datarax.core.sampler import SamplerModule
from datarax.core.sharder import SharderModule
from datarax.core.temporal import TimeSeriesSpec

# Distributed utilities
from datarax.distributed import prefetch_to_device

# Pipeline (DAG composition + iteration + scan)
from datarax.pipeline import Pipeline

# Samplers
from datarax.samplers.buffer_sampler import BufferSampler, BufferSamplerConfig
from datarax.samplers.sliding_window_sampler import (
    SlidingWindowSampler,
    SlidingWindowSamplerConfig,
)

# Streaming source
from datarax.sources.streaming_disk_source import (
    StreamingDiskSource,
    StreamingDiskSourceConfig,
)

# Types
from datarax.typing import Batch, Element

# Utilities
from datarax.utils.multirate import multirate_align


__version__ = "0.1.3"

__all__ = [
    # Type aliases
    "Batch",
    "Element",
    # Core modules
    "BatcherModule",
    "DataSourceModule",
    "OperatorModule",
    "SamplerModule",
    "SharderModule",
    # Pipeline (linear stages + Pipeline.from_dag for branching)
    "Pipeline",
    # Distributed utilities
    "prefetch_to_device",
    # Time-series contracts
    "TimeSeriesSpec",
    # Samplers
    "BufferSampler",
    "BufferSamplerConfig",
    "SlidingWindowSampler",
    "SlidingWindowSamplerConfig",
    # Streaming source
    "StreamingDiskSource",
    "StreamingDiskSourceConfig",
    # Utilities
    "multirate_align",
]
