"""Datarax: A high-performance data pipeline framework for JAX.

Datarax provides a JAX-native solution for constructing complex data pipelines
for machine learning with JAX, leveraging the full potential of JAX's
Just-In-Time (JIT) compilation, automatic differentiation, and hardware
acceleration capabilities.
"""

from importlib.metadata import version as _installed_version

# Host-to-device prefetching
from datarax.control.prefetcher import prefetch_to_device

# Core modules
from datarax.core.batcher import BatcherModule
from datarax.core.data_source import DataSourceModule

# Record and batch types
from datarax.core.element_batch import Batch, Element
from datarax.core.host_resources import HostResources
from datarax.core.operator import OperatorModule
from datarax.core.sampler import SamplerModule
from datarax.core.temporal import TimeSeriesSpec

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

# Utilities
from datarax.utils.multirate import multirate_align


__version__ = _installed_version("datarax")

__all__ = [
    # Record and batch types
    "Batch",
    "Element",
    # Core modules
    "BatcherModule",
    "DataSourceModule",
    "OperatorModule",
    "SamplerModule",
    # Pipeline (linear stages + Pipeline.from_dag for branching) and its host stage's budget
    "Pipeline",
    "HostResources",
    # Host-to-device prefetching
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
