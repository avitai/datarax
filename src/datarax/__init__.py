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

# Configuration
from datarax.dag.dag_config import DAGConfig

# DAG components
from datarax.dag.dag_executor import DAGExecutor, from_source, pipeline
from datarax.dag.nodes import (
    BatchNode,
    Branch,
    CacheNode,
    DataSourceNode,
    DifferentiableRebatchImpl,
    FastRebatchImpl,
    GradientTransparentRebatchImpl,
    Merge,
    Node,
    OperatorNode,
    Parallel,
    rebatch,
    RebatchNode,
    Sequential,
    ShuffleNode,
    SplitField,
)

# Distributed utilities
from datarax.distributed import prefetch_to_device

# Monitoring
from datarax.monitoring.dag_monitor import monitored_pipeline, MonitoredDAGExecutor

# Types
from datarax.typing import Batch, Element


__version__ = "0.1.1"

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
    # DAG API
    "DAGExecutor",
    "pipeline",
    "from_source",
    # Nodes
    "Node",
    "DataSourceNode",
    "BatchNode",
    "OperatorNode",
    "ShuffleNode",
    "CacheNode",
    "Sequential",
    "Parallel",
    "Branch",
    "Merge",
    "SplitField",
    "rebatch",
    "RebatchNode",
    "DifferentiableRebatchImpl",
    "FastRebatchImpl",
    "GradientTransparentRebatchImpl",
    # Configuration
    "DAGConfig",
    # Monitoring
    "MonitoredDAGExecutor",
    "monitored_pipeline",
    # Distributed utilities
    "prefetch_to_device",
]
