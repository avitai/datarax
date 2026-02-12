"""DAG node types for building data processing pipelines.

Provides the node graph primitives — Sequential, Parallel, Branch, Cache,
DataSource, Batch, Operator, Shuffle, Prefetch, Sampler, Sharder, DataLoader,
field-splitting, and rebatching nodes.
"""

from datarax.dag.nodes.base import Node
from datarax.dag.nodes.control_flow import (
    Identity,
    Sequential,
    Parallel,
    Branch,
    Merge,
    MergeBatchNode,
    FusedOperatorNode,
    branch,
    parallel,
)
from datarax.dag.nodes.caching import Cache, CacheNode
from datarax.dag.nodes.data_source import (
    DataSourceNode,
    BatchNode,
    OperatorNode,
    ShuffleNode,
    PrefetchNode,
    SamplerNode,
    SharderNode,
)
from datarax.dag.nodes.loaders import DataLoader, dataloader
from datarax.dag.nodes.field_operators import SplitFields, SplitField
from datarax.dag.nodes.rebatch import (
    RebatchNode,
    DifferentiableRebatchImpl,
    FastRebatchImpl,
    GradientTransparentRebatchImpl,
    rebatch,
)

__all__ = [
    "Node",
    "Identity",
    "Sequential",
    "Parallel",
    "Branch",
    "Merge",
    "MergeBatchNode",
    "FusedOperatorNode",
    "Cache",
    "CacheNode",
    "DataSourceNode",
    "BatchNode",
    "OperatorNode",
    "ShuffleNode",
    "PrefetchNode",
    "SamplerNode",
    "SharderNode",
    "DataLoader",
    "dataloader",
    "SplitFields",
    "SplitField",
    "RebatchNode",
    "DifferentiableRebatchImpl",
    "FastRebatchImpl",
    "GradientTransparentRebatchImpl",
    "rebatch",
    "branch",
    "parallel",
]
