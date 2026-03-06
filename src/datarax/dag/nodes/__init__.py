"""DAG node types for building data processing pipelines.

Provides the node graph primitives — Sequential, Parallel, Branch, Cache,
DataSource, Batch, Operator, Shuffle, Prefetch, Sampler, Sharder, DataLoader,
field-splitting, and rebatching nodes.
"""

from datarax.dag.nodes.base import Node
from datarax.dag.nodes.caching import Cache, CacheNode
from datarax.dag.nodes.control_flow import (
    Branch,
    branch,
    FusedOperatorNode,
    Identity,
    Merge,
    MergeBatchNode,
    Parallel,
    parallel,
    Sequential,
)
from datarax.dag.nodes.data_source import (
    BatchNode,
    DataSourceNode,
    OperatorNode,
    PrefetchNode,
    SamplerNode,
    SharderNode,
    ShuffleNode,
)
from datarax.dag.nodes.field_operators import SplitField, SplitFields
from datarax.dag.nodes.loaders import DataLoader, dataloader
from datarax.dag.nodes.rebatch import (
    DifferentiableRebatchImpl,
    FastRebatchImpl,
    GradientTransparentRebatchImpl,
    rebatch,
    RebatchNode,
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
