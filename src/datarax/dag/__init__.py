"""
DAG-based pipeline execution for Datarax.

This module provides components for building and executing
directed acyclic graph (DAG) pipelines for data processing.
"""

from datarax.dag.dag_executor import DAGExecutor, from_source, pipeline
from datarax.dag.dag_config import DAGConfig

# Import node types from dag.nodes
from datarax.dag.nodes import (
    Node,
    OperatorNode,
    DataSourceNode,
    BatchNode,
    Sequential,
    Parallel,
    Branch,
    Merge,
    Cache,
    Identity,
    ShuffleNode,
    CacheNode,
    DataLoader,
    SplitFields,
    SplitField,
)


__all__ = [
    "DAGConfig",
    "DAGExecutor",
    "from_source",
    "pipeline",
    # Node types
    "Node",
    "OperatorNode",
    "DataSourceNode",
    "BatchNode",
    "Sequential",
    "Parallel",
    "Branch",
    "Merge",
    "Cache",
    "Identity",
    "ShuffleNode",
    "CacheNode",
    "DataLoader",
    "SplitFields",
    "SplitField",
]
