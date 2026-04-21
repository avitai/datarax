"""
DAG-based pipeline execution for Datarax.

This module provides components for building and executing
directed acyclic graph (DAG) pipelines for data processing.
"""

from datarax.dag.dag_config import DAGConfig
from datarax.dag.dag_executor import build_source_pipeline, DAGExecutor, pipeline

# Import node types from dag.nodes
from datarax.dag.nodes import (
    BatchNode,
    Branch,
    Cache,
    CacheNode,
    DataLoader,
    DataSourceNode,
    Identity,
    Merge,
    Node,
    OperatorNode,
    Parallel,
    Sequential,
    ShuffleNode,
    SplitField,
    SplitFields,
)


__all__ = [
    "DAGConfig",
    "DAGExecutor",
    "build_source_pipeline",
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
