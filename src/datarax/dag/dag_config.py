"""
DAG-based configuration system for Datarax.

Supports both TOML/YAML configuration files and
programmatic configuration via Python dicts.
"""

from typing import Any
import toml
import yaml

from datarax.dag.dag_executor import DAGExecutor
from datarax.dag.nodes import (
    DataSourceNode,
    BatchNode,
    OperatorNode,
    ShuffleNode,
    CacheNode,
    Sequential,
    Parallel,
    Node,
)
from datarax.config.registry import create_component_from_config


class DAGConfig:
    """Configuration builder for DAG pipelines.

    Supports complex DAG topologies defined in configuration files.

    Example TOML configuration:
        ```toml
        [pipeline]
        name = "training_pipeline"

        [[nodes]]
        id = "source"
        type = "DataSource"
        class = "TFDSSource"
        params = {dataset = "mnist", split = "train"}

        [[nodes]]
        id = "batch"
        type = "BatchNode"
        params = {batch_size = 32, drop_remainder = true}

        [[nodes]]
        id = "augment"
        type = "Transform"
        class = "RandomFlip"

        [[edges]]
        from = "source"
        to = "batch"

        [[edges]]
        from = "batch"
        to = "augment"
        ```
    """

    @classmethod
    def from_toml(cls, filepath: str) -> DAGExecutor:
        """Load DAG configuration from TOML file.

        Args:
            filepath: Path to TOML configuration

        Returns:
            Configured DAGExecutor
        """
        with open(filepath, "r") as f:
            config = toml.load(f)

        return cls.from_dict(config)

    @classmethod
    def from_yaml(cls, filepath: str) -> DAGExecutor:
        """Load DAG configuration from YAML file.

        Args:
            filepath: Path to YAML configuration

        Returns:
            Configured DAGExecutor
        """
        with open(filepath, "r") as f:
            config = yaml.safe_load(f)

        return cls.from_dict(config)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> DAGExecutor:
        """Build DAG from dictionary configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Configured DAGExecutor
        """
        pipeline_config = config.get("pipeline", {})
        nodes_config = config.get("nodes", [])
        edges_config = config.get("edges", [])

        # Create executor
        executor = DAGExecutor(
            name=pipeline_config.get("name", "DAGPipeline"),
            enable_caching=pipeline_config.get("enable_caching", True),
            jit_compile=pipeline_config.get("jit_compile", False),
            enforce_batch=pipeline_config.get("enforce_batch", True),
        )

        # Build nodes
        nodes: dict[str, Node] = {}
        for node_cfg in nodes_config:
            node = cls._create_node(node_cfg, nodes)
            nodes[node_cfg["id"]] = node

        # Build graph from edges
        if edges_config:
            graph = cls._build_graph_from_edges(nodes, edges_config)
            executor.graph = graph
        else:
            # Linear pipeline - add nodes in order
            for node_cfg in nodes_config:
                executor.add(nodes[node_cfg["id"]])

        return executor

    @classmethod
    def _create_node(cls, node_config: dict[str, Any], nodes: dict[str, Node]) -> Node:
        """Create a node from configuration.

        Args:
            node_config: Node configuration dict

        Returns:
            Configured node
        """
        node_type = node_config["type"]
        params = node_config.get("params", {})

        if node_type == "DataSource":
            # Create data source from registry
            source_class = node_config["class"]
            source = create_component_from_config("source", source_class, params)
            return DataSourceNode(source, name=node_config.get("id"))

        elif node_type == "BatchNode":
            return BatchNode(**params, name=node_config.get("id"))

        elif node_type == "Operator":
            # Create operator from registry
            operator_class = node_config["class"]
            operator = create_component_from_config("operator", operator_class, params)
            return OperatorNode(operator, name=node_config.get("id"))

        elif node_type == "ShuffleNode":
            return ShuffleNode(**params, name=node_config.get("id"))

        elif node_type == "CacheNode":
            inner_id = params.pop("inner")
            inner: Node = nodes[inner_id]
            return CacheNode(inner, **params, name=node_config.get("id"))

        elif node_type == "Parallel":
            node_ids = params["nodes"]
            parallel_nodes: list[Node] = [nodes[nid] for nid in node_ids]
            return Parallel(parallel_nodes)

        elif node_type == "Sequential":
            node_ids = params["nodes"]
            seq_nodes: list[Node] = [nodes[nid] for nid in node_ids]
            return Sequential(seq_nodes)

        else:
            raise ValueError(f"Unknown node type: {node_type}")

    @classmethod
    def _build_graph_from_edges(cls, nodes: dict[str, Node], edges: list[dict[str, str]]) -> Node:
        """Build graph structure from edges.

        Args:
            nodes: Dictionary of node id -> node
            edges: List of edge definitions

        Returns:
            Root node of the graph
        """
        # Build adjacency list
        graph = {node_id: [] for node_id in nodes}

        for edge in edges:
            from_id = edge["from"]
            to_id = edge["to"]
            graph[from_id].append(to_id)

        # Find root (node with no incoming edges)
        incoming = set()
        for targets in graph.values():
            incoming.update(targets)

        roots = [nid for nid in nodes if nid not in incoming]

        if len(roots) != 1:
            raise ValueError(f"Expected single root, found {len(roots)}")

        # Build graph recursively
        def build_subgraph(node_id: str) -> Node:
            node = nodes[node_id]
            children = graph[node_id]

            if not children:
                return node
            elif len(children) == 1:
                # Sequential connection
                child = build_subgraph(children[0])
                return node >> child
            else:
                # Parallel branches
                branches = [build_subgraph(cid) for cid in children]
                return node >> Parallel(branches)

        return build_subgraph(roots[0])
